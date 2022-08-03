import torch.nn as nn
from gym import spaces
from gym.spaces import Discrete
import torch.nn.functional as F
import collections
import numpy as np

from env.HERec import HERec
from env.HAN import HAN
from env.MCRec import MCRec
from metrics import *
import time
import torch
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from sklearn.metrics import f1_score

from utils import load_data, EarlyStopping

from KGDataLoader import *

STOP = 0

NEG_SIZE_TRAIN = 20
NEG_SIZE_EVAL = 20
NEG_SIZE_RANKING = 499

USER_TYPE = 5
ITEM_TYPE = 0


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


class hgnn_env(object):
    def __init__(self, logger1, logger2, model_name, args, dataset='yelp_data', weight_decay=1e-5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cur_best = 0
        self.args = args
        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        lr = args.lr
        self.lr = lr
        self.task = args.task
        task = self.task
        dataset = args.data_name
        self.dataset = dataset
        self.past_performance = []
        self.init = -1
        self.weight_decay = weight_decay
        global USER_TYPE
        global ITEM_TYPE
        if dataset == 'yelp_data':
            USER_TYPE = 4
            ITEM_TYPE = 0
        elif dataset == 'double_movie':
            USER_TYPE = 5
            ITEM_TYPE = 0
        elif dataset.startswith('TCL'):
            USER_TYPE = 8
            ITEM_TYPE = 0

        if task == 'rec' or task == 'herec' or task == 'mcrec':
            self.etypes_lists = eval(args.mpset)
            self.data = DataLoaderHGNN(logger1, args, dataset)
            self.metapath_transform_dict = self.data.metapath_transform_dict
            data = self.data
            if task == 'rec':
                self.model = HAN(
                    in_size=data.entity_dim,
                    hidden_size=args.hidden_dim,
                    out_size=data.entity_dim,
                    num_heads=args.num_heads,
                    dropout=0,
                    threshold=0.75).to(
                    self.device)
            elif task == 'herec':
                self.model = HERec(data, self.etypes_lists, args, 1)
            elif task == 'mcrec':
                self.neg_num = 4 if self.dataset == 'TCL' else 1
                metapath_attrs = []
                for mp in self.etypes_lists[0]:
                    if len(mp) < 2 or len(mp) > 4:
                        continue
                    metapath_attrs.append((self.neg_num + 1, len(mp) + 1))
                self.model = MCRec(
                    latent_dim=data.entity_dim,
                    att_size=data.entity_dim,
                    feature_size=data.entity_dim,
                    negative_num=self.neg_num,
                    user_num=data.n_users,
                    item_num=data.n_items,
                    metapath_list_attributes=metapath_attrs,
                    layer_size=[512, 256, 128, 64]
                ).to(self.device)
                self.metapath_mcrec_dict = dict()
                self.metapath_mcrec_eval_dict = dict()
                self.metapath_mcrec_test_dict = dict()
                self.eval_user_ids = []
                self.test_user_ids = []
            self.train_data = data.train_graph
            self.train_data.x = self.train_data.x.to(self.device)
            self.train_data.node_idx = self.train_data.node_idx.to(self.device)
            self._set_action_space(len(data.metapath_transform_dict) + 1)
            self.user_policy = None
            self.item_policy = None
            self.eval_neg_dict = collections.defaultdict(list)
            self.test_neg_dict = collections.defaultdict(list)
        elif task == 'classification':
            self.etypes_lists = [[['pf', 'fp']]]
            g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
            val_mask, test_mask = load_data(dataset)
            if hasattr(torch, 'BoolTensor'):
                train_mask = train_mask.bool()
                val_mask = val_mask.bool()
                test_mask = test_mask.bool()

            self.train_data = g.to(self.device)
            self.train_data.x = features.to(self.device)
            self.train_data.e_n_dict = {'pa': ['p', 'a'], 'ap': ['a', 'p'], 'pf': ['p', 'f'], 'fp': ['f', 'p']}
            self.metapath_transform_dict = {1: ['pa', 'ap'], 2: ['pf', 'fp']}
            self.labels = labels.to(self.device)
            self.train_mask = train_mask.to(self.device)
            self.val_mask = val_mask.to(self.device)
            self.test_mask = test_mask.to(self.device)
            self.model = HAN(in_size=features.shape[1],
                             hidden_size=32,
                             out_size=num_classes,
                             num_heads=args.num_heads,
                             dropout=0.1,
                             threshold=0.8).to(self.device)
            self.embedding_func = nn.Linear(num_classes, 32).to(self.device)
            self._set_action_space(3)
            self.policy = None

        self.mpset_eval_dict = dict()
        self.nd_batch_size = args.nd_batch_size
        self.rl_batch_size = args.rl_batch_size
        if task != "herec":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        else:
            self.optimizer = None
        if task == "classification":
            self.optimizer.add_param_group({'params': self.embedding_func.parameters()})
        self.obs = self.reset()
        self._set_observation_space(self.obs)
        # self.W_R = torch.randn(self.data.n_relations + 1, self.data.entity_dim,
        #                        self.data.relation_dim).to(self.device)
        # nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        # self.kg_l2loss_lambda = args.kg_l2loss_lambda

        self.baseline_experience = 1
        logger1.info('Data initialization done')

    def reset_past_performance(self):
        if self.init == -1:
            if self.optimizer:
                self.model.train()
                self.optimizer.zero_grad()
            # self.etypes_lists = [[['2', '1']], [['1', '2']]]
            # self.train_GNN()
            self.init = self.eval_batch()
            if self.task == 'rec':
                self.model.reset()
        self.past_performance = [self.init]

    def evaluate(self, model, g, features, labels, mask, loss_func):
        self.model.eval()
        ids = torch.tensor(range(self.train_data.x.shape[0]))
        with torch.no_grad():
            logits = self.model(g, features, self.etypes_lists[0], self.optimizer, ids, test=True)
        loss = loss_func(logits[mask], labels[mask])
        accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

        return loss, accuracy, micro_f1, macro_f1

    def get_user_embedding(self, u_ids, test=False):
        h = self.model(self.train_data, self.train_data.x[self.data.node_type_list == USER_TYPE], self.etypes_lists[0],
                       self.optimizer, u_ids, test)
        # self.etypes_lists[0] = meta_paths
        return h

    def get_item_embedding(self, i_ids, test=False):
        h = self.model(self.train_data, self.train_data.x[self.data.node_type_list == ITEM_TYPE], self.etypes_lists[1],
                       self.optimizer, i_ids, test)
        # self.etypes_lists[1] = meta_paths
        return h

    def get_all_user_embedding(self, test=False):
        all_user_ids = torch.tensor(range(self.train_data.x[self.data.node_type_list == USER_TYPE].shape[0]))
        return self.get_user_embedding(all_user_ids, test)

    def get_all_item_embedding(self, test=False):
        all_item_ids = torch.tensor(range(self.train_data.x[self.data.node_type_list == ITEM_TYPE].shape[0]))
        return self.get_item_embedding(all_item_ids, test)

    def _set_action_space(self, _max):
        self.action_num = _max
        self.action_space = Discrete(_max)

    def _set_observation_space(self, observation):
        low = np.full(observation.shape, -np.float32('inf'))
        high = np.full(observation.shape, np.float32('inf'))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.etypes_lists = eval(self.args.mpset)
        if self.args.task == 'mcrec':
            self.etypes_lists = [[['2']], [[]]]
        state = self.get_user_state()
        # state = self.train_data.x[0]
        if self.task == 'classification':
            state = self.get_class_state()[0]
        if self.optimizer:
            self.optimizer.zero_grad()
        return state

    def cal_user_state(self):
        state = [0] * (self.data.n_relations + 1)
        for mp in self.etypes_lists[0]:
            for rel in mp:
                state[int(rel)] += 1
        v = np.array(state, dtype=np.float32)
        return np.expand_dims(v / (np.linalg.norm(v) + 1e-16), axis=0)

    def cal_item_state(self):
        state = [0] * (self.data.n_relations + 1)
        for mp in self.etypes_lists[1]:
            for rel in mp:
                state[int(rel)] += 1
        v = np.array(state, dtype=np.float32)
        return np.expand_dims(v / (np.linalg.norm(v) + 1e-16), axis=0)

    def reset_eval_dict(self):
        self.eval_neg_dict = collections.defaultdict(list)

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def sample_state(self, embeds, nodes):
        state = []
        for i in range(self.rl_batch_size):
            index = random.sample(nodes, min(self.nd_batch_size, len(nodes)))
            state.append(F.normalize(torch.mean(embeds[index], 0), dim=0).cpu().detach().numpy())
        return np.array(state)

    def get_user_state(self):
        # nodes = range(self.train_data.x[self.data.node_type_list == USER_TYPE].shape[0])
        # user_embeds = self.get_all_user_embedding()
        # return self.sample_state(user_embeds, nodes)
        return self.cal_user_state()
        # return np.concatenate([self.cal_user_state(), self.sample_state(user_embeds, nodes)], axis=1)

    def user_reset(self):
        self.etypes_lists = eval(self.args.mpset)
        if self.args.task == 'mcrec':
            self.etypes_lists = [[['2']], [[]]]
        state = self.get_user_state()
        # state = self.cal_user_state()
        if self.optimizer:
            self.optimizer.zero_grad()
        return state

    def get_item_state(self):
        # nodes = range(self.train_data.x[self.data.node_type_list == ITEM_TYPE].shape[0])
        # item_embeds = self.get_all_item_embedding()
        # return self.sample_state(item_embeds, nodes)
        return self.cal_item_state()
        # return np.concatenate([self.cal_item_state(), self.sample_state(item_embeds, nodes)], axis=1)

    def item_reset(self):
        self.etypes_lists = eval(self.args.mpset)
        state = self.get_item_state()
        # state = self.cal_item_state()
        if self.optimizer:
            self.optimizer.zero_grad()
        return state

    def get_class_state(self):
        nodes = range(self.train_data.x.shape[0])
        b_ids = torch.tensor(range(self.train_data.x.shape[0]))
        class_embeds = self.model(self.train_data, self.train_data.x, self.etypes_lists[0],
                                  self.optimizer, b_ids, test=False)
        # class_embeds = self.embedding_func(class_embeds)
        return self.sample_state(class_embeds, nodes)

    def class_reset(self):
        self.etypes_lists = [[['pf', 'fp']]]
        state = self.get_class_state()
        self.optimizer.zero_grad()
        return state

    def rec_step(self, actions, logger1, logger2, test, type):
        if self.optimizer:
            self.model.train()
            self.optimizer.zero_grad()
        tmpmp = copy.deepcopy(self.etypes_lists)
        done_list = [False] * len(actions)
        next_state, reward, val_acc = [], [], []
        if test:
            print(actions)
        for i, act in enumerate(actions):
            if act == STOP:
                done_list[i] = True
                if self.args.task == 'herec':
                    self.model = HERec(self.data, self.etypes_lists, self.args, steps=1)
                elif self.args.task == 'mcrec':
                    metapath_attrs = []
                    for mp in self.etypes_lists[0]:
                        if len(mp) < 2 or len(mp) > 4:
                            continue
                        metapath_attrs.append((self.neg_num + 1, len(mp) + 1))
                    self.model = MCRec(
                        latent_dim=self.data.entity_dim,
                        att_size=self.data.entity_dim,
                        feature_size=self.data.entity_dim,
                        negative_num=self.neg_num,
                        user_num=self.data.n_users,
                        item_num=self.data.n_items,
                        metapath_list_attributes=metapath_attrs,
                        layer_size=[512, 256, 128, 64]
                    ).to(self.device)
                    self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
                if test:
                    self.train_GNN(test=True)
                else:
                    self.train_GNN()
            else:
                augment_mp = self.metapath_transform_dict[act]
                for i in range(len(self.etypes_lists[type[0]])):
                    mp = self.etypes_lists[type[0]][i]
                    if len(mp) < (4 if self.args.task != 'mcrec' else 3):
                        if self.train_data.e_n_dict[mp[-1]][1] == self.train_data.e_n_dict[augment_mp[0]][0]:
                            self.etypes_lists[type[0]].append(mp[:])
                            mp.extend(augment_mp)
                        else:
                            if self.train_data.e_n_dict[mp[0]][0] == self.train_data.e_n_dict[augment_mp[-1]][1]:
                                self.etypes_lists[type[0]].append(mp[:])
                                mp[0:0] = augment_mp
                            else:
                                for inx in range(len(mp)):
                                    rel = mp[inx]
                                    if self.train_data.e_n_dict[rel][1] == self.train_data.e_n_dict[augment_mp[0]][0]:
                                        self.etypes_lists[type[0]].append(mp[:])
                                        mp[inx + 1:inx + 1] = augment_mp
                                        break

                if self.train_data.e_n_dict[augment_mp[0]][0] == type[1] and self.args.task != 'mcrec':
                    self.etypes_lists[type[0]].append(augment_mp)
                self.etypes_lists[type[0]] = list(
                    map(lambda x: list(x), set(map(lambda x: tuple(x), self.etypes_lists[type[0]]))))

                if self.args.task == 'herec':
                    self.model = HERec(self.data, self.etypes_lists, self.args, steps=1)
                    if len(self.eval_neg_dict) != 0:
                        self.model.eval_neg_dict = self.eval_neg_dict
                    if len(self.test_neg_dict) != 0:
                        self.model.test_neg_dict = self.test_neg_dict
                elif self.args.task == 'mcrec':
                    metapath_attrs = []
                    for mp in self.etypes_lists[0]:
                        if len(mp) < 2 or len(mp) > 4:
                            continue
                        metapath_attrs.append((self.neg_num + 1, len(mp) + 1))
                    self.model = MCRec(
                        latent_dim=self.data.entity_dim,
                        att_size=self.data.entity_dim,
                        feature_size=self.data.entity_dim,
                        negative_num=self.neg_num,
                        user_num=self.data.n_users,
                        item_num=self.data.n_items,
                        metapath_list_attributes=metapath_attrs,
                        layer_size=[512, 256, 128, 64]
                    ).to(self.device)
                    self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
                if str(self.etypes_lists) not in self.mpset_eval_dict:
                    self.train_GNN(act, test)
            if not test:
                if str(self.etypes_lists) not in self.mpset_eval_dict:
                    val_precision = self.eval_batch()
                    self.mpset_eval_dict[str(self.etypes_lists)] = val_precision
                else:
                    val_precision = self.mpset_eval_dict[str(self.etypes_lists)]
            else:
                # val_precision = self.eval_batch(NEG_SIZE_EVAL)
                val_precision = 0

            if len(self.past_performance) == 0:
                self.past_performance.append(val_precision)

            baseline = np.mean(np.array(self.past_performance[-self.baseline_experience:]))
            rew = 5 * (val_precision - baseline)
            if rew > 0.5:
                rew = 0.5
            elif rew < -0.5:
                rew = -0.5
            if actions[0] == STOP or len(self.past_performance) == 0:
                rew = 0
            reward.append(rew)
            val_acc.append(val_precision)
            self.past_performance.append(val_precision)
            logger1.info("Action: %d" % act)
            logger1.info("Val acc: %.5f  reward: %.5f" % (val_precision, rew))
            logger1.info("-----------------------------------------------------------------------")
        r = np.mean(np.array(reward))
        val_acc = np.mean(val_acc)

        if not self.optimizer:
            if len(self.model.eval_neg_dict) != 0 and len(self.eval_neg_dict) == 0:
                self.eval_neg_dict = self.model.eval_neg_dict
            if len(self.model.test_neg_dict) != 0 and len(self.test_neg_dict) == 0:
                self.test_neg_dict = self.model.test_neg_dict

        if actions[0] != STOP and self.meta_path_equal(tmpmp):
            r, reward = -0.5, [-0.5]
        logger2.info("Action: %d  Val acc: %.5f  reward: %.5f" % (actions[0], val_acc, r))
        logger2.info("Meta-path Set: %s" % str(self.etypes_lists))
        return done_list, r, reward, val_acc

    def meta_path_equal(self, tmp):
        mpset = copy.deepcopy(self.etypes_lists)
        tmp[0].sort()
        tmp[1].sort()
        mpset[0].sort()
        mpset[1].sort()
        if tmp[0] == mpset[0] and tmp[1] == mpset[1]:
            return True
        else:
            return False

    def user_step(self, logger1, logger2, actions, test=False,
                  type=(0, USER_TYPE)):  # type - (index_of_etpyes_list, index_of_node_type)
        done_list, r, reward, val_acc = self.rec_step(actions, logger1, logger2, test, type)
        next_state = self.get_user_state()
        if self.task == 'rec':
            self.model.reset()
        return next_state, reward, done_list, (val_acc, r)

    def item_step(self, logger1, logger2, actions, test=False, type=(1, ITEM_TYPE)):
        done_list, r, reward, val_acc = self.rec_step(actions, logger1, logger2, test, type)
        next_state = self.get_item_state()
        if self.task == 'rec':
            self.model.reset()
        return next_state, reward, done_list, (val_acc, r)

    def class_step(self, logger1, logger2, actions, test=False, type=(0, 'p')):
        done_list, r, reward, val_acc = self.rec_step(actions, logger1, logger2, test, type)
        next_state = self.get_class_state()

        return next_state, reward, done_list, (val_acc, r)

    def train_classifier(self, test=False):
        stopper = EarlyStopping(patience=50)
        loss_fcn = torch.nn.CrossEntropyLoss()
        ids = torch.tensor(range(self.train_data.x.shape[0]))

        if test:
            epoch = 200
        else:
            epoch = 100

        for epoch in range(epoch):
            self.model.train()
            logits = self.model(self.train_data, self.train_data.x, self.etypes_lists[0], self.optimizer, ids,
                                test=False)
            loss = loss_fcn(logits[self.train_mask], self.labels[self.train_mask])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_acc, train_micro_f1, train_macro_f1 = score(logits[self.train_mask], self.labels[self.train_mask])
            val_loss, val_acc, val_micro_f1, val_macro_f1 = self.evaluate(self.model, self.train_data,
                                                                          self.train_data.x,
                                                                          self.labels, self.val_mask, loss_fcn)
            early_stop = stopper.step(val_loss.data.item(), val_acc, self.model)

            if (epoch + 1) % 20 == 0:
                print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
                      'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
                    epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1,
                    val_macro_f1))

            if early_stop:
                break
        stopper.load_checkpoint(self.model)

    def train_GNN(self, act=STOP, test=False):
        if self.task == 'rec':
            self.train_recommender(test, act)
        elif self.task == 'classification':
            self.train_classifier(test)
        elif self.task == 'herec':
            self.model.recommend(test)
        elif self.task == 'mcrec':
            self.train_mcrec(test, act)

    def train_recommender(self, test, act=STOP):
        n_cf_batch = 2 * self.data.n_cf_train // self.data.cf_batch_size + 1
        # n_cf_batch = 1
        cf_total_loss = 0
        if test:
            n_cf_batch = 1
        for iter in range(1, n_cf_batch + 1):
            #     print("current iter: ", iter, " ", n_cf_batch)
            time1 = time.time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = self.data.generate_cf_batch(self.data.train_user_dict)
            time2 = time.time()

            self.optimizer.zero_grad()

            # print("generate batch: ", time2 - time1)
            cf_batch_loss = self.calc_cf_loss(cf_batch_user,
                                              cf_batch_pos_item,
                                              cf_batch_neg_item, test, act)

            time3 = time.time()
            # print("calculate loss: ", time3 - time2)

            cf_batch_loss.backward()

            time4 = time.time()
            # print("backward: ", time4 - time3)

            self.optimizer.step()

            time5 = time.time()
            # print("step: ", time5 - time4)

            cf_total_loss += float(cf_batch_loss)
        # cf_total_loss.backward()
        # self.optimizer.step()
        # print("total_cf_loss: ", float(cf_total_loss))

    # def calc_kg_loss(self, h, r, pos_t, neg_t):
    #     """
    #     h:      (kg_batch_size)
    #     r:      (kg_batch_size)
    #     pos_t:  (kg_batch_size)
    #     neg_t:  (kg_batch_size)
    #     """
    #     r_embed = self.train_data.relation_embed[r]  # (kg_batch_size, relation_dim)
    #
    #     W_r = self.W_R[r]  # (kg_batch_size, entity_dim, relation_dim)
    #
    #     pred = self.update_embedding().to(self.device)
    #
    #     h_embed = pred[h]  # (kg_batch_size, entity_dim)
    #     pos_t_embed = pred[pos_t]  # (kg_batch_size, entity_dim)
    #     neg_t_embed = pred[neg_t]  # (kg_batch_size, entity_dim)
    #
    #     r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
    #     r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
    #     r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
    #
    #     # Equation (1)
    #     pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
    #     neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)
    #
    #     # Equation (2)
    #     kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
    #     kg_loss = torch.mean(kg_loss)
    #
    #     l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
    #         r_mul_neg_t)
    #     loss = kg_loss + self.kg_l2loss_lambda * l2_loss
    #     return loss

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids, test=False, act=STOP):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        tim1 = time.time()
        # pred = self.update_embedding().to(self.device)
        unode_ids = torch.tensor([user_id - self.data.n_id_start_dict[USER_TYPE] for user_id in user_ids])

        # import pdb
        # pdb.set_trace()

        # user_embed = self.get_user_embedding(unode_ids)
        u_embeds = self.get_all_user_embedding(test)
        tim2 = time.time()
        # print("get user embedding: ", tim2 - tim1)

        # item_pos_embed = self.get_item_embedding(item_pos_ids)
        i_embeds = self.get_all_item_embedding(test)
        tim3 = time.time()
        # print("get item embedding: ", tim3 - tim2)

        # item_neg_embed = self.get_item_embedding(item_neg_ids)
        # tim4 = time.time()
        # print("get neg item embedding: ", tim4 - tim3)
        # print(u_embeds.shape, i_embeds.shape)
        # tim2 = time.time()
        # self.train_data.x.weight = nn.Parameter(pred)
        # all_embed = pred  # (n_users + n_entities, cf_concat_dim)
        # user_embed = all_embed[user_ids]  # (cf_batch_size, cf_concat_dim)
        # item_pos_embed = all_embed[item_pos_ids]  # (cf_batch_size, cf_concat_dim)
        # item_neg_embed = all_embed[item_neg_ids]  # (cf_batch_size, cf_concat_dim)
        user_embed = u_embeds[unode_ids]
        item_pos_embed = i_embeds[item_pos_ids]  # (cf_batch_size, cf_concat_dim)
        item_neg_embed = i_embeds[item_neg_ids]  # (cf_batch_size, cf_concat_dim)

        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)  # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)  # (cf_batch_size)

        # print("pos, neg: ", pos_score, neg_score)
        # print("user_embedding: ", user_embed)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)
        # print("cf_loss: ", float(cf_loss))

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss

    def eval_batch(self, neg_num=NEG_SIZE_TRAIN):
        if self.task == 'rec':
            return self.eval_recommender(neg_num)
        elif self.task == 'classification':
            return self.eval_classifier()
        elif self.task == 'herec':
            return self.model.eval_batch()
        elif self.task == 'mcrec':
            return self.eval_mcrec(neg_num)

    def eval_classifier(self):
        loss_fcn = torch.nn.CrossEntropyLoss()
        val_loss, val_precision, val_micro_f1, val_macro_f1 = self.evaluate(self.model, self.train_data,
                                                                            self.train_data.x,
                                                                            self.labels, self.val_mask, loss_fcn)
        print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
            val_loss.item(), val_micro_f1, val_macro_f1))

        return val_precision

    def eval_recommender(self, neg_num):
        self.model.eval()
        time1 = time.time()
        user_ids = list(self.data.train_user_dict.keys())
        self.seed(0)
        user_ids_batch = random.sample(user_ids, min(len(user_ids) - 2, self.args.train_batch_size))

        if not self.eval_neg_dict and neg_num == NEG_SIZE_EVAL:
            print("neg_sum: ", NEG_SIZE_EVAL)

        for u in user_ids_batch:
            if u not in self.eval_neg_dict:
                for _ in self.data.train_user_dict[u]:
                    nl = self.data.sample_neg_items_for_u(self.data.train_user_dict, u, neg_num)
                    self.eval_neg_dict[u].extend(nl)
        with torch.no_grad():
            u_embeds = self.get_all_user_embedding()
            i_embeds = self.get_all_item_embedding()

            time2 = time.time()

            pos_logits = torch.tensor([]).to(self.device)
            neg_logits = torch.tensor([]).to(self.device)

            cf_scores = torch.matmul(
                u_embeds[[user_id - self.data.n_id_start_dict[USER_TYPE] for user_id in user_ids_batch]],
                i_embeds.transpose(0, 1))
            for idx, u in enumerate(user_ids_batch):
                pos_logits = torch.cat([pos_logits, cf_scores[idx][self.data.train_user_dict[u]]])
                neg_logits = torch.cat([neg_logits, torch.unsqueeze(cf_scores[idx][self.eval_neg_dict[u]], 1)])
            # print(pos_logits.shape)
            # print(neg_logits.shape)
            time3 = time.time()
            NDCG10 = self.metrics(pos_logits, neg_logits)
            print(f"Evaluate: NDCG10 : {NDCG10.item():.5f}")
            time4 = time.time()
            # print("ALL time: ", time4 - time1)
        return NDCG10.cpu().item()

    def eval_mcrec(self, neg_num):
        self.model.eval()
        time1 = time.time()
        user_ids = list(self.data.train_user_dict.keys())
        if len(self.eval_user_ids) == 0:
            self.eval_user_ids = random.sample(user_ids, min(len(user_ids) - 2, 500))
        user_ids_batch = self.eval_user_ids

        if not self.eval_neg_dict and neg_num == NEG_SIZE_EVAL:
            print("neg_sum: ", NEG_SIZE_EVAL)

        for u in user_ids_batch:
            if u not in self.eval_neg_dict:
                for _ in self.data.train_user_dict[u]:
                    nl = self.data.sample_neg_items_for_u(self.data.train_user_dict, u, neg_num)
                    self.eval_neg_dict[u].extend(nl)

        u_ids = torch.tensor([u - self.data.n_id_start_dict[USER_TYPE] for u in user_ids_batch], dtype=torch.int)

        with torch.no_grad():
            user_ids_b = torch.split(u_ids, 5)
            time2 = time.time()
            ndcg = 0
            for user_ids_batch in user_ids_b:
                pos_logits = torch.tensor([]).to(self.device)
                neg_logits = torch.tensor([]).to(self.device)
                user_input = user_ids_batch.unsqueeze(1).expand(-1, self.data.n_items)
                item_input = torch.arange(self.data.n_items).expand(len(user_ids_batch), -1)
                metapaths = self.etypes_lists[0]
                metapath_input_list = []

                for metapath in metapaths:
                    if len(metapath) < 2 or len(metapath) > 4:
                        continue
                    metapath_input = torch.zeros(
                        (user_input.shape[0], self.data.n_items, self.neg_num + 1, len(metapath) + 1, 64),
                        dtype=torch.float32)

                    if str(metapath) not in self.metapath_mcrec_dict:
                        traces, types = dgl.sampling.random_walk(self.train_data,
                                                                 list(range(self.data.n_users)) * (self.neg_num + 1),
                                                                 metapath=metapath)

                        path_dict = collections.defaultdict(list)
                        for path in traces.tolist():
                            if path[-1] != -1:
                                path_dict[(path[0], path[-1])].append(path)
                        self.metapath_mcrec_dict[str(metapath)] = (path_dict, types)

                    # if str(metapath) not in self.metapath_mcrec_eval_dict:
                    for i in range(len(user_input)):
                        for j in range(len(user_input[i])):
                            uid = user_input[i][j]
                            iid = item_input[i][j]
                            path_dict, types = self.metapath_mcrec_dict[str(metapath)]
                            if (uid, iid) in path_dict:
                                for x in range(len(path_dict[(uid, iid)])):
                                    for y in range(len(path_dict[(uid, iid)][x])):
                                        nodeid = path_dict[(uid, iid)][x][y]
                                        typeid = types[y]
                                        metapath_input[i][j][x][y] = \
                                            self.train_data.x[self.data.node_type_list == typeid][
                                                nodeid]
                        # self.metapath_mcrec_eval_dict[str(metapath)] = metapath_input.cpu()
                    # metapath_input = self.metapath_mcrec_eval_dict[str(metapath)]
                    metapath_input_list.append(metapath_input)

                metapath_inputs = []
                for metapath_input in metapath_input_list:
                    metapath_inputs.append(metapath_input.to(self.device))

                cf_scores = self.model(user_input.to(self.device), item_input.to(self.device), metapath_inputs).reshape(
                    -1,
                    self.data.n_items)

                for idx, u in enumerate(user_ids_batch.tolist()):
                    pos_logits = torch.cat(
                        [pos_logits,
                         cf_scores[idx][self.data.train_user_dict[u + self.data.n_id_start_dict[USER_TYPE]]]])
                    neg_logits = torch.cat([neg_logits, torch.unsqueeze(
                        cf_scores[idx][self.eval_neg_dict[u + self.data.n_id_start_dict[USER_TYPE]]], 1)])
                NDCG10 = self.metrics(pos_logits, neg_logits)
                ndcg += NDCG10.cpu().item()
            ndcg /= len(user_ids_b)
            print(f"Evaluate: NDCG10 : {ndcg:.5f}")
            time4 = time.time()
        print("ALL time: ", time4 - time1)
        return ndcg

    def test_batch(self, logger2):
        if self.task == 'rec':
            return self.test_recommender(logger2)
        elif self.task == 'classification':
            return self.test_classifier(logger2)
        elif self.task == 'herec':
            return self.model.test_batch()
        elif self.task == 'mcrec':
            return self.test_mcrec(logger2)

    def test_classifier(self, logger2):
        loss_fcn = torch.nn.CrossEntropyLoss()
        test_loss, test_acc, test_micro_f1, test_macro_f1 = self.evaluate(self.model, self.train_data,
                                                                          self.train_data.x,
                                                                          self.labels, self.test_mask, loss_fcn)
        logger2.info('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
            test_loss.item(), test_micro_f1, test_macro_f1))
        print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
            test_loss.item(), test_micro_f1, test_macro_f1))
        return test_acc

    def test_recommender(self, logger2):
        self.model.eval()
        user_ids = list(self.data.test_user_dict.keys())
        user_ids_batch = user_ids[:]
        NDCG10 = 0
        with torch.no_grad():
            for u in user_ids_batch:
                if u not in self.test_neg_dict:
                    nl = self.data.sample_neg_items_for_u_test(self.data.train_user_dict, self.data.test_user_dict,
                                                               u, NEG_SIZE_RANKING)
                    for _ in self.data.test_user_dict[u]:
                        self.test_neg_dict[u].extend(nl)
            # self.train_data.x.weight = nn.Parameter(self.train_data.x.weight.to(self.device))
            # all_embed = self.update_embedding().to(self.device)

            u_embeds = self.get_all_user_embedding(True)
            i_embeds = self.get_all_item_embedding(True)

            pos_logits = torch.tensor([]).to(self.device)
            neg_logits = torch.tensor([]).to(self.device)

            cf_scores = torch.matmul(
                u_embeds[[user_id - self.data.n_id_start_dict[USER_TYPE] for user_id in user_ids_batch]],
                i_embeds.transpose(0, 1))
            for idx, u in enumerate(user_ids_batch):
                pos_logits = torch.cat([pos_logits, cf_scores[idx][self.data.test_user_dict[u]]])
                neg_logits = torch.cat([neg_logits, torch.unsqueeze(cf_scores[idx][self.test_neg_dict[u]], 1)])
            HR1, HR3, HR10, HR20, NDCG10, NDCG20 = self.metrics(pos_logits, neg_logits,
                                                                training=False)
            logger2.info(
                "HR1 : %.4f, HR3 : %.4f, HR10 : %.4f, HR20 : %.4f, NDCG10 : %.4f, NDCG20 : %.4f" % (
                    HR1, HR3, HR10, HR20, NDCG10.item(), NDCG20.item()))
            print(
                f"Test: HR1 : {HR1:.4f}, HR3 : {HR3:.4f}, HR10 : {HR10:.4f}, NDCG10 : {NDCG10.item():.4f}, NDCG20 : {NDCG20.item():.4f}")
        return NDCG10.cpu().item()

    def test_mcrec(self, logger2):
        self.model.eval()
        user_ids = list(self.data.test_user_dict.keys())
        # if len(self.test_user_ids) == 0:
        #     self.test_user_ids = random.sample(user_ids, min(len(user_ids) - 2, 5000))
        # user_ids = self.test_user_ids

        hr1, hr3, hr10, ndcg10, ndcg20 = 0, 0, 0, 0, 0

        for u in user_ids:
            if u not in self.test_neg_dict:
                nl = self.data.sample_neg_items_for_u_test(self.data.train_user_dict, self.data.test_user_dict,
                                                           u, NEG_SIZE_RANKING)
                for _ in self.data.test_user_dict[u]:
                    self.test_neg_dict[u].extend(nl)
        u_ids = torch.tensor([u - self.data.n_id_start_dict[USER_TYPE] for u in user_ids], dtype=torch.int)

        with torch.no_grad():
            user_ids_b = torch.split(u_ids, 5)
            for user_ids_batch in user_ids_b:
                pos_logits = torch.tensor([]).to(self.device)
                neg_logits = torch.tensor([]).to(self.device)
                user_input = user_ids_batch.unsqueeze(1).expand(-1, self.data.n_items)
                item_input = torch.arange(self.data.n_items).expand(len(user_ids_batch), -1)
                metapaths = self.etypes_lists[0]
                metapath_input_list = []
                time2 = time.time()
                for metapath in metapaths:
                    if len(metapath) < 2 or len(metapath) > 4:
                        continue
                    metapath_input = torch.zeros(
                        (user_input.shape[0], self.data.n_items, self.neg_num + 1, len(metapath) + 1, 64),
                        dtype=torch.float32)

                    if str(metapath) not in self.metapath_mcrec_dict:
                        traces, types = dgl.sampling.random_walk(self.train_data,
                                                                 list(range(self.data.n_users)) * (self.neg_num + 1),
                                                                 metapath=metapath)

                        path_dict = collections.defaultdict(list)
                        for path in traces.tolist():
                            if path[-1] != -1:
                                path_dict[(path[0], path[-1])].append(path)
                        self.metapath_mcrec_dict[str(metapath)] = (path_dict, types)

                    # if str(metapath) not in self.metapath_mcrec_test_dict:
                    for i in range(len(user_input)):
                        for j in range(len(user_input[i])):
                            uid = user_input[i][j]
                            iid = item_input[i][j]
                            path_dict, types = self.metapath_mcrec_dict[str(metapath)]
                            if (uid, iid) in path_dict:
                                for x in range(len(path_dict[(uid, iid)])):
                                    for y in range(len(path_dict[(uid, iid)][x])):
                                        nodeid = path_dict[(uid, iid)][x][y]
                                        typeid = types[y]
                                        metapath_input[i][j][x][y] = \
                                            self.train_data.x[self.data.node_type_list == typeid][
                                                nodeid]
                        # self.metapath_mcrec_test_dict[str(metapath)] = metapath_input.cpu()
                    # metapath_input = self.metapath_mcrec_test_dict[str(metapath)]
                    metapath_input_list.append(metapath_input)

                metapath_inputs = []
                for metapath_input in metapath_input_list:
                    metapath_inputs.append(metapath_input.to(self.device))

                cf_scores = self.model(user_input.to(self.device), item_input.to(self.device), metapath_inputs).reshape(
                    -1,
                    self.data.n_items)

                for idx, u in enumerate(user_ids_batch.tolist()):
                    pos_logits = torch.cat(
                        [pos_logits,
                         cf_scores[idx][self.data.test_user_dict[u + self.data.n_id_start_dict[USER_TYPE]]]])
                    neg_logits = torch.cat([neg_logits, torch.unsqueeze(
                        cf_scores[idx][self.test_neg_dict[u + self.data.n_id_start_dict[USER_TYPE]]], 1)])
                HR1, HR3, HR10, HR20, NDCG10, NDCG20 = self.metrics(pos_logits, neg_logits,
                                                                    training=False)
                hr1 += HR1
                hr3 += HR3
                hr10 += HR10
                ndcg10 += NDCG10.cpu().item()
                ndcg20 += NDCG20.cpu().item()
            hr1 /= len(user_ids_b)
            hr3 /= len(user_ids_b)
            hr10 /= len(user_ids_b)
            ndcg10 /= len(user_ids_b)
            ndcg20 /= len(user_ids_b)

            print(
                f"Test: HR1 : {hr1:.4f}, HR3 : {hr3:.4f}, HR10 : {hr10:.4f}, NDCG10 : {ndcg10:.4f}, NDCG20 : {ndcg20:.4f}")
        return ndcg10

    def metrics(self, batch_pos, batch_nega, training=True):
        hit_num1 = 0.0
        hit_num3 = 0.0
        hit_num10 = 0.0
        hit_num20 = 0.0
        # hit_num50 = 0.0
        # mrr_accu10 = torch.tensor(0)
        # mrr_accu20 = torch.tensor(0)
        # mrr_accu50 = torch.tensor(0)
        ndcg_accu10 = torch.tensor(0).to(self.device)
        ndcg_accu20 = torch.tensor(0).to(self.device)
        # ndcg_accu50 = torch.tensor(0)

        if training:
            batch_neg_of_user = torch.split(batch_nega, NEG_SIZE_TRAIN, dim=0)
        else:
            batch_neg_of_user = torch.split(batch_nega, NEG_SIZE_RANKING, dim=0)
        if training:
            for i in range(batch_pos.shape[0]):
                pre_rank_tensor = torch.cat((batch_pos[i].view(1, 1), batch_neg_of_user[i]), dim=0).to(self.device)
                _, indices = torch.topk(pre_rank_tensor, k=pre_rank_tensor.shape[0], dim=0)
                rank = torch.squeeze((indices == 0).nonzero().to(self.device))
                rank = rank[0]
                if rank < 10:
                    ndcg_accu10 = ndcg_accu10 + torch.log(torch.tensor([2.0]).to(self.device)) / torch.log(
                        (rank + 2).type(torch.float32))
            return ndcg_accu10 / batch_pos.shape[0]
        else:
            for i in range(batch_pos.shape[0]):
                pre_rank_tensor = torch.cat((batch_pos[i].view(1, 1), batch_neg_of_user[i]), dim=0).to(self.device)
                _, indices = torch.topk(pre_rank_tensor, k=pre_rank_tensor.shape[0], dim=0)
                rank = torch.squeeze((indices == 0).nonzero().to(self.device))
                rank = rank[0]
                # if rank < 50:
                #     ndcg_accu50 = ndcg_accu50 + torch.log(torch.tensor([2.0]).to(self.device)) / torch.log(
                #         (rank + 2).type(torch.float32))
                #     mrr_accu50 = mrr_accu50 + 1 / (rank + 1).type(torch.float32)
                #     hit_num50 = hit_num50 + 1
                if rank < 20:
                    ndcg_accu20 = ndcg_accu20 + torch.log(torch.tensor([2.0]).to(self.device)) / torch.log(
                        (rank + 2).type(torch.float32))
                    hit_num20 = hit_num20 + 1
                    # mrr_accu20 = mrr_accu20 + 1 / (rank + 1).type(torch.float32)
                if rank < 10:
                    ndcg_accu10 = ndcg_accu10 + torch.log(torch.tensor([2.0]).to(self.device)) / torch.log(
                        (rank + 2).type(torch.float32))
                    hit_num10 = hit_num10 + 1
                # if rank < 10:
                # mrr_accu10 = mrr_accu10 + 1 / (rank + 1).type(torch.float32)
                if rank < 3:
                    hit_num3 = hit_num3 + 1
                if rank < 1:
                    hit_num1 = hit_num1 + 1
            # return hit_num1 / batch_pos.shape[0], hit_num3 / batch_pos.shape[0], hit_num10 / batch_pos.shape[
            #     0], hit_num50 / \
            #        batch_pos.shape[0], mrr_accu10 / batch_pos.shape[0], mrr_accu20 / batch_pos.shape[0], mrr_accu50 / \
            #        batch_pos.shape[0], \
            #        ndcg_accu10 / batch_pos.shape[0], ndcg_accu20 / batch_pos.shape[0], ndcg_accu50 / batch_pos.shape[0]
            return hit_num1 / batch_pos.shape[0], hit_num3 / batch_pos.shape[0], hit_num10 / batch_pos.shape[
                0], hit_num20 / batch_pos.shape[
                       0], ndcg_accu10 / batch_pos.shape[0], ndcg_accu20 / batch_pos.shape[0]

    def train_mcrec(self, test, act=STOP):
        n_cf_batch = 2 * self.data.n_cf_train // self.data.cf_batch_size + 1
        if self.dataset == 'douban_movie':
            n_cf_batch = 10
        if test:
            n_cf_batch = 1
        for iter in range(1, n_cf_batch + 1):
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = self.data.generate_cf_batch(self.data.train_user_dict,
                                                                                              self.neg_num)
            self.optimizer.zero_grad()

            cf_batch_loss = self.calc_mcrec_loss(cf_batch_user,
                                                 cf_batch_pos_item,
                                                 cf_batch_neg_item, test)

    def calc_mcrec_loss(self, user_ids, item_pos_ids, item_neg_ids, test=False):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size * neg_num)
        """

        user_ids -= self.data.n_id_start_dict[USER_TYPE]
        loss_fn = nn.BCELoss()
        cumulative_loss = 0
        # user_input = torch.zeros((self.neg_num + 1) * self.data.cf_batch_size, dtype=torch.int)
        # item_input = torch.zeros((self.neg_num + 1) * self.data.cf_batch_size, dtype=torch.int)
        labels = torch.cat((torch.ones(self.data.cf_batch_size, 1), torch.zeros(self.data.cf_batch_size, self.neg_num)),
                           dim=1)

        user_input = user_ids.unsqueeze(1).expand(-1, self.neg_num + 1).int()
        item_input = torch.cat((item_pos_ids.unsqueeze(1), item_neg_ids.reshape(-1, self.neg_num)), dim=1).int()

        # if not os.path.exists('./data/' + self.dataset + '/metapaths/'):
        #     os.mkdir('./data/' + self.dataset + '/metapaths')

        metapaths = self.etypes_lists[0]
        metapath_input_list = []

        for metapath in metapaths:
            if len(metapath) < 2 or len(metapath) > 4:
                continue
            unode_ids = user_input.reshape(-1)
            # mpfile = './data/' + self.dataset + '/metapaths/' + ''.join(metapath) + '.pkl'
            if str(metapath) not in self.metapath_mcrec_dict:
                traces, types = dgl.sampling.random_walk(self.train_data,
                                                         list(range(self.data.n_users)) * (self.neg_num + 1),
                                                         metapath=metapath)

                path_dict = collections.defaultdict(list)
                for path in traces.tolist():
                    if path[-1] != -1:
                        path_dict[(path[0], path[-1])].append(path)
                self.metapath_mcrec_dict[str(metapath)] = (path_dict, types)

            metapath_input = torch.zeros(
                (self.data.cf_batch_size, self.neg_num + 1, self.neg_num + 1, len(metapath) + 1, 64),
                dtype=torch.float32)

            for i in range(len(user_input)):
                for j in range(len(user_input[i])):
                    uid = user_input[i][j]
                    iid = item_input[i][j]
                    path_dict, types = self.metapath_mcrec_dict[str(metapath)]
                    if (uid, iid) in path_dict:
                        for x in range(len(path_dict[(uid, iid)])):
                            for y in range(len(path_dict[(uid, iid)][x])):
                                nodeid = path_dict[(uid, iid)][x][y]
                                typeid = types[y]
                                metapath_input[i][j][x][y] = self.train_data.x[self.data.node_type_list == typeid][
                                    nodeid]
            metapath_input_list.append(metapath_input)
        # import pdb;pdb.set_trace()
        metapath_inputs = []
        for metapath_input in metapath_input_list:
            metapath_inputs.append(metapath_input.to(self.device))

        output = self.model(user_input.to(self.device), item_input.to(self.device), metapath_inputs)
        loss = loss_fn(output, labels.to(self.device).view(-1, 1))

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        cumulative_loss += loss.item()

        return cumulative_loss
