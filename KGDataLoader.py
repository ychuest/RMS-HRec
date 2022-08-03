import os
import random
import collections
import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import dgl
import time
import logging


def parse_args():
    parser = argparse.ArgumentParser(description="Run HGNN.")

    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for using multi GPUs.')

    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed.')

    parser.add_argument('--task', nargs='?', default='rec',
                        help='Choose a task from {rec, herec, mcrec, classification}')

    parser.add_argument('--data_name', nargs='?', default='TCL',
                        help='Choose a dataset from {yelp_data, douban_movie, TCL}')
    parser.add_argument('--data_dir', nargs='?', default='data/',
                        help='Input data path.')

    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')

    parser.add_argument('--cf_batch_size', type=int, default=90000,
                        help='CF batch size.')
    parser.add_argument('--kg_batch_size', type=int, default=10000,
                        help='KG batch size.')
    parser.add_argument('--nd_batch_size', type=int, default=5000,
                        help='node sampling batch size.')
    parser.add_argument('--rl_batch_size', type=int, default=1,
                        help='RL training batch size.')
    parser.add_argument('--train_batch_size', type=int, default=2000,
                        help='Eval batch size (the user number to test every batch).')
    parser.add_argument('--test_batch_size', type=int, default=20000,
                        help='Test batch size (the user number to test every batch).')

    parser.add_argument('--entity_dim', type=int, default=64,
                        help='User / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=32,
                        help='Relation Embedding size.')

    parser.add_argument('--aggregation_type', nargs='?', default='bi-interaction',
                        help='Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')

    parser.add_argument('--limit', type=float, default=1000,
                        help='Time Limit.')

    parser.add_argument('--cf_print_every', type=int, default=1,
                        help='Iter interval of printing CF loss.')
    parser.add_argument('--kg_print_every', type=int, default=1,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating CF.')

    parser.add_argument('--K', type=int, default=20,
                        help='Calculate metric@K when evaluating.')
    parser.add_argument('--episode', type=int, default=20,
                        help='episode')
    parser.add_argument('--feats-type', type=int, default=2,
                        help='Type of the node features used. ' +
                             '0 - loaded features; ' +
                             '1 - only target node features (zero vec for others); ' +
                             '2 - only target node features (id vec for others); ' +
                             '3 - all id vec. Default is 2.')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers. Default is 2.')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    parser.add_argument('--num-heads', type=list, default=[4], help='Number of the attention heads. Default is 8.')
    parser.add_argument('--attn-vec-dim', type=int, default=128,
                        help='Dimension of the attention vector. Default is 128.')
    parser.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    parser.add_argument('--patience', type=int, default=10, help='Patience. Default is 10.')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Repeat the training and testing for N times. Default is 1.')
    parser.add_argument('--log', default='',
                        help='Name in log')
    parser.add_argument('--mpset', default="[[['2', '1']], [['1', '2']]]",
                        help='Meta-path Set.')
    parser.add_argument('--init', default="RL",
                        help='Meta-path Set initialization method.')

    args = parser.parse_args()

    save_dir = 'trained_model/HGNN/{}/entitydim{}_relationdim{}_{}_{}_lr{}_pretrain{}/'.format(
        args.data_name, args.entity_dim, args.relation_dim, args.aggregation_type,
        '-'.join([str(i) for i in eval(args.conv_dim_list)]), args.lr, args.use_pretrain)
    args.save_dir = save_dir

    return args


class DataLoaderHGNN(object):

    def __init__(self, logging, args, dataset):
        # self.test_neg_dict = dict()
        self.args = args
        self.data_name = dataset
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir

        self.cf_batch_size = args.cf_batch_size
        if args.task == 'mcrec' and self.cf_batch_size > 15000:
            self.cf_batch_size = 15000
        self.kg_batch_size = args.kg_batch_size

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        data_dir = os.path.join(args.data_dir, self.data_name)
        train_file = os.path.join(data_dir, 'train.txt')
        test_file = os.path.join(data_dir, 'test.txt')
        kg_file = os.path.join(data_dir, "kg_final.txt")

        self.cf_train_data, self.train_user_dict = self.load_cf(train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(test_file)

        # for u in self.test_user_dict:
        #     self.test_neg_dict[u] = list(set(self.train_user_dict[u]) | set(self.test_user_dict[u]))

        self.statistic_cf()

        kg_data = self.load_kg(kg_file)
        self.construct_data(kg_data)

        logging.info("Constructed KG finished.")

        self.print_info(logging)
        self.train_graph = self.create_graph(self.kg_train_data, self.n_users_entities)
        # self.test_graph = self.create_graph(self.kg_test_data, self.n_users_entities)

        # if self.use_pretrain == 1:
        #     self.load_pretrained_data()

    def load_cf(self, filename):
        user = []
        item = []
        user_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))

                for item_id in item_ids:
                    user.append(user_id)
                    item.append(item_id)
                user_dict[user_id] = item_ids

        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict

    def statistic_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) - min(min(self.cf_train_data[0]),
                                                                                        min(self.cf_test_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])
        # print(self.n_users, self.n_items, self.n_cf_train, self.n_cf_test)

    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def construct_data(self, kg_data):
        # plus inverse kg data; relations: 1-12, STOP: 0

        n_relations = max(kg_data['r']) + 1
        reverse_kg_data = kg_data.copy()
        reverse_kg_data = reverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        reverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, reverse_kg_data], axis=0, ignore_index=True, sort=False)

        # re-map user id
        kg_data['r'] += 3
        self.n_relations = max(kg_data['r'])
        self.n_users_entities = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_entities = self.n_users_entities - self.n_users
        node_type_list = np.zeros(self.n_users_entities, dtype=np.int32)
        if self.args.data_name == 'yelp_data':
            '''
            Only for Yelp dataset
        
                Business: 0 (0 - 14283), Category: 1 (14284 - 14794), City: 2 (14795 - 14841),
                Compliment : 3 (14842 - 14852), User: 4 (14853 - 31091)
                
                1: B-U 2: U-B 3: B-Ca 4: B-Ci 5: U-Co 6: U-U 7: Ca-B 8: Ci-B 9: Co-U 10: U-U
                
            '''

            node_type_list[:14284] = 0
            node_type_list[14284:14795] = 1
            node_type_list[14795:14842] = 2
            node_type_list[14842:14853] = 3
            node_type_list[14853:] = 4
            self.node_type_list = node_type_list

            self.n_id_start_dict = {0: 0, 1: 14284, 2: 14795, 3: 14842, 4: 14853}

            self.metapath_transform_dict = {1: ['1', '2'], 2: ['2', '1'], 3: ['3', '7'], 4: ['4', '8'],
                                            5: ['5', '9'], 6: ['6']}

            self.n_types = max(node_type_list) + 1
        elif self.args.data_name == 'douban_movie':
            '''
            Only for Douban-Movie dataset
                1: M-U 2: U-M 3: M-A 4: M-D 5: M-T 6: U-G 7: U-U 8: A-M 9: D-M 10: T-M 11: G-U 12: U-U

                Movie: 0 (0 - 12676), Actor: 1 (12677 - 18987), Director: 2 (18988 - 21436),
                Type : 3 (21437 - 21474), Group: 4 (21475 - 24227), User: 5 (24228 - 37594)
            '''

            node_type_list[:12677] = 0
            node_type_list[12677:18988] = 1
            node_type_list[18988:21437] = 2
            node_type_list[21437:21475] = 3
            node_type_list[21475:24228] = 4
            node_type_list[24228:] = 5
            self.node_type_list = node_type_list

            self.n_id_start_dict = {0: 0, 1: 12677, 2: 18988, 3: 21437, 4: 21475, 5: 24228}

            self.metapath_transform_dict = {1: ['1', '2'], 2: ['2', '1'], 3: ['3', '8'], 4: ['4', '9'],
                                            5: ['5', '10'], 6: ['6', '11'], 7: ['7']}
            self.n_types = max(node_type_list) + 1
        elif self.args.data_name == 'TCL':
            '''
            Only for TCL dataset
                1: M-U 2: U-M 3: M-A 4: M-Di 5: M-R 6: M-C 7: M-L 8: M-T 9: M-De 10: A-M 11: Di-M 12: R-M
                13: C-M 14: L-M 15: T-M 16: De-M

                Movie: 0 (0 - 12636), Null: 10 (12637 - 30528), Actor: 1 (30529 - 82060),
                Region : 2 (82061 - 82157), Copyright: 3 (82157 - 82441), Language: 4 (82441 - 82521)
                Director: 5 (82521 - 103763), Decade: 6 (103763 - 103776), Tag: 7 (103776 - 105550), 
                User: 8 (105550 - 122284)
            '''
            node_type_list[:12637] = 0  # movie_df
            node_type_list[12637:30529] = 10  # null_df
            node_type_list[30529:82061] = 1  # actor_df
            node_type_list[82061:82157] = 2  # region_df
            node_type_list[82157:82441] = 3  # copyright_df
            node_type_list[82441:82521] = 4  # language_df
            node_type_list[82521:103763] = 5  # director_df
            node_type_list[103763:103776] = 6  # decade_df
            node_type_list[103776:105550] = 7  # tag_df
            node_type_list[105550:] = 8  # user_df
            self.node_type_list = node_type_list

            self.n_id_start_dict = {0: 0, 1: 30529, 2: 82061, 3: 82157, 4: 82441, 5: 82521, 6: 103763, 7: 103776,
                                    8: 105550}

            self.metapath_transform_dict = collections.defaultdict(list)
            self.metapath_transform_dict[1] = ['1', '2']
            self.metapath_transform_dict[2] = ['2', '1']

            for i in range(3, self.n_relations // 2 + 2):
                kg_relation = (self.n_relations - 2) // 2
                if i + kg_relation > self.n_relations:
                    self.metapath_transform_dict[i].append(str(i - kg_relation))
                    self.metapath_transform_dict[i].append(str(i))
                else:
                    self.metapath_transform_dict[i].append(str(i))
                    self.metapath_transform_dict[i].append(str(i + kg_relation))
            self.n_types = max(node_type_list) + 1
        self.cf_train_data = (
            np.array(list(self.cf_train_data[0])).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (
            np.array(list(self.cf_test_data[0])).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = {k: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.full((self.n_cf_train, 5), 2, dtype=np.int32),
                                        columns=['h', 'r', 't', 'ht', 'tt'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]
        cf2kg_train_data['ht'] = self.node_type_list[cf2kg_train_data['h']]
        cf2kg_train_data['tt'] = self.node_type_list[cf2kg_train_data['t']]

        reverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 5), dtype=np.int32),
                                                columns=['h', 'r', 't', 'ht', 'tt'])
        reverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        reverse_cf2kg_train_data['t'] = self.cf_train_data[0]
        reverse_cf2kg_train_data['ht'] = self.node_type_list[reverse_cf2kg_train_data['h']]
        reverse_cf2kg_train_data['tt'] = self.node_type_list[reverse_cf2kg_train_data['t']]

        cf2kg_test_data = pd.DataFrame(np.full((self.n_cf_test, 5), 2, dtype=np.int32),
                                       columns=['h', 'r', 't', 'ht', 'tt'])
        cf2kg_test_data['h'] = self.cf_test_data[0]
        cf2kg_test_data['t'] = self.cf_test_data[1]
        cf2kg_test_data['ht'] = self.node_type_list[cf2kg_test_data['h']]
        cf2kg_test_data['tt'] = self.node_type_list[cf2kg_test_data['t']]

        reverse_cf2kg_test_data = pd.DataFrame(np.ones((self.n_cf_test, 5), dtype=np.int32),
                                               columns=['h', 'r', 't', 'ht', 'tt'])
        reverse_cf2kg_test_data['h'] = self.cf_test_data[1]
        reverse_cf2kg_test_data['t'] = self.cf_test_data[0]
        reverse_cf2kg_test_data['ht'] = self.node_type_list[reverse_cf2kg_test_data['h']]
        reverse_cf2kg_test_data['tt'] = self.node_type_list[reverse_cf2kg_test_data['t']]

        kg_data['ht'] = self.node_type_list[kg_data['h']]
        kg_data['tt'] = self.node_type_list[kg_data['t']]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, reverse_cf2kg_train_data], ignore_index=True)
        self.kg_test_data = pd.concat([kg_data, cf2kg_test_data, reverse_cf2kg_test_data], ignore_index=True)

        self.n_kg_train = len(self.kg_train_data)
        self.n_kg_test = len(self.kg_test_data)
        # construct kg dict
        # self.train_kg_dict = collections.defaultdict(list)
        # self.train_relation_dict = collections.defaultdict(list)
        # # print("#Entites: ", self.n_users_entities)
        # # print(len(self.kg_train_data))
        # print(row[1])
        # h, r, t = row[1]
        # self.train_kg_dict[h].append((t, r))
        # self.train_relation_dict[r].append((h, t))
        #
        # self.test_kg_dict = collections.defaultdict(list)
        # self.test_relation_dict = collections.defaultdict(list)
        # for row in self.kg_test_data.iterrows():
        #     h, r, t = row[1]
        #     self.test_kg_dict[h].append((t, r))
        #     self.test_relation_dict[r].append((h, t))

    def print_info(self, logging):
        logging.info('n_users:            %d' % self.n_users)
        logging.info('n_items:            %d' % self.n_items)
        logging.info('n_entities:         %d' % self.n_entities)
        logging.info('n_users_entities:   %d' % self.n_users_entities)
        logging.info('n_relations:        %d' % self.n_relations)

        logging.info('n_cf_train:         %d' % self.n_cf_train)
        logging.info('n_cf_test:          %d' % self.n_cf_test)

        logging.info('n_kg_train:         %d' % self.n_kg_train)
        logging.info('n_kg_test:          %d' % self.n_kg_test)

    def create_graph(self, kg_data, n_nodes):
        '''
        Yelp_data:
            Business: 0 (0 - 14283), Category: 1 (14284 - 14794), City: 2 (14795 - 14841),
            Compliment : 3 (14842 - 14852), User: 4 (14853 - 31091)
        '''

        node_type_list = torch.from_numpy(self.node_type_list)

        node_feature = collections.defaultdict(lambda: torch.Tensor)
        '''
            edge_list: index the adjacancy matrix (time) by 
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        # edge_list = collections.defaultdict(  # target_type
        #     lambda: collections.defaultdict(  # source_type
        #         lambda: collections.defaultdict(  # relation_type
        #             lambda: collections.defaultdict(  # target_id
        #                 lambda: []  # source_id
        #             ))))
        # adjM = np.zeros((n_nodes, n_nodes), dtype=int)
        relations = collections.defaultdict(list)
        e_n_dict = collections.defaultdict(tuple)
        for row in zip(*kg_data.to_dict("list").values()):
            relations[('n' + str(row[3]), str(row[1]), 'n' + str(row[4]))].append(
                (row[0] - self.n_id_start_dict[row[3]], row[2] - self.n_id_start_dict[row[4]]))
            # adjM[row[0], row[2]] = 1
            # adjM[row[2], row[0]] = 1
            e_n_dict[str(row[1])] = [row[3], row[4]]
        graph = dgl.heterograph(relations)

        x = torch.randn(n_nodes, self.entity_dim)
        nn.init.xavier_uniform_(x, gain=nn.init.calculate_gain('relu'))
        for i in range(5):
            node_feature[i] = x[node_type_list == i]

        edge_index = torch.tensor([kg_data['t'], kg_data['h']], dtype=torch.long)
        edge_attr = torch.tensor(kg_data['r'])
        # graph.adjM = adjM
        graph.e_n_dict = e_n_dict
        graph.x = x
        graph.edge_index = edge_index
        graph.edge_attr = edge_attr
        # graph.relation_embed = torch.randn(self.n_relations + 1, self.relation_dim)
        graph.node_idx = torch.arange(n_nodes, dtype=torch.long)
        graph.node_types = node_type_list
        return graph
        # g = dgl.DGLGraph()
        # g.add_nodes(n_nodes)
        # g.add_edges(kg_data['t'], kg_data['h'])
        # g.readonly()
        # g.ndata['id'] = torch.arange(n_nodes, dtype=torch.long)
        # g.edata['type'] = torch.LongTensor(kg_data['r'])
        # return g

    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)

        sample_pos_items = []
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break

            pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_idx]
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)
        return sample_pos_items

    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items

    def sample_neg_items_for_u_test(self, user_dict, test_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]
        pos_items_2 = test_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in pos_items_2 and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items

    def generate_cf_batch(self, user_dict, neg_num=1):
        exist_users = list(user_dict.keys())
        if self.cf_batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, self.cf_batch_size)
        else:
            batch_user = [random.choice(exist_users) for _ in range(self.cf_batch_size)]

        batch_pos_item, batch_neg_item = [], []
        for u in batch_user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, neg_num)

        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item

    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=self.n_users_entities, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails

    def generate_kg_batch(self, kg_dict):
        exist_heads = list(kg_dict.keys())
        if self.kg_batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, self.kg_batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(self.kg_batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1)
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail

    # def load_pretrained_data(self):
    #     pre_model = 'mf'
    #     pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir, self.data_name, pre_model)
    #     pretrain_data = np.load(pretrain_path)
    #     self.user_pre_embed = pretrain_data['user_embed']
    #     self.item_pre_embed = pretrain_data['item_embed']
    #
    #     assert self.user_pre_embed.shape[0] == self.n_users
    #     assert self.item_pre_embed.shape[0] == self.n_items
    #     assert self.user_pre_embed.shape[1] == self.args.entity_dim
    #     assert self.item_pre_embed.shape[1] == self.args.entity_dim
