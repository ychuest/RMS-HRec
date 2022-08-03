import numpy as np
import time
import random
import os
from math import sqrt, fabs, log, exp
import sys
import collections
import dgl

import torch


class HERec:
    def __init__(self, data, metapaths, args, steps):
        self.unum = data.n_users
        self.inum = data.n_items
        self.graph = data.train_graph
        self.ratedim = 10
        self.userdim = 30
        self.itemdim = 10
        self.steps = steps
        self.delta = 0.02
        self.beta_e = 0.1
        self.beta_h = 0.1
        self.beta_p = 2
        self.beta_w = 0.1
        self.beta_b = 0.1
        self.reg_u = 1
        self.reg_v = 1
        self.optimal_i = 0
        self.dataset = args.data_name
        self.metapaths = metapaths

        user_metapaths = metapaths[0]
        item_metapaths = metapaths[1]

        self.user_metapathnum = len(user_metapaths)
        self.item_metapathnum = len(item_metapaths)

        self.train_user_dict = collections.defaultdict(list)
        self.test_user_dict = collections.defaultdict(list)

        self.X, self.user_metapathdims = self.load_embedding(user_metapaths, self.unum)
        # print('Load user embeddings finished.')

        self.Y, self.item_metapathdims = self.load_embedding(item_metapaths, self.inum)
        # print('Load user embeddings finished.')

        data_dir = os.path.join(args.data_dir, args.data_name)
        trainfile = os.path.join(data_dir, 'ub_0.8.train')
        testfile = os.path.join(data_dir, 'ub_0.8.test')

        self.R, self.T, self.ba = self.load_rating(trainfile, testfile)
        # print('Load rating finished.')
        # print('train size : ', len(self.R))
        # print('test size : ', len(self.T))

        self.eval_neg_dict = collections.defaultdict(list)
        self.test_neg_dict = collections.defaultdict(list)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.initialize()

    def write_graph(self, graph, metapath, mpfile):
        if not os.path.exists('./data/' + self.dataset + '/metapaths/'):
            os.mkdir('./data/' + self.dataset + '/metapaths')
        U, V = graph.edges()
        U = U.tolist()
        V = V.tolist()
        with open(mpfile, 'w') as infile:
            for i in range(len(U)):
                infile.write(str(U[i]) + '\t' + str(V[i]) + '\n')
        infile.close()

    def load_embedding(self, metapaths, num):
        dim = 64
        walk_len = 3
        win_size = 2
        num_walk = 5

        X = np.zeros((num, len(metapaths), 64))
        metapathdims = []

        ctn = 0
        for metapath in metapaths:
            sourcefile = './data/' + self.dataset + '/embedding/' + ''.join(metapath) + '.embedding'
            mpfile = './data/' + self.dataset + '/metapaths/' + ''.join(metapath) + '.txt'
            if not os.path.exists(sourcefile):
                subgraph = dgl.metapath_reachable_graph(self.graph, metapath)
                self.write_graph(subgraph, metapath, mpfile)

                cmd = 'deepwalk --format edgelist --input ' + mpfile + ' --output ' + sourcefile + \
                      ' --max-memory-data-size 0 --walk-length ' + str(walk_len) + ' --window-size ' + str(
                    win_size) + ' --number-walks ' \
                      + str(num_walk) + ' --representation-size ' + str(dim)
                os.system(cmd)
                # print('Metapath ' + str(metapath) + ' Embedding Finish')
            # print sourcefile
            with open(sourcefile) as infile:

                k = int(infile.readline().strip().split(' ')[1])
                metapathdims.append(k)

                n = 0
                for line in infile.readlines():
                    n += 1
                    arr = line.strip().split(' ')
                    i = int(arr[0]) - 1
                    for j in range(k):
                        X[i][ctn][j] = float(arr[j + 1])
                # print('metapath: ', metapath, 'numbers: ', n)
            ctn += 1
        return X, metapathdims

    def load_rating(self, trainfile, testfile):
        R_train = []
        R_test = []
        ba = 0.0
        n = 0
        user_test_dict = dict()
        with open(trainfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                R_train.append([int(user) - 1, int(item) - 1, int(rating)])
                ba += int(rating)
                n += 1
                self.train_user_dict[int(user) - 1].append(int(item) - 1)
        ba = ba / n
        ba = 0
        with open(testfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                R_test.append([int(user) - 1, int(item) - 1, int(rating)])
                self.test_user_dict[int(user) - 1].append(int(item) - 1)

        return R_train, R_test, ba

    def initialize(self):
        self.E = np.random.randn(self.unum, self.itemdim) * 0.1
        self.H = np.random.randn(self.inum, self.userdim) * 0.1
        self.U = np.random.randn(self.unum, self.ratedim) * 0.1
        self.V = np.random.randn(self.inum, self.ratedim) * 0.1

        self.pu = np.ones((self.unum, self.user_metapathnum)) * 1.0 / self.user_metapathnum
        self.pv = np.ones((self.inum, self.item_metapathnum)) * 1.0 / self.item_metapathnum

        self.Wu = {}
        self.bu = {}
        for k in range(self.user_metapathnum):
            self.Wu[k] = np.random.randn(self.userdim, self.user_metapathdims[k]) * 0.1
            self.bu[k] = np.random.randn(self.userdim) * 0.1

        self.Wv = {}
        self.bv = {}
        for k in range(self.item_metapathnum):
            self.Wv[k] = np.random.randn(self.itemdim, self.item_metapathdims[k]) * 0.1
            self.bv[k] = np.random.randn(self.itemdim) * 0.1

    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def cal_u(self, i):
        ui = np.zeros(self.userdim)
        for k in range(self.user_metapathnum):
            ui += self.pu[i][k] * self.sigmod((self.Wu[k].dot(self.X[i][k]) + self.bu[k]))
        return self.sigmod(ui)

    def cal_v(self, j):
        vj = np.zeros(self.itemdim)
        for k in range(self.item_metapathnum):
            vj += self.pv[j][k] * self.sigmod((self.Wv[k].dot(self.Y[j][k]) + self.bv[k]))
        return self.sigmod(vj)

    def get_rating(self, i, j):
        ui = self.cal_u(i)
        vj = self.cal_v(j)
        return self.U[i, :].dot(self.V[j, :]) + self.reg_u * ui.dot(self.H[j, :]) + self.reg_v * self.E[i, :].dot(vj)

    def cal_us(self):
        us = np.zeros((self.unum, self.userdim))
        for k in range(self.user_metapathnum):
            us += self.pu[:, k].reshape(self.unum, 1) * (self.sigmod((self.Wu[k].dot(self.X[:, k].T).T + self.bu[k])))
        return self.sigmod(us)

    def cal_vs(self):
        vs = np.zeros((self.inum, self.itemdim))
        for k in range(self.item_metapathnum):
            vs += self.pv[:, k].reshape(self.inum, 1) * (self.sigmod((self.Wv[k].dot(self.Y[:, k].T).T + self.bv[k])))
        return self.sigmod(vs)

    def get_ratings(self):
        us = self.cal_us()
        vs = self.cal_vs()
        return self.U.dot(self.V.T) + self.reg_u * us.dot(self.H.T) + self.reg_v * self.E.dot(vs.T)

    def maermse(self):
        m = 0.0
        mae = 0.0
        rmse = 0.0
        n = 0
        scores = self.get_ratings()
        for t in self.T:
            n += 1
            i = t[0]
            j = t[1]
            r = t[2]
            r_p = scores[i, j]
            m = fabs(r_p - r)
            mae += m
            rmse += m * m
        mae = mae * 1.0 / n
        rmse = sqrt(rmse * 1.0 / n)
        return mae, rmse

    def recommend(self, test=False):
        print("Current Metapath: ", str(self.metapaths))
        if test:
            print("Predicting meta-path set. Return now.")
            return
        mae = []
        rmse = []
        ndcg = []
        starttime = time.time()
        perror = 99999
        cerror = 9999
        # if self.dataset == 'douban_movie':
        # Since dataset is too large, we must use sampling.
        random.seed(0)
        if self.steps == 1:
            train_R = random.sample(self.R, min(len(self.R), 100000))
        else:
            train_R = random.sample(self.R, min(len(self.R),
                                                300000 if self.dataset == 'yelp_data' else 900000))
        # else:
        #     train_R = self.R

        n = len(train_R)
        min_mae = 9999999

        for step in range(self.steps):
            total_error = 0.0
            for t in train_R:
                i = t[0]
                j = t[1]
                rij = t[2]

                rij_t = self.get_rating(i, j)
                eij = rij - rij_t
                total_error += eij * eij

                U_g = -eij * self.V[j, :] + self.beta_e * self.U[i, :]
                V_g = -eij * self.U[i, :] + self.beta_h * self.V[j, :]

                self.U[i, :] -= self.delta * U_g
                self.V[j, :] -= self.delta * V_g

                ui = self.cal_u(i)
                for k in range(self.user_metapathnum):
                    x_t = self.sigmod(self.Wu[k].dot(self.X[i][k]) + self.bu[k])

                    pu_g = self.reg_u * -eij * (ui * (1 - ui) * self.H[j, :]).dot(x_t) + self.beta_p * self.pu[i][k]

                    Wu_g = self.reg_u * -eij * self.pu[i][k] * np.array(
                        [ui * (1 - ui) * x_t * (1 - x_t) * self.H[j, :]]).T.dot(
                        np.array([self.X[i][k]])) + self.beta_w * self.Wu[k]
                    bu_g = self.reg_u * -eij * ui * (1 - ui) * self.pu[i][k] * self.H[j, :] * x_t * (
                            1 - x_t) + self.beta_b * self.bu[k]
                    # print pu_g
                    self.pu[i][k] -= 0.1 * self.delta * pu_g
                    self.Wu[k] -= 0.1 * self.delta * Wu_g
                    self.bu[k] -= 0.1 * self.delta * bu_g

                H_g = self.reg_u * -eij * ui + self.beta_h * self.H[j, :]
                self.H[j, :] -= self.delta * H_g

                vj = self.cal_v(j)
                for k in range(self.item_metapathnum):
                    y_t = self.sigmod(self.Wv[k].dot(self.Y[j][k]) + self.bv[k])
                    pv_g = self.reg_v * -eij * (vj * (1 - vj) * self.E[i, :]).dot(y_t) + self.beta_p * self.pv[j][k]
                    Wv_g = self.reg_v * -eij * self.pv[j][k] * np.array(
                        [vj * (1 - vj) * y_t * (1 - y_t) * self.E[i, :]]).T.dot(
                        np.array([self.Y[j][k]])) + self.beta_w * self.Wv[k]
                    bv_g = self.reg_v * -eij * vj * (1 - vj) * self.pv[j][k] * self.E[i, :] * y_t * (
                            1 - y_t) + self.beta_b * self.bv[k]

                    self.pv[j][k] -= 0.1 * self.delta * pv_g
                    self.Wv[k] -= 0.1 * self.delta * Wv_g
                    self.bv[k] -= 0.1 * self.delta * bv_g

                E_g = self.reg_v * -eij * vj + 0.01 * self.E[i, :]
                self.E[i, :] -= self.delta * E_g

            perror = cerror
            cerror = total_error / n

            self.delta = 0.93 * self.delta

            if (abs(perror - cerror) < 0.0001):
                break
            # print('step ', step, 'crror : ', sqrt(cerror))
            if self.steps == 1:
                NDCG = self.eval_batch()
            else:
                NDCG = self.test_batch()
            ndcg.append(NDCG)
            # print('NDCG@10: ', NDCG)
            endtime = time.time()
            print('Recommend time of step ', step, ': ', (endtime - starttime) / 60, 'min')
        # print('MAE: ', min(mae), ' RMSE: ', min(rmse))
        if self.steps != 1:
            print('NDCG@10: ', max(ndcg), "index: ", ndcg.index(max(ndcg)))
        # st = time.time()
        # self.test_batch()
        # et = time.time()
        # print('test time: ', et - st)

    def eval_batch(self):
        time1 = time.time()
        user_ids = list(self.train_user_dict.keys())
        user_ids_batch = user_ids[:]

        for u in user_ids_batch:
            if u not in self.eval_neg_dict:
                for _ in self.train_user_dict[u]:
                    nl = self.sample_neg_items_for_u_test(self.train_user_dict, self.train_user_dict, u, 10)
                    self.eval_neg_dict[u].extend(nl)

        pos_logits = torch.tensor([])
        neg_logits = torch.tensor([])

        scores = self.get_ratings()

        for u in user_ids_batch:
            pos_logits = torch.cat([pos_logits, torch.from_numpy(scores[u][self.test_user_dict[u]])])
            neg_logits = torch.cat([neg_logits, torch.unsqueeze(torch.from_numpy(scores[u][self.eval_neg_dict[u]]), 1)])

        NDCG10 = self.metrics(pos_logits, neg_logits)
        print("Evaluate NDCG10 : %.4f" % (NDCG10.item()))

        return NDCG10.cpu().item()

    def test_batch(self):
        user_ids = list(self.test_user_dict.keys())
        user_ids_batch = user_ids[:]

        for u in user_ids_batch:
            if u not in self.test_neg_dict:
                nl = self.sample_neg_items_for_u_test(self.train_user_dict, self.test_user_dict, u, 499)
                for _ in self.test_user_dict[u]:
                    self.test_neg_dict[u].extend(nl)

        pos_logits = torch.tensor([])
        neg_logits = torch.tensor([])

        scores = self.get_ratings()

        for u in user_ids_batch:
            pos_logits = torch.cat([pos_logits, torch.from_numpy(scores[u][self.test_user_dict[u]])])
            neg_logits = torch.cat([neg_logits, torch.unsqueeze(torch.from_numpy(scores[u][self.test_neg_dict[u]]), 1)])

        HR1, HR3, HR10, HR20, NDCG10, NDCG20 = self.metrics(pos_logits, neg_logits, training=False)
        print("Test: HR1 : %.4f, HR3 : %.4f, HR10 : %.4f, HR20 : %.4f, NDCG10 : %.4f, NDCG20 : %.4f" % (
            HR1, HR3, HR10, HR20, NDCG10.item(), NDCG20.item()))

        return NDCG10.cpu().item()

    def metrics(self, batch_pos, batch_nega, training=True):
        hit_num1 = 0.0
        hit_num3 = 0.0
        hit_num10 = 0.0
        hit_num20 = 0.0
        ndcg_accu10 = torch.tensor(0).to(self.device)
        ndcg_accu20 = torch.tensor(0).to(self.device)

        if training:
            batch_neg_of_user = torch.split(batch_nega, 10, dim=0)
        else:
            batch_neg_of_user = torch.split(batch_nega, 499, dim=0)
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
                if rank < 20:
                    ndcg_accu20 = ndcg_accu20 + torch.log(torch.tensor([2.0]).to(self.device)) / torch.log(
                        (rank + 2).type(torch.float32))
                    hit_num20 = hit_num20 + 1
                if rank < 10:
                    ndcg_accu10 = ndcg_accu10 + torch.log(torch.tensor([2.0]).to(self.device)) / torch.log(
                        (rank + 2).type(torch.float32))
                    hit_num10 = hit_num10 + 1
                if rank < 3:
                    hit_num3 = hit_num3 + 1
                if rank < 1:
                    hit_num1 = hit_num1 + 1
            return hit_num1 / batch_pos.shape[0], hit_num3 / batch_pos.shape[0], hit_num10 / batch_pos.shape[
                0], hit_num20 / batch_pos.shape[
                       0], ndcg_accu10 / batch_pos.shape[0], ndcg_accu20 / batch_pos.shape[0]

    def sample_neg_items_for_u_test(self, user_dict, test_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]
        pos_items_2 = test_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break
            neg_item_id = np.random.randint(low=0, high=self.inum, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in pos_items_2 and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items
