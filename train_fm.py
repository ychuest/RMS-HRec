import collections

import torch

from KGDataLoader import parse_args
from dqn_agent_pytorch import DQNAgent
import numpy as np
import os
import random
import time
from copy import deepcopy
import logging
import torch.nn as nn

from env.hgnn import hgnn_env


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

def seed(random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def get_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    return logging.getLogger(logger_name)


def use_pretrain(env, dataset='yelp_data'):
    if dataset == 'yelp_data':
        print('./data/yelp_data/embedding/user.embedding_' + str(env.data.entity_dim))
        fr1 = open('./data/yelp_data/embedding/user.embedding_' + str(env.data.entity_dim), 'r')
        fr2 = open('./data/yelp_data/embedding/item.embedding_' + str(env.data.entity_dim), 'r')
    elif dataset == 'douban_movie':
        print('./data/douban_movie/embedding/user.embedding_' + str(env.data.entity_dim))
        fr1 = open('./data/douban_movie/embedding/user.embedding_' + str(env.data.entity_dim), 'r')
        fr2 = open('./data/douban_movie/embedding/item.embedding_' + str(env.data.entity_dim), 'r')
    else:
        print('./data/' + dataset + '/embedding/user.embedding_' + str(env.data.entity_dim))
        fr1 = open('./data/' + dataset + '/embedding/user.embedding_' + str(env.data.entity_dim), 'r')
        fr2 = open('./data/' + dataset + '/embedding/item.embedding_' + str(env.data.entity_dim), 'r')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    emb = env.train_data.x
    emb.requires_grad = False

    for line in fr1.readlines():
        embeddings = line.strip().split()
        id, embedding = int(embeddings[0]), embeddings[1:]
        embedding = list(map(float, embedding))
        emb[id] = torch.tensor(embedding)

    for line in fr2.readlines():
        embeddings = line.strip().split()
        id, embedding = int(embeddings[0]), embeddings[1:]
        embedding = list(map(float, embedding))
        emb[id] = torch.tensor(embedding)

    # emb.requires_grad = True
    env.train_data.x = emb.to(device)


def main():
    seed(0)
    tim1 = time.time()
    torch.backends.cudnn.deterministic = True

    args = parse_args()
    dataset = args.data_name
    max_timesteps = 4 if args.task == 'rec' else 3
    if args.task == 'mcrec':
        max_timesteps = 5

    infor = 'rl_' + str(args.data_name) + '_' + str(args.task) + '_' + str(args.log)
    model_name = 'model_' + infor + '.pth'

    episode = int(args.episode)
    u_max_episodes = episode
    i_max_episodes = episode

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger1 = get_logger('log', 'log/logger_' + infor + '.log')
    logger2 = get_logger('log2', 'log/logger2_' + infor + '.log')

    env = hgnn_env(logger1, logger2, model_name, args, dataset=dataset)
    use_pretrain(env, dataset)

    user_agent = DQNAgent(scope='dqn',
                          action_num=env.action_num,
                          replay_memory_size=int(1e4),
                          replay_memory_init_size=500,
                          norm_step=1,
                          batch_size=1,
                          state_shape=env.obs.shape,
                          mlp_layers=[32, 64, 32],
                          learning_rate=0.001,
                          device=torch.device(device)
                          )
    env.user_policy = user_agent
    best_user_val = 0.0
    best_user_i = 0


    for i_episode in range(1, u_max_episodes + 1):
        env.reset_past_performance()
        loss, reward, (val_acc, reward) = user_agent.user_learn(logger1, logger2, env,
                                                                max_timesteps)  # debug = (val_acc, reward)
        logger2.info("Generated meta-path set: %s" % str(env.etypes_lists))
        print("Generated meta-path set: %s" % str(env.etypes_lists))
        if val_acc > best_user_val:  # check whether gain improvement on validation set
            best_user_policy = deepcopy(user_agent)  # save the best policy
            best_user_val = val_acc
            best_user_i = i_episode
        logger2.info("Training User Meta-policy: %d    Val_Acc: %.5f    Avg_reward: %.5f    Best_Acc:  %.5f    Best_i: %d "
                     % (i_episode, val_acc, reward, best_user_val, best_user_i))
        for i in range(4):
            user_agent.train()

    tim_1 = time.time()
    for i in range(50):
        user_agent.train()

    if args.task != 'mcrec':
        item_agent = DQNAgent(scope='dqn',
                              action_num=env.action_num,
                              replay_memory_size=int(1e4),
                              replay_memory_init_size=500,
                              norm_step=1,
                              batch_size=1,
                              state_shape=env.obs.shape,
                              mlp_layers=[32, 64, 32],
                              learning_rate=0.001,
                              device=torch.device(device)
                              )

        # item_agent = user_agent
        env.item_policy = item_agent
        best_item_val = 0.0
        best_item_i = 0

        # Training: Learning meta-policy
        logger2.info("Training Meta-policy on Validation Set")

        for i_episode in range(1, i_max_episodes + 1):
            env.reset_past_performance()
            loss, reward, (val_acc, reward) = item_agent.item_learn(logger1, logger2, env,
                                                                    max_timesteps)  # debug = (val_acc, reward)
            logger2.info("Generated meta-path set: %s" % str(env.etypes_lists))
            print("Generated meta-path set: %s" % str(env.etypes_lists))
            if val_acc > best_item_val:  # check whether gain improvement on validation set
                best_item_policy = deepcopy(item_agent)  # save the best policy
                best_item_val = val_acc
                best_item_i = i_episode
            logger2.info("Training Item Meta-policy: %d    Val_Acc: %.5f    Avg_reward: %.5f    Best_Acc:  %.5f    Best_i: %d "
                         % (i_episode, val_acc, reward, best_item_val, best_item_i))
            for i in range(4):
                item_agent.train()
        for i in range(50):
            item_agent.train()

    print('Reinforced training time: ', time.time() - tim_1, 's')

    # del env
    tim2 = time.time()

    print("RL agent training time: ", (tim2 - tim1) / 60, "min")

    # Predicting:
    logger2.info("Training GNNs with learned meta-policy. Evaluate NDCG10")
    print("Training GNNs with learned meta-policy. Evaluate NDCG10")
    # new_env = hgnn_env(logger1, logger2, model_name, args, dataset=dataset)
    # use_pretrain(new_env)

    best_user_policy = user_agent
    env.user_policy = user_agent

    torch.save({'q_estimator_qnet_state_dict': env.user_policy.q_estimator.qnet.state_dict(),
                'target_estimator_qnet_state_dict': env.user_policy.target_estimator.qnet.state_dict(),
                'Val': best_user_val},
               'model/a-best-user-' + str(best_user_val) + '-' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                               time.localtime()) + '.pth.tar')

    user_state = env.user_reset()

    if args.task != 'mcrec':
        best_item_policy = item_agent
        env.item_policy = item_agent
        if env.optimizer:
            env.model.reset()

        torch.save({'q_estimator_qnet_state_dict': env.item_policy.q_estimator.qnet.state_dict(),
                    'target_estimator_qnet_state_dict': env.item_policy.target_estimator.qnet.state_dict(),
                    'Val': best_item_val},
                   'model/a-best-item-' + str(best_item_val) + '-' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                   time.localtime()) + '.pth.tar')
        item_state = env.item_reset()
    env.reset_eval_dict()
    mp_set = []
    for i_episode in range(max_timesteps):
        user_action = env.user_policy.eval_step(user_state)
        # if env.optimizer:
        #     env.model.reset()
        user_state, _, user_done, (val_acc, _) = env.user_step(logger1, logger2, user_action, True)
        logger2.info("Meta-path set: %s" % (str(env.etypes_lists)))
        print("Meta-path set: %s" % (str(env.etypes_lists)))
        if args.task != 'mcrec':
            item_action = env.item_policy.eval_step(item_state)
            item_state, _, item_done, (val_acc, _) = env.item_step(logger1, logger2, item_action, True)
            logger2.info("Meta-path set: %s" % (str(env.etypes_lists)))
            print("Meta-path set: %s" % (str(env.etypes_lists)))
    mp_set = deepcopy(env.etypes_lists)
    del env

    tim3 = time.time()

    print("RL agent Predicting time: ", (tim3 - tim2) / 60, "min")
    print("RL agent total time: ", (tim3 - tim1) / 60, "min")

    # Testing
    logger2.info("Start testing meta-path set generated by RL agent")
    logger2.info("Generated meta-path set: %s" % str(mp_set))
    print("Start testing meta-path set generated by RL agent. Generated meta-path set: %s" % str(mp_set))

    args.mpset = str(mp_set)
    test_env = hgnn_env(logger1, logger2, model_name, args, dataset=dataset)
    use_pretrain(test_env, dataset)
    test_env.etypes_lists = mp_set
    if args.task != 'herec':
        best = 0
        best_i = 0
        m_episode = 80 if dataset == 'yelp_data' else 150
        for i in range(m_episode):
            print('Current epoch: ', i)
            if i % 1 == 0:
                acc = test_env.test_batch(logger2)
                if acc > best:
                    best = acc
                    best_i = i
                    if os.path.exists(model_name):
                        os.remove(model_name)
                    torch.save({'state_dict': test_env.model.state_dict(),
                                'optimizer': test_env.optimizer.state_dict(),
                                'Embedding': test_env.train_data.x},
                               model_name)
                logger2.info('Best Accuracy: %.5f\tBest_i : %d' % (best, best_i))
                print('Best: ', best, 'Best_i: ', best_i)
            test_env.train_GNN()
    else:
        test_env.model.steps = 15
        test_env.model.recommend()

    if args.task != 'herec':
        logger2.info(
            "---------------------------------------------------\nStart the performance testing on test dataset:")
        model_checkpoint = torch.load(model_name)
        test_env.model.load_state_dict(model_checkpoint['state_dict'])
        test_env.train_data.x = model_checkpoint['Embedding']
        test_env.test_batch(logger2)


if __name__ == '__main__':
    main()
