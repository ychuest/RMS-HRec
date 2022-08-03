import torch

from KGDataLoader import parse_args
import os
import logging

from env.hgnn import hgnn_env


def get_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    return logging.getLogger(logger_name)


if __name__ == '__main__':
    args = parse_args()
    dataset = args.data_name

    # mp_set = eval(args.mpset)

    print("Meta-path Set: ", args.mpset)

    infor = 'herec_' + str(args.data_name) + '_spec_' + str(args.log)
    model_name = 'model_' + infor + '.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger1 = get_logger('log', 'logger_' + infor + '.log')
    logger2 = get_logger('log2', 'logger2_' + infor + '.log')

    test_env = hgnn_env(logger1, logger2, model_name, args, dataset=dataset)
    # test_env.etypes_lists = mp_set

    test_env.model.steps = 20
    test_env.model.recommend()
