'''
Created on Apr 15, 2016

An Example to run the MFbpr

@author: hexiangnan
'''
from dataloader import LoadRatingFile_HoldKOut
from MFbpr import MFbpr
import multiprocessing as mp
import sys

if __name__ == '__main__':
    # Load data
    dataset = sys.argv[1] if len(sys.argv) > 1 else "douban"
    # splitter = "\t"
    # hold_k_out = 1
    train, test, num_user, num_item, num_ratings = LoadRatingFile_HoldKOut(dataset)
    # print("Load data (%s) done." % (dataset))
    print("#users: %d, #items: %d, #ratings: %d" % (num_user, num_item, num_ratings))

    # MFbpr parameters
    factors = 64
    learning_rate = 0.3
    reg = 0.01
    init_mean = 0
    init_stdev = 0.01
    maxIter = 101
    num_thread = 4

    # Run model
    bpr = MFbpr(train, test, num_user, num_item,
                factors, learning_rate, reg, init_mean, init_stdev)
    bpr.build_model(maxIter, num_thread, batch_size=32)
