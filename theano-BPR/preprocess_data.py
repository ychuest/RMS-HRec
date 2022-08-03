# let user id and item id start with 1.
import random

if __name__ == '__main__':
    f1 = open('../data/douban_movie/train.txt')
    f2 = open('../data/douban_movie/test.txt')
    fw1 = open('./data/douban.train.rating', 'w')
    fw2 = open('./data/douban.test.rating', 'w')

    train_list = []
    lines = f1.readlines()
    for line in lines:
        words = line.split()
        for i in range(1, len(words)):
            train_list.append([int(words[0]) - 24227, int(words[i]) + 1, 5])
    random.shuffle(train_list)

    for a in train_list:
        fw1.write("%d\t%d\t%d\n" % (a[0], a[1], a[2]))

    f1.close()
    fw1.close()

    lines = f2.readlines()
    for line in lines:
        words = line.split()
        fw2.write(str(int(words[0]) - 24227) + ' ')
        for i in range(1, len(words)):
            fw2.write(str(int(words[i]) + 1) + ' ')
        fw2.write('\n')
    f2.close()
    fw2.close()
