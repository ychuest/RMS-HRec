import random


fw = open("kg_final.txt", "w")
f = open("movie_actor.dat")

max_id = 0
num_movie = 12677
act_start = num_movie
lines = f.readlines()
for line in lines:
    words = line.split("\t")
    # print(int(words[0]), int(words[1]))
    if max_id < int(words[1]):
        max_id = int(words[1])
    fw.write("%d %d %d\n" % (int(words[0]) - 1, 0, int(words[1]) + act_start - 1))
f.close()

print(max_id)
director_start = max_id + act_start
max_id = 0

f = open("movie_director.dat")
lines = f.readlines()
for line in lines:
    words = line.split("\t")
    # print(int(words[0]), int(words[1]))
    if max_id < int(words[1]):
        max_id = int(words[1])
    fw.write("%d %d %d\n" % (int(words[0]) - 1, 1, int(words[1]) + director_start - 1))
f.close()

print(max_id)
type_start = max_id + director_start
max_id = 0

f = open("movie_type.dat")
lines = f.readlines()
for line in lines:
    words = line.split("\t")
    # print(int(words[0]), int(words[1]))
    if max_id < int(words[1]):
        max_id = int(words[1])
    fw.write("%d %d %d\n" % (int(words[0]) - 1, 2, int(words[1]) + type_start - 1))
f.close()

print(max_id)
group_start = max_id + type_start
num_group = 2753
user_start = num_group + group_start

f = open("user_group.dat")
lines = f.readlines()
for line in lines:
    words = line.split("\t")
    fw.write("%d %d %d\n" % (int(words[0]) + user_start - 1, 3, int(words[1]) + group_start - 1))
f.close()

f = open("user_user.dat")
lines = f.readlines()
for line in lines:
    words = line.split("\t")
    fw.write("%d %d %d\n" % (int(words[0]) + user_start - 1, 4, int(words[1]) + user_start - 1))
f.close()

fw.close()

fw1 = open("train.txt", "w")
fw2 = open("test.txt", "w")

import collections
train_dict = collections.defaultdict(list)
test_dict = collections.defaultdict(list)

f = open("user_movie.dat")
text = f.read()
lines = text.split("\n")
del lines[-1]
random.shuffle(lines)

i = 0
for line in lines:
    words = line.split("\t")
    if i < len(lines) * 0.8:
        train_dict[int(words[0])].append(int(words[1]))
    else:
        if int(words[0]) in train_dict:
            test_dict[int(words[0])].append(int(words[1]))
        else:
            train_dict[int(words[0])].append(int(words[1]))
    i += 1

len_train = 0
len_test = 0


for user in train_dict:
    if len(train_dict[user]) > 0:
        fw1.write("%d " %(user_start + user - 1))
        len_train += len(train_dict[user])
        for item in train_dict[user]:
            fw1.write("%d " %(item - 1))
        fw1.write("\n")

for user in test_dict:
    if len(test_dict[user]) > 9:
        fw2.write("%d " %(user_start + user - 1))
        len_test += len(test_dict[user])
        for item in test_dict[user]:
            fw2.write("%d " %(item - 1))
        fw2.write("\n")    

print(len_train, len_test)
print(user_start)
f.close()
fw1.close()
fw2.close()