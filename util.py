# -*- coding: utf-8 -*-
#/usr/bin/python2
import sys
import copy
import random
import numpy as np
from collections import defaultdict
import time

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    TS = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i, ymd, hms, timestamp = line.rstrip().split(' ')
        a=time.localtime(float(timestamp))
        
        u = int(u)
        i = int(i)
        long_date = str(ymd+" "+hms)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        TS[u].append(a.tm_yday)
        

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_train[str(user)+"_t"] = TS[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

            user_train[str(user)+"_t"] = TS[user][:-2]
            
            user_valid[str(user)+"_t"] = []
            user_valid[str(user)+"_t"].append(TS[user][-2])
            
            user_test[str(user)+"_t"] = []
            user_test[str(user)+"_t"].append(TS[user][-1])
            
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(xrange(1, usernum + 1), 10000)
    else:
        users = xrange(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        ts_valid = valid[str(u)+"_t"]
        ts_train = train[str(u)+"_t"]
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        seq_t = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1  #199
        seq[idx] = valid[u][0]  #seq[199] = valid item
        seq_t[idx] = ts_valid[0]
        
        j=len(train[u]) -1
        
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            seq_t[idx] = ts_train[j]
            
            j -=1
            idx -= 1
            
            if idx == -1: break
        
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]  # ground-truth
        for _ in range(100):     # 剩下的100个neg
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

#        print( [u], [seq], [seq_t], item_idx )
        
        predictions = -model.predict(sess, [u], [seq], [seq_t], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print 't.',
            sys.stdout.flush()
    print("evaluate",NDCG / valid_user, HT / valid_user)
    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(xrange(1, usernum + 1), 10000)
    else:
        users = xrange(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        ts_train = train[str(u)+"_t"]
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        seq_t = np.zeros([args.maxlen], dtype=np.int32)
        
        idx = args.maxlen - 1
        j=len(train[u]) -1
        
        for i in reversed(train[u]):
            seq[idx] = i
            seq_t[idx] = ts_train[j]
            
            j -= 1
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], [seq_t],item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print 'v.',
            sys.stdout.flush()
            
    print("evaluate_valid",NDCG / valid_user, HT / valid_user)
    return NDCG / valid_user, HT / valid_user
