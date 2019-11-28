# -*- coding: utf-8 -*-
#/usr/bin/python2
import numpy as np
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        #已经padding好了
        seq_t = np.zeros([maxlen], dtype=np.int32)
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]  #最后一个item
        idx = maxlen - 1            #序列长度减一   199


        ts = user_train[str(user)+"_t"][:-1]
        j = len(ts)-2
        
        # user_train[user] = [1，2，3，4]
        # seq = [1,2,3]  pos = [2,3,4] neg = [6,7,8]
        tempset = set(user_train[user])
        for i in reversed(user_train[user][:-1]): #从倒数第二个开始倒叙
            seq_t[idx] = ts[j]
            
            seq[idx] = i
            pos[idx] = nxt # 作为预测结果 正例
            if nxt != 0: 
                neg[idx] = random_neq(1, itemnum + 1, tempset) # 采样负例
            nxt = i
            idx -= 1
            j -= 1
            
            if idx == -1: break  # idx  从199 处理到0，  就是取最近的200个item
#        print(seq_t.shape,seq.shape)
#        print("========",int(seq_t[-1])-int(seq_t[0]))   
        return (user, seq, pos, neg, seq_t)  

    # 一个batch里面有很多user 的数据，所有的数据里面，可能有相同的user的数据
    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
