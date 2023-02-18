import numpy as np
import pandas as pd
import time, datetime
from tqdm import tqdm
from sklearn.model_selection import KFold,StratifiedKFold
import json,os
from sklearn.model_selection import train_test_split
import sys

last_n = 30

file_path = '/data/shshen/AAAI_reasoning/ncs/AAAI23/data/'
all_data =  pd.read_csv(file_path + 'train_valid_sequences.csv')  

user = np.array(all_data['uid'])
user = list(set(user))

data_dict= {}

for item in np.array(all_data):
    problem = item[2]
    skill = item[3]
    answer = item[4]
    repeat = item[7]
    p1 = [int(x) for x in problem.strip().split(',')]
    s1 = [int(x) for x in skill.strip().split(',')]
    a1 = [int(x) for x in answer.strip().split(',')]
    r1 = [int(x) for x in repeat.strip().split(',')]
    if item[1] in data_dict.keys():
        data_dict[item[1]].append([p1, s1, a1, r1])
    else:
        data_dict[item[1]] = []
        data_dict[item[1]].append([p1, s1, a1, r1])



user_df_train = []
user_df_valid = []



np.random.seed(2022)
np.random.shuffle(user)
aaa = int(0.2*len(user))
folds = int(sys.argv[1])


train_id = user[:aaa*(folds)] + user[aaa*(folds+1):]
valid_id = user[aaa*(folds):aaa*(folds+1)]

for item in tqdm(train_id):
    temp = data_dict[item]
    ppp = []
    sss = []
    aaa = []
    rrr = []
    for iii in temp:
        ppp += iii[0]
        sss += iii[1]
        aaa += iii[2]
        rrr += iii[3]
    aaa = [x for x in aaa if x != -1]
    ppp = ppp[:len(aaa)]
    sss = sss[:len(aaa)]
    rrr = rrr[:len(aaa)]

    history = []
    history_kc = []
    his_a = []

    for one in range(len(aaa)):
        if rrr[one] !=0:
            continue


        
        if len(history) >= last_n:
            user_df_train.append([int(item) , ppp[one], sss[one], aaa[one], history[-last_n:], history_kc[-last_n:], his_a[-last_n:]])
        his_a.append(aaa[one])
        history.append(ppp[one])
        history_kc.append(sss[one])


for item in tqdm(valid_id):
    temp = data_dict[item]
    ppp = []
    sss = []
    aaa = []
    rrr = []
    for iii in temp:
        ppp += iii[0]
        sss += iii[1]
        aaa += iii[2]
        rrr += iii[3]
    aaa = [x for x in aaa if x != -1]
    ppp = ppp[:len(aaa)]
    sss = sss[:len(aaa)]
    rrr = rrr[:len(aaa)]
    
    history = []
    history_kc = []
    his_a = []

    for one in range(int(0.8*len(aaa))):
        if rrr[one] !=0:
            continue
        if len(history) >= last_n:
            user_df_train.append([int(item) , ppp[one], sss[one], aaa[one], history[-last_n:], history_kc[-last_n:], his_a[-last_n:]])
        his_a.append(aaa[one])
        history.append(ppp[one])
        history_kc.append(sss[one])

    for one in range(int(0.8*len(aaa)), len(aaa)):
        if rrr[one] !=0:
            continue
        if len(history) >= last_n:
            user_df_valid.append([int(item) , ppp[one], sss[one], aaa[one], history[-last_n:], history_kc[-last_n:], his_a[-last_n:]])


np.save('data/train.npy', np.array(user_df_train))
np.save('data/valid.npy', np.array(user_df_valid))


print("complete")
