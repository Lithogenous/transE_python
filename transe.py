import pandas as pd
import numpy as np
import random

def distanceL2(h, t, r):
    s = h + r - t
    sum = (s*s).sum(axis=1)
    return sum

def distanceL1(h, t ,r):
    s = h + r - t
    sum = np.fabs(s).sum(axis=1)
    return sum

def l2norm(mat):
    return mat / np.linalg.norm(mat, axis=1)[:,None]

def initalize(n_relation, n_entity, emb_dim=20):
    relation_vec = np.random.uniform(-6.0 / np.sqrt(emb_dim), 6.0 / np.sqrt(emb_dim), [n_relation, emb_dim])
    entity_vec = np.random.uniform(-6.0 / np.sqrt(emb_dim), 6.0 / np.sqrt(emb_dim), [n_entity, emb_dim])
    return l2norm(relation_vec), l2norm(entity_vec)

def construct_corrupted_triplet(batch_df):
    tmp_df = batch_df.copy()
    
    corrupted_triple_list = []
    for each in tmp_df.values:
        i = random.uniform(-1, 1)
        if i < 0:#小于0，打坏三元组的第一项
            while True:
                entityTemp = random.sample(list(entity2id['entity'].values), 1)[0]
                if entityTemp != each[0]:
                    break
            corruptedTriplet = [entityTemp, each[1]]
        else:#大于等于0，打坏三元组的第二项
            while True:
                entityTemp = random.sample(list(entity2id['entity'].values), 1)[0]
                if entityTemp != each[1]:
                    break
            corruptedTriplet = [each[0], entityTemp]

        corrupted_triple_list.append(corruptedTriplet)

    return pd.DataFrame(np.concatenate([tmp_df.values, np.array(corrupted_triple_list)], axis=1), columns=['e1', 'e2', 'rel', 'fake_e1', 'fake_e2'])
    

trn = pd.read_csv('../fb15k/train.txt', sep='\t',names=['e1', 'e2', 'rel'])
vld = pd.read_csv('../fb15k/valid.txt', sep='\t',names=['e1', 'e2', 'rel'])
tst = pd.read_csv('../fb15k/test.txt', sep='\t',names=['e1', 'e2', 'rel'])
relation2id = pd.read_csv('../fb15k/relation2id.txt', sep='\t',names=['rel', 'id'])
entity2id = pd.read_csv('../fb15k/entity2id.txt', sep='\t',names=['entity', 'id'])

EMB_DIM = 20
N_EPOCH = 10
BATCH_SIZE = 150

n_entity = len(entity2id)
n_relation = len(relation2id)

####construct the 
relation_emb_dict = {}
entity_emb_dict = {}

rel_vec, ent_vec = initalize(n_relation=n_relation, n_entity=n_entity, emb_dim=EMB_DIM)

for i, each in enumerate(relation2id['rel'].values):
    relation_emb_dict[each] = rel_vec[i]
for i, each in enumerate(entity2id['entity'].values):
    entity_emb_dict[each] = ent_vec[i]


for iter in range(N_EPOCH):
    S_batch_index = random.sample(list(trn.index), BATCH_SIZE)
    S_batch = trn.loc[S_batch_index]
    T_batch = construct_corrupted_triplet(S_batch)
    print(T_batch)