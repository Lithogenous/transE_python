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

def constructCorruptedTriplet(batch_df):
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
    T_batch = pd.DataFrame(np.concatenate([tmp_df.values, np.array(corrupted_triple_list)], axis=1), columns=['e1', 'e2', 'rel', 'fake_e1', 'fake_e2'])
    T_batch['e1_index'] = T_batch['e1'].apply(lambda x: entity_emb_dict[x])
    T_batch['e2_index'] = T_batch['e2'].apply(lambda x: entity_emb_dict[x])
    T_batch['rel_index'] = T_batch['rel'].apply(lambda x: relation_emb_dict[x])
    T_batch['fake_e1_index'] = T_batch['fake_e1'].apply(lambda x: entity_emb_dict[x])
    T_batch['fake_e2_index'] = T_batch['fake_e2'].apply(lambda x: entity_emb_dict[x])
    
    return T_batch
    

trn = pd.read_csv('../fb15k/train.txt', sep='\t',names=['e1', 'e2', 'rel'])
vld = pd.read_csv('../fb15k/valid.txt', sep='\t',names=['e1', 'e2', 'rel'])
tst = pd.read_csv('../fb15k/test.txt', sep='\t',names=['e1', 'e2', 'rel'])
relation2id = pd.read_csv('../fb15k/relation2id.txt', sep='\t',names=['rel', 'id'])
entity2id = pd.read_csv('../fb15k/entity2id.txt', sep='\t',names=['entity', 'id'])

EMB_DIM = 20
N_EPOCH = 10000
BATCH_SIZE = 150
MARGIN = 1.0
LearningRate = 10
LOSS_TYPE = 'L1'

n_entity = len(entity2id)
n_relation = len(relation2id)

####construct the 
relation_emb_dict = {}
entity_emb_dict = {}

#algorithm line 1 - 3 using the Xavier initialazation to initialize the embedding vector and l2 normalize it
rel_emb_mat, ent_emb_mat = initalize(n_relation=n_relation, n_entity=n_entity, emb_dim=EMB_DIM)
#algotrithm line 1 - 3 end rel_emb_mat and ent_emb_mat save the embedding vector respectively.

for each in relation2id.values:
    relation_emb_dict[each[0]] = each[1]
for each in entity2id.values:
    entity_emb_dict[each[0]] = each[1]


#loss = []

#algorithm line 4 start
for epoch in range(N_EPOCH):
    #algorithm line 6 sample the minibatch
    S_batch_index = random.sample(list(trn.index), BATCH_SIZE)
    S_batch = trn.loc[S_batch_index]
    #algorithm line 6 end S_batch is the sampled minibatch
    
    #algorithm line 7 - 11 constuct the corrupted triplet
    T_batch = constructCorruptedTriplet(S_batch)
        
    e1_emb = ent_emb_mat[T_batch['e1_index'].values,:]
    e2_emb = ent_emb_mat[T_batch['e2_index'].values,:]
    rel_emb = rel_emb_mat[T_batch['rel_index'].values,:]
    fake_e1_emb = ent_emb_mat[T_batch['fake_e1_index'].values,:]
    fake_e2_emb = ent_emb_mat[T_batch['fake_e2_index'].values,:]
    #algorithm line 7 - 11 end e1_emb is the h, e2_emb is t, rel_emb is l, fake_e1_emb is h' fake_e2_emb is t'


    if LOSS_TYPE == 'L1':
        #algorithm line 12 the minibatch SGD
        dist_triplet1 = distanceL1(e1_emb, e2_emb, rel_emb)
        dist_corrupted_triplet1 = distanceL1(fake_e1_emb, fake_e2_emb, rel_emb)
            
        eg = MARGIN + dist_triplet1 - dist_corrupted_triplet1
        gre_positive = ((e2_emb - e1_emb - rel_emb) >0).astype(np.float64) -  ((e2_emb - e1_emb - rel_emb) <0).astype(np.float64)
        gre_negative = ((fake_e2_emb - fake_e1_emb - rel_emb) >0).astype(np.float64) -  ((fake_e2_emb - fake_e1_emb - rel_emb) <0).astype(np.float64)

    else:
        #algorithm line 12 the minibatch SGD
        dist_triplet2 = distanceL2(e1_emb, e2_emb, rel_emb)
        dist_corrupted_triplet2 = distanceL2(fake_e1_emb, fake_e2_emb, rel_emb)
            
        eg = MARGIN + dist_triplet2 - dist_corrupted_triplet2
        #loss.append(eg)

        gre_positive = 2 * LearningRate * (e2_emb - e1_emb - rel_emb) * (eg >0)[:,None]
        gre_negative = 2 * LearningRate * (fake_e2_emb - fake_e1_emb - rel_emb) *(eg>0)[:,None]
    
    
    e1_update_emb = e1_emb + gre_positive
    e2_update_emb = e2_emb - gre_positive
    relation_update_emb = rel_emb + gre_positive - gre_negative
    fake_e1_update_emb = fake_e1_emb - gre_negative
    fake_e2_update_emb = fake_e2_emb + gre_negative
    
    ent_emb_mat[T_batch['e1_index'].values, :] = l2norm(e1_update_emb)
    ent_emb_mat[T_batch['e2_index'].values, :] = l2norm(e2_update_emb)
    ent_emb_mat[T_batch['fake_e1_index'].values, :] = l2norm(fake_e1_update_emb)
    ent_emb_mat[T_batch['fake_e2_index'].values, :] = l2norm(fake_e2_update_emb)
    rel_emb_mat[T_batch['rel_index'].values, :] = l2norm(relation_update_emb)
    
    print(np.mean(eg[eg > 0]))