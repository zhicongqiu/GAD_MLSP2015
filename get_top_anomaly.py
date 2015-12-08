import numpy as np
import sklearn
import math
import itertools
from GAD_module import *
#learn background models
#learning pairwise gmms

#calculate dependence tree
def get_top_anomaly(DATA,GMM_pairwise,MI_pairwise,max_order,
                    top_list=500,all=True,start_TA=float('inf'),TA_num=500):

    BEST = []
    SEQ = []
    N,K = DATA.shape
    index_set = range(0,N)
    feature_set = range(0,K)
    while len(SEQ)<top_list:
        
        temp_score = 0
        N,K = DATA.shape #N samples and K features
        for i in range(1,max_order+1):
            TA_list = []
            if i<=start_TA or all == True:
                for f_subset in itertools.combinations(feature_set,i):
                    #get the subset matrices
                    DATA_subset,GMM_subset,MI_subset = \
                    get_subset_DATA_GMM_MI(DATA,GMM_pairwise,MI_pairwise,f_subset)
                    #calculate the subset score
                    subset_score, subset_seq = \
                    get_subset_score(DATA_subset,MI_subset,GMM_subset,N,K)
                    if subset_score < temp_score:
                        temp_score = subset_score
                        temp_seq = subset_seq
                        temp_order = i
                        temp_fsubset = f_subset
                    #save the top-k if we start from i
                    if i==start_TA:
                        
                        '''
                        #insertion approach later
                        if len(TA_list)<TA_num:
                            TA_list.append([f_subset,subset_score])
                        elif len(TA_list)==TA_num:
                            TA_list.sort(key=lambda x:x[1])
                        else:
                            insert_into_list(TA_list,f_subset,subset_score) 
                        ''' 
                        TA_list.append([f_subset,subset_score])
                if i==start_TA:
                    trial_subset = sorted(TA_list,key=lambda x:x[1])[0:TA_num]
                    del TA_list
            elif i>start_TA: #later this part
                #trial-add feature from the best i-1 candidates
                tried_list = []
                for temp_trial in trial_subset:
                    remain_indicator = np.array(range(0,K))
                    remain_indicator[list(temp_trial)]=-1
                    for j in range(remain_indicator):
                        if remain_indicator[j]!=-1:
                            #a valid trial passed, set as a list
                            valid_trial = list(temp_trial)
                            for jj in range(len(valid_trial)):
                                if valid_trial[jj]<remain_indicator[j]:
                                    valid_trial.insert(jj,remain_indicator[j])
                                    break
                                elif jj==len(valid_trial)-1:
                                    valid_trial.extend(remain_indicator[j])
                            #turn into a tuple
                            valid_trial = tuple(valid_trial)
                            if valid_trial not in tried_list:
                                tried_list.append(valid_trial)
                                #get the subset matrices
                                DATA_subset,GMM_subset,MI_subset = \
                                get_subset_DATA_GMM_MI(DATA,GMM_pairwise,
                                                       MI_pairwise,valid_trial)
                                #calculate the subset score
                                subset_score, subset_seq = \
                                get_subset_score(DATA_subset,MI_subset,GMM_subset,N,K)
                                if subset_score < temp_score:
                                    temp_score = subset_score
                                    temp_seq = subset_seq
                                    temp_order = i
                                    temp_fsubset = valid_trial    
                                TA_list.append([valid_trial,subset_score])
                trial_subset = sorted(TA_list,key=lambda x:x[1])[0:TA_num]
                del TA_list
        BEST.append([len(temp_seq),temp_order,temp_score,temp_fsubset])
        print str(len(temp_seq))+' samples added into the list\n'
        SEQ.extend([index_set[i] for i in temp_seq])
        #remove these samples
        for i in temp_seq:
            np.delete(DATA,i,axis=0)
            del index_set[i]

if __name__ == '__main__':
    
    TRAIN = numpy.loadtxt('train.txt')
    DATA = numpy.loadtxt('data.txt')
    LABEL = numpy.loadtxt('label.txt')
    normal_cat = 1
    #max number of component per gmm models
    M_max = 50
    _,K = TRAIN.shape
    #get pairwise gmm clusters from DATA
    num_comp,GMM_pair = get_all_pairwise_gmm(TRAIN,M_max) 
    #get mutual info for each gm pair, by mc sampling
    MI_pair = get_all_pairwise_MI(GMM_pair)
    #set the order be the number of features to ensure monotonicity
    anomaly_list = get_top_anomaly(DATA,GMM_pair,MI_pair,K/2)
    roc_auc = calculate_roc(anomaly_list,LABEL,normal_cat)
    print 'the final roc is ' + str(roc_auc)
