import numpy as np
import sklearn
import math

from GAD_module import *
#learn background models
#learning pairwise gmms

#calculate dependence tree
def get_top_anomaly(DATA,GMM_pairwise,MI_pairwise,max_order,
                    top_list=500,all=True,start_TA=float('inf')):

    BEST = []
    SEQ = []
    #max_order = 20
    N,K = DATA.shape
    index_set = range(0,N)
    
    while len(SEQ)<top_list:
        temp_score = 0
        N,K = DATA.shape #N samples and K features
        for i in range(0,max_order):
            if i<start_TA or all == True:
                trial_comb = combntns(range(0,i)) # needs corrections
                for f_subset in trial_comb:
                    #get the subset matrices
                    DATA_subset = DATA[f_subset][:,f_subset]
                    GMM_subset = []
                    MI_subset = []
                    for j in range(0,len(f_subset)):
                        GMM_subset.append([])
                        MI_subset.append([])
                        for k in range(0,len(f_subset)):
                            if j<k:
                                GMM_subset[j].append(GMM_subset[j][i])
                                MI_subset[j].append(MI_subset[j][i])
                            else:
                                GMM_subset[j].append(
                                    GMM_pairwise[f_subset[j]][f_subset[k]])
                                MI_subset[j].append(
                                    MI_pairwise[f_subset[j]][f_subset[k]])
                    #calculate the subset score
                    subset_score, subset_seq = get_subset_score(DATA_subset,
                                                                MI_subset,
                                                                GMM_subset,
                                                                N,K)
                    if subset_score < temp_score:
                        temp_score = subset_score
                        temp_seq = subset_seq
                        temp_order = i
                        temp_index = f_subset

            elif i>start_TA: #later this part
                #trial-add feature from the best i-1 candidates

        BEST.append([len(temp_seq),temp_order,temp_score,temp_index])
        print str(len(temp_seq))+' samples added into the list\n'
        SEQ.extend([index_set[i] for i in temp_seq])
        #remove these samples
        for i in temp_seq:
            np.delete(DATA,i,axis=0)
            del index_set[i]

if __name__ == '__main__':

    DATA = numpy.loadtxt('data.txt')
    LABEL = numpy.loadtxt('label.txt')
    #max number of component per gmm models
    M_max = 50
    #set the max order be the number of features
    _, max_order = DATA.shape
    #get pairwise gmms
    GMM_pairwise = get_all_pairwise_gmm(DATA,M_max) 
    #get mutual info for each pair, by mc sampling
    MI_pairwise = get_all_pairwise_MI(GMM_pair)
    anomaly_list = get_top_anomaly(DATA,GMM_pairwise,MI_pairwise,
                                   max_order,500,True,float('inf'))
    roc_auc = calculate_roc(anomaly_list,LABEL)
    print 'the final roc is ' + str(roc_auc)
