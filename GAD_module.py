import math
import numpy as np
from scipy.stats import mvn
from sklearn import mixture
from gmm_module import *
from MinimumSpanningTree import *

#generate gmm pairwise models
def get_all_pairwise_gmm(DATA,M_max):
    N,K = DATA.shape
    GMM_pairwise = []
    for i in range(0,K):
        for j in range(0,K):
            if j<i:
                GMM_pairwise[i].append(GMM_pairwise[j][i])
            elif j==i:
                _,gmm = GMM_BIC(DATA[:,i],M_max)
                GMM_pairwise[i].append(gmm) 
            else:
                num_comp,gmm = GMM_BIC(np.hstack((DATA[:,i],DATA[:,j])),M_max)
                GMM_pairwise[i].append(gmm)
    return GMM_pairwise

#get mutual information of a pair of r.v.
#by MC sampling, number of generated samples fixed to 1e6
def get_pair_mutual_info(gm_model):

    #total mc trials
    total_trials = 1e6
    #get the sorted weight index
    M = len(gm_model.weights_)
    weight_index_sorted = sorted(range(M), 
                                 key=lambda k: gm_model.weights_[k])
    weight_cumsum = np.cumsum(np.asarray(gm_model.weights_)\
                              [weight_index_sorted])
    #simulate 1e6 MC
    count = 0
    MI = 0.
    while count<total_trials:
        #generate a random number
        temp = np.random.uniform()
        for i in range(M):
            if weight_cumsum[i]>=temp:
                comp_sel = weight_index_sorted[i]
                break
        #based on the selected component, generate a mvn number
        rand_num = np.random.multivariate_normal(gm_model.means_[comp_sel],
                                                 gm_model.covars_[comp_sel])
        #get its score under gmm
        temp_score_xy = gm_model.score(rand_num)
        #x and y score
        temp_score_x,temp_score_y = get_single_gm_score(gm_model)
        #sample mutual info
        if temp_score_x!=0 and temp_score_y!=0 and temp_score_xy!=0:
            MI += (temp_score_xy-temp_score_x-temp_score_y)
        count+=1
    return MI/total_trials

#get all pairwise gmms' MI
#for the same feature index, MI = 0
def get_all_pairwise_MI(GMM_pair):
    MI_pair = [[0 for i in range(GMM_pair)] for j in range(len(GMM_pair))]
    count = 0
    for i in range(len(GMM_pair)):
        for j in range(i+1,len(GMM_pair)):
            MI_pair[i,j] = get_pair_mutual_info(GMM_pair[count])
            if MI_pair<0:
                raise ValueError('Mutual Information cannot be negative')
            elif MI_pair<1e-10:
                #avoid assigning too small MI
                MI_pair = 1e-10
            MI_pair[j][i] = MI_pair[i][j]
            count+=1
    return MI_pair

#get dependence tree structure for a feature subset
def get_DT(MI_subset):
    #we use Eppstein's MST implementation
    #we negate MI, turning into a MST problem
    min_span_tree = MinimumSpanningTree(-MI_subset)
    #rearrange the mst s.t. all 1st axis is parents of 2nd axis
    N = len(min_span_tree)
    #indicates which nodes are added, hence they are not children
    indicator = np.zeros((N+1,1)) #add 1 to compensate for N edges and N+1 nodes
    rearranged = [min_span_tree.pop(0)]
    indicator[[rearrange[0][0],rearrange[0][1]]] = 1 #default parents
    while not min_span_tree:
        for ins in len(min_span_tree):
            if indicator[min_span_tree[ins][0]]==1:
                temp1,temp2 = min_span_tree.pop(ins)
                rearrange.append(temp1,temp2)
                indicator[temp2] = 1
            elif indicator[min_span_tree[ins][1]]==1:
                temp1,temp2 = min_span_tree.pop(ins)
                rearrange.append((temp2,temp1))
                indicator[temp1] = 1
    return rearranged

#get the subset score specified by node (feature) index
#Note: we should not consider more than N/2 samples or more than K/2 features.
#This is because the Bonferroni score is monotonic in [0, N/2] and [0, K/2]
def get_subset_score(data,mi,gm_model,N,K):
    #N: total number of samples
    #K: total number of original features

    #get the number of subset features
    _,KK = data.shape
    #learn a DT on the feature subset
    DT = get_DT(mi)
    #calculate DT p-value for each sample
    data_logpval = calculate_logpval_DT(data,gm_model,DT)
    # sorted pval in ascending order
    logpval_index_sorted = sorted(range(N), 
                               key=lambda k: data_logpval[k])
    #calculate the Bonferroni corrected score
    sub_seq = [logpval_index_sorted[0]]
    #extract at least one sample
    log_p = data_logpval[logpval_index_sorted[0]]
    count = 1
    if N>1:
        for i in range(1,logdata_pval):
            if (N-count)*np.exp(data_logpval[logpval_index_sorted[i]])<1:
                count+=1
                sub_seq.append(logpval_index_sorted[i]))
                log_p+=data_logpval[logpval_index_sorted[i]]
            else:
                break
    #calculate the log of score
    log_score = log_p
    #number of samples correction
    log_score -= np.cumsum(np.log(range(2,N-count+1)))
    #number of features correction
    log_score -= np.cumsum(np.log(range(2,KK+1)))
    log_score -= np.cumsum(np.log(range(2,K-KK+1)))

    return log_score,sub_seq
                
def calculate_logpval_DT(data,gm_model,DT):
    
    #the root node is at DT[0][0]
    #get responsibilities, n*m matrix
    temp_id0,temp_id1 = DT[0][0],DT[0][1]
    temp_gmm = gm_model[temp_id0][temp_id0]

    post = temp_gmm.predict_proba(data[:][:,temp_id0])
    #get double-sided p-value
    temp_logpval = get_single_logpval(data[:,temp_id0],
                                      temp_gmm.means_,temp_gmm.covars_,post)
    for i in range(0,len(DT)):
        temp_id0,temp_id1 = DT[i][0],DT[i][1]
        #use idx0,idx1 to indicate which of the (x,y) is parent
        if temp_id0>temp_id1:
            data_gmm_idx0,data_gmm_idx1 = temp_id1,temp_id0
            idx0,idx1 = 1,0
        else:
            data_gmm_idx0,data_gmm_idx1 = temp_id0,temp_id1
            idx0,idx1 = 0,1
        temp_gmm = gm_model[temp_id0][temp_id1]
        if temp_id0<temp_id1:
            post = temp_gmm.predict_proba(np.hstack(
                (data[:,temp_id0],data[:,temp_id1])))
        else:
            post = temp_gmm.predict_proba(np.hstack(
                (data[:,temp_id1],data[:,temp_id0])))

        N,M = post.shape
        #calculate parent log pvalue

        co = []
        for j in range(0,M):
            co.append(temp_gmm.covars_[j,idx0,idx0])
        temp_single = get_single_logpval(data[:,temp_id0],
                                         temp_gmm.means_[:,idx0],
                                         co,post)
        temp_double = get_double_logpval(np.hstack((data[:,data_gmm_idx0],
                                                    data[:,data_gmm_idx1])),
                                         temp_gmm,post)
        temp_logpval += (temp_double-temp_single)
                                         
    return temp_logpval
def get_single_logpval(data,gm_means,gm_covars,post):

    #get double-sided p-value
    N,M = post.shape
    single_pval = np.zeros((N,1))
    for i in range(0,M):
        temp_ef = math.erfc((data-gm_means[i])/
                                   np.sqrt(gm_covars[i]))
        for j in range(0,len(temp_ef)):
            if temp_ef[j]<=0.5:
                single_pval[j]+=temp_ef[j]*2
            else:
                single_pval[j]+=2*(1-temp_ef[j])
    return np.log(single_pval)

def get_double_logpval(data,gmm,post):
    #calculate joint pvalue for each data sample
    #four possibilities relative to mu and data
    N,M = post.shape
    temp_double = np.zeros((N,1))
    for n in range(0,N):
        for j in range(0,M):
            mean_target = gmm.means_[j,:]
            cov_target = gmm.covars_[j,:,:]
            #first quadrant w.r.t. mu[j,:]
            if (data[n,0]>=mean_target[0]
                and data[n,1]>=mean_target[1]):
                temp_double[n]+=\
                4*post(n,j)*mvn.mvnun(np.array([data[n,0],data[n,1]]),
                                      np.array([1e6,1e6]),
                                      mean_target,cov_target)
            elif (data[n,0]<=mean_target[0]
                  and data[n,1]<=mean_target[1]):
                temp_double[n]+=\
                4*post(n,j)*mvn.mvnun(np.array([-1e6,-1e6]),
                                      np.array([data[n,0],data[n,1]]),
                                      mean_target,cov_target)
            elif (data[n,0]<=mean_target[0]
                  and data[n,1]>=mean_target[1]):
                temp_double[n]+=\
                4*post(n,j)*mvn.mvnun(np.array([-1e6,data[n,1]]),
                                      np.array([data[n,0],1e6]),
                                      mean_target,cov_target)
            elif (data[n,0]>=mean_target[0]
                  and data[n,1]<=mean_target[1]):
                temp_double[n]+=\
                4*post(n,j)*mvn.mvnun(np.array([data[n,0],-1e6]),
                                      np.array([1e6,data[n,1]]),
                                      mean_target,cov_target)
    return np.log(temp_double)