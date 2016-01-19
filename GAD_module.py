import math
import numpy as np
from scipy.stats import mvn,norm
from sklearn import mixture
from gmm_module import *
import networkx as nx

#generate gmm pairwise models
def get_all_pairwise_gmm(DATA,M_max):
    N,K = DATA.shape
    GMM_pair = []
    num_comp = []
    for i in range(0,K):
        GMM_pair.append([])
        for j in range(0,K):
            if j<i:
                GMM_pair[i].append(GMM_pair[j][i])
            elif j==i:
                temp_num,gmm = GMM_BIC(DATA[:,i].reshape(-1,1),M_max)
                GMM_pair[i].append(gmm) 
                num_comp.append(temp_num)
            else:
                temp_num,gmm = GMM_BIC(DATA[:,[i,j]],M_max)
                GMM_pair[i].append(gmm)
                num_comp.append(temp_num)
    return num_comp,GMM_pair

#get all pairwise gmms' MI
#for the same feature index, MI = 0
def get_all_pairwise_MI(GMM_pair,total_trials):
    MI_pair = [[0 for i in range(len(GMM_pair))] for j in range(len(GMM_pair))]
    occur =  [[0 for i in range(len(GMM_pair))] for j in range(len(GMM_pair))]
    count = 0
    for i in range(len(GMM_pair)):
        for j in range(i+1,len(GMM_pair)):
            print 'starting evaluating MI pairs '+str(i)+' '+str(j)+'\n'
            MI_pair[i][j],occur[i][j] = get_pair_mutual_info(GMM_pair[i][j],total_trials)
            if MI_pair[i][j]<0:
                print 'MI is negative: '+str(MI_pair[i][j])
                #raise ValueError('Mutual Information cannot be negative')
            elif MI_pair[i][j]<1e-10 and MI_pair[i][j]>=0:
                #avoid assigning too small MI
                MI_pair[i][j] = 1e-10
            MI_pair[j][i] = MI_pair[i][j]
            count+=1
    return MI_pair

#get mutual information of a pair of r.v.
#by MC sampling, number of generated samples fixed to 1e6
def get_pair_mutual_info(gm_model,total_trials):

    #total mc trials is 1e5
    #total_trials = 1e5
    #get the sorted weight index
    M = len(gm_model.weights_)
    weight_index_sorted = sorted(range(M), 
                                 key=lambda k: gm_model.weights_[k])
    weight_cumsum = np.cumsum(gm_model.weights_[weight_index_sorted])
    count = 0
    occur = 0
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
        temp_score_x = get_single_gm_score(rand_num,gm_model,0)
        temp_score_y = get_single_gm_score(rand_num,gm_model,1)

        #sample mutual info
        if temp_score_x!=0 and temp_score_y!=0 and temp_score_xy!=0:
            sample_MI = temp_score_xy-temp_score_x-temp_score_y
            if sample_MI<0:
                #print str(sample_MI)+' is negative \n'
                occur+=1
                #raise ValueError('MI sample cannot be negative')
            MI += sample_MI
        count+=1
    return MI/total_trials,occur

#generate score on a single dim specified by which
def get_single_gm_score(sample,gm_double,which):
    M = gm_double.weights_.size
    g_x = mixture.GMM(n_components = M)
    g_x.weights_ = gm_double.weights_
    g_x.means_ = gm_double.means_[:,which].reshape(-1,1)
    #print str(g_x.means_.shape)
    temp_covar = []
    for i in range(M):
        temp_covar.append(gm_double.covars_[i,which,which])
    g_x.covars_ = np.array(temp_covar).reshape(-1,1)
    #print str(g_x.covars_.shape)
    return g_x.score(sample[which])

#get dependence tree structure for a feature subset
def get_DT(MI_subset):
    #we use networkx package
    #create a Graph object
    G = nx.Graph()
    G.add_nodes_from(range(len(MI_subset)))
    edge_list = []
    for i in range(len(MI_subset)):
        for j in range(i+1,len(MI_subset)):
            #we negate MI, turning into a MST problem
            edge_list.extend([(i,j,-MI_subset[i][j]),(j,i,-MI_subset[j][i])])
    G.add_weighted_edges_from(edge_list)
    
    min_span_tree = sorted(list(nx.minimum_spanning_edges(G)))
    #rearrange the mst s.t. all 1st axis is parents of 2nd axis
    N = len(min_span_tree)
    #indicates which nodes are added, hence they are not children
    indicator = np.zeros((N+1,1)) #add 1 to compensate for N edges and N+1 nodes
    temp1,temp2,_ = min_span_tree.pop(0)
    rearranged = [[temp1,temp2]]
    indicator[[rearranged[0][0],rearranged[0][1]]] = 1 #default parents
    while min_span_tree:
        for ins in range(len(min_span_tree)):
            if indicator[min_span_tree[ins][0]]==1:
                temp1,temp2,_ = min_span_tree.pop(ins)
                rearranged.append([temp1,temp2])
                indicator[temp2] = 1
                break
            elif indicator[min_span_tree[ins][1]]==1:
                temp1,temp2,_ = min_span_tree.pop(ins)
                rearranged.append([temp2,temp1])
                indicator[temp1] = 1
                break
    return rearranged

#get the subset score specified by node (feature) index
#Note: we should not consider more than N/2 samples or more than K/2 features.
#This is because the Bonferroni score is monotonic in [0, N/2] and [0, K/2]
def get_subset_score(DT,data_logpval,N,K,KK):
    #N: total number of samples
    #K: total number of original features

    #get the number of subset features
    #_,KK = data.shape
    #learn a DT on the feature subset
    #DT = get_DT(mi)
    #calculate DT p-value for each sample
    #data_logpval = calculate_logpval_DT(data,gm_model,DT)
    # sorted pval in ascending order
    logpval_index_sorted = sorted(range(N), 
                               key=lambda k: data_logpval[k])
    #calculate the Bonferroni corrected score
    sub_seq = [logpval_index_sorted[0]]
    #extract at least one sample
    log_p = data_logpval[logpval_index_sorted[0]]
    count = 1
    if N>1:
        for i in range(1,data_logpval.size):
            if (N-count)*np.exp(data_logpval[logpval_index_sorted[i]])<1:
                count+=1
                sub_seq.append(logpval_index_sorted[i])
                log_p+=data_logpval[logpval_index_sorted[i]]
            else:
                break
    #calculate the log of score
    log_score = log_p
    #number of samples correction
    log_score -= np.sum(np.log(range(2,N-count+1)))
    #number of features correction
    log_score -= np.sum(np.log(range(2,KK+1)))
    log_score -= np.sum(np.log(range(2,K-KK+1)))

    return log_score,sub_seq
                
def calculate_logpval_DT(data,gm_model,DT):
    
    #the root node is at DT[0][0]
    #get responsibilities, n*m matrix
    temp_id0,temp_id1 = DT[0][0],DT[0][1]
    temp_gmm = gm_model[temp_id0][temp_id0]

    post = temp_gmm.predict_proba(data[:,temp_id0].reshape(-1,1))
    #get double-sided p-value
    temp_logpval = np.log(get_single_pval(data[:,temp_id0],
                                          temp_gmm.means_,
                                          temp_gmm.covars_,post))
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
        post = temp_gmm.predict_proba(data[:,[data_gmm_idx0,data_gmm_idx1]])

        N,M = post.shape
        #calculate parent log pvalue
        '''
        co = []
        for j in range(0,M):
            co.append(temp_gmm.covars_[j,idx0,idx0])
        #get single dimension pval with uniform weights
        temp_single = get_single_pval(data[:,temp_id0],
                                         temp_gmm.means_[:,idx0],
                                         co,np.ones((N,M)))
        temp_double = get_double_pval(data[:,[data_gmm_idx0,data_gmm_idx1]],
                                      temp_gmm,post)
        #sanity check, joint is less than single
        for n in range(0,N):
            for j in range(0,M):
                if (temp_single[n,j]>=temp_double[n,j])==False:
                    print ((temp_double[n,j],temp_single[n,j],
                            temp_double[n,j]/temp_single[n,j]))
                    temp_double[n,j] = temp_single[n,j]
        temp_logpval+=np.log(np.sum(temp_double/temp_single,axis = 1))
        '''
        temp_cond = get_cond_pval(data[:,[data_gmm_idx0,data_gmm_idx1]],
                                  temp_gmm,idx1,idx0,post)
        temp_logpval+=np.log(temp_cond)
            
                                         
    return temp_logpval

#return N*M p-value
def get_single_pval(data,gm_means,gm_covars,post):

    #get double-sided p-value
    N,M = post.shape
    #print((N,M))
    single_pval = np.zeros((N,1))
    for j in range(0,N):
        for i in range(0,M):
            temp_cdf = norm.cdf((data[j]-gm_means[i])/
                                np.sqrt(gm_covars[i]))
            if data[j]<=gm_means[i]:
                single_pval[j] += 2*temp_cdf*post[j][i]
            else:
                single_pval[j] += 2*(1-temp_cdf)*post[j][i]
            #smallest p-value used is 1e-100
        if single_pval[j]<1e-10:
            single_pval[j] = 1e-10

    return single_pval

def get_cond_pval(data,gmm,x,y,post):
    #we want pval(x|y)
    N,_ = data.shape
    M = gmm.weights_.size
    row = []
    #M*N matrix
    cond_mean = []
    #1*M matrix
    cond_var = []
    for i in range(M):
        row.append(gmm.covars_[i,x,y]/
                   (np.sqrt(gmm.covars_[i,x,x]*gmm.covars_[i,y,y])))
    
        cond_mean.append(gmm.means_[i,x]-
                         row[i]*np.sqrt(gmm.covars_[i,x,x]/gmm.covars_[i,y,y])*
                         (data[:,y].reshape(-1,1)-gmm.means_[i,y]))
        cond_var.append(gmm.covars_[i,x,x]*(1-row[i]**2))
    cond_pval = np.zeros((N,1))
    for j in range(N):
        for i in range(M):
            temp_cdf = norm.cdf((data[j,x]-cond_mean[i][j])/
                                np.sqrt(cond_var[i]))
            if data[j,x]<=cond_mean[i][j]:
                cond_pval[j]+=(post[j][i]*2*temp_cdf)
            else:
                cond_pval[j]+=(post[j][i]*2*(1-temp_cdf))
        if cond_pval[j]<1e-10:
            cond_pval[j] = 1e-10

    return cond_pval

'''
#potential problems with mvn.mvnun
def get_double_pval(data,gmm,post):
    #calculate joint pvalue for each data sample
    #four possibilities relative to mu and data
    N,M = post.shape
    temp_double = np.zeros((N,M))
    
    for n in range(0,N):
        for j in range(0,M):
            #print n,j
            mean_target = gmm.means_[j]
            cov_target = gmm.covars_[j]
            #first quadrant w.r.t. mu[j,:]
            if (data[n,0]>=mean_target[0]
                and data[n,1]>=mean_target[1]):
                temp_double[n][j],_ = mvn.mvnun(np.array([data[n,0],data[n,1]]),
                                                np.array([1e4,1e4]),
                                                mean_target,cov_target)
            elif (data[n,0]<=mean_target[0]
                  and data[n,1]<=mean_target[1]):
                temp_double[n][j],_ = mvn.mvnun(np.array([-1e2,-1e2]),
                                                np.array([data[n,0],data[n,1]]),
                                                mean_target,cov_target)
            elif (data[n,0]<=mean_target[0]
                  and data[n,1]>=mean_target[1]):
                temp_double[n][j],_ = mvn.mvnun(np.array([-1e2,1e4]),
                                                np.array([data[n,0],data[n,1]]),
                                                mean_target,cov_target)
            elif (data[n,0]>=mean_target[0]
                  and data[n,1]<=mean_target[1]):
                temp_double[n][j],_ = mvn.mvnun(np.array([data[n,0],data[n,1]]),
                                                np.array([1e4,-1e2]),
                                                mean_target,cov_target)
            temp_double[n][j] = temp_double[n][j]*4
            #smallest p-value used is 1e-100
            if temp_double[n][j]>=1e-10:
                temp_double[n][j] = 1e-10
            temp_double[n][j] = temp_double[n][j]*post[n][j]
    return temp_double
'''

def calculate_roc(anomaly_list,LABEL,normal_cat):
    #calculate roc auc given the index of LABEL
    TD,FA = 0,0
    roc_auc = 0.
    for i in anomaly_list:
        if LABEL[i]!=normal_cat:
            TD+=1
        else:
            roc_auc+=TD
            FA+=1
    if TD==0:
        return 0
    elif FA==0:
        return 1
    else:
        return roc_auc/(TD*FA)

def get_subset_DATA_GMM_MI(DATA,GMM_pairwise,MI_pairwise,f_subset):
    DATA_subset = DATA[:,list(f_subset)]
    GMM_subset = []
    MI_subset = []
    for j in range(0,len(f_subset)):
        GMM_subset.append([])
        MI_subset.append([])
        for k in range(0,len(f_subset)):
            if j>k:
                GMM_subset[j].append(GMM_subset[k][j])
                MI_subset[j].append(MI_subset[k][j])
            else:
                GMM_subset[j].append(
                    GMM_pairwise[f_subset[j]][f_subset[k]])
                MI_subset[j].append(
                    MI_pairwise[f_subset[j]][f_subset[k]])
    return DATA_subset,GMM_subset,MI_subset
