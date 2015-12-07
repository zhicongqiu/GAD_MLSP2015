import numpy as np
import sklearn
import math

from GAD_module import *
#learn background models
#learning pairwise gmms

#calculate dependence tree
BEST = []
max_order = 20
while not exhausted:
    temp_score = 0
    for i in range(0,max_order):
        if i<start_TA or all == True:
            trial_comb = combntns(range(0,i))
            for f_subset in trial_comb:
                subset_score, subset_seq = get_subset_score(f_subset)
                if subset_score < temp_score:
                    temp_score = subset_score
                    temp_seq = subset_seq
                    temp_order = i
        elif i>start_TA: #later this part

    if count_while == 0:
        BEST = [temp_order,len(temp_seq),temp_score]
        SEQ = temp_seq
    else:
        np.vstack((BEST,[temp_order,len(temp_seq),temp_score]))
        np.vstack((SEQ,temp_seq))

