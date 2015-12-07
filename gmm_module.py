import numpy as np
from sklearn import mixture

def GMM_BIC(X,M_max):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, M_max)
    #cv_types = ['spherical', 'tied', 'diag', 'full']
    cv_types = ['full']   
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                num_comp = n_components
    #score = best_gmm.score(X)
    return num_comp,best_gmm

def GMM_get_input_prob(x,GMM):
    return GMM.score(x)

