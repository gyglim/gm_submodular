'''
 Implementation of the objectives used in
 M. Gygli, H. Grabner, L. Van Gool - Video Summarization by learning submodular mixtures of objectives. CVPR 2015
'''
__author__ = "Michael Gygli"
__maintainer__ = "Michael Gygli"
__email__ = "gygli@vision.ee.ethz.ch"

import numpy as np
from IPython.core.debugger import Tracer


def intersect_complement_loss(S,selection):
    '''
    :param S: A DataElement
    :param selection: a list of selected indices
    :return: the loss (in  [0; 1])
    '''

    #set intersection is much faster that numpy intersect1d
    return (len(selection)-len(set(S.y_gt).intersection(selection)))/float(len(S.y_gt))

def representativeness_shell(S):
    '''
        Representativeness shell Eq. (8)
    :param S: DataElement with function getDistances()
    :return: representativeness objective
    '''
    tempDMat=S.getDistances()
    norm=tempDMat.mean()
    return (lambda X: (1 - kmedoid_loss(X,tempDMat,float(norm))))

def kmedoid_loss(X,distMat,norm):
    '''
    Implements Eq (7)

    :param X: selected indices
    :param distMat: distance matrix
    :param norm: normalizer. defined the distance to the phantom element
    :return: k-medoid loss
    '''
    if len(X)>0:
        min_dist=distMat[:,X].min(axis=1)
        min_dist[min_dist>norm] = norm
        return min_dist.mean()/norm
    else:
        return 1

def random_shell(S):
    '''
     Random shell (to check noise sensitivity).
     Assigns each element in S.Y a random value.
     The score of a solution is the sum over the random values of this solution
    :param S: DataElement
    :return: random objective
    '''
    #np.random.seed(0)
    rand_scores=np.random.rand(len(S.Y))
    return (lambda X: np.sum(rand_scores[X]) /  float(S.budget))


def x_coord_shell(S):
    return (lambda X: np.sum(S.x[np.array(X),0]) /  (S.budget*S.x[:,0].max()))

def earliness_shell(S):
    '''
    :param S: DataElement
    :return: earliness objective
    '''
    return (lambda X: (np.max(S.Y)*len(X) - np.sum(S.Y[X]))/float(S.budget*np.max(S.Y)))
