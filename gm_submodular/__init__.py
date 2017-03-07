'''
This package contains code for submodular maximization and
structured learning using stochastic gradient decent.
It allows to learn an objective function as a linear combination of simpler functions f, i.e.

.. math:: o(\mathbf{x_\mathcal{V}},\mathbf{y})=\mathbf{w^\mathrm{T}}\mathbf{f}(\mathbf{x_\mathcal{V},y}).
This is known as the structured SVM problem.

In this package, we use stochastic gradient descent in combination with specialized algorithms for submodular maximization.
In particular, it implements the algorithms of [1,2,4] and allows to use AdaGrad [6,7] in the optimization.
Furthermore it allows to use supermodular loss functions, by approximating them using a variant
of a submodular-supermodular procedure based on [5].


You can find an example on how to do submodular maximization and structured learning
`HERE <http://www.vision.ee.ethz.ch/~gyglim/gm_submodular/gm_submodular_usage.html>`_.


If you use this code for your research, please cite [3]:

@inproceedings{GygliCVPR15,
   author ={Gygli, Michael and Grabner, Helmut and Van Gool, Luc},
   title = {Video Summarization by Learning Submodular Mixtures of Objectives},
   booktitle = {CVPR},
   year = {2015}
}

REFERENCES:

[1] Lin, H. & Bilmes, J. Learning mixtures of submodular shells with application to document summarization. UAI 2012

[2] Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J., & Glance, N. Cost-effective outbreak detection in networks. ACM SIGKDD 2007

[3] Gygli, M., Grabner, H., & Gool, L. Van. Video Summarization by Learning Submodular Mixtures of Objectives. CVPR 2015

[4] Minoux, M. . Accelerated greedy algorithms for maximizing submodular set functions. Optimization Techniques. 1978

[5] Narasimhan, M., & Bilmes, J. A submodular-supermodular procedure with applications to discriminative structure learning. UAI. 2005

[6] Duchi, J., Hazan, E., & Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. Journal of Machine Learning Research 2011

[7] Dyer, C. Notes on AdaGrad
'''
__author__ = "Michael Gygli"
__maintainer__ = "Michael Gygli"
__email__ = "gygli@vision.ee.ethz.ch"
__version__="0.1"
__license__='BSD licence. If you use this code, please cite Gygli et al. [3]'

import numpy as np
import random
import logging
import warnings
import scipy.optimize
import scipy.linalg
import utils
import time
from IPython.core.debugger import Tracer
logger = logging.getLogger('gm_submodular')
logger.setLevel(logging.ERROR)
skipAssertions=False


class DataElement:
    '''
    Defines a DataElement.
    For inference, this needs the function getCosts(), and a set Y (candidate elements).

    '''
    def __init__(self):

        Y=[]

    def getCosts(self):
        raise NotImplementedError

    def __str__(self):
        return 'DataElement'


def leskovec_maximize(S,w,submod_fun,budget,loss_fun=None):
    '''
    Implements the submodular maximization algorithm of [2]

    :param S: data object containing information on needed in the objective functions
    :param w: weights of the objectives
    :param submod_fun: submodular functions
    :param budget: budget
    :param loss_fun: optional loss function (for learning)
    :return: y, score: selected indices y and the score of the solution
    '''

    logger.debug('Uniform cost greedy')
    y,score,minoux_bound=lazy_greedy_maximize(S,w,submod_fun,budget,loss_fun,False)
    if len(np.unique(S.getCosts()))>1:
        logger.debug('Cost benefit greedy')

        y_cost,score_cost,minoux_bound_cost=lazy_greedy_maximize(S,w,submod_fun,budget,loss_fun,True)
        if score_cost>score :
            if minoux_bound_cost>0:
                logger.debug('Score: %.3f (%.1f%% of Minoux bound; 31%% of Leskovec bound)' % (score, 100*(score / float(minoux_bound_cost))))
            return y_cost,score_cost,minoux_bound_cost
        else:
            if minoux_bound>0:
                logger.debug('Score: %.3f (%.1f%% of Minoux bound; 31%% of Leskovec bound)' % (score, 100*(score / float(minoux_bound))))
    elif minoux_bound>0:
        logger.debug('Score: %.3f (%.1f%% of the Minoux bound; 63%% of Nemhauser bound)' % (score, 100*(score / float(minoux_bound))))

    return y,score,minoux_bound

def modular_approximation(loss,pi,S):
    '''
    Computes a modular approximation of a loss function. Algorithm based on [5]

    :param loss: the supermodular loss function we want to approximate
    :param pi: an ordering on S.Y
    :param S: DataElement. needs S.Y
    :return:
    '''
    W_old=[]
    scores=np.zeros(len(S.Y))
    for i in range(0,len(S.Y)):
        W = W_old[:]
        W.append(pi[i])
        scores[pi[i]]=loss(S, W) - loss(S, W_old)
        W_old = W
    return lambda S, X: scores[X].sum()#+loss(S,[])

def submodular_supermodular_maximization(S,w,submod_fun,budget,loss,delta=10**-100):
    '''
    Does submodular maximization with a supermodular loss. Thus
    Optmizes it using a submodular-supermodular procedure.
    Algorithm based on [5].
    Adapted such that the supermodular loss is apprixmated rather then the submodular objectives

    :param S: DataElement
    :param w: objective weights
    :param submod_fun: objective functions
    :param budget: budget
    :param loss: the supermodular loss function
    :return: list of selected indices, (approximate) score
    '''
    #FIXME: recheck for correctness. Is the modular approximation really an upper bound on the correct
    # submodular loss?
    n = 0
    pi = S.Y[np.random.permutation(len(S.Y))]
    improvementFound = True
    maxVal = -np.inf
    A = []
    A_old=[]

    iter=0
    while improvementFound:
        iter+=1
        # Get a modular approximation of the loss at pi
        #logger.info('Get modular approximation of the loss')
        h = modular_approximation(loss,pi,S)

        #Solve submodular minimization using the previous solution A to approximate h
        A_old=A
        A,val,online_bound=leskovec_maximize(S,w,submod_fun,budget,loss_fun=h)
        logger.debug('Selected %d elements: [%s]' % (len(A),' '.join(map(lambda x: str(x),A))))
        assert (len(A) == S.budget)

        # update pi
        D = np.setdiff1d(S.Y,A)
        pi = A[:]
        pi.extend(D[np.random.permutation(len(D))])
        n += 1

        if val - delta > maxVal:
            logger.debug('Have improvement: From %.3f to %.3f ' % (maxVal,val))
            maxVal=val
            improvementFound=True
        else:
            improvementFound=False
    logger.debug('Took %d iteations.' % iter)
    if len(A_old) < S.budget:
        logger.warn('Selected solution is smaller than the budget (%d of %d' % (len(A_old),S.budget))
    return A_old,maxVal



def lazy_greedy_maximize(S,w,submod_fun,budget,loss_fun=None,useCost=False,randomize=True):
    '''
    Implements the submodular maximization algorithm of [4]

    :param S: data object containing information on needed in the objective functions
    :param w: weights of the objectives
    :param submod_fun: submodular functions
    :param budget: budget
    :param loss_fun: optional loss function (for learning)
    :param useCost: boolean. Take into account the costs per element or not
    :param randomize: randomize marginals brefore getting the maximum. This results in selecting a random element among the top scoring ones, rather then taking the one with the lowest index.
    :return: y, score: selected indices y and the score of the solution
    '''

    sel_indices=[]
    type='UC'
    if useCost:
        type='CB'

    ''' Init arrays to keep track of marginal benefits '''
    marginal_benefits = np.ones(len(S.Y),np.float32)*np.Inf
    mb_indices = np.arange(len(S.Y))
    isUpToDate = np.zeros((len(S.Y),1))

    costs = S.getCosts()


    currCost  = 0.0
    currScore = 0.0
    i = 0

    if loss_fun is None:
        #FIXME: this is not actually a zero loss, but just a loss that is the same for all elements
        # This is a hack to ensure that, in case all weights w are zero, a non empty set is selected
        # i.e., just a random subset of size S.budget
        loss_fun=utils.zero_loss

    ''' Select as long as we are within budget and have elements to select '''
    while True:
        ''' Find the highest scoring element '''
        while (isUpToDate[mb_indices[0]]==0):
            cand=list(sel_indices)
            cand.append(mb_indices[0])
            if useCost:
                t_marg=((np.dot(w,utils.evalSubFun(submod_fun,cand,False,w)) + loss_fun(S,cand)) - currScore) / float(costs[mb_indices[0]])
            else:
                t_marg=(np.dot(w,utils.evalSubFun(submod_fun,cand,False,w)) + loss_fun(S,cand) - currScore)

            if not skipAssertions:
                assert marginal_benefits[mb_indices[0]]-t_marg >=-10**-5, ('%s: Non-submodular objective at element %d!: Now: %.3f; Before: %.3f' % (type,mb_indices[0],t_marg,marginal_benefits[mb_indices[0]]))
            marginal_benefits[mb_indices[0]]=t_marg
            isUpToDate[mb_indices[0]]=True

            if randomize:
                idx1=np.random.permutation(len(marginal_benefits))
                idx2=(-marginal_benefits[idx1]).argsort(axis=0)
                mb_indices=idx1[idx2]
            else:
                mb_indices=(-marginal_benefits).argsort(axis=0)

            if not skipAssertions:
                assert marginal_benefits[-1]> -10**-5,'Non monotonic objective'

        # Compute upper bound (see [4])
        if i==0:
            best_sel_indices=np.where(costs[mb_indices].cumsum()<=budget)[0]
            minoux_bound = marginal_benefits[mb_indices][best_sel_indices].sum()


        ''' Select the highest scoring element '''
        if marginal_benefits[mb_indices[0]] > 0.0:
            logger.debug('Select element %d (gain %.3f)' % (mb_indices[0],marginal_benefits[mb_indices[0]]))
            sel_indices.append(mb_indices[0])

            if useCost:
                currScore=currScore + marginal_benefits[mb_indices[0]] * float(costs[mb_indices[0]])
            else:
                currScore=currScore + marginal_benefits[mb_indices[0]]
            currCost=currCost+ costs[mb_indices[0]]

            # Set the selected element to -1 (so that it is not becoming a candidate again)
            # Set all others to not up to date (so that the marignal gain will be recomputed)
            marginal_benefits[mb_indices[0]] = 0#-np.inf
            isUpToDate[isUpToDate==1]=0
            isUpToDate[mb_indices[0]]=-1

            mb_indices=(-marginal_benefits).argsort()

        else:
            logger.debug(' If the best element is zero, we are done ')
            logger.debug(sel_indices)
            return sel_indices,currScore,minoux_bound

        ''' Check if we still have budget to select something '''
        for elIdx in range(0,len(S.Y)):
            if costs[elIdx]+currCost>budget:
                marginal_benefits[elIdx]=0
                isUpToDate[elIdx]=1

        if marginal_benefits.max()==0:
            logger.debug('no elements left to select. Done')
            logger.debug('Selected %d elements with a cost of %.1f (max: %.1f)' % (len(sel_indices),currCost,budget))
            logger.debug(sel_indices)
            return sel_indices,currScore,minoux_bound
        ''' Increase iteration number'''
        i+=1

class SGDparams:
    '''
        Class for the parameters of stochastic gradient descent used for learnSubmodularMixture
    '''
    def __init__(self,**kwargs):
        self.momentum=0.0  #: defines the momentum used. Default: 0.0
        self.use_l1_projection=False #: project the weights into an l_1 ball (leads to sparser solutions). Default: False
        self.use_ada_grad=False #: use adaptive gradient [6]? Default: False
        self.max_iter=10 #: number of passes throught the dataset (3-10 should do). Default: 10
        self.norm_objective_scores=False #: normalize the objective scores to sum to one. This improved the learnt weights and can be considered to be the equivalent to l1 normalization of feature points in a standard SVM
        self.learn_lambda=None #: learning rate. Default: Estimated using [1]
        self.nu=lambda t,T: 1.0/np.sqrt(t+1) #: Function nu(t,T) to compute nu for each iteration, given the current iteration t and the maximal number of iterations T. Default: 1/sqrt(t+1)
        for k,v in kwargs.items():
            setattr(self,k,v)

    def __str__(self):
        return 'SGDparams\n-----\n%s' % '\n'.join(map(lambda x: '%22s:\t%s' % (x, str(self.__dict__[x])),self.__dict__.keys()))



def learnSubmodularMixture(training_data, submod_shells, loss_fun, params=None, loss_supermodular=False):
    '''
    Learns mixture weights of submodular functions. This code implements algorithm 1 of [1]

    :param training_data: training data. S[t].Y:             indices of possible set elements
                      S[t].y_gt:          indices selected in the ground truth solution
                      S[t].budget:        The budget of for this example
    :param submod_shells:    A cell containing submodular shell functions
                      They need to be of the format submodular_function = shell_function(S[t])
    :param   loss_function:    A (submodular) loss
    :param   maxIter:          Maximal number of iterations
    :param   loss_supermodular: True, if the loss is supermodular. Then, [5] is used for loss-augmented inference
    :return: learnt weights, weights per iteration
    '''

    if params == None:
        params = SGDparams()
    logger.info('%s' % params)

    if len(training_data) ==0:
        raise IOError('No training examples given')
    # Make a copy of the training samples so that is doesn't shuffle the  input list
    training_examples=training_data[:]

    ''' Initialize the weights '''
    function_list,names=utils.instaciateFunctions(submod_shells,training_examples[0])
    w_0=np.ones(len(function_list),np.float)
    #w_0=np.random.rand(len(function_list))

    learn_lambda = params.learn_lambda
    T = len(training_examples)*params.max_iter
    if learn_lambda is None:
        ''' Set learning rate according to theorem from
            "Learning Mixtures of Submodular Shells" - Lin & Bilmes 2012 '''
        M=len(submod_shells)
        G=1.0
        ''' Assume:
         - w_i,f_i are all upperbounded by 1
         - loss l <= B for some B
         - ||g_t|| <= G, for some G
         then, we use learning rate nu=2/ (lambda*t)
        with '''
        learn_lambda=G/M * np.sqrt((2+(1+np.log(T)) / float(T)))

    # fudge factor as in http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
    fudge_factor = 1e-6 #for numerical stability
    logger.debug('Training using %d samples' % T)

    if len(function_list)<=1:
        logger.info('Just 1 function. No work for me here :-)\n')
        return 1

    ''' Start training '''
    logger.info('regularizer lambda: %.3f' % learn_lambda)

    it=0
    w=[]
    exitTraining=False

    g_t_old=np.zeros(len(function_list))
    if params.use_ada_grad:
        historical_grad=np.zeros(len(function_list))

    while exitTraining==False:
        
        start_time = time.time()

        if it==0:
            w.append(w_0);
        else:
            w.append(w[it-1])

        t=np.mod(it,len(training_examples))

        ''' Before each iteration: shuffle training examples '''
        if t==0:
            logger.debug('Suffle training examples')

            training_examples=training_examples
            random.shuffle(training_examples)

        if np.mod(it,50)==0:
            logger.info('Example %d of %d' % (it,T))
        logger.debug('%s (budget: %d)' % (training_examples[t],training_examples[t].budget))
        logger.debug(training_examples[t].y_gt)

        ''' Instanciate the shells to submodular functions '''
        function_list,names=utils.instaciateFunctions(submod_shells,training_examples[t])

        ''' Approximate loss augmented inference
        (this is equivalent to a greedy submodular optimization) '''
        if loss_supermodular:
            y_t,score = submodular_supermodular_maximization(training_examples[t],w[it],function_list,training_examples[t].budget,loss_fun)
        else:
            y_t,score,online_bound = leskovec_maximize(training_examples[t],w[it],function_list,training_examples[t].budget,loss_fun)
        assert(len(y_t)==training_examples[t].budget)


        ''' Subgradient '''
        score_t  = utils.evalSubFun(function_list,y_t,False)
        score_gt = utils.evalSubFun(function_list,list(training_examples[t].y_gt),True)

        if params.norm_objective_scores:
            score_t /= score_t.sum()
            score_gt /= score_gt.sum()


        if params.use_l1_projection:
            g_t = score_t - score_gt
        else: # Lin et al. use an l2 regularized formulation, and have thus a different gradient
            g_t = learn_lambda*w[it] + (score_t - score_gt)
        g_t = ((1 - params.momentum) * g_t + params.momentum * g_t_old)

        if params.use_ada_grad:
            # See [6,7]
            g_t_old=g_t
            historical_grad+= g_t**2
            g_t= g_t / (fudge_factor + np.sqrt(historical_grad))
        logger.debug('Gradient:')
        logger.debug(g_t)

        ''' Update weights '''
        if params.nu is None:
            nu = 2.0 / float(learn_lambda*(it+1))
        else:
            if hasattr(params.nu,'__call__'):
                nu=params.nu(it,T)
            else:
                nu=params.nu
        if np.mod(it,10)==0:
            logger.info('Nu: %.3f; Gradient: %s; Grad magnitue (abs): %.4f' % (nu, ', '.join(map(str,g_t)),nu*np.sum(np.abs(g_t))))

        w[it]=w[it]-nu*g_t

        ''' Project to feasible set'''
        if params.use_l1_projection:
            # We want to keep the euclidean distance between the initial and the projected weight minimal
            if params.use_ada_grad:
                # See [7]
                obj=lambda w_t: (np.multiply(w_t-w[it],w_t-w[it]) / (fudge_factor + historical_grad)).sum()
            else:
                obj=lambda w_t: np.inner(w_t-w[it],w_t-w[it])
            cons=[]
            bnds=[]
            # Define the bounds such that w[it]>0
            for idx in range(0,len(function_list)):
                bnds.append((0, None))

            # Define the l1-ball inequality
            cons.append({'type': 'ineq','fun' : lambda x: 1-np.abs(x).sum()})
            cons=tuple(cons)
            bnds=tuple(bnds)

            # Optimize for the best projection into the l-1 ball
            if it==0:
                res=scipy.optimize.minimize(obj,w_0,constraints=cons,bounds=bnds)#, options={'maxiter':10**3})
            else:
                res=scipy.optimize.minimize(obj,w[it-1],constraints=cons,bounds=bnds)#, options={'maxiter':10**3})
            if res.success:
                assert (res.x<-10**-5).any()==False
                w[it]=res.x

                # Note: We need to re-normalize the weights to sum to one, in order to give each SGD step the same weight
                if np.sum(w[it])>0:
                    w[it]=w[it]/np.sum(w[it])
            else:
                logger.warn('Iteration %d: l1: Failed to find constraint solution on w' % it)
                w[it][w[it]<0]=0
                if w[it].sum()>0:
                    w[it]=w[it]/w[it].sum()

        else: # projection of [1]
            ''' update the weights accoring to  [1] algorithm 1'''
            w[it][w[it]<0]=0
            if w[it].sum()>0:
                w[it]=w[it]/np.sum(np.abs(w[it]))
            #w[it][np.isnan(w[it])]=0

        if np.mod(it,10)==0:
            logger.info('w[it]:\t%s' % ', '.join(map(str,w[it])))
        it=it+1
        logger.debug(it)
        if it>=len(training_examples)*params.max_iter:
            logger.warn('Break without convergence\n')
            exitTraining=True
        logger.debug("--- %.1f seconds ---" % (time.time() - start_time))

    ''' Return the averaged weights (See [1] algorithm 1) '''
    w_res = np.asarray(w).mean(axis=0)
    w_res/=np.abs(w_res).sum()

    logger.info('----------------------------\n')
    logger.info('Weights:\n')
    for w_idx in range(len(w_res)):
        logger.info(' %20s: %2.3f%%' % (names[w_idx],round(10000*w_res[w_idx]) / 100))
    logger.info('----------------------------\n')

    return w_res,w
