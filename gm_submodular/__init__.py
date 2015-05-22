'''
This package contains code for submodular maximization and
structured learning using stochastic gradient decent
if you use this code, please cite Gygli et al. [3]

    [1] Lin, H. & Bilmes, J. Learning mixtures of submodular shells with application to document summarization. UAI 2012
    [2] Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J., & Glance, N. Cost-effective outbreak detection in networks. ACM SIGKDD 2007
    [3] Gygli, M., Grabner, H., & Gool, L. Van. Video Summarization by Learning Submodular Mixtures of Objectives. CVPR 2015
    [4] Minoux, M. . Accelerated greedy algorithms for maximizing submodular set functions. Optimization Techniques 1978
'''
__author__ = "Michael Gygli"
__maintainer__ = "Michael Gygli"
__email__ = "gygli@vision.ee.ethz.ch"
import numpy as np
import random
import logging
import warnings
import scipy.optimize
import scipy.linalg
import utils
import time
import gradient_functions
logger = logging.getLogger('gm_submodular')
logger.setLevel(logging.INFO)
skipAssertions=False


class DataElement:
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
    y,score=lazy_greedy_optimize(S,w,submod_fun,budget,loss_fun,False)

    if len(np.unique(S.getCosts()))>1:
        logger.debug('Cost benefit greedy')

        y_cost,score_cost=lazy_greedy_optimize(S,w,submod_fun,budget,loss_fun,True)
        if score_cost>score:
            return y_cost,score_cost

    return y,score



def lazy_greedy_optimize(S,w,submod_fun,budget,loss_fun=None,useCost=False,randomize=True):
    '''
    Implements the submodular maximization algorithm of [4]
    :param S: data object containing information on needed in the objective functions
    :param w: weights of the objectives
    :param submod_fun: submodular functions
    :param budget: budget
    :param loss_fun: optional loss function (for learning)
    :param useCost: boolean. Take into account the costs per element or not
    :param randomize: randomize marginals brefore getting the maximum. This results in selecting a random
    element among the top scoring ones, rather then taking the one with the lowest index.
    TODO: This can definitely be done faster than it is now
    :return: y, score: selected indices y and the score of the solution
    '''

    sel_indices=[]
    type='UC'
    if useCost:
        type='CB'

    ''' Init arrays to keep track of marginal benefits '''
    marginal_benefits=np.ones(len(S.Y),np.float32)*np.Inf
    mb_indices=np.arange(len(S.Y))
    isUpToDate=np.zeros((len(S.Y),1))

    costs=S.getCosts()


    currCost=0.0
    currScore=0.0
    i=0

    if loss_fun is None:
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

            #if verbose:
            #    print('%d has gain %.3f (before: %.3f)\n' % (mb_indices[0],t_marg,marginal_benefits[mb_indices[0]]))
            if not skipAssertions:
                assert marginal_benefits[mb_indices[0]]-t_marg > -10**-3, ('%s: Non-submodular objective at element %d!: Now: %.3f; Before: %.3f' % (type,mb_indices[0],t_marg,marginal_benefits[mb_indices[0]]))
            marginal_benefits[mb_indices[0]]=t_marg
            isUpToDate[mb_indices[0]]=True

            if randomize:
                idx1=np.random.permutation(len(marginal_benefits))
                idx2=(-marginal_benefits[idx1]).argsort(axis=0)
                mb_indices=idx1[idx2]
            else:
                mb_indices=(-marginal_benefits).argsort(axis=0)

            if marginal_benefits[-1]<0:
                warnings.warn('Non monotonic objective')


        ''' Select the highest scoring element '''
        if marginal_benefits[mb_indices[0]] > 0.0:
            logger.debug('Select element %d (gain %.3f)' % (mb_indices[0],marginal_benefits[mb_indices[0]]))
            sel_indices.append(mb_indices[0])
            if useCost:
                currScore=currScore + marginal_benefits[mb_indices[0]] * float(costs[mb_indices[0]])
            else:
                currScore=currScore + marginal_benefits[mb_indices[0]]
            currCost=currCost+ costs[mb_indices[0]]
            marginal_benefits[mb_indices[0]]=0
            mb_indices=(-marginal_benefits).argsort()
            isUpToDate[:]=0

        else:
            logger.debug(' If the best element is zero, we are done ')
            logger.debug(sel_indices)
            return sel_indices,currScore

        ''' Check if we still have budget to select something '''
        for elIdx in range(0,len(S.Y)):
            if costs[elIdx]+currCost>budget:
                marginal_benefits[elIdx]=0
                isUpToDate[elIdx]=1

        if marginal_benefits.max()==0:
            logger.debug('no elements left to select. Done')
            logger.debug('Selected %d elements with a cost of %.1f (max: %.1f)' % (len(sel_indices),currCost,budget))
            logger.debug(sel_indices)
            return sel_indices,currScore
        ''' Increase iteration number'''
        i+=1


def learnSubmodularMixture(training_examples,submod_shells,loss_fun,maxIter,use_l1_inequality=False):
    '''
    This code implements algorithm 1 of [1]
    :param S: training data. S[t].Y:             indices of possible set elements
                      S[t].y_gt:          indices selected in the ground truth solution
                      S[t].budget:        The budget of for this example
    :param submod_shells:    A cell containing submodular shell functions
                      They need to be of the format submodular_function = shell_function(S[t])
    :param   loss_function:    A submodular loss
    :param   maxIter:          Maximal number of iterations
    :return: learnt weights, weights per iteration
    '''
    if len(training_examples) ==0:
        raise IOError('No training examples given')


    ''' Initialize the weights '''
    function_list,names=utils.instaciateFunctions(submod_shells,training_examples[0])
    w_0=np.zeros(len(function_list),np.float)

    ''' Set learning rate according to theorem from
        "Learning Mixtures of Submodular Shells" - Lin & Bilmes 2012 '''
    T = len(training_examples)*maxIter
    M=len(submod_shells)
    G=1.0
    ''' Assume:
     - w_i,f_i are all upperbounded by 1
     - loss l <= B for some B
     - ||g_t|| <= G, for some G
     then, we use learning rate nu=2/ (lambda*t)
    with '''
    learn_lambda=G/M * np.sqrt((2+(1+np.log(T)) / float(T)))
    logger.debug('Training using %d samples' % T)

    if len(function_list)<=1:
        print('Just 1 function. No work for me here :-)\n')
        return 1

    ''' Start training '''
    logger.info('regularizer lambda: %.3f' % learn_lambda)
        
    logger.debug('Training running...')

    it=0
    w=[]
    exitTraining=False

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
            random.shuffle(training_examples)

        if np.mod(it,50)==0:
            logger.info('Example %d of %d' % (it,T))
        logger.debug('%s (budget: %d)' % (training_examples[t],training_examples[t].budget))
        logger.debug(training_examples[t].y_gt)

        ''' Instanciate the shells to submodular functions '''
        function_list,names=utils.instaciateFunctions(submod_shells,training_examples[t])

        ''' Approximate loss augmented inference
        (this is equivalent to a greedy submodular optimization) '''
        y_t,score = leskovec_maximize(training_examples[t],w[it],function_list,training_examples[t].budget,loss_fun)


        ''' Subgradient '''        
        if use_l1_inequality:
            score_t  = utils.evalSubFun(function_list,y_t,False)
            score_gt = utils.evalSubFun(function_list,list(training_examples[t].y_gt),True)
            g_t = score_t - score_gt
        else: # Lin et al. use an l2 regularized formulation, and have thus a different gradient
            g_t = learn_lambda*w[it] + utils.evalSubFun(function_list,y_t,False)
            g_t= g_t - utils.evalSubFun(function_list,list(training_examples[t].y_gt),True)
        logger.debug('Gradient:')
        logger.debug(g_t)

        ''' Update weights '''
        nu = 2.0 / float(learn_lambda*(it+1))
        logger.debug('Nu: %.3f' % nu)
        w[it]=w[it]-nu*g_t;

        ''' Project to feasible set'''
        if use_l1_inequality:
            # We want to keep the euclidean distance between the initial and the projected weight minimal
            obj=lambda w_t: np.inner(w_t-w[it],w_t-w[it])
            cons=[]
            bnds=[]
            # Define the bounds such that w[it]>0
            for idx in range(0,len(function_list)):
                bnds.append((0, None))

            # Define the l1-ball inequality
            cons.append({'type': 'ineq','fun' : lambda x: 1-x.sum()})
            cons=tuple(cons)
            bnds=tuple(bnds)

            # Optimize for the best projection into the l-1 ball
            if it==0:
                res=scipy.optimize.minimize(obj,w_0,constraints=cons,bounds=bnds)#, options={'maxiter':10**3})
            else:
                res=scipy.optimize.minimize(obj,w[it-1],constraints=cons,bounds=bnds)#, options={'maxiter':10**3})
            assert (res.x<-10**-5).any()==False
            w[it]=res.x

            # Note: We need to re-normalize the weights to sum to one, in order to give each SGD step the same weight
            if np.sum(w[it])>0:
                w[it]=w[it]/np.sum(w[it])

            if res.success==False:
                logger.error('Iteration %d: l1: Failed to find constraint solution on w' % it)
                w[it][w[it]<0]=0
                if w[it].sum()>0:
                    w[it]=w[it]/w[it].sum()
        else: # projection of [1]
            ''' update the weights accoring to  [1] algorithm 1'''
            w[it][w[it]<0]=0
            w[it]=w[it]/np.sum(w[it])
            w[it][np.isnan(w[it])]=0
            

        logger.debug('w[it]:\n')
        logger.debug(w[it])
        it=it+1
        logger.debug(it)
        if it>=len(training_examples)*maxIter:
            logger.warn('Break without convergence\n')
            exitTraining=True
        logger.debug("--- %.1f seconds ---" % (time.time() - start_time))

    ''' Return the averaged weights (See [1] algorithm 1) '''
    w_res = np.asarray(w).mean(axis=0)
    w_res/=w_res.sum()

    logger.info('----------------------------\n')
    logger.info('Weights:\n')
    for w_idx in range(len(w_res)):
        logger.info(' %20s: %2.3f%%' % (names[w_idx],round(10000*w_res[w_idx]) / 100))
    logger.info('----------------------------\n')

    return w_res,w