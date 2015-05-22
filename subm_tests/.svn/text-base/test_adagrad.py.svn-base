# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 15:15:52 2015

@author: gyglim
"""

import time                                                
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import numpy as np
import sys
from IPython.core.debugger import Tracer

# Load submodular package
import gm_submodular
import gm_submodular.utils
from gm_submodular.utils import timeit
import gm_submodular.example_objectives as ex
gm_submodular.logger.setLevel('ERROR')
import logging
logger = logging.getLogger('test_adagrad')


class ClusterData(gm_submodular.DataElement):            
    def __init__(self):
        num_points=100
        x=np.random.rand(num_points/5,2)
        self.cluster_assignment=np.zeros(x.shape[0])
        for i in range(0,5):
            new_points=np.random.rand(2)*np.random.rand(2)*30+(np.random.randn(num_points/5,2))
            x=np.vstack((x,new_points))
            self.cluster_assignment=np.hstack((self.cluster_assignment,np.ones(new_points.shape[0])*(i+1)))
       
        self.dist_v=dist.pdist(x)
        
        self.Y=np.arange(0,x.shape[0])
        self.x=x
        self.budget=10
    def getCosts(self):
        return np.ones(self.dist_v.shape[0])
    def getDistances(self):
        return dist.squareform(self.dist_v)**2

        
def plotExample(S,objectives,weights, title, res_ada=None, res_simple=None):   
    # Maximize the objectives
    selected_elements,score=gm_submodular.leskovec_maximize(S,weights,objectives,S.budget)
    
    # Plot results
    plt.figure(figsize=(15,10)) # Definition of a larger figure (in inches)
    colors=['g','y','black','c','m','grey']
    for i in set(S.cluster_assignment):
        plt.scatter(S.x[S.cluster_assignment==i,0],S.x[S.cluster_assignment==i,1], c=colors[int(i)], alpha=0.66, s=50, linewidths=1)
    plt.legend(map(lambda x: str(int(x)),set(S.cluster_assignment)))
    plt.scatter(S.x[selected_elements,0],S.x[selected_elements,1], c='green', s=100, alpha=1, linewidths=2)
    if res_ada is not None:
        plt.scatter(S.x[res_ada,0],S.x[res_ada,1], c='blue', s=120, alpha=0.5, linewidths=2)
    if res_simple is not None:
        plt.scatter(S.x[res_simple,0],S.x[res_simple,1], c='red', s=120, alpha=0.5, linewidths=2)
    #print('Selected points: %s' % ' '.join(map(lambda x: str(x),selected_elements)))
    plt.title(title)




import gm_submodular
import sys
import gm_submodular.utils
from gm_submodular.utils import timeit
import gm_submodular.example_objectives as ex
gm_submodular.logger.setLevel('ERROR')
import logging
logger = logging.getLogger('test_adagrad')

regularization={'delta':0,'l2_regularizer': False, 'lambda_regularizer': 1, 'l1_constraint': True,'l2_constraint': False, 'l1_regularizer':False, 'l2_inequality': False, 'l1_inequality': False}

class ClusterData(gm_submodular.DataElement):
    def __init__(self):
        num_points=100
        x=np.random.rand(num_points/5,2)
        self.cluster_assignment=np.zeros(x.shape[0])
        for i in range(0,5):
            new_points=np.random.rand(2)*np.random.rand(2)*30+(np.random.randn(num_points/5,2))
            x=np.vstack((x,new_points))
            self.cluster_assignment=np.hstack((self.cluster_assignment,np.ones(new_points.shape[0])*(i+1)))

        self.dist_v=dist.pdist(x)

        self.Y=np.arange(0,x.shape[0])
        self.x=x
        self.budget=10
    def getCosts(self):
        return np.ones(self.dist_v.shape[0])
    def getDistances(self):
        return dist.squareform(self.dist_v)**2
def test_l1_constraint(weights=[1,1,0,0,0],output=True):
    shells=[ex.representativeness_shell,ex.earliness_shell,ex.random_shell,ex.random_shell,ex.random_shell]
    # Create tranining data (use result of the k-modoid objective)
    training_examples=[]
    for i in range(0,50):
        S=ClusterData()
        S.budget=5
        objectives,obj_names=gm_submodular.utils.instaciateFunctions(shells,S)
        selected_elements,score=gm_submodular.leskovec_maximize(S,weights,objectives,S.budget*2)
        S.y_gt=list(np.array(selected_elements)[np.random.permutation(S.budget*2)][0:S.budget])
        training_examples.append(S)

    # Learn the weights. Given that we used the k-medoid results as ground truth, this objective should get all the weight
    regularization={'l2_inequality': False, 'l1_inequality': False}
    weights_simple,dummy=gm_submodular.learnSubmodularMixture(training_examples, shells,None, 2,'simple',regularization)
    #print('Simple:')
    regularization={'l2_inequality': False, 'l1_inequality': True}
    #print('l1-ball:')
    weights_l1,dummy=gm_submodular.learnSubmodularMixture(training_examples, shells,None, 2,'simple',regularization)

    #regularization={'delta':0,'l1_constraint': True,'l2_constraint': False,'l1_inequality':False,'l2_inequality':False,'l2_regularizer':True,'l1_regularizer':False}
    #weights_lin,dummy=gm_submodular.learnSubmodularMixture(training_examples, shells,None, 2,'AdaGrad',regularization)
    
    
    weights=np.array(weights,np.float32)
    weights_l1/=weights_l1.sum()
    weights_simple/=weights_simple.sum()
    #weights_lin/=weights_lin.sum()
    weights/=float(weights.sum())
    diff_l1=np.abs(weights-weights_l1)
    diff_simple=np.abs(weights-weights_simple)
    #diff_lin=np.abs(weights_lin-weights)
    if output:
        print('SSE to target weights:')
        print('l1 projection: %.5f %% (weights: %s)' % (100*diff_l1.sum(),', '.join(map(lambda x: str(round(1000*x)/10),weights_l1))))
        print('Simple:        %.5f %% (weights: %s)' % (100*diff_simple.sum(),', '.join(map(lambda x: str(round(1000*x)/10),weights_simple))))
        #print('Lin   :        %.5f %% (weights: %s)' % (100*diff_lin.sum(),', '.join(map(lambda x: str(round(1000*x)/10),weights_lin))))
        print('\n')
        sys.stdout.flush()          
    return 100*diff_l1, 100*diff_simple#,100*diff_lin
    
def compare_l1_2_lin():
    l1=[]
    lin=[]
    simple=[]
    weight_set=[
             [1,0,0,0,0],
             [0.75,0.25,0,0,0],
             [0.5,0.5,0,0,0],
             [0.25,0.75,0,0,0],
             [0,1,0,0,0]]
             
    for wIdx in range(0,len(weight_set)):
        for runNr in range(0,5):
            #diff_l1,diff_simple,diff_lin=test_l1_constraint(weight_set[wIdx],True)
            diff_l1,diff_simple=test_l1_constraint(weight_set[wIdx],True)
            l1.append(diff_l1)
            simple.append(diff_simple)
            #lin.append(diff_lin)
    print('l1-ball:%.10f; squared: %.10f' % (np.array(l1).mean(),(np.array(l1)**2).mean()))
    print('Simple:%.10f; squared: %.10f' % (np.array(simple).mean(),(np.array(simple)**2).mean()))
    print('Lin:    %.10f; squared: %.10f' % (np.array(lin).mean(),(np.array(lin)**2).mean()))
    #print('E,diff_linrror of l1-ball wrt Lin: %.f%%' % (100*(np.array(l1).mean()/np.array(lin).mean())))
    print('Error of l1-ball wrt Simple: %.f%%' % (100*(np.array(l1).mean()/np.array(simple).mean())))
    
@timeit
def test_run_clusters():
    shells=[ex.representativeness_shell,ex.earliness_shell,ex.random_shell,ex.random_shell,ex.random_shell,ex.random_shell,ex.random_shell]    
    # Create tranining data (use result of the k-modoid objective)
    training_examples=[]         
    for i in range(0,50):
        S=ClusterData()
        S.budget=5        
        objectives,obj_names=gm_submodular.utils.instaciateFunctions(shells,S)
        weights=[1.0,0,0,0,0,0,0]
        selected_elements,score=gm_submodular.leskovec_maximize(S,weights,objectives,S.budget)
        S.y_gt=selected_elements
        training_examples.append(S)
    title='k-medoids'    
    
    regularization_l1={'l1_inequality': True}
    weights_simple,dummy=gm_submodular.learnSubmodularMixture(training_examples, shells,None, 1,'simple',regularization_l1)    
    weights_ada,dummy=gm_submodular.learnSubmodularMixture(training_examples, shells,None, 1,'AdaGrad',regularization)    
    
    print('SSE to target weights:')
    weights=np.array(weights,np.float32)
    weights_ada/=weights_ada.sum()
    weights_simple/=weights_simple.sum()
    weights/=float(weights.sum())
    diff_ada=np.abs(weights-weights_ada)
    diff_simple=np.abs(weights-weights_simple)
    print('AdaGrad: %.5f (percent)' % (100*np.inner(diff_ada,diff_ada)))
    print('Simple:  %.5f (percent)' % (100*np.inner(diff_simple,diff_simple)))
    print('\n')
    sys.stdout.flush()    
    
    # Learn the weights. Given that we used the k-medoid results as ground truth, this objective should get all the weight
    weights_ada,dummy=gm_submodular.learnSubmodularMixture(training_examples, shells,None, 1,'AdaGrad',regularization)
    weights_simple,dummy=gm_submodular.learnSubmodularMixture(training_examples, shells,None, 1,'simple',regularization)    
    
    # Plot
    plt.figure(figsize=(12,7))
    width = 0.3
    plt.bar(np.arange(0,len(weights_ada)),weights_ada,width)
    plt.bar(np.arange(0,len(weights_ada))+width,weights_simple,width,color='r')
    plt.xticks(np.arange(len(shells))+width/2., map(lambda x: x.func_name, shells))
    plt.title('Weights %s proplem' % title)
    plt.grid()    
    
    objectives,obj_names=gm_submodular.utils.instaciateFunctions(shells,S)
    res_ada,score=gm_submodular.leskovec_maximize(S,weights_ada,objectives,S.budget)
    res_simple,score=gm_submodular.leskovec_maximize(S,weights_simple,objectives,S.budget)
    plotExample(S,objectives,weights,title, res_ada, res_simple)    
    
    print('SSE to target weights:')
    weights=np.array(weights,np.float32)
    weights_ada/=weights_ada.sum()
    weights_simple/=weights_simple.sum()
    weights/=float(weights.sum())
    diff_ada=np.abs(weights-weights_ada)
    diff_simple=np.abs(weights-weights_simple)
    print('AdaGrad: %.5f (percent)' % (100*np.inner(diff_ada,diff_ada)))
    print('Simple:  %.5f (percent)' % (100*np.inner(diff_simple,diff_simple)))
    print('\n')
    sys.stdout.flush()    
    return weights_ada, weights_simple ,shells
    
@timeit
def test_run_earliness(mix=False):
    # Create tranining data (use result of the k-modoid objective)
    training_examples=[]     

    shells=[ex.representativeness_shell,ex.earliness_shell,ex.random_shell,ex.random_shell,ex.random_shell,ex.random_shell,ex.random_shell]    
    for i in range(0,50):
        S=ClusterData()
        S.budget=5
        if mix:           
            objectives,obj_names=gm_submodular.utils.instaciateFunctions(shells,S)
            weights=[1,1,0,0,0,0,0]
            selected_elements,score=gm_submodular.leskovec_maximize(S,weights,objectives,S.budget)
            S.y_gt=selected_elements
            #S.y_gt.extend(np.random.random_integers(0,10,3))
            #S.y_gt.extend([0,1,2])
            title='Mixed'
        else:
            S.y_gt=np.arange(0,10)
            objectives,obj_names=gm_submodular.utils.instaciateFunctions(shells,S)
            weights=[0,1,0,0,0,0,0]
            title='Eearliness'
        training_examples.append(S)
    #plotExample(S,objectives,weights, title)
        
        
    # Learn the weights. Given that we used the k-medoid results as ground truth, this objective should get all the weight    
    weights_ada,dummy=gm_submodular.learnSubmodularMixture(training_examples, shells,None, 1,'AdaGrad',regularization)
    regularization_l1={'l1_inequality': True}
    weights_simple,dummy=gm_submodular.learnSubmodularMixture(training_examples, shells,None, 1,'simple',regularization_l1)    
    
    
    # Plot
    plt.figure(figsize=(12,7))
    width = 0.3
    plt.bar(np.arange(0,len(weights_ada)),weights_ada,width)
    plt.bar(np.arange(0,len(weights_ada))+width,weights_simple,width,color='r')
    plt.xticks(np.arange(len(shells))+width/2., map(lambda x: x.func_name, shells))
    plt.title('Weights %s proplem' % title)
    plt.grid()

    #Tracer()()
    objectives_noisy,obj_names=gm_submodular.utils.instaciateFunctions(shells,S)
    res_ada,score=gm_submodular.leskovec_maximize(S,list(weights_ada),objectives_noisy,S.budget)
    res_simple,score=gm_submodular.leskovec_maximize(S,list(weights_simple),objectives_noisy,S.budget)
    plotExample(S,objectives,weights,title, res_ada, res_simple)    
    
    print('SSE to target weights:')
    weights=np.array(weights,np.float32)
    weights_ada/=float(weights_ada.sum())
    weights_simple/=float(weights_simple.sum())
    weights/=float(weights.sum())
    diff_ada=np.abs(weights-weights_ada)
    diff_simple=np.abs(weights-weights_simple)
    print('AdaGrad: %.5f (percent)' % (100*np.inner(diff_ada,diff_ada)))
    print('Simple:  %.5f (percent)' % (100*np.inner(diff_ada,diff_simple)))
    print('\n')
    sys.stdout.flush()
    
    
    return weights_ada, weights_simple,shells
    
#plotExample()

if __name__=='__main__':
    ''' k-means example
    weights,weights_s,shells=test_run_clusters()
    
    weights,weights_s,shells=test_run_earliness(mix=False)
    
    weights,weights_s,shells=test_run_earliness(mix=True)
    
    '''
    #test_l1_constraint()
    compare_l1_2_lin()
    
