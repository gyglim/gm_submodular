'''
This module provides functions for comparing the method of
    Lin, H. & Bilmes, J. Learning mixtures of submodular shells with application to document summarization. UAI 2012
to our own (l1-contraint formulation)
In particular it provides code to generate toy datasets and contains plotting functions.
'''
import gm_submodular.utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import gm_submodular.example_objectives as ex
gm_submodular.logger.setLevel('ERROR')
import pickle
import sys
import pystruct.learners
import subm_tests.test_pystruct as tp

def representativeness_shell_x(S):
    '''
        Representativeness shell Eq. (8)
    :param S: DataElement with function getDistances()
    :return: representativeness objective
    '''
    tempDMat=S.getDistancesX()
    norm=tempDMat.mean()
    return (lambda X: (1 - ex.kmedoid_loss(X,tempDMat,float(norm))))

def representativeness_shell_y(S):
    '''
        Representativeness shell Eq. (8)
    :param S: DataElement with function getDistances()
    :return: representativeness objective
    '''
    tempDMat=S.getDistancesY()
    norm=tempDMat.mean()
    return (lambda X: (1 - ex.kmedoid_loss(X,tempDMat,float(norm))))


def intersect_complement_loss(S,selection):
    '''
    :param S: A DataElement
    :param selection: a list of selected indices
    :return: the loss (in  [0; 1])
    '''

    #set intersection is much faster that numpy intersect1d
    return 1-len(set(S.y_gt).intersection(selection))/float(len(S.y_gt))

def plotExample(S,objectives,weights, title):
    '''
    Plots the given smaple dataset in S
    :param S: DataElement
    :param objectives: objectives used
    :param weights: their weights
    :param title: figure title
    :return: None
    '''
    # Maximize the objectives
    selected_elements,score=gm_submodular.leskovec_maximize(S,weights,objectives,S.budget)

    # Plot results
    plt.figure(figsize=(16,8)) # Definition of a larger figure (in inches)
    colors=['y','black','c','m','grey','red']
    for i in set(S.cluster_assignment):
        plt.scatter(S.x[S.cluster_assignment==i,0],S.x[S.cluster_assignment==i,1], c=colors[int(i)], alpha=0.66, s=50, linewidths=1)
        plt.hold(True)

    plt.scatter(S.x[selected_elements,0],S.x[selected_elements,1], c='green', s=100, alpha=1, linewidths=2)
    legend=list(map(lambda x: 'Cluster %d' %x,set(S.cluster_assignment)))
    legend.append('Ground Truth')
    plt.legend(legend,fontsize=18,loc=4)
    plt.title(title,fontsize=22)


class ClusterData(gm_submodular.DataElement):
    '''
        A DatasetElement containg some randomly generated cluster data.
        (see ipython notebook for an example)
        Derives from gm_submodular.DataElement
    '''
    def __init__(self):
        num_points=100
        x=np.random.rand(num_points/5,2)
        self.cluster_assignment=np.zeros(x.shape[0])
        for i in range(0,4):
            new_points=np.random.rand(2)*np.random.rand(2)*30+(np.random.randn(num_points/5,2))
            x=np.vstack((x,new_points))
            self.cluster_assignment=np.hstack((self.cluster_assignment,np.ones(new_points.shape[0])*(i+1)))

        self.dist_v=dist.pdist(x)
        self.dist_x=dist.pdist(x[:,0].reshape(len(x[:,0]),1))
        self.dist_y=dist.pdist(x[:,1].reshape(len(x[:,1]),1))

        self.Y=np.arange(0,x.shape[0])
        self.x=x
        self.budget=5

    def getCosts(self):
        return np.ones(self.dist_v.shape[0])
    def getDistances(self):
        return dist.squareform(self.dist_v)
    def getDistancesX(self):
        return dist.squareform(self.dist_x)
    def getDistancesY(self):
        return dist.squareform(self.dist_y)


def createTrainingData(weights=[1,0],num_noise_obj=0, gt_variability=0,num_datasets=50):
    shells=[ex.representativeness_shell,ex.earliness_shell]
    weights_gt=list(weights)
    for i in range(num_noise_obj):
        shells.append(ex.random_shell)
        weights_gt.append(0)

    # Create tranining data (use result of the k-medoid objective)
    training_examples=[]
    for i in range(0,num_datasets):
        S=ClusterData()
        S.budget=5
        objectives,obj_names=gm_submodular.utils.instaciateFunctions(shells,S)
        test_b=int(S.budget*(gt_variability+1))
        selected_elements,score=gm_submodular.leskovec_maximize(S,weights_gt,objectives,test_b)
        S.y_gt=list(np.array(selected_elements)[np.random.permutation(test_b)][0:S.budget])
        training_examples.append(S)
    return training_examples,shells,weights_gt

def getError(weights=[1,0],num_noise_obj=0, gt_variability=0, num_runs=100):
    l1_error=[]
    lin_error=[]
    pystruct_error=[]
    for runNr in range(0,num_runs):
        training_examples,shells,weights_gt = createTrainingData(np.array(weights).copy(),num_noise_obj, gt_variability)
        m=tp.SubmodularSSVM(shells)
        sg_ssvm=pystruct.learners.SubgradientSSVM(m,max_iter=3,shuffle=True,averaging='linear')
        res_ps=sg_ssvm.fit(training_examples,map(lambda x: x.y_gt,training_examples))

        weights_simple,dummy = gm_submodular.learnSubmodularMixture(training_examples,
                                                                    shells,
                                                                    intersect_complement_loss,
                                                                    3, use_l1_inequality=False)
        weights_l1,dummy = gm_submodular.learnSubmodularMixture(training_examples,
                                                                shells,
                                                                intersect_complement_loss,
                                                                3, use_l1_inequality=True)
        weights_pystruct=np.array(res_ps.w,np.float32)
        weights_pystruct[weights_pystruct<0]=0
        weights_pystruct/=weights_pystruct.sum()
        # Compute the relative deviation from the target weights
        weights_gt=np.array(weights_gt,np.float32)
        weights_l1/=weights_l1.sum()    
        weights_simple/=weights_simple.sum()
        weights_gt/=float(weights_gt.sum())
        diff_l1=np.abs(weights_gt-weights_l1)
        diff_simple=np.abs(weights_gt-weights_simple)
        diff_pystruct=np.abs(weights_gt-weights_pystruct)
        l1_error.append(diff_l1.sum())
        lin_error.append(diff_simple.sum())
        pystruct_error.append(diff_pystruct.sum())

    # report and return deviation from the target weights
    l1_error=(np.array(l1_error).mean(),np.array(l1_error).std())
    lin_error=np.array(lin_error).mean(),np.array(lin_error).std()
    pystruct_error=np.array(pystruct_error).mean(),np.array(pystruct_error).std()
    print('l1-ball: %.10f; lin: %.10f; pystruct: %.10f' % (l1_error[0],lin_error[0],pystruct_error[0]))
    sys.stdout.flush()
    return l1_error,lin_error,pystruct_error

def get_noisy_objective_plot(num_runs=100):
    '''
        This experiment tests how the methods are affected by objectives only contributing noise
    :return:
    '''
    l1_error=[]
    lin_error=[]
    num_noise_obj=range(0,5)
    print('testing range %s' % ', '.join(map(str,num_noise_obj)))
    for num_noise in num_noise_obj:
        l1,lin,pystruct=getError(num_noise_obj=num_noise,num_runs=num_runs)
        l1_error.append(l1)
        lin_error.append(lin)
    plt.figure(figsize=(10,10))
    plt.errorbar(num_noise_obj,np.array(l1_error)[:,0]*100,yerr=np.array(l1_error)[:,1]*100,linewidth=3)
    plt.hold(True)
    plt.errorbar(num_noise_obj,np.array(lin_error)[:,0]*100,yerr=np.array(lin_error)[:,1]*100,color='red',linewidth=3)
    plt.title('Robustness w.r.t. noise objectives',fontsize=22)
    plt.legend(['l1 inequality (ours)','Lin et al.'],fontsize=18)
    plt.xlabel('# of noise objectives',fontsize=18)
    plt.ylabel('Deviations from the ground truth weights [%]',fontsize=18)
    plt.grid()
    plt.show()
    data={'num_runs':num_runs,'num_noise_obj':num_noise_obj,'l1_error':l1_error,'lin_error':lin_error}
    with open('noisy_gt.pickle','w') as f:
        pickle.dump(data,f)
    return num_noise_obj,l1_error,lin_error

def get_noisy_ground_truth_plot(num_runs=100):
    '''
    This experiment tests how the methods are affected by having noisy ground truth (i.e. there is not always
    the optimal element selected when creating the ground truth)
    :param num_runs:
    :return:
    '''
    l1_error=[]
    lin_error=[]
    gt_variability=np.arange(0,1.001,0.2)
    print('testing range %s' % ', '.join(map(str,gt_variability)))
    for gt_var in gt_variability:
        l1,lin,pystruct=getError(gt_variability=gt_var,num_runs=num_runs)
        l1_error.append(l1)
        lin_error.append(lin)
    plt.figure(figsize=(10,10))
    plt.errorbar(gt_variability*100,np.array(l1_error)[:,0]*100,yerr=np.array(l1_error)[:,1]*100,linewidth=3)
    plt.hold(True)
    plt.errorbar(gt_variability*100,np.array(lin_error)[:,0]*100,yerr=np.array(lin_error)[:,1]*100,color='red',linewidth=3)
    plt.title('Robustness w.r.t. noise objectives',fontsize=22)
    plt.legend(['l1 inequality (ours)','Lin et al.'],fontsize=18)
    plt.xlabel('noise in ground truth [%]',fontsize=18)
    plt.ylabel('Deviations from the ground truth weights [%]',fontsize=18)
    plt.grid()
    plt.show()
    data={'num_runs':num_runs,'gt_variability':gt_variability,'l1_error':l1_error,'lin_error':lin_error}
    with open('noisy_gt.pickle','w') as f:
        pickle.dump(data,f)
    return gt_variability,l1_error,lin_error

def get_multiobjective_plot(num_runs=100,num_noise_obj=1):
    '''
    This experiment tests how well the methods are able to learn a weighted combination of 2 objectives
    (potentially while additional noise objectives are present)
    :param num_runs:
    :param num_noise_obj:
    :return:
    '''
    l1_error=[]
    lin_error=[]
    pystruct_error=[]
    obj_ratio=np.arange(0,1.001,0.1)
    print('testing ratios %s' % ', '.join(map(str,obj_ratio)))
    for obj_r in obj_ratio:
        l1,lin,pystruct=getError(weights=[1-obj_r, obj_r],num_runs=num_runs,num_noise_obj=num_noise_obj)
        l1_error.append(l1)
        lin_error.append(lin)
        pystruct_error.append(pystruct)

    # Save to file
    data={'num_runs':num_runs,'obj_ratio':obj_ratio,'l1_error':l1_error,'lin_error':lin_error}
    with open('multiobjective.pickle','w') as f:
        pickle.dump(data,f)

    #Plot
    plt.figure(figsize=(10,10))
    plt.errorbar(obj_ratio*100,np.array(l1_error)[:,0]*100,yerr=np.array(l1_error)[:,1]*100,linewidth=3)
    plt.hold(True)
    plt.errorbar(obj_ratio*100,np.array(lin_error)[:,0]*100,yerr=np.array(lin_error)[:,1]*100,color='red',linewidth=3)
    plt.hold(True)
    plt.errorbar(obj_ratio*100,np.array(pystruct_error)[:,0]*100,yerr=np.array(pystruct_error)[:,1]*100,color='green',linewidth=3)
    plt.title('Fidelity regarding different target weight',fontsize=22)
    plt.legend(['l1 inequality (ours)','Lin et al.','pystruct'],fontsize=18)
    plt.xlabel('relative weight importance',fontsize=18)
    plt.ylabel('Deviations from the ground truth weights [%]',fontsize=18)
    plt.grid()
    plt.hold(False)
    plt.show()
    return obj_ratio,l1_error,lin_error