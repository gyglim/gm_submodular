import pystruct
import pystruct.models
import gm_submodular
import numpy as np


def intersection_loss(S,y):
    if type(S) is list or type(S) is np.ndarray:
        return 1-len(set(S).intersection(y))/float(len(S))
    else:
        return 1-len(set(S.y_gt).intersection(y))/float(len(S.y_gt))

class SubmodularSSVM(pystruct.models.StructuredModel):

    def __init__(self,submod_fun, loss_function=None, budget=5):
        if loss_function is None:
            self.loss_f=intersection_loss
        else:
            self.loss_f=loss_function
        self.budget=budget
        self.submod_fun=submod_fun
        self.size_joint_feature=len(submod_fun)
        self.inference_calls=0

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        self.inference_calls+=1
        w=np.array(w)
        w[w<0]=0
        return gm_submodular.leskovec_maximize(x,w,gm_submodular.utils.instaciateFunctions(self.submod_fun,x)[0],self.budget,loss_fun=self.loss_f)[0]

    def inference(self, x, w, relaxed=None):
        self.inference_calls+=1
        w=np.array(w)
        w[w<0]=0
        return gm_submodular.leskovec_maximize(x,w,gm_submodular.utils.instaciateFunctions(self.submod_fun,x)[0],self.budget,loss_fun=None)[0]


    def loss(self, y, y_hat):
        return self.loss_f(y,y_hat)

    def joint_feature(self,x,y):
        return np.array(map(lambda sf: sf(y),gm_submodular.utils.instaciateFunctions(self.submod_fun,x)[0]))