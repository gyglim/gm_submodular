import numpy as np
import time

def evalSubFun(submod_fun,y,isGt,w=None):
    res=np.zeros(len(submod_fun))
    for idx in range(len(submod_fun)):
        if w is not None and w[idx]==0:
            res[idx]=0
        else:
            res[idx]=submod_fun[idx](y)
    return res

def zero_loss(d,d2):
    return len(d2)*0.0001

    
def instaciateFunctions(submod_fun,s):
    fun_list=[]
    name_list=[]
    for idx in range(0,len(submod_fun)):
        res=submod_fun[idx](s)
        if type(res) is tuple:
            objective=res[0]
            names=res[1]
        else:
            objective=res
            names=submod_fun[idx].__name__
            
        if type(objective) is list:            
            fun_list.extend(objective)
            if names is not list:
                names=map(lambda x: '%s-%d' % (names,x),np.arange(len(objective)))
            name_list.extend(names)
        else:
            fun_list.append(objective)
            if names is None:
                name_list.append(submod_fun[idx].func_name)
            else:
                name_list.append(names)

    return fun_list,name_list
    
    

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts)
        return result

    return timed    