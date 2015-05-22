# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:17:29 2015

@author: gyglim
"""


def getFunction(dim, l2_regularizer=True, l1_constraint=False):
    if l2_regularizer:
        obj=lambda w_t: nu*np.inner(np.array(g_t).mean(axis=0), w_t) +nu*np.sqrt(np.inner(w_t,w_t)) - 1/(2*(it+1))*np.inner(np.array(w_t),np.inner(H_t,w_t))            
    else:
        obj=lambda w_t: nu*np.inner(np.array(g_t).mean(axis=0), w_t) - 1/(2*(it+1))*np.inner(np.array(w_t),np.inner(H_t,w_t))            
    cons=[]
    for idx in range(0,dim):
        cons.append({'type': 'ineq','fun' : lambda x: x[idx]})
    cons.append({'type': 'eq','fun' : lambda x: x.sum()-1})
    cons=tuple(cons)
        
    bnds = (0, None)
    
    return obj, cons, bnds