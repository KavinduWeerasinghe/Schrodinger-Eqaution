import numpy as np

def Infinity_well(x,infinity_value=2000000000000):
    L=len(x)
    V_x=np.ones(L)*infinity_value
    V_x[L//2-L//10:L//2+L//10]=0
    return V_x
    
def finite_well(x,finite_value=10):
    L=len(x)
    V_x=np.ones(L)*finite_value
    V_x[L//2-L//10:L//2+L//10]=0
    return V_x

def harmonic_well(x,k=1):
    V_x=k*(x-5)**2
    return V_x

def potential_barrier(x):
    L=len(x)
    V_x=np.ones(L)*0
    V_x[L//2-L//40:L//2+L//40]=2
    return V_x