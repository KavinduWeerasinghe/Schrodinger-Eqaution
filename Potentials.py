import numpy as np

def Infinity_well(x,infinity_value=2000000000000):
    L=len(x)
    V_x=np.ones(L)*infinity_value
    V_x[L//2-L//4:L//2+L//4]=0
    return V_x
    