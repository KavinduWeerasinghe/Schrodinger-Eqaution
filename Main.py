import scipy
import numpy as np

def Hamiltonian(x,psi,V,dx=0.02,m=1,h=1)->np.ndarray:
    D_matrix = scipy.sparse.diags([1, -2, 1],
                                  [-1, 0, 1],
                                  shape=(x.size, x.size))/dx**2
    H=[-0.5*(h/(2*m))*D_matrix.dot(psi)+V/h*psi]*(-1j)
    return H

def setup_Vector(V):
    V_matrix=np.diag(V)
    return V_matrix
    
def setup_base_vector(start=0,end=10,dx=0.02):
    x = np.arange(start, end, dx)
    return x