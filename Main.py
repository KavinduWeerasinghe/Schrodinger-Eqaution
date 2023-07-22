import scipy
import numpy as np
from Potentials import Infinity_well
import matplotlib.pyplot as plt

def Hamiltonian(x,V,dx,m=1,h=1)->np.ndarray:
    D_matrix = scipy.sparse.diags([1, -2, 1],
                                  [-1, 0, 1],
                                  shape=(x.size, x.size))/dx**2
    H2=-(h**2/(2*m))*D_matrix+V
    H=H2*(-1j)
    return H

def setup_psi(H,dx):
    eig_val,eig_vec=scipy.linalg.eigh(H)
    prob=np.abs(eig_vec)**2
    prob/=np.sum(prob,axis=0)*dx
    return prob[:,:10]

def setup_Vector(V):
    V_matrix=np.diag(V)
    return V_matrix
    
def setup_base_vector(start=0,end=10,dx=0.01):
    x = np.arange(start, end, dx)
    return x,dx

def plot_potential(x,V_x,lable=None):
    axs.plot(x,V_x,label=lable)

def plot_prob(x,psi,lable=None):
    axs.plot(x,psi,label=lable)

x,dx=setup_base_vector()
x0 = 3.0  #center
V_x=Infinity_well(x)

V=setup_Vector(V_x)
H=Hamiltonian(x,V,dx)
psi=setup_psi(H,dx)
fig, axs = plt.subplots()
plot_potential(x,V_x,lable="Potential")
plot_prob(x,psi,lable="Probability")

plt.ylim([0, 2*max(psi[:,0])])
axs.legend()
plt.show()