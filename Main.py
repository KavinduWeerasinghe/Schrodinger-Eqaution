import scipy
import numpy as np
from Potentials import Infinity_well,finite_well,harmonic_well,potential_barrier
import matplotlib.pyplot as plt

def Hamiltonian(x,V,dx,m=1,h=1)->np.ndarray:
    D_matrix = scipy.sparse.diags([1, -2, 1],
                                  [-1, 0, 1],
                                  shape=(x.size, x.size))/dx**2
    H2=-(h**2/(2*m))*D_matrix+V
    H=H2
    return H

def setup_psi(H,dx):
    eig_val,eig_vec=scipy.linalg.eigh(H)
    
    prob=(np.abs(eig_vec))**2
    prob/=np.sum(prob,axis=0)*dx
    
    eig_vec/=np.sqrt(np.sum(prob,axis=0)*dx)
    return prob[:,:5],eig_vec[:,:5]

def setup_Vector(V):
    V_matrix=np.diag(V)
    return V_matrix
    
def setup_base_vector(start=0,end=10,dx=0.01):
    x = np.arange(start, end, dx)
    return x,dx

def plot_potential(x,V_x,lable=None):
    axs[ 0].plot(x,V_x,label=lable,linestyle="--",linewidth=0.5)
    axs[ 1].plot(x,V_x,label=lable,linestyle="--",linewidth=0.5)

def plot_prob(x,psi,lable=None):
    L=len(psi[0])
    M0=max(psi[:,0])
    for i in range(L):
        M=max(psi[:,i])
        y=psi[:,i]+i*M*1.5
        lable_new=lable+" : n = "+str(i+1)
        axs[0].plot(x,y,label=lable_new)
    return (L+1)*M0*1.5

def plot_wave(x,psi,lable=None):
    L=len(psi[0])
    M0=max(psi[:,0])
    for i in range(L):
        M=max(psi[:,i])
        y=psi[:,i]+i*M*1.5
        lable_new=lable+" : n = "+str(i+1)
        axs[1].plot(x,y,label=lable_new)
    return (L+1)*M0*1.5

x,dx=setup_base_vector()
x0 = 3.0  #center
V_x=harmonic_well(x) #Infinity_well,finite_well,harmonic_well,potential_barrier

V=setup_Vector(V_x)
H=Hamiltonian(x,V,dx)
psi_p,psi_w=setup_psi(H,dx)
fig, axs = plt.subplots(1,2)
plot_potential(x,V_x,lable="Potential")
K1=plot_prob(x,psi_p,lable="Probability")
K2=plot_wave(x,psi_w,lable="Wave Function")

#plt.ylim([0, K])
axs[0].set_ylim([0,K1])
axs[1].set_ylim([0,K2])
axs[0].legend()
axs[1].legend()
plt.show()