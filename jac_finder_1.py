import time
import numpy as np
import scipy.io


import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer

import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc

import h5py
import hdf5storage
import pickle


L = 100
N = 1024
dx = L/N
x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]

kappa = 2 * np.pi*np.fft.fftfreq(N,d=dx)

lambdas = [1.0,1.0,1.0]

u0 = -np.cos(x*2*np.pi/100)*(1+np.sin(-x*2*np.pi/100))
print(u0)
dt = 1e-3



def ETRNK4intKS(u0,t,kappa,N,lambdas):
    
    dt = t[1]-t[0]
    uhat = np.fft.fft(u0)

    # ETDRK4 constants
    L = lambdas[1]*kappa**2 - lambdas[2]*kappa**4
    E = np.exp(dt*L)
    E_2 = np.exp(dt*L/2)
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = dt*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
    Q =  dt*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = dt*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = dt*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = dt*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
    
    g = -0.5j*kappa
    
    uhat_ri = np.zeros((len(t),N))
    uhat_ri[0,:] = u0
    #Arthur: I'm gonna change the range, since it seems that his overwrite the u0 row.
    for tcount in range(1,len(t)):
        Nuhat = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(uhat))**2)
        a = E_2*uhat + Q*Nuhat
        Na = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(a))**2)
        b = E_2*uhat + Q*Na
        Nb = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(b))**2)
        c = E_2*a + Q*(2*Nb-Nuhat)
        Nc = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(c))**2)
        uhat = E*uhat + Nuhat*f1 + 2*(Na+Nb)*f2 + Nc*f3
        uhat_ri[tcount,:]  = np.fft.ifft(uhat).real
        

    return uhat_ri.real.T






with open("/arthurresearch/training_data/KS_1024.pkl",'rb') as f:
    u = pickle.load(f)
print("done reading u")
print(len(u[0,:]))


#find the std of each row of u
sd_pos_vec_all = np.zeros(N) 
for pos in np.arange(N):
        #sd_pos is the std of the pos row of the u matrix
    sd_pos = np.std(u[pos,:])
    sd_pos_vec_all[pos] = sd_pos


#array of the Jacobians
#find how many columns u has. That is len(u[0,:]), the length of a row.
Jac = np.zeros((int(len(u[0,:])/10),N,N))
#Note: the u matrix has len(t) columns, since it adds in the u_0 for the first column.
#For u, the each column is a time, and each row is a position
#There are len(t)-1 number of jacobians, since we cannot find the jacobian at the last time.
# record start time
start = time.time()

Leng = int(len(u[0,:])/10)
for i in np.arange(Leng):
    Jac_i = np.zeros((N,N))
    for pos in np.arange(N):
        #sd_pos is the std of the pos row of the u matrix
        sd_pos = sd_pos_vec_all[pos]
        #id_pos is the unit vector with 1 in the pos position
        id_pos = np.zeros(N)
        id_pos[pos] = 1
        sd_pos_vec = id_pos * sd_pos
        

        #v_i_pos : ith column of u, which is our current u_i, varied by the pos position

        v_i_pos = u[:,i] +sd_pos_vec * 0.001 
        v_i_pos_next = ETRNK4intKS(v_i_pos ,[0,dt],kappa,N,lambdas)[:,1]
        #populate the jac_i
        #Note that the row we are populating right now if the pos row, since that is the term we are varying in u_i
        for row in np.arange(N):
            Jac_i[row, pos] = (v_i_pos_next[row] - u[row,i+1]) / (sd_pos*0.001)
    Jac[i,:,:] = Jac_i
    #find the eigenvalues
    print(np.max(np.abs(np.linalg.eigvals(Jac_i))))
    print(np.max(np.abs(np.linalg.eigvals(Jac[i,:,:]))))
    # record end time
    end = time.time()

    print("1: ",end-start)
    scipy.io.savemat('jac_1.mat', {'jac1_'+str(i): Jac_i})

print(np.abs(Jac[1,:,:] - Jac[2,:,:]).max())

#hdf5 stuff

#Let's write the Jac into a file
Jac_mat_lab= {}
Jac_mat_lab['jac'] = Jac
hdf5storage.write(Jac_mat_lab,'.', '/arthurresearch/jac_thing.hdf5', matlab_compatible=True)
#hdf5storage.savemat('/arthurresearch/jac.mat',Jac_mat_lab,format='7.3')






#f = open('/examplevol/home/exouser/mount/conrad_stability/jacobians.pkl', 'wb')
#pickle.dump(Jac, f)
#f.close()
