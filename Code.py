# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:31:50 2018

@author: prash
"""

import scipy.io
import numpy as np 
from scipy import linalg as LA

#Input: data for the left and right hand respectively. Format for each type: channel X time
#Output: 6 Features corresponding to the first 3 and last 3 eigen values
def CSP(data_left,data_right):
#Another implementation taken from artvicle titled :
#"Model based generalization analysis of common spatial pattern in brain computer interfaces"
    cov_left=np.dot(data_left,data_left.T)/np.trace(np.dot(data_left,data_left.T))
    cov_right=np.dot(data_right,data_right.T)/np.trace(np.dot(data_right,data_right.T))
    #sum of covariances for all instances needs to be added
    C=cov_left+cov_right
    eigvals,eigvecs=LA.eig(C)
    diag=np.zeros((C.shape[1],C.shape[1]))
    diag_inv=np.zeros((C.shape[1],C.shape[1]))
    sort_eigvals=np.sort(eigvals) #when to sort eigen values
    sort_eigvecs=eigvecs[0,np.argsort(eigvals)] 
    eigvecs=sort_eigvecs
    for i in range(eigvals.shape[0]):
        diag[i,i]=np.float(sort_eigvals[i].real)
        diag_inv[i,i]=(1/np.abs(sort_eigvals[i].real)) #considering absolute value of the real parts! need to verify if approach is correct  
    P=np.sqrt(diag_inv)*eigvecs.T
    S_l=P*cov_left*P.T
    S_r=P*cov_right*P.T
    #Here, the whitening transform are used for simultaneous diagonalization
    eigval_l,eigvec_l=LA.eig(S_l)
    eigval_r,eigvec_r=LA.eig(S_r)
    #print(eigval_l+eigval_r) # This sum should be equal to identity but isnt: DEBUG
    # for some reason the eigvec_l and eigvec_r are Identity
    W=np.dot(eigvec_r.T,P)#projection matrix
    #consider the first 10 columns of W as the required feautres 
    #ideally you want to pick the first three features and the last three
    J_first=3
    W_select=np.zeros([np.shape(W)[0],6])
    W_select[:,0:J_first]=W[:,0:J_first]
    W_select[:,J_first::]=W[:,np.shape(W)[1]-3:np.shape(W)[1]]
    feat=np.log(np.var(np.dot(W_select.T,data_left),axis=1))
    return feat

feature=[]
for subject in range(52):
    if (subject+1)<10:
        file='s0%d.mat'% (subject+1)
    else:
        file='s%d.mat'% (subject+1)
    mat=scipy.io.loadmat(file)
    eeg=mat['eeg']
    temp = eeg[0][0]
    movement_left=temp[3]
    movement_right=temp[4]
    movement_noise=temp[5]
    imagery_left=temp[7]
    imagery_right=temp[8]
    imagery_event=temp[11]
    feature_temp=CSP(imagery_left,imagery_right)
    feature.append(feature_temp)

feature=np.array(feature)

