#!/usr/bin/env python
import glob
import argparse
import os
import numpy as np
import pickle
import sys
from tqdm import tqdm
import random
import json
import math
'''
Global Parameters
'''
n_ae_epochs= 1000+1
batch_size = 1
ae_lr      = 0.00001
z_size     = 500
leak_value = 0.2
reg_map    = 5.0e-7
reg_kl     = 1.0e-6
reg_mse    = 1.0
ae_inter   = 142
maxbatch   = 10981
trbatch    = 10000
outnum     = 981
lvlist     = []
dim=1000

filelist = glob.glob('*.npy')
plist=np.loadtxt('property2000.txt')

def mean(z1):
    mean=z1[:500]
    return mean
    
def meanlist(lvlist):
    ML=[]
    for i in range(len(lvlist)):
        ML.append(mean(lvlist[i]))
    print(len(ML))
    return ML

def distance(z1,z2):
    z3=z1.reshape(1000)
    meanz3=z3[:500]
    z4=z2.reshape(1000)
    meanz4=z4[:500]
    dist = np.linalg.norm(meanz3-meanz4)
    return dist
    
def nearest (zini,zlist):
    d=100000
    idx=100000
    for i in range(len(zlist)):
        if distance(zini,zlist[i])!=0:
            if distance(zini,zlist[i])<d:
                d=distance(zini,zlist[i])
                idx=i
    return idx
    
    
    
def SphereSpaceS (zlist):
    maxd=0
    for i in range(len(zlist)):
        for j in range(len(zlist)):
            if distance(zlist[i],zlist[j])>maxd:
                maxd=distance(zlist[i],zlist[j])
                print(maxd)
    return maxd

def RecSpaceSMin (zlist):
    mind=np.zeros(dim)
    for h in range(dim):
        for i in range(len(zlist)):
            if zlist[i][0][h]<mind[h]:
                mind[h]=zlist[i][0][h]
    #print(mind)
    return mind

def RecSpaceSMax (zlist):
    maxd=np.zeros(dim)
    for h in range(dim):
        for i in range(len(zlist)):
            if zlist[i][0][h]>maxd[h]:
                maxd[h]=zlist[i][0][h]
    #print(maxd)
    return maxd




def norm_pdf_multivariate(x, mu, sigma,prod):
    size = len(x)
    det = np.abs(prod)
    norm_const = 1.0/ ( math.pow((2*math.pi),float(size)/2) * math.pow(det,1.0/2) )
    x_mu = np.matrix(x - mu)
    inv = np.linalg.inv(sigma)       
    result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
    return norm_const * result


#print (norm_pdf_multivariate(x, mu, sigma))






def KDE (x,inilist):
    mlist=[]
    varlist=[]
    kdelist=[]
    kdesum=0
    a=inilist[0].reshape(1000)
    revar=a[500:]
    for i in range(len(inilist)):
        print(i)
        prod=1
        temp=inilist[i].reshape(1000)
        mlist.append(temp[:500])
        varlist.append(temp[500:])
        for j in range(500):
            prod=prod*varlist[i][j]/revar[j]
        sig=np.eye(500)
        row,col=np.diag_indices_from(sig)
        sig[row,col]=varlist[i]
        if prod!=0:
            kde = norm_pdf_multivariate(x, mlist[i], sig,prod)
            kdelist.append(kde)
            kdesum=kdesum+kde
    return kdesum





def UniformR (mind,maxd):
    g=np.zeros(dim)
    for i in range (dim):
        g[i]=random.uniform(mind[i],maxd[i])
    #print(g)
    return g



    
for filename in tqdm(filelist):
    lv = np.array(json.load(open(filename))['encode'])
    lvlist.append(lv)
    #print (len(lvlist))

#UniformR(RecSpaceSMin(lvlist),RecSpaceSMax(lvlist))

#print(plist,lvlist)

def RS (inip,lvlist,plist):
    lvfoundlist=[]
    lvremainlist=lvlist
    lvfoundlist.append(lvlist[inip])
    lvremainlist=lvremainlist[:inip]+lvremainlist[inip+1:]
    

    
    idxp=inip
    
    while len(lvremainlist)!=0:
        
        randv=UniformR(RecSpaceSMin(lvlist),RecSpaceSMax(lvlist))
        NearestID=nearest(randv,lvremainlist)
        if plist[NearestID]<plist[idxp]:
            idxp=NearestID
            

            lvfoundlist.append(lvlist[idxp])
            lvremainlist=lvremainlist[:idxp]+lvremainlist[idxp+1:]
            print(len(lvremainlist))
        else :

            lvfoundlist.append(lvlist[NearestID])
            lvremainlist=lvremainlist[:NearestID]+lvremainlist[NearestID+1:]
            print(len(lvremainlist))
        print(plist[idxp])
        #np.savetxt('p.txt',plist[idxp],delimiter=" ",fmt="%s")
    return plist[idxp]
    

KDE(np.zeros(500),meanlist(lvlist))

#RS(1999,lvlist,plist)



    
		
		
		
		