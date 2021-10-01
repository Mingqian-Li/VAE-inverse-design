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

for filename in tqdm(filelist):
    lv = np.array(json.load(open(filename))['encode'])
    lvlist.append(lv)
    
    
slerpinilist=lvlist[:31]
'''
def mean(z1):
    mean=z1[:500]
    return mean
    
def meanlist(lvlist):
    for i in range(len(lvlist)):
        meanlist[i]=mean[lvlist[i]]
return meanlist
'''
def angle(a,b):
    a1=a.reshape(1000)[500:]
    b1=b.reshape(1000)[500:]
    a2=(a1/np.linalg.norm(a1))
    b2=(b1/np.linalg.norm(b1))
    dot=np.dot(a2,b2)
    omega=np.arccos(dot)
    return omega


def slerp(inilist,impnum):
    slerplist=[]
    for i in range (len(inilist)):
        for j in range (len(inilist)):
            if i<j:
                omega=angle(inilist[i],inilist[j])
                for h in range (impnum):
                    f1=h/(impnum+1)
                    f2=1-f1
                    lvg=inilist[i]*np.sin(f1*omega)/np.sin(omega)+inilist[j]*np.sin(f2*omega)/np.sin(omega)
                    slerplist.append(lvg)
                    #print(h)
    return slerplist
        
        
        
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

def UniformR (mind,maxd):
    g=np.zeros(dim)
    for i in range (dim):
        g[i]=random.uniform(mind[i],maxd[i])
    #print(g)
    return g



    

    #print (len(lvlist))

#UniformR(RecSpaceSMin(lvlist),RecSpaceSMax(lvlist))

#print(plist,lvlist)

def SS (slerplist,lvlist,plist):

    

    idxp=0
    for i in range (len(slerplist)):
        NearestID=nearest(slerplist[i],lvlist)
        if plist[NearestID]<plist[idxp]:
            idxp=NearestID
        print(plist[idxp])
        #np.savetxt('p.txt',plist[idxp],delimiter=" ",fmt="%s")
    return plist[idxp]
    
#print(len(slerp(slerpinilist,19)))
SS(slerp(slerpinilist,19),lvlist,plist)



    
		
		
		
		