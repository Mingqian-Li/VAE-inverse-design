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
trbatch    = 8000
outnum     = 981
lvlist     = []
dim=1000

filelist = glob.glob('*.npy')
plist=np.loadtxt('property2000.txt')

def mean(z1):
    mean=z1[:500]
    return mean
    
def meanlist(lvlist):
    for i in range(len(lvlist)):
        meanlist[i]=mean[lvlist[i]]
    return meanlist

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
    


RS(1999,lvlist,plist)



    
		
		
		
		