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


filelist = glob.glob('*.npy')
plist=np.loadtxt('property2000.txt')

def distance(z1,z2):
    dist = np.linalg.norm(z1-z2)
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
    
    
    
def space (zlist):
    maxd=0
    for i in range(len(zlist)):
        for j in range(len(zlist)):
            if distance(zlist[i],zlist[j])>maxd:
                maxd=distance(zlist[i],zlist[j])
                print(maxd)
    return maxd





    
for filename in tqdm(filelist):
    lv = np.array(json.load(open(filename))['encode'])
    lvlist.append(lv)
    #print (len(lvlist))



#print(plist,lvlist)

def RS (inip,lvlist,plist):
    lvfoundlist=[]
    lvremainlist=lvlist
    lvfoundlist.append(lvlist[inip])
    lvremainlist=lvremainlist[:inip]+lvremainlist[inip+1:]
    

    
    idxp=inip
    
    while len(lvremainlist)!=0:
        if plist[nearest(idxp,lvremainlist)]<plist[idxp]:
            idxp=nearest(idxp,lvremainlist)
            

            lvfoundlist.append(lvlist[idxp])
            lvremainlist=lvremainlist[:idxp]+lvremainlist[idxp+1:]
            print(len(lvremainlist))
        else :

            lvfoundlist.append(lvlist[nearest(idxp,lvremainlist)])
            lvremainlist=lvremainlist[:nearest(idxp,lvremainlist)]+lvremainlist[nearest(idxp,lvremainlist)+1:]
            #print(len(lvremainlist))
        print(plist[idxp])
        #np.savetxt('p.txt',plist[idxp],delimiter=" ",fmt="%s")
    return plist[idxp]
    
space(lvlist)
RS(1999,lvlist,plist)



    
		
		
		
		