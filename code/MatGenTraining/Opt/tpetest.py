import glob
import argparse
import os
import numpy as np
import pickle
import sys
from tqdm import tqdm
import random
import optuna
import json
from hyperopt import tpe, hp, fmin
import numpy as np
#define objective funtion

#method one DFT
#def objective(params):
#    for i in range(500)
#        x[i] = params['x'+str(i)]
#    return DFT(x)

#method two deePMD
#def objective(params):
#    for i in range(500)
#        x[i] = params['x'+str(i)]
#    return DFT(x)
filelist = glob.glob('*.npy')
plist=np.loadtxt('property2000.txt')
lvlist     = []

for filename in tqdm(filelist):
    lv = np.array(json.load(open(filename))['encode'])
    lvlist.append(lv)

ztag=[]
for i in range(500):
    ztag.append('x'+str(i))
#print(ztag)

    
#def objective(lvnum):
    
#    return plist[lvnum]
    
def objective(params):
    x, y, z = params['x'], params['y'], params['z'], params['w'], params['e'], params['r']
    return x**2+y**2+z**2+w**2+e**2+r**2





''''def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    z = trial.suggest_float("z", -10, 10)
    w = trial.suggest_float("z", -10, 10)
    e = trial.suggest_float("z", -10, 10)
    r = trial.suggest_float("z", -10, 10)
    return x ** 2+y**2+z**2+w**2+e**2+r**2



sampler = optuna.samplers.TPESampler(multivariate=True)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=10000)
'''

#define search space, we can use 0 to 1 as search space or min to max 
#space = {'x'+str(x): x for x in range(500)}
space = {
    'x': hp.uniform('x', -6, 6),
    'y': hp.uniform('y', -6, 6),
    'z': hp.uniform('z', -6, 6),
    'w': hp.uniform('w', -6, 6),
    'e': hp.uniform('e', -6, 6),
    'r': hp.uniform('r', -6, 6),
}




best = fmin(
    fn=objective, # Objective Function to optimize
    space=space, # Hyperparameter's Search Space
    algo=tpe.suggest, # Optimization algorithm (representative TPE)
    max_evals=1000 # Number of optimization attempts
)
print(best)
