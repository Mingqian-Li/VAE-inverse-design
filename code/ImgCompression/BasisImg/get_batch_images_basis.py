#!/qcfs/jackjack5/3party-software/anaconda/kdft/anaconda2/bin/python

import glob
import argparse
import os
import numpy as np
import pickle
import sys
from tqdm import tqdm
import random
import json

def makebatch(args):
	filelist = glob.glob('*.npy')
	#filelist=list(filelist)
	print(filelist)
	batchsize = args.batchsize
	savedir = args.savedir
	batch_iter = 0
	batch_images = []


	tt = np.array(filelist)
	if not os.path.isdir(savedir):
		os.mkdir(savedir)
	sum_nc = 0
	atlb=0

	for filename in tqdm(filelist):

		images1=np.zeros((1,1,200,1))
		images2=np.zeros((1,2,200,1))
		images3=np.zeros((1,1,400,1))
		if atlb==0:
			images = np.array(json.load(open(filename))['image'])
			atlb=1
		else:
			images1 = np.array(json.load(open(filename))['image'])
			images2 = np.append(images,images1,1)

			images3=images2.reshape(1,1,400,1)
			atlb=0


			dim = images3.shape[2]
			nc = images.shape[-1]

			sum_nc += nc
			for c in range(nc):
				batch_images.append(images3[c,c,:,c].reshape(1,1,dim,1))
				
				if len(batch_images)==batchsize:
					print ('saving batch #',batch_iter)


					batch_savefilename2 = savedir+'/'+str(batch_iter)+'_pvals_basis.npy'
					fin_batch = np.array(batch_images)
					(p,q,r,s,t) = np.where(fin_batch)
					pvals = fin_batch[p,q,r,s,t]
					fin_batch2 = np.array([p,q,r,s,t],np.int32)


					



					pvals=pvals.reshape(1,1,200,2)
					pvals=np.pad(pvals,((0,0),(0,0),(0,0),(0,3)))
					
					batch_iter += 1
					batch_images = []
					np.save(batch_savefilename2, np.array(pvals))
					print(pvals.shape)
                
                
                

	return 1

def main():
	parser = argparse.ArgumentParser(description='script for making single image batches')
	parser.add_argument('--savedir',type=str,default='batch_images/',
		help = 'save destination for batch images')
	parser.add_argument('--files',type=str,
		help = 'input image files, xx.npy')
	parser.add_argument('--batchsize',type=int,default=1,
		help = 'the size of batches')
	args = parser.parse_args()

	makebatch(args)
	
	return



if __name__=='__main__':
	main()	


