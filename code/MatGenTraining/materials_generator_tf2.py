#!/usr/bin/env python
import os
import sys
import pickle,glob
import json
import numpy as np
import tensorflow as tf
import torch
import itertools
from tqdm import *
from utils import *
from numpy import *
from numpy.linalg import norm
'''
Global Parameters
'''
n_ae_epochs= 1000+1
batch_size = 1
ae_lr      = 0.0001
z_size     = 500
leak_value = 0.2
reg_map    = 0.1
reg_kl     = 5.0e-4
reg_mse    = 1.0
ae_inter   = 142
maxbatch   = 1800
trbatch    = 1600

batch_directory = './../vo-2nd-batch/'
train_sample_directory = './ae_sample/'
model_directory = './ae_models/'

weights = {}

def slerp(list1, t:'int'):

		
		omega=(list1[0]-list1[1])/t
		a=[0 for i in range(t-1)]
		for i in range (t-1):
			
			a[i]=list1[1]+omega*(i+1)
		return a
		
def sampling(mu, log_var):
  eps = tf.random.normal(shape=tf.shape(input=mu))
  return mu + tf.exp(log_var / 2) * eps

def encoder(inputs, phase_train=True, reuse=False):

	strides = [1,1,2,1]
	with tf.compat.v1.variable_scope("enc",reuse=reuse):
		e_1 = tf.nn.conv2d(input=inputs, filters=weights['we1'], strides=[1,1,2,1], padding="SAME")
		e_1 = lrelu(e_1, leak_value)

		e_2 = tf.nn.conv2d(input=e_1, filters=weights['we2'], strides=[1,1,2,1], padding="SAME") 
		e_2 = lrelu(e_2, leak_value)

		e_3 = tf.nn.conv2d(input=e_2, filters=weights['we3'], strides=[1,1,2,1], padding="SAME")  
		e_3 = lrelu(e_3, leak_value) 

		e_4 = tf.nn.conv2d(input=e_3, filters=weights['we4'], strides=[1,1,1,1], padding="SAME")
		e_4 = lrelu(e_4, leak_value)
		e_4_flat = tf.keras.layers.Flatten()(e_4)

		e_5 = tf.matmul(e_4_flat, weights['we5']) 

		return e_5

def decoder(inputs, phase_train=True, reuse=False):
	strides = [1,1,2,1]
	with tf.compat.v1.variable_scope("dec",reuse=reuse):
		d_1 = tf.matmul(inputs, weights['wd1'])
		d_1 = lrelu(d_1)
		d_1_reshape = tf.reshape(d_1, tf.stack([batch_size,1,25,50]))

		d_2 = tf.nn.conv2d_transpose(d_1_reshape, weights['wd2'],(batch_size,1,25,100), strides=[1,1,1,1], padding="SAME") 
		d_2 = lrelu(d_2, leak_value)
        
		d_3 = tf.nn.conv2d_transpose(d_2, weights['wd3'],(batch_size,1,50,100), strides=[1,1,2,1], padding="SAME")  
		d_3 = lrelu(d_3, leak_value) 

		d_4 = tf.nn.conv2d_transpose(d_3, weights['wd4'], (batch_size,1,100,100),strides=[1,1,2,1], padding="SAME")     
		d_4 = lrelu(d_4, leak_value)

		d_5 = tf.nn.conv2d_transpose(d_4, weights['wd5'], (batch_size,1,200,6),strides=[1,1,2,1], padding="SAME")
		d_5 = tf.nn.tanh(d_5)

		return d_5

def mapping(inputs):
	with tf.compat.v1.variable_scope("map"):
		m_1 = tf.matmul(inputs, weights['wm1'])
		m_1 = tf.nn.tanh(m_1)

		m_2 = tf.matmul(m_1, weights['wm2'])
	return m_2


def initialiseWeights():

	global weights
	xavier_init = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")

	weights['we1'] = tf.compat.v1.get_variable("we1", shape=[1, 4, 6, 100], initializer=xavier_init)
	weights['we2'] = tf.compat.v1.get_variable("we2", shape=[1, 4, 100, 100], initializer=xavier_init)
	weights['we3'] = tf.compat.v1.get_variable("we3", shape=[1, 4, 100, 100], initializer=xavier_init)
	weights['we4'] = tf.compat.v1.get_variable("we4", shape=[1, 1, 100, 50], initializer=xavier_init)
	weights['we5'] = tf.compat.v1.get_variable("we5", shape=[1250,z_size*2], initializer=xavier_init)

	weights['wd1'] = tf.compat.v1.get_variable("wd1", shape=[z_size,1250], initializer=xavier_init) 
	weights['wd2'] = tf.compat.v1.get_variable("wd2", shape=[1, 1, 100, 50], initializer=xavier_init)
	weights['wd3'] = tf.compat.v1.get_variable("wd3", shape=[1, 4, 100, 100], initializer=xavier_init)
	weights['wd4'] = tf.compat.v1.get_variable("wd4", shape=[1, 4, 100, 100], initializer=xavier_init)
	weights['wd5'] = tf.compat.v1.get_variable("wd5", shape=[1, 4, 6, 100], initializer=xavier_init)    

	weights['wm1'] = tf.compat.v1.get_variable("wm1", shape=[z_size,500], initializer=xavier_init)
	weights['wm2'] = tf.compat.v1.get_variable("wm2", shape=[500,1], initializer=xavier_init)

	return weights

def trainGAN(loadmodel,model_checkpoint):
	tf.compat.v1.disable_eager_execution()
	weights = initialiseWeights()
	x_vector = tf.compat.v1.placeholder(shape=[batch_size,1,200,5],dtype=tf.float32)
	z_vector = tf.compat.v1.placeholder(shape=[batch_size,z_size],dtype=tf.float32)
	c_vector = tf.compat.v1.placeholder(shape=[batch_size,1,1,5],dtype=tf.float32)
	cell_vector = tf.compat.v1.placeholder(shape=[batch_size,1,200,1],dtype=tf.float32)
	p_vector = tf.compat.v1.placeholder(shape=[1,1],dtype=tf.float32)

  # Weights for autoencoder pretraining
	xavier_init = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")

	x_cell = tf.concat([x_vector,cell_vector],3)

	with tf.compat.v1.variable_scope('encoders') as scope1:
		encoded = encoder(x_cell, phase_train=True, reuse=False)

		scope1.reuse_variables()
		encoded2 = encoder(x_cell, phase_train=False, reuse=True)

	z_mu, z_logvar = tf.split(encoded,[z_size,z_size],1)
	z_sampled = 0.0001 * sampling(z_mu,z_logvar)

	z_mu2, z_logvar2 = tf.split(encoded2,[z_size,z_size],1)
	z_sampled2 = 0.0001 * sampling(z_mu,z_logvar)

	with tf.compat.v1.variable_scope('decoders') as scope4:
		decoded = decoder(z_sampled, phase_train=True, reuse=False)
		scope4.reuse_variables()
		decoded2 = decoder(z_sampled2, phase_train=False, reuse=True)

	with tf.compat.v1.variable_scope('mappings') as scope3:
		mapped = mapping(z_sampled)

		scope3.reuse_variables()
		mapped2 = mapping(z_sampled2)
		

	# Compute MSE Loss and L2 Loss
	all_loss = tf.reduce_mean(input_tensor=tf.pow((decoded-x_cell),2))
	all_loss2 = tf.reduce_mean(input_tensor=tf.pow((decoded2-x_cell),2))

	map_loss = tf.reduce_mean(input_tensor=tf.pow((mapped-p_vector),2))
	map_loss2 = tf.reduce_mean(input_tensor=tf.pow((mapped2-p_vector),2))

	kl_loss = tf.reduce_mean(input_tensor=0.5*((tf.exp(z_logvar)+(z_mu**2)) - 1. - z_logvar))
	kl_loss2 = tf.reduce_mean(input_tensor=0.5*((tf.exp(z_logvar2)+(z_mu2**2)) - 1. - z_logvar2))

	para_ae1 = [var for var in tf.compat.v1.trainable_variables() if any(x in var.name for x in ['we','wd','wm'])]

	l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in para_ae1])
	ae_loss = all_loss + reg_kl*kl_loss+reg_map*map_loss

	optimizer_ae1 = tf.compat.v1.train.AdamOptimizer(learning_rate=ae_lr, name="Adam_AE").minimize(ae_loss,var_list=para_ae1)

	saver = tf.compat.v1.train.Saver(max_to_keep=50) 
	orienergylist=[]
	preenergylist=[]
	with tf.compat.v1.Session() as sess:  
		sess.run(tf.compat.v1.global_variables_initializer())        
		if loadmodel == True:
			saver.restore(sess, model_checkpoint)

		np.random.seed(333)
		fname = [str(n) for n in range(maxbatch)]
		np.random.shuffle(fname)
		tr_name = fname[:trbatch]
		test_name = fname[trbatch:]
		
		np.random.shuffle(tr_name)
		for epoch in range(n_ae_epochs):
			orienergylist=[]
			preenergylist=[]
			mse_train = 0; mse_test = 0;
			kl_train = 0; kl_test = 0;
			map_train = 0; map_test = 0;
			plist=np.loadtxt('800k.txt')
			for idx_tr in tqdm(range(len(tr_name))):
				x = np.load(batch_directory+tr_name[idx_tr]+'_pvals_basis.npy',allow_pickle=True)#basis vector
				cell = np.load(batch_directory+tr_name[idx_tr]+'_cell.npy',allow_pickle=True)#cell vector

				p = plist[idx_tr]
				
				map_l, mse_l, kl_l, _ ,energypre,energyori= sess.run([map_loss,all_loss, kl_loss, optimizer_ae1,mapped,p_vector],feed_dict={x_vector:x.reshape(1,1,200,5), cell_vector:cell.reshape(-1,1,200,1),p_vector:p.reshape(-1,1)})
				mse_train += mse_l; kl_train += kl_l; map_train += map_l
				
				
				energypre_value=energypre.reshape(1,1)
				pretemenergy=energypre.tolist()
				preenergylist.append(pretemenergy)
				print(map_l, mse_l, kl_l, _ ,energypre,energyori,tr_name[idx_tr])
				
					
					
				energyori_value=energyori.reshape(1,1)
				oritemenergy=energyori.tolist()
				orienergylist.append(oritemenergy)
					
					
				
			for idx_t in range(len(test_name)):
				x_test = np.load(batch_directory+test_name[idx_t]+'_pvals_basis.npy',allow_pickle=True)

				cell_test = np.load(batch_directory+test_name[idx_t]+'_cell.npy',allow_pickle=True)
				p_test = plist[idx_tr]
				
				map_t, mse_t, kl_t = sess.run([map_loss2,all_loss2, kl_loss2], feed_dict={x_vector:x_test.reshape(1,1,200,5), cell_vector:cell_test.reshape(-1,1,200,1),p_vector:p_test.reshape(-1,1)})
				mse_test += mse_t; kl_test += kl_t;map_test += map_t
				
			print (epoch,' ',mse_train/len(tr_name),' ',mse_test/len(test_name),' ',kl_train/len(tr_name),' ',kl_test/len(test_name),' ',map_train/len(tr_name),' ',map_test/len(test_name))
			np.random.shuffle(tr_name)

			if epoch % ae_inter == 0:
				if not os.path.exists(model_directory):
					os.makedirs(model_directory)      
				saver.save(sess, save_path = model_directory + '/' + str(epoch) + '_ae.ckpt')
				
	predic = {'image':preenergylist}
	savefilenamepre = 'final_preenergy.npy'
	with open(savefilenamepre,'w') as f:
		json_dicts=json.dump(predic,f)
		
	oridic = {'image':orienergylist}
	savefilenameori = 'final_orienergy.npy'
	with open(savefilenameori,'w') as f:
		json_dicts=json.dump(oridic,f)








if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
	loadmodel = bool(int(sys.argv[2]))
	model_checkpoint = sys.argv[3]
	trainGAN(loadmodel,model_checkpoint)
	
	'''arraycom=[]
	fname = [str(n) for n in range(20)]
	tr_name = fname
	p_list=np.loadtxt('300k.txt')
	for idx_tr in tqdm(range(len(tr_name))):
		x = np.load(batch_directory+tr_name[idx_tr]+'_pvals_basis - 副本.npy',allow_pickle=True)#basis vector
		cell = np.load(batch_directory+tr_name[idx_tr]+'_cell - 副本.npy',allow_pickle=True)
		cell=cell.reshape(1,1,200,1)
		
		x1 = np.concatenate([x,cell],3); 
		x2 = torch.tensor(x1,dtype=torch.float32)
		outcellnew=encoder(x2)
		x2_mu, x2_logvar = tf.split(outcellnew,[500,500],1)
		x2_sampled = 0.0001 * sampling(x2_mu,x2_logvar)
		energy=mapping(x2_sampled)


		with tf.compat.v1.Session() as sess:
			sess.run(tf.compat.v1.global_variables_initializer()) 
			outcellnew_value=sess.run(outcellnew)
			
		with tf.compat.v1.Session() as sess:
			sess.run(tf.compat.v1.global_variables_initializer()) 
			energy_value=sess.run(energy)
		
			
			

			
			
		outcellnew2=outcellnew_value.reshape(1,1000)
		dic = {'image':outcellnew2.tolist()}
		savefilename = str(idx_tr)+'_encoded.npy'
		with open(savefilename,'w') as f:
			json_dicts=json.dump(dic,f)
		arraycom.append(outcellnew2)
		
		energy_value2=energy_value.reshape(1,1)
		dic = {'energy':energy_value2.tolist()}
		savefilenameen = str(idx_tr)+'_energy.npy'
		with open(savefilenameen,'w') as f:
			json_dicts=json.dump(dic,f)
		
		
		
		
		
		
		
		
	list2 = list(itertools.combinations(arraycom, 2))
	slerpsample=[]
	for i in range (len(list2)):
		slerpsample.extend(slerp((list2[i]),10))

	
	
	for idy_tr in tqdm(range(len(slerpsample))):
		dic = {'image':slerpsample[idy_tr].tolist()}
		savefilename = str(idx_tr)+'_encoded_slerp.npy'
		x3 = torch.tensor(slerpsample[idy_tr],dtype=torch.float32)
		
		x3_mu, x3_logvar = tf.split(x3,[500,500],1)
		x3_sampled = 0.0001 * sampling(x3_mu,x3_logvar)
		samplenew=decoder(x3_sampled)
		with tf.compat.v1.Session() as sess:
			sess.run(tf.compat.v1.global_variables_initializer()) 
			outsamplenew_value=sess.run(samplenew)
		outsamplenew2=outsamplenew_value.reshape(1,1,200,6)
		outsamplebasis1=outsamplenew2[:,:,:,0]
		outsamplebasis2=outsamplenew2[:,:,:,0]
		outsamplecell=outsamplenew2[:,:19,:,5]
		
		
		dic = {'image':outsamplenew2.tolist()}
		savefilename2 = str(idy_tr)+'_decoded.npy'
		with open(savefilename2,'w') as f:
			json_dicts=json.dump(dic,f)
		
		
		
		dic = {'image':outsamplebasis1.tolist()}
		savefilename3 = str(idy_tr)+'_decoded_basis1.npy'
		with open(savefilename3,'w') as f:
			json_dicts=json.dump(dic,f)
		
		
		dic = {'image':outsamplebasis2.tolist()}
		savefilename4 = str(idy_tr)+'_decoded_basis2.npy'
		with open(savefilename4,'w') as f:
			json_dicts=json.dump(dic,f)
		
		
		
		dic = {'image':outsamplecell.tolist()}
		savefilename5 = str(idy_tr)+'_decoded_cell.npy'
		with open(savefilename5,'w') as f:
			json_dicts=json.dump(dic,f)'''
		
		
		
		
		