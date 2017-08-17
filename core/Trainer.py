from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm

from utils import *
from bleu import evaluate

class Trainer(object):
    def __init__(self, model, optimizer, learning_rate = 1e-4):
        self.model = model
        if optimizer == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate)
        else:
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

    def train(self, data, support_data, val_data, epochs, pretrain_epochs, batch_size = 100, print_bleu = True):
        n_example = data['captions'].shape[0]
        if n_example%batch_size == 0:
            n_batch = int(n_example/batch_size)
        else:
            n_batch = int(n_example/batch_size)+1

        features = data['features']
        captions = data['captions']
        image_idx = data['image_idxs']

        val_features = val_data['features']
        n_val_example = val_features.shape[0]
        if n_val_example%batch_size==0:
            n_val_batch = int(n_val_example/batch_size)
        else:
            n_val_batch = int(n_val_example/batch_size)+1

        pretrain_opt = self.optimizer.minimize(self.model.pretrain_loss)
        train_opt = self.optimizer.minimize(self.model.loss)
        loss = self.model.loss
        pretrain_loss = self.model.pretrain_loss
        sample_caption = self.model.build_sampler()
        init = tf.global_variables_initializer()

        print('='*80)
        print('The number of epoch: %d'%epochs)
        print('Iteration per epoch: %d'%n_batch)
        print('The batch size: %d'%batch_size)
        print('The number of training example: %d'%n_example)
        print('The number of validation example: %d'%n_val_example)
        print('='*80)

        # config = tf.ConfigProto(allow_soft_placement = True)
        # config.gpu_options.allow_growth = True
        with tf.Session() as sess:
            print('model is initialized.')
            sess.run(init)
            saver = tf.train.Saver(max_to_keep=40)
            start_time = time.time()
            
            print('Start to pretrain...')
            for pre_ep in xrange(pretrain_epochs):
                pre_ep_start = time.time()
                rand_idxs = np.random.permutation(n_example)
                #captions = captions[rand_idxs]
                #support_data = support_data[rand_idxs]
                #image_idx = image_idx[rand_idxs]
                pretrain_cost = 0.
                for itr in tqdm(xrange(n_batch),desc='Pretrain Epoch:%d'%(pre_ep+1)):
                    start = itr*batch_size
                    if (itr+1)*batch_size>n_example:
                        end = n_example
                    else:
                        end = (itr+1)*batch_size
		    rand_idxs_batch = rand_idxs[start:end]
                    caption_batch = captions[rand_idxs_batch]
                    support_data_batch = support_data[rand_idxs_batch]
                    image_idx_batch = image_idx[rand_idxs_batch]
                    features_batch = features[image_idx_batch]
                                        
                    feed_dict={
                        self.model.img_feature:features_batch,
                        self.model.support_context:support_data_batch,
                        self.model.captions:caption_batch
                    }
                    _,pre_loss_batch = sess.run([pretrain_opt,pretrain_loss],feed_dict=feed_dict)
                    pretrain_cost += pre_loss_batch/n_batch
                pre_ep_end = time.time()
                pre_ep_sec = pre_ep_end-pre_ep_start
                pre_ep_min = int(pre_ep_sec/60)
                pre_ep_sec = pre_ep_sec%60

                print('='*80)
                print('Pretrain Epoch: %d'%(pre_ep+1))
                print('Pretrain loss: %.4f'%pretrain_cost)
                print('Cost time %d:%d'%(pre_ep_min,pre_ep_sec))
                print('='*80)

            
            print('Start to train...')
            for ep in xrange(epochs):
                train_cost=0.
                ep_start = time.time()
                rand_idxs = np.random.permutation(n_example)
                #captions = captions[rand_idxs]
                #support_data = support_data[rand_idxs]
                #image_idx = image_idx[rand_idxs]

                for itr in tqdm(xrange(n_batch),desc='Epoch:%d'%(ep+1)):
                    start = itr*batch_size
                    if (itr+1)*batch_size>n_example:
                        end = n_example
                    else:
                        end = (itr+1)*batch_size
	            rand_idxs_batch = rand_idxs[start:end]
                    caption_batch = captions[rand_idxs_batch]
                    image_idx_batch = image_idx[rand_idxs_batch]
                    features_batch = features[image_idx_batch]
                    support_data_batch = support_data[rand_idxs_batch]

                    feed_dict = {
                        self.model.img_feature:features_batch,
                        self.model.support_context:support_data_batch,
                        self.model.captions:caption_batch
                    }
                    _,loss_batch = sess.run([train_opt,loss],feed_dict=feed_dict)
                    train_cost += loss_batch/n_batch
                ep_end = time.time()
                ep_sec = ep_end-ep_start
                ep_min = int(ep_sec/60)
                ep_sec = ep_sec%60

                saver.save(sess,'./model_ckpt/model',global_step=ep+1)

                print('='*80)
                print('Epoch: %d'%(ep+1))
                print('Training loss: %.4f'%train_cost)
                print('Cost time %d:%d'%(ep_min,ep_sec))
                print('model-%d is saved'%(ep+1))
                print('='*80)

                if print_bleu:
                    all_gen_cap = np.ndarray((val_features.shape[0], 16))
                    for i in xrange(n_val_batch):
                        start = i*batch_size
                        if (i+1)*batch_size>n_val_example:
                            end = n_val_example
                        else:
                            end = (i+1)*batch_size
                        features_batch = val_features[start:end]
                        feed_dict = {self.model.img_feature: features_batch}
                        gen_cap = sess.run(sample_caption, feed_dict=feed_dict)  
                        all_gen_cap[start:end] = gen_cap
                    
                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                    scores = evaluate(data_path='./data', split='val', get_scores=True)
                    write_bleu(scores=scores, path='./model_ckpt', epoch=ep)
            
            end_time = time.time()
            total_sec = end_time-start_time
            total_hr = int(total_sec/3600)
            total_min = int((total_sec%3600)/60)
            total_sec = total_sec%60
            print('\n')
            print('Total cost time %d:%d:%d'%(total_hr,total_min,total_sec))
