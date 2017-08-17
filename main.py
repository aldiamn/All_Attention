from __future__ import print_function

import numpy as np
import tensorflow as tf

from core.utils import *
from core.Model import *
from core.Trainer import *

data = load_coco_data(data_path = './data', split='train')
val_data = load_coco_data(data_path = './data', split='val')
#data = np.load('./data/train/data.npz')
#val_data = np.load('./data/val/data.npz')
word_to_idx = data['word_to_idx']
support_context = np.load('data/context/context_model_19_v2.npy')

model = AttentionNet(word_to_idx=word_to_idx,
                    feature_dim=[196,512],
                    emb_dim = 512,
                    hidden_dim = 2048,
                    n_time_steps = 16)
print('Model is built')
trainer = Trainer(model,optimizer='adam', learning_rate = 1e-3)
trainer.train(data = data,
              support_data = support_context,
              val_data = val_data,
              epochs = 20,
              pretrain_epochs = 0,
              batch_size = 100)
