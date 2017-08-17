import tensorflow as tf
from Layer import *

class Encoder(object):
    def __init__(self,input_dim,data_num,name):
        self.input_dim = input_dim
        self.data_num = data_num
        self.name = name

        self._build_varibles()

    def _build_varibles(self):
        with tf.variable_scope(self.name):
            self.slf_att = Self_Attention(input_dim = self.input_dim,input_num = self.data_num,name='self_attention')
            self.feedforward_h1 = Tensor_Dense(input_dim=self.input_dim,output_dim=self.input_dim,activation=tf.nn.softplus,name = 'transform_h1')
            self.feedforward_h2 = Tensor_Dense(input_dim=self.input_dim,output_dim=self.input_dim,activation=tf.nn.softplus,name = 'transform_h2')
            self.batch_norm = BatchNorm(name=self.name+'/'+'batch_norm')

    def __call__(self,image_features,mode):
        _,img_context,key = self.slf_att(image_features)
        #img_context = tf.add(img_context,image_features)
        #img_rep = self.feedforward_h1(img_context)
        #img_rep = self.feedforward_h2(img_rep)
        #img_rep = tf.add(img_rep,img_context)
        #img_rep = self.batch_norm(img_context,mode)
        img_rep = img_context
        return img_rep, key

class Decoder(object):
    def __init__(self,feature_dim,feature_num,emb_dim,name):
        self.input_feature_num = feature_num
        self.input_feature_dim = feature_dim
        self.emb_dim = emb_dim
        self.name = name

        self._build_varibles()
        
    def _build_varibles(self):
        with tf.variable_scope(self.name):
            self.feature_to_key = FeatureProj(self.input_feature_dim,self.input_feature_num,self.input_feature_dim,name = 'feature_proj')
            self.text_to_key = Dense(2*self.emb_dim,self.input_feature_dim,None,name='text_to_key')
            self.hid_to_key = Dense(self.input_feature_dim,self.input_feature_dim,None,name='hid_to_key')
            self.batch_norm_feat = BatchNorm(name=self.name+'/'+'batch_norm_feat')            

            # Attention
            self.controller_h1 = Attention_Product(self.input_feature_dim,self.emb_dim,self.input_feature_num,name='controller_h1')
            self.controller_h2 = Dense(self.input_feature_dim,self.input_feature_dim,activation=tf.nn.relu,name='controller_h2')
            self.controller_h3 = Dense(self.input_feature_dim,self.input_feature_dim,activation=tf.nn.relu,name='controller_h3')
            self.batch_norm_att = BatchNorm(name=self.name+'/'+'batch_norm_att')

            # Multihop Attention
            self.controller_2_h1 = Attention_Product(self.input_feature_dim,self.emb_dim,self.input_feature_num,name='controller_2_h1')
            self.controller_2_h2 = Dense(self.input_feature_dim,self.input_feature_dim,activation=tf.nn.relu,name='controller_2_h2')
            self.controller_2_h3 = Dense(self.input_feature_dim,self.input_feature_dim,activation=tf.nn.relu,name='controller_2_h3')
            self.batch_norm_matt = BatchNorm(name=self.name+'/'+'batch_norm_matt')

            # Text only
            #self.text_pred_h1 = Dense(self.emb_dim,self.input_feature_dim,activation=tf.nn.relu,name='text_pred_h1')
            #self.text_pred_h2 = Dense(self.input_feature_dim,self.input_feature_dim,activation=tf.nn.relu,name='text_pred_h2')
            #self.batch_norm_t = BatchNorm(name=self.name+'/'+'batch_norm_t')

            # image vs text gate
            self.gate = Dense(2*self.input_feature_dim,1,activation=tf.nn.sigmoid,name='gate')

            # i2i vs w2i
            self.gate_i2i = Dense(self.input_feature_dim,1,activation=tf.nn.sigmoid,name='gate_i2i')

            # reconstruct
            self.reconstruct_h1 = Dense(self.input_feature_dim,1024,tf.nn.relu,'reconstruct_h1')
            self.reconstruct_h2 = Dense(1024,self.input_feature_dim,None,'reconstruct_h2')


    def __call__(self,text,feature,encoder_output,encoder_key,mode):
        key = self.feature_to_key(feature)
        #feature = self.batch_norm_feat(feature,mode)
        text_key = self.text_to_key(text)
        alpha,context = self.controller_h1(feature,key,text_key)
        
        
        # Attention
        hidden_c = self.controller_h2(context)
        hidden_c = self.controller_h3(hidden_c)
        #hidden_c = tf.add(hidden_c,context)
        #hidden_c = self.batch_norm_att(hidden_c,mode)
        hidden_c = tf.nn.dropout(hidden_c,0.5)
        # hidden key
        hidden_c_key = self.hid_to_key(hidden_c)

    
        # Multihop Attention
        slf_alpha, context_2 = self.controller_2_h1(encoder_output,encoder_key,hidden_c_key)
        hidden_c2 = self.controller_2_h2(context_2)
        hidden_c2 = self.controller_2_h3(hidden_c2)
        #hidden_c2 = tf.concat(axis=1,values=[hidden_c2,hidden_c])
        #hidden_c2 = tf.multiply(1-hidden_g_i2i,hidden_c2)+tf.multiply(hidden_g_i2i,hidden_c)
        #hidden_c2 = self.batch_norm_matt(hidden_c2,mode)
        #hidden_c2 = tf.multiply(hidden_g_i2i,hidden_c2)
        hidden_c2 = tf.nn.dropout(hidden_c2,0.5)

        # Text only
        #hidden_t = self.text_pred_h1(text)
        #hidden_t = self.text_pred_h2(hidden_t)
        #hidden_t = tf.add(hidden_t,text)
        #hidden_t = self.batch_norm_t(hidden_t,mode)
        #hidden_t = tf.nn.dropout(hidden_t,0.5)
        
        # i2i vs w2i
        hidden_g_i2i = self.gate_i2i(hidden_c)
        
        # image vs text gate
        hidden_g = self.gate(text)
        
        hidden_c = tf.multiply(hidden_g,hidden_c)
        hidden_c2 = tf.multiply(hidden_g_i2i,hidden_c2)

        # generate state
        #state = tf.multiply(hidden_g,hidden_c2)+tf.multiply(1-hidden_g,text)
        #state = tf.multiply(hidden_g,hidden_c2)
        #state = tf.add(hidden_c2,text)
        state = tf.concat(axis=1,values=[hidden_c2,hidden_c])

        # Reconstruction
        recon = self.reconstruct_h1(hidden_c)
        recon = self.reconstruct_h2(recon)

        return state, recon
