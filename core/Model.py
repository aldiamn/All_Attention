import tensorflow as tf
from Submodel import *
from Layer import *

class AttentionNet(object):
    def __init__(self,word_to_idx,feature_dim,emb_dim,hidden_dim,n_time_steps):
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i:w for w,i in word_to_idx.iteritems()}
        self.voc_size = len(word_to_idx)
        self.input_feature_num = feature_dim[0]
        self.input_feature_dim = feature_dim[1]
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_time_steps = n_time_steps
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        self._build_input()
        self._build_variables()
        self._build_model()

    def _build_input(self):
        self.img_feature = tf.placeholder(tf.float32,[None,self.input_feature_num,self.input_feature_dim])
        self.captions = tf.placeholder(tf.int32,[None,self.n_time_steps+1])
        self.support_context = tf.placeholder(tf.float32,[None,self.n_time_steps,self.input_feature_dim])

    def _build_variables(self):
        self.word_embedding = Embedding(self.voc_size,self.emb_dim,name = 'word_embedding')
        self.beginner = InitStateGen(self.input_feature_dim,self.emb_dim,name='beginner')
        self.state_tranducer = LSTM(self.emb_dim+2*self.input_feature_dim,2*self.input_feature_dim,activation=tf.nn.relu,name='state_tranducer')
        self.batch_norm = BatchNorm(name='batch_norm_model')
        self.encoder = Encoder(self.input_feature_dim,self.input_feature_num,name='encoder')
        self.decoder = Decoder(self.input_feature_dim,self.input_feature_num,self.emb_dim,name='decoder')
        self.classifier_h1 = Dense(2*self.input_feature_dim,self.hidden_dim,activation=tf.nn.relu,name='classifier_h1')
        self.classifier_h2 = Dense(self.hidden_dim,self.voc_size,activation=None,name='classifier_h2')
    
    def _build_model(self):
        caption_in = self.captions[:,:self.n_time_steps]
        caption_out = self.captions[:,1:]
        mask = tf.to_float(tf.not_equal(caption_out,self._null))
        word_vec = self.word_embedding(caption_in)
        features = self.batch_norm(self.img_feature,'train')
        latent,latent_key = self.encoder(features,'train')
        cell, state = self.beginner(features)
        state = tf.concat(axis=1,values=[state,state])
        cell = tf.concat(axis=1,values=[cell,cell])
        cell = tf.nn.dropout(cell,0.5)
        state = tf.nn.dropout(state,0.5)
        
        pred_loss = 0.0
        recon_loss = 0.0
        #loss=0.0
        #alpha_list = []
        for t in xrange(self.n_time_steps):
            if t == 0:
                context,recon = self.decoder(state,features,latent,latent_key,'train')
            else:
                context,recon = self.decoder(state,features,latent,latent_key,'train_iter')
            input_vec = tf.concat(axis = 1,values=[word_vec[:,t,:],context])
            state,cell = self.state_tranducer(input_vec,state,cell)
            cell = tf.nn.dropout(cell,0.5)
            state = tf.nn.dropout(state,0.5)
            logit = self.classifier_h1(state)
            logit = self.classifier_h2(logit)

            recon_loss += tf.reduce_sum(tf.squared_difference(self.support_context[:,t,:],recon))
            pred_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,labels=caption_out[:,t])*mask[:,t])

        self.loss = pred_loss#+recon_loss
        self.pretrain_loss = recon_loss

    def build_sampler(self):
        sample_word_list = []
        batch_size = tf.shape(self.img_feature)[0]

        features = self.batch_norm(self.img_feature,'test')
        latent, latent_key = self.encoder(features,mode='test')
        cell, state = self.beginner(features)
        state = tf.concat(axis=1,values=[state,state])
        cell = tf.concat(axis=1,values=[cell,cell])
        for t in xrange(self.n_time_steps):
            if t == 0:
                x = self.word_embedding(tf.fill([batch_size],self._start))
            else:
                x = self.word_embedding(sample_word)

            context,_ = self.decoder(state,features,latent,latent_key,mode='test')
            input_vec = tf.concat(axis=1,values=[x,context])
            cell, state = self.state_tranducer(input_vec,state,cell)
            logit = self.classifier_h1(state)
            logit = self.classifier_h2(logit)
            sample_word = tf.arg_max(logit,1)
            sample_word_list.append(sample_word)

        sample_caption = tf.transpose(tf.stack(sample_word_list),(1,0))

        return sample_caption
