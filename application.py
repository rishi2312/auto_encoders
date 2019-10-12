import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input,Dense
from keras.callbacks import TensorBoard
import tensorflow as tf
import os

def seedy(s):
    tf.random.set_seed(s)

class AutoEncoder:
    '''
    def __init__(self,ec_dim=3,hidn_dim=3,hidn_lyr=3):
        self.encoding_dims = ec_dim
        self.hidden_layers = hidn_lyr
        self.hidden_dims = hidn_dim
        self.x = pd.read_csv('data2.csv').to_numpy()
        print(self.x.shape)
    '''
    def __init__(self,layers):
        self.layers = layers
        self.x = pd.read_csv('data.csv').to_numpy()
        
    def _encoder(self):
        inputs = Input(shape=(self.x[0].shape))
        temp = [None for i in range(len(self.layers)-2)]
        temp[0] = Dense(self.layers[1],activation='sigmoid')(inputs)
        for i in range(1,len(self.layers)-2):
            temp[i] = Dense(self.layers[i+1],activation='sigmoid')(temp[i-1])
        encoded = Dense(self.layers[len(self.layers)-1],activation='sigmoid')(temp[len(temp)-1])
        model = Model(inputs,encoded)
        #model.summary()
        self.encoder = model
        return model
        '''
        layers = [None for i in range(self.hidden_layers)]
        layers[0] = Dense(self.hidden_dims,activation='linear',use_bias=True)(inputs)
        for i in range(1,self.hidden_layers):
            layers[i] = Dense(self.hidden_dims,activation='linear',use_bias=True)(layers[i-1])
        encoded = Dense(self.encoding_dims,activation='linear',use_bias=True)(layers[self.hidden_layers-1])
        model = Model(inputs,encoded)
        self.encoder = model
        return model
        '''
                                   
    def _decoder(self):
        self.layers.reverse()
        inputs = Input(shape=(self.layers[0],))
        temp = [None for i in range(len(self.layers)-2)]
        temp[0] = Dense(self.layers[1],activation='sigmoid')(inputs)
        for i in range(1,len(self.layers)-2):
            temp[i] = Dense(self.layers[i+1],activation='sigmoid')(temp[i-1])
        decoded = Dense(self.layers[len(self.layers)-1],activation='softmax')(temp[len(temp)-1])
        model = Model(inputs,decoded)
        #model.summary()
        self.decoder = model
        return model
        '''
        inputs = Input(shape=(self.encoding_dims,))
        layers =[None for i in range(self.hidden_layers)]
        layers[0] = Dense(self.hidden_dims,activation='linear',use_bias=True)(inputs)
        for i in range(1,self.hidden_layers):
            layers[i] = Dense(self.hidden_dims,activation='linear',use_bias=True)(layers[i-1])
        decoded = Dense(len(self.x[0]),activation='softmax',use_bias=True)(layers[self.hidden_layers-1])
        model = Model(inputs,decoded)
        self.decoder = model
        return model
        '''
        
    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()
        inputs = Input(shape=(self.x[0].shape))
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model=Model(inputs,dc_out)
        #model.summary()
        self.model = model
        return model
    
    def fit(self,batch_size=10,epochs=300,loss='binary_crossentropy',optimizer='adadelta'):
        self.model.compile(optimizer=optimizer,loss=loss)
        log_dir = '.\\log\\'
        tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq = 0,write_graph=True,write_images = True)
        self.model.fit(self.x,self.x,epochs=epochs,batch_size=batch_size,callbacks=[tbCallBack])

    def save(self):
        if not os.path.exists(r'.\\weights'):
            os.mkdir(r'.\\weights')

        else:
            self.encoder.save(r'.\\weights\\encoder_weights.h5')
            self.decoder.save(r'.\\weights\\decoder_weights.h5')
            self.model.save(r'.\\weights\\ae_weights.h5')

if __name__ == '__main__':
    layers = [784,500,300,150,50,10,1]
    ae = AutoEncoder(layers)
    ae.encoder_decoder()
    ae.fit(batch_size=250,epochs=100)
    ae.save()

    
        
