from matplotlib.cbook import flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM,GRU, Activation, RNN, Bidirectional
from tensorflow.keras.layers import Embedding, Masking
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2,l1
import numpy as np
import logging 
import tensorflow as tf
from tensorflow.keras import regularizers
module_logger = logging.getLogger('Training')

class Simple_lstm(object): 
    def __init__(self,data,units,class_number,dropout_rate=0.3, model_type ='lstm'): 
        self.units= units
        self.data_shape = data.shape
        self.class_number = class_number
        self.dropout_rate = dropout_rate
        if model_type == 'lstm' : 
            self.model = self.lstm_model(data)
        elif model_type =='gru': 
            self.model =self.gru_model(data)
        elif model_type =='gru_fc': 
            self.model =self.gru_with_fc_model(data)
        elif model_type =='lstm_fc': 
            self.model =self.lstm_with_fc_model(data)
        elif model_type =='other': 
            self.model = self.lstm_model(data)
        
    
            
    # def training_process(self,X_train, Y_train,Y_train_class, Y_train_oh ,X_val, Y_val,Y_val_class, Y_val_oh ,leave_out = '' ):
    #     X = X_train 
    #     print('shape of  lstm input', X_train.shape)
    #     Y = np.asarray(Y_train_oh)
    #     val_X = np.asarray(X_val)
    #     val_Y = np.asarray(Y_val_oh)
    #     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)  
    #     myoptim=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-020, decay=0.1)
    #     all_weights = self.model.get_weights()
    #     #l2 = lambda_loss_amount * sum( tf.nn.l2_loss(tf_var) for tf_var in all_weights)
    #     self.model.compile(loss='sparse_categorical_crossentropy', optimizer=myoptim, metrics=['categorical_accuracy'])
    #     self.model.fit(X, Y,epochs=100, batch_size= 15, shuffle=True,  validation_data=(val_X, val_Y),verbose=1)
    #     loss,acc  = self.model.evaluate(np.array(val_X),np.array(Y_val_oh))
    #     val_result= self.model.predict(val_X)
    #     val_result= np.argmax(val_result,axis=1)
    #     print(val_result,Y_val)

    def lstm_model(self,data):
        """
        Function creating the neural network model's graph. 
        Arguments:
        input - data
        Returns:
        model -- a model instance in Keras
        """
        input_data = Input(data.shape,dtype = 'float32')
        print('data shape: ', data.shape)
        units = self.units
        model = Sequential()
        model.add(keras.layers.Masking(mask_value=0,input_shape=(data.shape)))   
        model.add(Bidirectional(LSTM(units, return_sequences=True,kernel_regularizer=regularizers.L2(0.05)),merge_mode='concat')) 
        model.add(Dropout(self.dropout_rate))
        # model.add(Bidirectional(LSTM(units, return_sequences=True,kernel_regularizer=regularizers.L2(0.05)),merge_mode='concat')) 
        # model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(LSTM(units, return_sequences=False ),merge_mode='concat')) 
       # model.add(keras.layers.Masking(mask_value=0) )
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.class_number)) 
        # Add a softmax activation
        model.add(Activation('softmax'))
        return model


    def lstm_with_fc_model(self,data):
        """
        Function creating the neural network model's graph. 
        Arguments:
        input - data
        Returns:
        model -- a model instance in Keras
        """
        input_data = Input(data.shape,dtype = 'float32')
        print('data shape: ', data.shape)
        units = self.units
        model = Sequential()
        model.add(keras.layers.Masking(mask_value=0,input_shape=(data.shape)))   
        model.add(keras.layers.Conv1D(filters= 32, kernel_size=50, activation= 'relu',padding = 'same',  data_format= 'channels_first'))
        model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(LSTM(units, return_sequences=True,kernel_regularizer=regularizers.L2(0.05)),merge_mode='concat')) 
        model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(LSTM(units, return_sequences=False ,kernel_regularizer=regularizers.L2(0.05)),merge_mode='concat')) 
        model.add(Dense(400))
        model.add(Dense(self.class_number)) 
        model.add(Activation('softmax'))
        return model

    def gru_with_fc_model(self,data):
        """
        Function creating the neural network model's graph. 
        Arguments:
        input - data
        Returns:
        model -- a model instance in Keras
        """
        input_data = Input(data.shape,dtype = 'float32')
        print('data shape: ', data.shape)
        units = self.units
        model = Sequential()
        model.add(keras.layers.Masking(mask_value=0,input_shape=(data.shape)))   
        model.add(keras.layers.Conv1D(filters= 32, kernel_size=50, activation= 'relu',padding = 'same',  data_format= 'channels_first'))
        model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(GRU(units, return_sequences=True,kernel_regularizer=regularizers.L2(0.05)),merge_mode='concat')) 
        model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(GRU(units, return_sequences=False ,kernel_regularizer=regularizers.L2(0.05)),merge_mode='concat')) 
        model.add(Dense(400))
        # Add a softmax activation
        model.add(Dense(self.class_number)) 
        model.add(Activation('softmax'))
        return model


    def gru_model(self,data):
        """
        Function creating the neural network model's graph. 
        Arguments:
        input - data
        Returns:
        model -- a model instance in Keras
        """
        input_data = Input(data.shape,dtype = 'float32')
        print(data.shape)
        units =self.units
        model = Sequential()
        #model.add(keras.layers.Masking(mask_value=0,input_shape=data.shape ) )   
        model.add(Bidirectional(GRU(units, return_sequences=True,kernel_regularizer=regularizers.L2(0.1)),merge_mode='concat', input_shape=data.shape )) 
        model.add(Dropout(self.dropout_rate))
        # model.add(Bidirectional(GRU(units, return_sequences=True,kernel_regularizer=regularizers.L2(0.1)),merge_mode='concat')) 
        # model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(GRU(units, return_sequences=False,kernel_regularizer=regularizers.L2(0.1)),merge_mode='concat')) 
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.class_number)) 
        # Add a softmax activation
        model.add(Activation('softmax'))
        return model

    def visualize_model(self,data,model_name): 
        if model_name =='lstm': 
            model = self.lstm_model(data)
        elif model_name == 'gru': 
            model = self.gru_model(data)
        elif model_name == 'lstm_fc': 
            model = self.lstm_with_fc_model(data)
        elif model_name == 'gru_fc': 
            model = self.gru_with_fc_model(data)
        return model.summary()

myoptim=Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-020, decay=1e-6)
def get_callbacks(filepath, patience=2):
    es = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(filepath, save_best_only =True, monitor = 'val_loss', mode ='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, verbose=1, min_delta=1e-4, mode='min') #epsilon
    return [es, mcp_save, reduce_lr_loss]

