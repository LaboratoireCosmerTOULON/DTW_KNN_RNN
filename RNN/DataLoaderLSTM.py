
import numpy as np 
import pandas as pd 
import pickle
# import lstm_model as model
# import tensorflow as tf
# from tensorflow import keras 
from scipy import signal as sig
# import matplotlib.pyplot as plt
import itertools
import yaml
from sklearn.preprocessing import StandardScaler
#TODO implement merger with DataLoader in KnnDTW project
#TODO implemment a class for all the signal manipulation part based on the frequency part of the KNN DTW project
from scipy import interpolate

class DataLoader : 
    def __init__(self,training_data, testing_data, **kwargs ): 
        arguments = {'gesture_array' : ['goup','godown', 'halfpressure'], 
        'meta_columns' :['signal_id','Name','label','twohands','h'],
        'padding' :False,
        'resample':True,
        'padding_length':400}
        arguments ={**arguments,**kwargs}
        
        self.meta_columns = arguments['meta_columns']
        self.training_loc = training_data
        self.testing_loc = testing_data
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.padding = arguments['padding']
        self.repad_size = arguments['padding_length']
        self.resample = arguments['resample']
        self.resample_size=arguments['padding_length']

    def initialize_from_yaml(self,file_path): 
        '''initilazation code for paramaters that can be stored in a yaml file'''
        with open(file_path, 'r') as yaml_file:
            config_data = yaml.load(yaml_file, Loader=yaml.FullLoader)   
        self.gesture_array = config_data['gestures']
        self.class_names={element:index for index,element in enumerate(self.gesture_array)}
        self.meta_columns = config_data['meta_columns']
        print(self.class_names)
  
    def load_data(self): 
        with open (self.training_loc, 'rb') as f: 
            df = pickle.load(f)
            print(df.columns)
            print(self.training_loc, 'contains:', len(df), 'elements')
        self.train_df = df
        with open (self.testing_loc, 'rb') as f: 
            df = pickle.load(f)
            print(df.columns)
        self.test_df = df
    
    @staticmethod
    def interpolate_signal(signal, new_length):
        # Create an array representing the original time points
        original_time_points = np.linspace(0, 1, len(signal))

        # Create a linear interpolation function
        interpolation_function = interpolate.interp1d(original_time_points, signal, kind='linear', fill_value='extrapolate')

        # Generate the new signal by evaluating the interpolation function
        new_time_points = np.linspace(0, 1, new_length)
        new_signal = interpolation_function(new_time_points)

        return new_signal
    def resample_signal(self,df): 
        #TODO seperate both dfs 
        resample_length = self.repad_size
        df2 =df.drop(columns=self.meta_columns,errors= 'ignore')
        for column_name in df2.columns: 
            column = df2[column_name]
            print
            column = column.values
            new_column = []
            for  signal in column:
                if np.isnan(np.array(signal)).any(): 
                    print('we have a nan signal the signal is:', signal)
                #resampled_signal= sig.resample(signal,resample_length)
                resampled_signal=DataLoader.interpolate_signal(signal,resample_length)
                new_column.append(resampled_signal)
            df[column_name] = new_column
        return df
    
    def resample_2dfs(self): 
        self.train_df = self.resample_signal(self.train_df)
        self.test_df  = self.resample_signal(self.test_df) 

    def drop_meta_data(self) : 
        train_data,train_data_label = self.train_df.drop(columns=self.meta_columns, errors='ignore').values ,self.train_df['label']
        test_data,test_data_label = self.test_df.drop(columns=self.meta_columns, errors='ignore').values, self.test_df['label']
        return train_data,train_data_label ,test_data,test_data_label

    @staticmethod
    def convert_to_one_hot(inputY, C):
        "Convert Y to one hot representation"
        N= inputY.size
        Y=np.zeros((N,C),dtype= int)
        for i in range (0, inputY.size):
            Y[i, int(inputY[i])] = 1
        return Y

    def repad(self,array_np): 
        features, timestep = array_np.shape
        array_np = array_np[:,::3]
        features, timestep = array_np.shape
        array_np = np.pad(array_np,((0,0),(0,self.repad_size -timestep)) )
        return array_np

    def transform_X_data(self,train_data): 
        '''transfrom X data from the df shape to 3d numpy'''   
        input_np = ( train_data.values)
        input_np = [np.vstack(item) for item in (input_np)]
        if self.padding ==True :
            input_np = [DataLoader.repad(self,item) for item in input_np]
        input_np = np.stack(input_np)
        #input_np = input_np.swapaxes(1,2)
        return input_np
    
    def get_nan_index(self,data):
        input_np = (data.values)
        input_np = [np.vstack(item) for item in (input_np)]
        if self.padding ==True :
            input_np = [DataLoader.repad(self,item) for item in input_np]
        input_np = np.stack(input_np)
        input_np = input_np.swapaxes(1,2)
        ashta = (np.sum(np.isnan(input_np),axis=(1,2)))
        nan_index = np.where(ashta!=0)
        return nan_index
    
    @staticmethod 
    def encode_label(array_to_encode , accepted_labels ):
        encoding_dict = {gesture: index  for index, gesture in enumerate(accepted_labels)}
        encoded_list = [encoding_dict[label] if label in encoding_dict else 0 for label in array_to_encode]
        return encoded_list
    
    def split_x_y(self,df): 
        gesture_array=self.gesture_array
        gesture_list = self.class_names
        #df = df[df['label'].isin(gesture_array)]
        X = df.drop(columns=self.meta_columns)
        nan_index = DataLoader.get_nan_index(self,X)
        print(nan_index)
        df =df.drop(df.index[nan_index])
        X = df.drop(columns=self.meta_columns)
        
        Y = df['label']
        #Y_class = np.array([gesture_list[x] for x in Y])
        Y_class = np.array(DataLoader.encode_label(Y, gesture_array))
        print(Y_class)
        Y_oh    = DataLoader.convert_to_one_hot(Y_class,len(self.gesture_array) ).tolist()
        return X,Y,Y_class,Y_oh

    def split_x_y_with_id(self,df ): 
        gesture_list = self.class_names
        X = df.drop(columns=self.meta_columns)
        nan_index = DataLoader.get_nan_index(self,X)
        print(nan_index)
        Y = df['label']
        signal_id = df['signal_id']
        Y_class = np.array([gesture_list[x] for x in Y])
        print(Y_class)
        Y_oh    = DataLoader.convert_to_one_hot(Y_class,len(self.gesture_array) ).tolist()
        return X,Y,Y_class,Y_oh,signal_id
    
    # standarize signals
    @staticmethod
    def standarize_signal(x): 
        """lambda signal used to standarize one signal"""
        return (x-np.mean(x))/np.std(x)
    
    @staticmethod
    def mask_signal(sig, mask): 
        new_sig = [sig[i] for i in mask]
        new_sig = np.vstack(new_sig)
        return new_sig
    
    def standarize_df(self,df):
        dfcopy = df.copy(deep=True)
        dfcopy = dfcopy.drop(columns=self.meta_columns)
        print(dfcopy.columns)
        for column_name  in dfcopy: 
            df.loc[:,column_name] = df[column_name].apply(lambda x:DataLoader.standarize_signal(x))
        return df

    def standarize_2df(self): 
        self.train_df = self.standarize_df(self.train_df)
        self.test_df  = self.standarize_df(self.test_df) 

    def standard_scale_2df(self): 
        scaler = StandardScaler()
        train_copy = self.train_df.drop(columns= self.meta_columns)
        test_copy  = self.test_df.drop( columns= self.meta_columns)
        scalers = [StandardScaler() for _ in train_copy.columns]
        for i in range  (len(train_copy.columns)): 
            col = train_copy.columns[i]
            scaler = scalers[i]
            channel = (train_copy[col].values)
            longest_array= max (channel,key=len)
            filled_channel_np = np.array([np.concatenate([array, [np.nan] * (len(longest_array) - len(array))]) for array in channel]  )
            flattened_channel_series = filled_channel_np.flatten()
            flattened_channel_series = flattened_channel_series.reshape(-1,1)
            channel_scaled = scaler.fit_transform(flattened_channel_series)
            channel_scaled_reshape = channel_scaled.reshape(len(channel),-1)
            channel_unpadded = [row[~np.isnan(row)] for row in channel_scaled_reshape]
            self.train_df[col] = channel_unpadded

            channel = (test_copy[col].values)
            longest_array= max (channel,key=len)
            filled_channel_np = np.array([np.concatenate([array, [np.nan] * (len(longest_array) - len(array))]) for array in channel]  )
            flattened_channel_series = filled_channel_np.flatten()
            flattened_channel_series = flattened_channel_series.reshape(-1,1)
            channel_scaled = scaler.transform(flattened_channel_series)
            channel_scaled_reshape = channel_scaled.reshape(len(channel),-1)
            channel_unpadded = [row[~np.isnan(row)] for row in channel_scaled_reshape]
            self.test_df[col] = channel_unpadded
