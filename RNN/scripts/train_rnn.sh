#!/bin/bash 

cd ../

yaml='./yaml/config.yaml'
#the yaml file contains some general configuration such  as the gestures labels, the meta data name(to ignore during classification) and the dropout rate

subject='subject3'
# the choice of the subject to be left out

data_type='duo'
# the data type choice is one of three 
# nosep : to classify the entire diver gestures. 
# duo : to classify only two armed gestures 
# mono : to classify only one armed gestures
model_type='lstm'
# the model type can be one out of four
# lstm : The LSTM model, 
# gru: the GRU model, 
# lstm_fc : the LSTM-CL model 
# gru_fc: the GRU-CL model 


output='./../output/'
# a confusion matrix will be generated to the output directory
data='./../data/'
epochs=3
# number of training epochs (50 in our paper)

python3 rnn_training.py --yaml $yaml --subject $subject --data_type $data_type --output_path $output --model $model_type --epochs $epochs
