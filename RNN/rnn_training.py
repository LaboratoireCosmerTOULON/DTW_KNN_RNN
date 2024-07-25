from DataLoaderLSTM import DataLoader
import lstm_model as model
import numpy as np 
import argparse
import tensorflow as tf
from tensorflow import keras 
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import yaml
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,args, normalize=False,title='Confusion matrix',cmap=plt.cm.Blues, nan_conv=False,subject='subject'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm_non = cm
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    if(nan_conv): 
        cm = np.nan_to_num(cm,nan=0.0)
   
    plt.figure(figsize = (10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=15)
    plt.yticks(tick_marks, classes,rotation=45,fontsize=15)

    fmt = '.0%' if normalize else 'd'
     
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize: 
            text_to_write = str(format(cm[i, j], fmt)) #+'\n'+str(cm_non[i,j])
        else : text_to_write = str(format(cm[i, j], fmt))+'\n'

        plt.text(j, i,text_to_write, horizontalalignment="center",verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black",fontsize=15)
        
    plt.title(subject)
    plt.tight_layout()
    plt.ylabel('True label',fontsize=20)
    plt.xlabel('Predicted label',fontsize=20)
    plt.savefig(args.output+'confusion_matrix'+args.subject+'.jpg')

def one_by_one_induction(dataloader,model,test_data): 
    X_test,Y_test,Y_class_test,Y_oh_test, signal_id_test =   dataloader.split_x_y_with_id(dataloader.test_df)
    X_test= dataloader.transform_X_data(X_test)
    signal_id_test= signal_id_test.array
    print('sampleId','correct class', 'predicted class')
    for i in range(len(X_test)): 
        sample_test = np.array([X_test[i]])
        sample_id = signal_id_test[i]
        sample_class = Y_class_test[i]
        result = np.argmax(model.predict(sample_test))
        print(sample_id , sample_class,result)

def main():
    print('main') 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', required = False, dest  = 'data_location', type=str, help='directory containing the data',default='./../data/')
    parser.add_argument('--data_type', required = False, dest  = 'data_type', type=str, help='nosep mono or duo',default='nosep')
    parser.add_argument('--output_path', required = False, dest  = 'output', type=str, help='contains the output path',default='./../output/')
    parser.add_argument('--subject', required = True, dest  = 'subject', type=str, help='which subject to leave out')
    parser.add_argument('--units', required = False , dest  = 'units', type=int, default=120, help='width of lstm')
    parser.add_argument('--model', required = False , dest  = 'model', type=str, default='lstm', help='RNN model choice can be one of 4 : lstm lstm_fc gru gru_fc')
    parser.add_argument('--yaml',dest='yaml', required =True, help ='yaml file location')
    parser.add_argument('--epochs', dest='epoch',required=False,type=int,default=5, help = 'number of epochs  of training' )
    args =parser.parse_args() 
    with open(args.yaml,'r') as yaml_file: 
        data = yaml.load(yaml_file,Loader=yaml.FullLoader)
        gesture_list = data['gestures']
        numberofclasses = len(gesture_list)
        dropout_rate = data['dropout']
    model_type =args.model
    print('Parameters  used during this training:\n')
    print(yaml.dump(data,default_flow_style = False))
    train_data = args.data_location +args.data_type +'/'+args.subject+ '_train.pickle'
    test_data = args.data_location +args.data_type +'/'+args.subject+ '_test.pickle'
    dataLoader = DataLoader(training_data=train_data,testing_data=test_data, resample = True, padding= False,padding_length=400)
    dataLoader.initialize_from_yaml(file_path=args.yaml)
    
    dataLoader.load_data()
    dataLoader.drop_meta_data()
    dataLoader.standarize_2df()
    dataLoader.resample_2dfs()
    
    X_train,Y_train,Y_class_train,Y_oh_train = dataLoader.split_x_y(dataLoader.train_df)
    X_train = dataLoader.transform_X_data(X_train)
    X_test,Y_test,Y_class_test,Y_oh_test = dataLoader.split_x_y(dataLoader.test_df)
    X_test = dataLoader.transform_X_data(X_test)
    
    X_val = X_test
    Y_val_oh = Y_oh_test
    Y_oh_train = np.asarray(Y_oh_train)
    val_X = np.asarray(X_val)
    val_Y = np.asarray(Y_val_oh)
    rnn_model = model.Simple_lstm(X_train[0],args.units,model_type=model_type,class_number= numberofclasses,dropout_rate= dropout_rate)
    print('The model used : \n',model_type )
    rnn_model.visualize_model(X_train[0],model_name=model_type)
    rnn_model =rnn_model.model
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)  
    myoptim=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-040, decay=1e-6)
    rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=myoptim, metrics=['accuracy','sparse_categorical_crossentropy'])
    rnn_model.fit(X_train, Y_class_train,epochs=args.epoch, batch_size= 100, shuffle=True,  validation_data=(X_test, Y_class_test),verbose=2)
    #loss,acc,loss1,k2,k3,k4  = rnn_model.evaluate(val_X,Y_class_test)
    loss,acc,loss1  = rnn_model.evaluate(val_X,Y_class_test)
    print('The results  on the  validation data are\n loss: ',loss, ' the accuracy: ',acc )
    val_result= rnn_model.predict(val_X)
    val_result= np.argmax(val_result,axis=1)
    print(val_result,Y_test)
    
    print(classification_report(Y_class_test, val_result))
    cm = confusion_matrix(Y_class_test,val_result,labels = list(range(len(gesture_list) )  ))   
    print(cm)
    plot_confusion_matrix(cm, classes= gesture_list, args= args, subject= args.subject)
    one_by_one_induction(dataloader=dataLoader, model = rnn_model, test_data=X_test)

if __name__ == "__main__":
    main()
