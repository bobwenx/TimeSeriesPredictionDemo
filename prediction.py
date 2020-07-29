#coding=utf-8

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf

f=open('dataset_1.csv')  
df=pd.read_csv(f)     
data=np.array(df['max'])
data=data[::-1]      

normalize_data=(data-np.mean(data))/np.std(data)  
normalize_data=normalize_data[:,np.newaxis]       
tf.compat.v1.reset_default_graph()

time_step=20      
rnn_unit=10      
lstm_layers=2
batch_size=60     
input_size=1      
output_size=1     
lr=0.0006         
train_x,train_y=[],[]   
for i in range(len(normalize_data)-time_step-1):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist()) 


X=tf.compat.v1.placeholder(tf.float32, [None,time_step,input_size])    
Y=tf.compat.v1.placeholder(tf.float32, [None,time_step,output_size])   

weights={
         'in':tf.Variable(tf.random.normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random.normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }


def lstm(batch):      
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])   
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(rnn_unit) for i in range(lstm_layers)])
    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



def prediction():
    with tf.compat.v1.variable_scope("sec_lstm",reuse=tf.compat.v1.AUTO_REUSE):
        pred,_=lstm(1)    
    saver=tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, 'model_save/model.ckpt') 
        
        prev_seq=train_x[-1]
        predict=[]
        for i in range(100):
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            predict.append(next_seq[-1])
            prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
        
        print(predict[:10])

        
prediction() 
