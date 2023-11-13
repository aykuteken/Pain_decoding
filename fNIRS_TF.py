#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:12:58 2022
Explainable fNIRS Based Pain Decoding Under Pharmacological Conditions 
via Deep Transfer Learning Approach 

Aykut Eken, Sinem Burcu Erdoğan, Murat Yüce, Gülnaz Yükselen

@author: Aykut Eken, PhD, TOBB University of Economics and Technology
, e-mail: aykuteken@etu.edu.tr
"""
## Dataset that was previously collected by 
# Peng K, Yücel MA, Steele SC, Bittner EA, Aasted CM, 
# Hoeft MA, Lee A, George EE, Boas DA, Becerra L and Borsook D (2018) 
# Morphine Attenuates fNIRS Signal Associated With Painful Stimuli in the Medial 
# Frontopolar Cortex (medial BA 10). Front. Hum. Neurosci. 12:394. doi: 10.3389/fnhum.2018.00394
import pandas as pd
import scipy as sp
from scipy import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from numpy import random
import os 
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Flatten, Conv1D, MaxPooling1D, Dropout
from keras import Sequential
import matplotlib.pyplot as plt
import shap
import pingouin as pg
import keras.backend as K


plt.rcParams['lines.linewidth'] = 20
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['figure.figsize'] = [64,32]
plt.rcParams['axes.labelsize']=80
plt.rcParams['axes.labelweight']='bold'

plt.rcParams['font.size']=80
plt.rcParams['font.weight']='bold'
plt.rcParams['figure.dpi']=100
plt.rcParams["legend.loc"] = 'right' 

tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()
# First, import the variables 

dirname = '/Users/aykuteken/Documents/MATLAB'
fname = os.path.join(dirname, 'Feature_fNIRS_TF.mat')
FVs = io.loadmat(fname,mat_dtype=True,matlab_compatible=True)

Morphine_HbO_pre_vas3=FVs['Morphine_HbO_pre']['vas3'][0][0]
Morphine_HbO_pre_vas7=FVs['Morphine_HbO_pre']['vas7'][0][0]
Morphine_Hb_pre_vas3=FVs['Morphine_Hb_pre']['vas3'][0][0]
Morphine_Hb_pre_vas7=FVs['Morphine_Hb_pre']['vas7'][0][0]
Morphine_pre_subj_vas3=FVs['Morphine_pre']['vas3_subj'][0][0]
Morphine_pre_subj_vas7=FVs['Morphine_pre']['vas7_subj'][0][0]


Placebo_HbO_pre_vas3=FVs['Placebo_HbO_pre']['vas3'][0][0]
Placebo_HbO_pre_vas7=FVs['Placebo_HbO_pre']['vas7'][0][0]
Placebo_Hb_pre_vas3=FVs['Placebo_Hb_pre']['vas3'][0][0]
Placebo_Hb_pre_vas7=FVs['Placebo_Hb_pre']['vas7'][0][0]
Placebo_pre_subj_vas3=FVs['Placebo_pre']['vas3_subj'][0][0]
Placebo_pre_subj_vas7=FVs['Placebo_pre']['vas7_subj'][0][0]

Morphine_HbO_post30_vas3=FVs['Morphine_HbO_post30']['vas3'][0][0]
Morphine_HbO_post30_vas7=FVs['Morphine_HbO_post30']['vas7'][0][0]
Morphine_Hb_post30_vas3=FVs['Morphine_Hb_post30']['vas3'][0][0]
Morphine_Hb_post30_vas7=FVs['Morphine_Hb_post30']['vas7'][0][0]
Morphine_post30_subj_vas3=FVs['Morphine_post30']['vas3_subj'][0][0]
Morphine_post30_subj_vas7=FVs['Morphine_post30']['vas3_subj'][0][0]

Placebo_HbO_post30_vas3=FVs['Placebo_HbO_post30']['vas3'][0][0]
Placebo_HbO_post30_vas7=FVs['Placebo_HbO_post30']['vas7'][0][0]
Placebo_Hb_post30_vas3=FVs['Placebo_Hb_post30']['vas3'][0][0]
Placebo_Hb_post30_vas7=FVs['Placebo_Hb_post30']['vas7'][0][0]
Placebo_post30_subj_vas3=FVs['Placebo_post30']['vas3_subj'][0][0]
Placebo_post30_subj_vas7=FVs['Placebo_post30']['vas7_subj'][0][0]

Morphine_HbO_post60_vas3=FVs['Morphine_HbO_post60']['vas3'][0][0]
Morphine_HbO_post60_vas7=FVs['Morphine_HbO_post60']['vas7'][0][0]
Morphine_Hb_post60_vas3=FVs['Morphine_Hb_post60']['vas3'][0][0]
Morphine_Hb_post60_vas7=FVs['Morphine_Hb_post60']['vas7'][0][0]
Morphine_post60_subj_vas3=FVs['Morphine_post60']['vas3_subj'][0][0]
Morphine_post60_subj_vas7=FVs['Morphine_post60']['vas7_subj'][0][0]

Placebo_HbO_post60_vas3=FVs['Placebo_HbO_post60']['vas3'][0][0]
Placebo_HbO_post60_vas7=FVs['Placebo_HbO_post60']['vas7'][0][0]
Placebo_Hb_post60_vas3=FVs['Placebo_Hb_post60']['vas3'][0][0]
Placebo_Hb_post60_vas7=FVs['Placebo_Hb_post60']['vas7'][0][0]
Placebo_post60_subj_vas3=FVs['Placebo_post60']['vas3_subj'][0][0]
Placebo_post60_subj_vas7=FVs['Placebo_post60']['vas7_subj'][0][0]

Morphine_HbO_post90_vas3=FVs['Morphine_HbO_post90']['vas3'][0][0]
Morphine_HbO_post90_vas7=FVs['Morphine_HbO_post90']['vas7'][0][0]
Morphine_Hb_post90_vas3=FVs['Morphine_Hb_post90']['vas3'][0][0]
Morphine_Hb_post90_vas7=FVs['Morphine_Hb_post90']['vas7'][0][0]
Morphine_post90_subj_vas3=FVs['Morphine_post90']['vas3_subj'][0][0]
Morphine_post90_subj_vas7=FVs['Morphine_post90']['vas7_subj'][0][0]

Placebo_HbO_post90_vas3=FVs['Placebo_HbO_post90']['vas3'][0][0]
Placebo_HbO_post90_vas7=FVs['Placebo_HbO_post90']['vas7'][0][0]
Placebo_Hb_post90_vas3=FVs['Placebo_Hb_post90']['vas3'][0][0]
Placebo_Hb_post90_vas7=FVs['Placebo_Hb_post90']['vas7'][0][0]
Placebo_post90_subj_vas3=FVs['Placebo_post90']['vas3_subj'][0][0]
Placebo_post90_subj_vas7=FVs['Placebo_post90']['vas7_subj'][0][0]




## We will try to classify the painful condition in sessions seperately. 
## For instance, model knowledge obtained from pre-Drug data will be used for post Drug sessions
## separately (30min, 60 min, 90 min).

## Due to the not having enough data, data augmentation will be performed. 

#### ------ DATA AUGMENTATION -----######



def data_split_aug(pre_data,stim_type,fold_type, norm):
    
    # Z-score normalization
    test_data = []
    val_data=[]
    test_label=[]
    val_label =[]

    if norm==1:
        
        for i in range(0,len(pre_data)):
            
            pre_data[i,:,:]=sp.stats.zscore(pre_data[i,:,:],axis=1)
                
    if fold_type=='holdout':
    
        train_data,test_data,train_label, test_label = train_test_split(pre_data,stim_type, test_size=0.2, random_state=None, shuffle=True, stratify=stim_type)
        train_data, val_data, train_label, val_label = train_test_split(train_data,train_label, test_size=0.25, random_state=None, shuffle=True,stratify=train_label)
        
    elif fold_type=='kfold':
        
        train_data = pre_data
        train_label = stim_type
        
    test_data = test_data.reshape(np.size(test_data,0),np.size(test_data,2),np.size(test_data,1))
    train_data = train_data.reshape(np.size(train_data,0),np.size(train_data,2),np.size(train_data,1))
    val_data = val_data.reshape(np.size(val_data,0),np.size(val_data,2),np.size(val_data,1))
        
    s = np.shape(train_data)

    opt =[0,2] # 0 is injecting Gaussian noise, #1 is adding spike #2 is adding trend
    
    aug_data = []
    aug_labels = []
    for j in range(0,25):
        for i in range(0,len(train_label)):
            
            sel =random.choice(opt)
            
            if sel==0:
            
                sigma = [0.01, 0.05, 0.1]
                mu= 0    
        
                ss=random.choice(sigma)
        
                noise = np.random.normal(mu, ss, (1,s[1],s[2]))
        
                new_created_data=train_data[i,:,:]+noise
            
                aug_data.append(np.squeeze(new_created_data))
            
                aug_labels.append(train_label[i])
                
            elif sel==1:
                
                
                std_data=np.std(train_data[i,:,:],axis=0)
                std_data=std_data*1.5
                
                ind = np.arange(0,24)
                
                sel_ind=random.choice(ind)
                
                direct=[0,1]
                
                sel_direct = random.choice(direct)
                
                if sel_direct ==0:
                    
                    train_data[i,:,sel_ind]=-1*std_data[sel_ind]
                    
                else:
                    
                    train_data[i,:,sel_ind]=std_data[sel_ind]
                
                new_created_data=train_data[i,:,:]
            
                aug_data.append(np.squeeze(new_created_data))
            
                aug_labels.append(train_label[i])
                
                
            elif sel==2:
                
                m = [0.01, 0.05, 0.1]
                
                x=np.arange(0,24)
                
                x = np.tile(x,(31,1))
                
                
                trend=random.choice(m)*x
                
                new_created_data=train_data[i,:,:]+trend
            
                aug_data.append(np.squeeze(new_created_data))
            
                aug_labels.append(train_label[i])
                
                
                

    aug_train_data = np.stack(aug_data,axis=0)
    aug_train_label = np.stack(aug_labels,axis=0)
    aug_train_label = np.array(aug_train_label,dtype=int)
    
    
    return aug_train_data, test_data, val_data, aug_train_label, test_label, val_label

def Pre_Drug_Model(train_data,val_data,train_label, val_label):
    
    train_label = np.asarray(train_label).astype('float32').reshape((-1,1))
    val_label = np.asarray(val_label).astype('float32').reshape((-1,1))
    
    callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_pre_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.01, patience=10, min_lr=1e-6
    ),
    #keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=0),
    ]
    drp=0.4
    # design network
    model = Sequential()
    model.add(Conv1D(32, 2, activation='relu', input_shape=(31, 24)))
    model.add(MaxPooling1D(2)) 
    model.add(Dropout(drp))# depending on the overfitting status I can add dropout.
    model.add(Conv1D(64, 2, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(drp))
    model.add(Conv1D(128, 2, activation='relu',name='grad_conv_layer'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(drp))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(drp))
    model.add(Dense(1, activation='sigmoid'))

    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="binary_crossentropy",
              metrics=["binary_accuracy"])
    # fit network
    history = model.fit(train_data, train_label, epochs=100, batch_size=16, callbacks=callbacks, validation_data=(val_data, val_label), verbose=0, shuffle=True)
    # plot history
    
    return model,history


def SensAndSpe(y_true,y_pred):
    
    fpr = np.linspace(0, 1, 100)
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    sens = true_positives / (possible_positives + K.epsilon())
    spe = true_negatives / (possible_negatives + K.epsilon())
    
    sens=sens.numpy()
    spe=spe.numpy()
    
    fprs, tprs,_ = roc_curve(y_true, y_pred, pos_label=1)
    auc = roc_auc_score (y_true, y_pred)
    
    tpr = np.interp(fpr,fprs,tprs)
    
    return sens, spe, fpr, tpr, auc

def PostDrug_HoldOutModel(model,train_data,val_data, train_label, val_label,data_type, exp_time):
    
    train_label = np.asarray(train_label).astype('float32').reshape((-1,1))
    val_label = np.asarray(val_label).astype('float32').reshape((-1,1))

    # get the base model and don't train it

    model.trainable=False
    print('Before TF')
    print(model.summary())
    model = tf.keras.models.Sequential(model.layers[:-4])

    inputs = keras.Input(shape=(31,24))
    x=model(inputs, training=False)
    x=keras.layers.Flatten()(x)
    x=keras.layers.Dense(256,activation='relu')(x)
    x=keras.layers.Dropout(0.4)(x)
    outputs=keras.layers.Dense(1,activation='sigmoid')(x)
    model = keras.Model(inputs,outputs)

    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))
    print('After TF')
    print(model.summary())
            
    callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_post_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
         monitor="val_loss", factor=0.1, patience=10, min_lr=1e-6
    ),
    #keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=0),
    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="binary_crossentropy",
              metrics=["binary_accuracy"])
    
    history = model.fit(train_data, train_label, epochs=100, batch_size=16, callbacks=callbacks, validation_data=(val_data, val_label), verbose=0, shuffle=True)

    # plot history

    fname = 'Post_Drug_HoldOut_'+data_type+exp_time+'.h5'
    model.save(fname)
        
    return model,history

n=30
pre_hist_tr =[]
pre_hist_val =[]
pre_acc=[]
pre_sens=[]
pre_spe=[]
pre_tpr=[]
pre_fpr=[]
pre_auc=[]

post30_morph_hist_tr=[]
post30_morph_hist_val=[]
post30_morph_acc =[]
post30_morph_sens=[]
post30_morph_spe=[]
post30_morph_tpr=[]
post30_morph_fpr=[]
post30_morph_auc=[]

post60_morph_hist_tr=[]
post60_morph_hist_val=[]
post60_morph_acc =[]
post60_morph_sens=[]
post60_morph_spe=[]
post60_morph_tpr=[]
post60_morph_fpr=[]
post60_morph_auc=[]

post90_morph_hist_tr=[]
post90_morph_hist_val=[]
post90_morph_acc =[]
post90_morph_sens=[]
post90_morph_spe=[]
post90_morph_tpr=[]
post90_morph_fpr=[]
post90_morph_auc=[]

post30_place_hist_tr=[]
post30_place_hist_val=[]
post30_place_acc =[]
post30_place_sens=[]
post30_place_spe=[]
post30_place_tpr=[]
post30_place_fpr=[]
post30_place_auc=[]

post60_place_hist_tr=[]
post60_place_hist_val=[]
post60_place_acc =[]
post60_place_sens=[]
post60_place_spe=[]
post60_place_tpr=[]
post60_place_fpr=[]
post60_place_auc=[]

post90_place_hist_tr=[]
post90_place_hist_val=[]
post90_place_acc =[]
post90_place_sens=[]
post90_place_spe=[]
post90_place_tpr=[]
post90_place_fpr=[]
post90_place_auc=[]

list_shap_values_pre=[]
list_test_sets_pre=[]
list_shap_values_post30morphine=[]
list_test_sets_post30morphine=[]
list_shap_values_post30placebo=[]
list_test_sets_post30placebo=[]
list_shap_values_post60morphine=[]
list_test_sets_post60morphine=[]
list_shap_values_post60placebo=[]
list_test_sets_post60placebo=[]
list_shap_values_post90morphine=[]
list_test_sets_post90morphine=[]
list_shap_values_post90placebo=[]
list_test_sets_post90placebo=[]
explainer_pre =[]
explainer_post30morphine =[]
explainer_post30placebo =[]
explainer_post60morphine =[]
explainer_post60placebo =[]
explainer_post90morphine =[]
explainer_post90placebo=[]

shap_test_data_pre=[]
shap_test_data_post30morphine=[]
shap_test_data_post30placebo=[]
shap_test_data_post60morphine=[]
shap_test_data_post60placebo=[]
shap_test_data_post90morphine=[]
shap_test_data_post90placebo=[]


for i in range(0,n):
    
    print('##############------------ITERATION '+str(i) +' ---------------##############')
    
    pre_data=np.concatenate([Morphine_HbO_pre_vas3,Placebo_HbO_pre_vas3,Morphine_HbO_pre_vas7,Placebo_HbO_pre_vas7])
    ## vas3 labeled as 0, vas7 labeled as 1
    stim_type = np.concatenate([np.zeros(np.shape(Morphine_HbO_pre_vas3)[0]), np.zeros(np.shape(Placebo_HbO_pre_vas3)[0]),
                           np.ones(np.shape(Morphine_HbO_pre_vas7)[0]), np.ones(np.shape(Placebo_HbO_pre_vas7)[0])])

    aug_train_pre_data, test_pre_data, val_pre_data, aug_train_pre_label, test_pre_label, val_pre_label=data_split_aug(pre_data, 
                                                                                      stim_type,'holdout',1)

    ##-------- Model Development-------- ####


    Pre_Model,pre_model_history = Pre_Drug_Model(aug_train_pre_data, val_pre_data, aug_train_pre_label, val_pre_label)
    
    pre_hist_tr.append(pre_model_history.history['binary_accuracy'])
    pre_hist_val.append(pre_model_history.history['val_binary_accuracy'])

    print('\n')
    print('######----Pre Drug Model------########--')
    print('Training accuracy: '+str(pre_model_history.history['binary_accuracy'][-1]))
    print('Validation accuracy: '+str(pre_model_history.history['val_binary_accuracy'][-1]))
    print('Test accuracy: '+str(Pre_Model.evaluate(test_pre_data, test_pre_label)))
    test_pre_label = np.asarray(test_pre_label).astype('float32').reshape((-1,1))
    y_pred = np.round(Pre_Model.predict(test_pre_data))
    y_true = test_pre_label
    sens,spe,fpr,tpr,auc=SensAndSpe(y_true,y_pred)
    pre_sens.append(sens)
    pre_spe.append(spe)
    pre_tpr.append(tpr)
    pre_fpr.append(fpr)
    pre_auc.append(auc)
    pre_model_Acc = Pre_Model.evaluate(test_pre_data, test_pre_label)
    pre_acc.append(pre_model_Acc)
    
    ## SHAP ##
    explainer=[]
    sel_aug_train_pre_data = aug_train_pre_data[np.random.choice(aug_train_pre_data.shape[0], 1000, replace=False)]
    explainer=shap.DeepExplainer(Pre_Model,sel_aug_train_pre_data)
    sel_test_pre_data = test_pre_data[np.random.choice(test_pre_data.shape[0], 30, replace=False)]
    shap_values = explainer.shap_values(sel_test_pre_data)
    list_shap_values_pre.append(shap_values)
    list_test_sets_pre.append(test_pre_label)
    explainer_pre.append(explainer)
    shap_test_data_pre.append(test_pre_data)
    
    print('\n')
    ## Pre-trained model using Pre-Drug data
    ## Now we transfer the knowledge to a base_model
    ## But first create the data for post 30 min

    post30_data_morphine=np.concatenate([Morphine_HbO_post30_vas3,Morphine_HbO_post30_vas7])
    stim_type_morphine = np.concatenate([np.zeros(np.shape(Morphine_HbO_post30_vas3)[0]),np.ones(np.shape(Morphine_HbO_post30_vas7)[0])])

    post30_data_placebo = np.concatenate([Placebo_HbO_post30_vas3,Placebo_HbO_post30_vas7])
    stim_type_placebo = np.concatenate([np.zeros(np.shape(Placebo_HbO_post30_vas3)[0]),np.ones(np.shape(Placebo_HbO_post30_vas7)[0])])


    aug_train_morphine_post30_data, test_morphine_post30_data, val_morphine_post30_data, aug_train_morphine_post30_label, test_morphine_post30_label, val_morphine_post30_label=data_split_aug(post30_data_morphine, 
                                                                                      stim_type_morphine,'holdout',1)
    
    aug_train_placebo_post30_data, test_placebo_post30_data, val_placebo_post30_data, aug_train_placebo_post30_label, test_placebo_post30_label, val_placebo_post30_label=data_split_aug(post30_data_placebo, 
                                                                                      stim_type_placebo,'holdout',1)


    post30_Model_Serial_Morphine, post30_History_Serial_Morphine= PostDrug_HoldOutModel(Pre_Model,aug_train_morphine_post30_data, val_morphine_post30_data, aug_train_morphine_post30_label, val_morphine_post30_label,'Morphine','30')
    post30_Model_Serial_Placebo,post30_History_Serial_Placebo = PostDrug_HoldOutModel(Pre_Model,aug_train_placebo_post30_data, val_placebo_post30_data, aug_train_placebo_post30_label, val_placebo_post30_label,'Placebo','30')
    
    post30_morph_hist_tr.append(post30_History_Serial_Morphine.history['binary_accuracy'])
    post30_morph_hist_val.append(post30_History_Serial_Morphine.history['val_binary_accuracy'])
    post30_place_hist_tr.append(post30_History_Serial_Placebo.history['binary_accuracy'])
    post30_place_hist_val.append(post30_History_Serial_Placebo.history['val_binary_accuracy'])  
    
    print('\n')
    print('######----Morphine Post 30 Min TF Network------########')
    print('Training accuracy: '+str(post30_History_Serial_Morphine.history['binary_accuracy'][-1]))
    print('Validation accuracy: '+str(post30_History_Serial_Morphine.history['val_binary_accuracy'][-1]))
    print('Test accuracy: '+str(post30_Model_Serial_Morphine.evaluate(test_morphine_post30_data, test_morphine_post30_label)[1]))
    test_morphine_post30_label = np.asarray(test_morphine_post30_label).astype('float32').reshape((-1,1))
    post30_Serial_Morph_Acc=post30_Model_Serial_Morphine.evaluate(test_morphine_post30_data, test_morphine_post30_label)[1]
    
    y_pred = np.round(post30_Model_Serial_Morphine.predict(test_morphine_post30_data))
    y_true = test_morphine_post30_label
    sens,spe,fpr,tpr,auc=SensAndSpe(y_true,y_pred)
    post30_morph_sens.append(sens)
    post30_morph_spe.append(spe)
    post30_morph_tpr.append(tpr)
    post30_morph_fpr.append(fpr)
    post30_morph_auc.append(auc)
    
    print('\n')
    print('######----Placebo Post 30 Min TF Network------########')
    print('Training accuracy: '+str(post30_History_Serial_Placebo.history['binary_accuracy'][-1]))
    print('Validation accuracy: '+str(post30_History_Serial_Placebo.history['val_binary_accuracy'][-1]))
    print('Test accuracy:' +str(post30_Model_Serial_Placebo.evaluate(test_placebo_post30_data, test_placebo_post30_label)[1]))
    test_placebo_post30_label = np.asarray(test_placebo_post30_label).astype('float32').reshape((-1,1))
    post30_Serial_Placebo_Acc=post30_Model_Serial_Placebo.evaluate(test_placebo_post30_data, test_placebo_post30_label)[1]
    
    y_pred = np.round(post30_Model_Serial_Placebo.predict(test_placebo_post30_data))
    y_true = test_placebo_post30_label
    sens,spe,fpr,tpr,auc=SensAndSpe(y_true,y_pred)
    post30_place_sens.append(sens)
    post30_place_spe.append(spe)
    post30_place_tpr.append(tpr)
    post30_place_fpr.append(fpr)
    post30_place_auc.append(auc)
    print('\n')
    
    post30_morph_acc.append(post30_Serial_Morph_Acc)
    post30_place_acc.append(post30_Serial_Placebo_Acc)
    
    ## SHAP post 30 morphine ##
    explainer=[]
    sel_aug_train_morphine_post30_data = aug_train_morphine_post30_data[np.random.choice(aug_train_morphine_post30_data.shape[0], 1000, replace=False)]
    explainer=shap.DeepExplainer(post30_Model_Serial_Morphine,sel_aug_train_morphine_post30_data)
    sel_test_morphine_post30_data = test_morphine_post30_data[np.random.choice(test_morphine_post30_data.shape[0], 15, replace=False)]
    shap_values = explainer.shap_values(sel_test_morphine_post30_data)
    
    list_shap_values_post30morphine.append(shap_values)
    list_test_sets_post30morphine.append(test_morphine_post30_label)
    explainer_post30morphine.append(explainer)
    shap_test_data_post30morphine.append(test_morphine_post30_data)
    
    ## SHAP post 30 placebo ##
    explainer=[]
    
    sel_aug_train_placebo_post30_data = aug_train_placebo_post30_data[np.random.choice(aug_train_placebo_post30_data.shape[0], 1000, replace=False)]
    explainer=shap.DeepExplainer(post30_Model_Serial_Placebo,sel_aug_train_placebo_post30_data)
    sel_test_placebo_post30_data = test_placebo_post30_data[np.random.choice(test_placebo_post30_data.shape[0], 15, replace=False)]
    shap_values = explainer.shap_values(sel_test_placebo_post30_data)
    

    list_shap_values_post30placebo.append(shap_values)
    list_test_sets_post30placebo.append(test_placebo_post30_label)
    explainer_post30placebo.append(explainer)
    shap_test_data_post30placebo.append(test_placebo_post30_data)


    ## Post 60 Morphine and Placebo Using Pre Model
    
    post60_data_morphine=np.concatenate([Morphine_HbO_post60_vas3,Morphine_HbO_post60_vas7])
    stim_type_morphine = np.concatenate([np.zeros(np.shape(Morphine_HbO_post60_vas3)[0]),np.ones(np.shape(Morphine_HbO_post60_vas7)[0])])
    
    post60_data_placebo = np.concatenate([Placebo_HbO_post60_vas3,Placebo_HbO_post60_vas7])
    stim_type_placebo = np.concatenate([np.zeros(np.shape(Placebo_HbO_post60_vas3)[0]),np.ones(np.shape(Placebo_HbO_post60_vas7)[0])])

    aug_train_morphine_post60_data, test_morphine_post60_data, val_morphine_post60_data, aug_train_morphine_post60_label, test_morphine_post60_label, val_morphine_post60_label=data_split_aug(post60_data_morphine, 
                                                                                      stim_type_morphine,'holdout',1)

    aug_train_placebo_post60_data, test_placebo_post60_data, val_placebo_post60_data, aug_train_placebo_post60_label, test_placebo_post60_label, val_placebo_post60_label=data_split_aug(post60_data_placebo, 
                                                                                      stim_type_placebo,'holdout',1)


    post60_Model_Serial_Morphine, post60_History_Serial_Morphine = PostDrug_HoldOutModel(Pre_Model,aug_train_morphine_post60_data, val_morphine_post60_data, aug_train_morphine_post60_label, val_morphine_post60_label,'Morphine','60')
    post60_Model_Serial_Placebo, post60_History_Serial_Placebo = PostDrug_HoldOutModel(Pre_Model,aug_train_placebo_post60_data, val_placebo_post60_data, aug_train_placebo_post60_label, val_placebo_post60_label,'Placebo','60')

    post60_morph_hist_tr.append(post60_History_Serial_Morphine.history['binary_accuracy'])
    post60_morph_hist_val.append(post60_History_Serial_Morphine.history['val_binary_accuracy'])
    post60_place_hist_tr.append(post60_History_Serial_Placebo.history['binary_accuracy'])
    post60_place_hist_val.append(post60_History_Serial_Placebo.history['val_binary_accuracy'])    

    print('\n')
    print('######----Morphine Post 60 Min TF Network------########')
    print('Training accuracy: '+str(post60_History_Serial_Morphine.history['binary_accuracy'][-1]))
    print('Validation accuracy: '+str(post60_History_Serial_Morphine.history['val_binary_accuracy'][-1]))
    print('Test accuracy: '+str(post60_Model_Serial_Morphine.evaluate(test_morphine_post60_data, test_morphine_post60_label)[1]))
    test_morphine_post60_label = np.asarray(test_morphine_post60_label).astype('float32').reshape((-1,1))
    post60_Serial_Morph_Acc=post60_Model_Serial_Morphine.evaluate(test_morphine_post60_data, test_morphine_post60_label)[1]
    y_pred = np.round(post60_Model_Serial_Morphine.predict(test_morphine_post60_data))
    y_true = test_morphine_post60_label
    sens,spe,fpr,tpr,auc=SensAndSpe(y_true,y_pred)
    post60_morph_sens.append(sens)
    post60_morph_spe.append(spe)
    post60_morph_tpr.append(tpr)
    post60_morph_fpr.append(fpr)
    post60_morph_auc.append(auc)
    print('\n')
    print('######----Placebo Post 60 Min TF Network------########')
    print('Training accuracy: '+str(post60_History_Serial_Placebo.history['binary_accuracy'][-1]))
    print('Validation accuracy: '+str(post60_History_Serial_Placebo.history['val_binary_accuracy'][-1]))
    print('Test accuracy:' +str(post60_Model_Serial_Placebo.evaluate(test_placebo_post60_data, test_placebo_post60_label)[1]))
    test_placebo_post60_label = np.asarray(test_placebo_post60_label).astype('float32').reshape((-1,1))
    post60_Serial_Placebo_Acc=post60_Model_Serial_Placebo.evaluate(test_placebo_post60_data, test_placebo_post60_label)[1]
    y_pred = np.round(post60_Model_Serial_Placebo.predict(test_placebo_post60_data))
    y_true = test_placebo_post60_label
    sens,spe,fpr,tpr,auc=SensAndSpe(y_true,y_pred)
    post60_place_sens.append(sens)
    post60_place_spe.append(spe)
    post60_place_tpr.append(tpr)
    post60_place_fpr.append(fpr)
    post60_place_auc.append(auc)
    print('\n')
    ## Post 60 Morphine and Placebo Pre Model
    post60_morph_acc.append(post60_Serial_Morph_Acc)
    post60_place_acc.append(post60_Serial_Placebo_Acc)
    
    ## SHAP post 60 morphine ##
    explainer=[]
    sel_aug_train_morphine_post60_data = aug_train_morphine_post60_data[np.random.choice(aug_train_morphine_post60_data.shape[0], 1000, replace=False)]
    explainer=shap.DeepExplainer(post60_Model_Serial_Morphine,sel_aug_train_morphine_post60_data)
    sel_test_morphine_post60_data = test_morphine_post60_data[np.random.choice(test_morphine_post60_data.shape[0], 15, replace=False)]
    shap_values = explainer.shap_values(sel_test_morphine_post60_data)

    list_shap_values_post60morphine.append(shap_values)
    list_test_sets_post60morphine.append(test_morphine_post60_label)
    explainer_post60morphine.append(explainer)
    shap_test_data_post60morphine.append(test_morphine_post60_data)
    
    ## SHAP post 60 placebo ##
    explainer=[]
    sel_aug_train_placebo_post60_data = aug_train_placebo_post60_data[np.random.choice(aug_train_placebo_post60_data.shape[0], 1000, replace=False)]
    explainer=shap.DeepExplainer(post60_Model_Serial_Placebo,sel_aug_train_placebo_post60_data)
    sel_test_placebo_post60_data = test_placebo_post60_data[np.random.choice(test_placebo_post60_data.shape[0], 15, replace=False)]
    shap_values = explainer.shap_values(sel_test_placebo_post60_data)
    
    list_shap_values_post60placebo.append(shap_values)
    list_test_sets_post60placebo.append(test_placebo_post60_label)
    explainer_post60placebo.append(explainer)
    shap_test_data_post60placebo.append(test_placebo_post60_data)
    

    post90_data_morphine=np.concatenate([Morphine_HbO_post90_vas3,Morphine_HbO_post90_vas7])
    stim_type_morphine = np.concatenate([np.zeros(np.shape(Morphine_HbO_post90_vas3)[0]),np.ones(np.shape(Morphine_HbO_post90_vas7)[0])])

    post90_data_placebo = np.concatenate([Placebo_HbO_post90_vas3,Placebo_HbO_post90_vas7])
    stim_type_placebo = np.concatenate([np.zeros(np.shape(Placebo_HbO_post90_vas3)[0]),np.ones(np.shape(Placebo_HbO_post90_vas7)[0])])

    aug_train_morphine_post90_data, test_morphine_post90_data, val_morphine_post90_data, aug_train_morphine_post90_label, test_morphine_post90_label, val_morphine_post90_label=data_split_aug(post90_data_morphine, 
                                                                                      stim_type_morphine,'holdout',1)

    aug_train_placebo_post90_data, test_placebo_post90_data, val_placebo_post90_data, aug_train_placebo_post90_label, test_placebo_post90_label, val_placebo_post90_label=data_split_aug(post90_data_placebo, 
                                                                                      stim_type_placebo,'holdout',1)


    post90_Model_Serial_Morphine,post90_History_Serial_Morphine  = PostDrug_HoldOutModel(Pre_Model,aug_train_morphine_post90_data, val_morphine_post90_data, aug_train_morphine_post90_label, val_morphine_post90_label,'Morphine','90')
    post90_Model_Serial_Placebo,post90_History_Serial_Placebo = PostDrug_HoldOutModel(Pre_Model,aug_train_placebo_post90_data, val_placebo_post90_data, aug_train_placebo_post90_label, val_placebo_post90_label,'Placebo','90')

    print('\n')
    print('######----Morphine Post 90 Min TF Network------########')
    print('Training accuracy: '+str(post90_History_Serial_Morphine.history['binary_accuracy'][-1]))
    print('Validation accuracy: '+str(post90_History_Serial_Morphine.history['val_binary_accuracy'][-1]))
    print('Test accuracy: '+str(post90_Model_Serial_Morphine.evaluate(test_morphine_post90_data, test_morphine_post90_label)[1]))
    test_morphine_post90_label = np.asarray(test_morphine_post90_label).astype('float32').reshape((-1,1))
    post90_Serial_Morph_Acc=post90_Model_Serial_Morphine.evaluate(test_morphine_post90_data, test_morphine_post90_label)[1]
    y_pred = np.round(post90_Model_Serial_Morphine.predict(test_morphine_post90_data))
    y_true = test_morphine_post90_label
    sens,spe,fpr,tpr,auc=SensAndSpe(y_true,y_pred)
    post90_morph_sens.append(sens)
    post90_morph_spe.append(spe)
    post90_morph_tpr.append(tpr)
    post90_morph_fpr.append(fpr)
    post90_morph_auc.append(auc)
    print('\n')
    print('######----Placebo Post 90 Min TF Network------########')
    print('Training accuracy: '+str(post90_History_Serial_Placebo.history['binary_accuracy'][-1]))
    print('Validation accuracy: '+str(post90_History_Serial_Placebo.history['val_binary_accuracy'][-1]))
    print('Test accuracy:' +str(post90_Model_Serial_Placebo.evaluate(test_placebo_post90_data, test_placebo_post90_label)[1]))
    test_placebo_post90_label = np.asarray(test_placebo_post90_label).astype('float32').reshape((-1,1))
    post90_Serial_Placebo_Acc=post90_Model_Serial_Placebo.evaluate(test_placebo_post90_data, test_placebo_post90_label)[1]
    y_pred = np.round(post90_Model_Serial_Placebo.predict(test_placebo_post90_data))
    y_true = test_placebo_post90_label
    sens,spe,fpr,tpr,auc=SensAndSpe(y_true,y_pred)
    post90_place_sens.append(sens)
    post90_place_spe.append(spe)
    post90_place_tpr.append(tpr)
    post90_place_fpr.append(fpr)
    post90_place_auc.append(auc)
    print('\n')
    
    post90_morph_acc.append(post90_Serial_Morph_Acc)
    post90_place_acc.append(post90_Serial_Placebo_Acc)
    
    ## SHAP post 90 morphine ##
    
    explainer=[]
    sel_aug_train_morphine_post90_data = aug_train_morphine_post90_data[np.random.choice(aug_train_morphine_post90_data.shape[0], 1000, replace=False)]
    explainer=shap.DeepExplainer(post90_Model_Serial_Morphine,sel_aug_train_morphine_post90_data)
    sel_test_morphine_post90_data = test_morphine_post90_data[np.random.choice(test_morphine_post90_data.shape[0], 15, replace=False)]
    shap_values = explainer.shap_values(sel_test_morphine_post90_data)
    
    list_shap_values_post90morphine.append(shap_values)
    list_test_sets_post90morphine.append(test_morphine_post90_label)
    explainer_post90morphine.append(explainer)
    shap_test_data_post90morphine.append(test_morphine_post90_data)
    
    
    ## SHAP post 90 placebo ##
    
    explainer=[]
    sel_aug_train_placebo_post90_data = aug_train_placebo_post90_data[np.random.choice(aug_train_placebo_post90_data.shape[0], 1000, replace=False)]
    explainer=shap.DeepExplainer(post90_Model_Serial_Placebo,sel_aug_train_placebo_post90_data)
    sel_test_placebo_post90_data = test_placebo_post90_data[np.random.choice(test_placebo_post90_data.shape[0], 15, replace=False)]
    shap_values = explainer.shap_values(sel_test_placebo_post90_data)
    
    list_shap_values_post90placebo.append(shap_values)
    list_test_sets_post90placebo.append(test_placebo_post90_label)
    explainer_post90placebo.append(explainer)
    shap_test_data_post90placebo.append(test_placebo_post90_data)
    
    post90_morph_hist_tr.append(post30_History_Serial_Morphine.history['binary_accuracy'])
    post90_morph_hist_val.append(post30_History_Serial_Morphine.history['val_binary_accuracy'])
    post90_place_hist_tr.append(post30_History_Serial_Placebo.history['binary_accuracy'])
    post90_place_hist_val.append(post30_History_Serial_Placebo.history['val_binary_accuracy'])

pre_acc = np.array(pre_acc)[:,1]
pre_mean_tr=np.mean(np.vstack(pre_hist_tr),axis=0)
pre_std_tr=np.std(np.vstack(pre_hist_tr),axis=0)##--##
pre_mean_val=np.mean(np.vstack(pre_hist_val),axis=0)
pre_std_val=np.std(np.vstack(pre_hist_val),axis=0)##--##

epc = np.arange(0,len(pre_mean_tr))

plt.title('Pre',fontweight='bold')
plt.plot(epc, pre_mean_tr, label='train')
plt.fill_between(epc,pre_mean_tr-pre_std_tr,pre_mean_tr+pre_std_tr,alpha=.2)
plt.plot(epc, pre_mean_val, label='validation')
plt.fill_between(epc, pre_mean_val-pre_std_val,pre_mean_val+pre_std_val,alpha=.2)
plt.xlabel('# of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#### Post 30 Placebo ####

post30_placebo_mean_tr=np.mean(np.vstack(post30_place_hist_tr),axis=0)
post30_placebo_std_tr=np.std(np.vstack(post30_place_hist_tr),axis=0)##--##
post30_placebo_mean_val=np.mean(np.vstack(post30_place_hist_val),axis=0)
post30_placebo_std_val=np.std(np.vstack(post30_place_hist_val),axis=0)##--##

epc = np.arange(0,len(post30_placebo_mean_tr))

plt.title('Post 30 Placebo',fontweight='bold')
plt.plot(epc, post30_placebo_mean_tr, label='train')
plt.fill_between(epc,post30_placebo_mean_tr-post30_placebo_std_tr,post30_placebo_mean_tr+post30_placebo_std_tr,alpha=.2)
plt.plot(epc, post30_placebo_mean_val, label='validation') 
plt.fill_between(epc,post30_placebo_mean_val-post30_placebo_std_val,post30_placebo_mean_val+post30_placebo_std_val,alpha=.2)
plt.xlabel('# of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#### Post 30 Morphine ####

post30_morphine_mean_tr=np.mean(np.vstack(post30_morph_hist_tr),axis=0)
post30_morphine_std_tr=np.std(np.vstack(post30_morph_hist_tr),axis=0)##--##
post30_morphine_mean_val=np.mean(np.vstack(post30_morph_hist_val),axis=0)
post30_morphine_std_val=np.std(np.vstack(post30_morph_hist_val),axis=0)##--##

epc = np.arange(0,len(post30_morphine_mean_tr))

plt.title('Post 30 Morphine',fontweight='bold')
plt.plot(epc, post30_morphine_mean_tr, label='train')
plt.fill_between(epc,post30_morphine_mean_tr-post30_morphine_std_tr,post30_morphine_mean_tr+post30_morphine_std_tr,alpha=.2)
plt.plot(epc, post30_morphine_mean_val, label='validation')
plt.fill_between(epc,post30_morphine_mean_val-post30_morphine_std_val,post30_morphine_mean_val+post30_morphine_std_val,alpha=.2)
plt.xlabel('# of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#### Post 60 Placebo ####

post60_placebo_mean_tr=np.mean(np.vstack(post60_place_hist_tr),axis=0)
post60_placebo_std_tr=np.std(np.vstack(post60_place_hist_tr),axis=0)##--##
post60_placebo_mean_val=np.mean(np.vstack(post60_place_hist_val),axis=0)
post60_placebo_std_val=np.std(np.vstack(post60_place_hist_val),axis=0)##--##

epc = np.arange(0,len(post60_placebo_mean_tr))

plt.title('Post 60 Placebo',fontweight='bold')
plt.plot(epc,post60_placebo_mean_tr, label='train')
plt.fill_between(epc, post60_placebo_mean_tr-post60_placebo_std_tr,post60_placebo_mean_tr+post60_placebo_std_tr,alpha=.2)
plt.plot(epc,post60_placebo_mean_val, label='validation')
plt.fill_between(epc, post60_placebo_mean_val-post60_placebo_std_val,post60_placebo_mean_val+post60_placebo_std_val,alpha=.2)
plt.xlabel('# of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#### Post 60 Morphine ####

post60_morphine_mean_tr=np.mean(np.vstack(post60_morph_hist_tr),axis=0)
post60_morphine_std_tr=np.std(np.vstack(post60_morph_hist_tr),axis=0)##--##
post60_morphine_mean_val=np.mean(np.vstack(post60_morph_hist_val),axis=0)
post60_morphine_std_val=np.std(np.vstack(post60_morph_hist_val),axis=0)##--##

epc = np.arange(0,len(post60_morphine_mean_tr))

plt.title('Post 60 Morphine',fontweight='bold')
plt.plot(epc,post60_morphine_mean_tr, label='train')
plt.fill_between(epc,post60_morphine_mean_tr-post60_morphine_std_tr,post60_morphine_mean_tr+post60_morphine_std_tr,alpha=.2)
plt.plot(epc,post60_morphine_mean_val, label='validation')
plt.fill_between(epc,post60_morphine_mean_val-post60_morphine_std_val,post60_morphine_mean_val+post60_morphine_std_val,alpha=.2)
plt.xlabel('# of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#### Post 90 Placebo ####

post90_placebo_mean_tr=np.mean(np.vstack(post90_place_hist_tr),axis=0)
post90_placebo_std_tr=np.std(np.vstack(post90_place_hist_tr),axis=0)##--##
post90_placebo_mean_val=np.mean(np.vstack(post90_place_hist_val),axis=0)
post90_placebo_std_val=np.std(np.vstack(post90_place_hist_val),axis=0)##--##

epc = np.arange(0,len(post90_placebo_mean_tr))

plt.title('Post 90 Placebo',fontweight='bold')
plt.plot(epc, post90_placebo_mean_tr, label='train')
plt.fill_between(epc,post90_placebo_mean_tr-post90_placebo_std_tr,post90_placebo_mean_tr+post90_placebo_std_tr,alpha=.2)
plt.plot(epc, post90_placebo_mean_val,label='validation')
plt.fill_between(epc,post90_placebo_mean_val-post90_placebo_std_val,post90_placebo_mean_val+post90_placebo_std_val,alpha=.2)
plt.xlabel('# of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#### Post 90 Morphine ####

post90_morphine_mean_tr=np.mean(np.vstack(post90_morph_hist_tr),axis=0)
post90_morphine_std_tr=np.std(np.vstack(post90_morph_hist_tr),axis=0)##--##
post90_morphine_mean_val=np.mean(np.vstack(post90_morph_hist_val),axis=0)
post90_morphine_std_val=np.std(np.vstack(post90_morph_hist_val),axis=0)##--##

epc = np.arange(0,len(post90_morphine_mean_tr))

plt.title('Post 90 Morphine',fontweight='bold')
plt.plot(epc,post90_morphine_mean_tr, label='train')
plt.fill_between(epc,post90_morphine_mean_tr-post90_morphine_std_tr,post90_morphine_mean_tr+post90_morphine_std_tr,alpha=.2)
plt.plot(epc,post90_morphine_mean_val, label='validation') 
plt.fill_between(epc,post90_morphine_mean_val-post90_morphine_std_val,post90_morphine_mean_val+post90_morphine_std_val,alpha=.2)
plt.xlabel('# of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("Mean Pre Model Acc / Sens / Spe : %.2f +/- %.2f, %.2f +/- %.2f, %.2f +/- %.2f" % (np.mean(pre_acc),np.std(pre_acc),np.mean(pre_sens),np.std(pre_sens),np.mean(pre_spe),np.std(pre_spe)))
print("Mean Post 30 Model Morphine Acc / Sens / Spe : %.2f +/- %.2f, %.2f +/- %.2f, %.2f +/- %.2f" % (np.mean(post30_morph_acc), np.std(post30_morph_acc),np.mean(post30_morph_sens), np.std(post30_morph_sens), np.mean(post30_morph_spe), np.std(post30_morph_spe)))
print("Mean Post 30 Model Placebo Acc / Sens / Spe : %.2f +/- %.2f, %.2f +/- %.2f, %.2f +/- %.2f" % (np.mean(post30_place_acc), np.std(post30_place_acc),np.mean(post30_place_sens), np.std(post30_place_sens),np.mean(post30_place_spe), np.std(post30_place_spe)))
print("Mean Post 60 Model Morphine Acc / Sens / Spe : %.2f +/- %.2f, %.2f +/- %.2f, %.2f +/- %.2f" % (np.mean(post60_morph_acc), np.std(post60_morph_acc),np.mean(post60_morph_sens), np.std(post60_morph_sens),np.mean(post60_morph_spe), np.std(post60_morph_spe)))
print("Mean Post 60 Model Placebo Acc / Sens / Spe : %.2f +/- %.2f, %.2f +/- %.2f, %.2f +/- %.2f" % (np.mean(post60_place_acc),np.std(post60_place_acc),np.mean(post60_place_sens),np.std(post60_place_sens),np.mean(post60_place_spe),np.std(post60_place_spe)))
print("Mean Post 90 Model Morphine Acc / Sens / Spe : %.2f +/- %.2f, %.2f +/- %.2f, %.2f +/- %.2f" % (np.mean(post90_morph_acc),np.std(post90_morph_acc),np.mean(post90_morph_sens),np.std(post90_morph_sens),np.mean(post90_morph_spe),np.std(post90_morph_spe)))
print("Mean Post 90 Model Placebo Acc / Sens / Spe : %.2f +/- %.2f, %.2f +/- %.2f, %.2f +/- %.2f" % (np.mean(post90_place_acc),np.std(post90_place_acc),np.mean(post90_place_sens),np.std(post90_place_sens),np.mean(post90_place_spe),np.std(post90_place_spe)))

def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('')
    #ax.set_title(region)

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def plot_perf_graphs(pre,post30_morph,post60_morph,post90_morph,post30_place,post60_place,post90_place, plt_type):
    
    
    data=[]
    data.append(pre)
    data.append(post30_morph)
    data.append(post60_morph)
    data.append(post90_morph)
    data.append(post30_place)
    data.append(post60_place)
    data.append(post90_place)
    
    fig, axs = plt.subplots(1,1,figsize=(60, 20),sharex=False, sharey=True,constrained_layout=False)
    names = [
    "Pre-Drug",
    "Post Drug \n Morphine \n 30 min",
    "Post Drug \n Morpihine \n 60 min",
    "Post Drug \n Morphine \n  90 min",
    "Post Drug \n Placebo \n  30 min",
    "Post Drug \n Placebo \n  60 min",
    "Post Drug \n Placebo \n  90 min",
    ]
    k=0
    fs=60
    fs2=60
    lw=5

    pp=axs.violinplot(data,showmeans=True,showextrema=True,showmedians=False,vert=True, widths=0.5)
    for pc in pp['bodies']:
        pc.set_facecolor('red')
        pc.set_edgecolor('red')
        pc.set_linewidth(lw)
    pp['cmeans'].set_color('k')
    pp['cmeans'].set_linewidth(lw)
    pp['cmaxes'].set_color('k')
    pp['cmaxes'].set_linewidth(lw)
    pp['cmins'].set_color('k')
    pp['cmins'].set_linewidth(lw)
    pp['cbars'].set_color('k')
    pp['cbars'].set_linewidth(lw)
    plt.setp(axs.get_xticklabels(), fontsize=fs2, fontweight="bold")
    plt.setp(axs.get_yticklabels(), fontsize=fs2, fontweight="bold")
    #axs[i,j].set_xlabel('Feature Set' '\n' '\n' + alph[k] + '\n' ,fontsize=fs,fontweight='bold')
    axs.set_ylabel(plt_type,fontsize=fs2,fontweight='bold')
    axs.set_xlabel(names,fontsize=fs2,fontweight='bold')
    axs.set_title('Transfer Learning Results',fontsize=60,fontweight='bold')
    set_axis_style(axs, names)
    plt.show()

plot_perf_graphs(pre_acc, post30_morph_acc, post60_morph_acc, post90_morph_acc, post30_place_acc, post60_place_acc, post90_place_acc, 'Accuracy')
plot_perf_graphs(pre_sens, post30_morph_sens, post60_morph_sens, post90_morph_sens, post30_place_sens, post60_place_sens, post90_place_sens, 'Sensitivity')
plot_perf_graphs(pre_spe, post30_morph_spe, post60_morph_spe, post90_morph_spe, post30_place_spe, post60_place_spe, post90_place_spe, 'Specificity')





def average_shap_wrt_reg(av_shap_values, cond):
    
    av_shap=[]
    
    if cond=='morphine':
        av_shap.append(av_shap_values[:,0])
        av_shap.append(np.mean(av_shap_values[:,[16,18,19,22]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[1,4,5,6,7,9]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[13,15]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[8,10,12]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[11,14]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[2,3]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[1,23]],axis=1))
        av_shap.append(av_shap_values[:,20])
        av_shap.append(av_shap_values[:,19])
    elif cond=='placebo':
        
        av_shap.append(np.mean(av_shap_values[:,[0,1]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[16,18,19,22]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[4,5,6,7,9,12]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[13,15]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[8,10]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[11,14]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[2,3]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[21,23]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[17,20]],axis=1))
        
    elif cond=='pre':
        
        av_shap.append(av_shap_values[:,0])
        av_shap.append(np.mean(av_shap_values[:,[16,18,19,22]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[1,4,5,6,7,9,12]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[13,15]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[8,10]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[11,14]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[2,3]],axis=1))
        av_shap.append(np.mean(av_shap_values[:,[21,23]],axis=1))
        av_shap.append(av_shap_values[:,20])
        av_shap.append(av_shap_values[:,17])
        
    
    av_shap=np.transpose(np.asarray(av_shap))
    
    return av_shap



# ## Averaging shap values for Pre-Model

## Regions are  L PMC Channels# 1
#               R PMC Channels# 17,19,20,23
#               L DLPFC Channels# 2,5,6,7,8,10,13
#               R DLPFC Channels# 14,16
#               L FPA Channels# 9,11
#               R FPA Channels# 12,15
#               L IFG Channels# 3,4
#               R SMG Channels# 22,24
#               R SI Channels# 21
#               R MI Channels# 18
### ------------ Pre Model -------######

roi_list_pre =['L PMC','R PMC','L DLPFC','R DLPFC','L FPA','R FPA','L IFG','R SMG','R SI','R MI']

features=[]

for i in range(0,len(roi_list_pre)):
    
    features.append(roi_list_pre[i])


shap_av_val_pre=[]
shap_av_test_data_pre=[]

for i in range(0,n):
        
    shap_av_val_pre.append(np.abs(list_shap_values_pre[i][0]).mean(axis=0))
    shap_av_test_data_pre.append(shap_test_data_pre[i].mean(axis=0))
        

av_shap_values=np.mean(shap_av_val_pre,axis=0)
av_shap_test_data=np.mean(shap_av_test_data_pre,axis=0)

av_shap_values=average_shap_wrt_reg(av_shap_values,'pre')
av_shap_test_data=average_shap_wrt_reg(av_shap_test_data,'pre')
av_shap_values_pre=av_shap_values

av_shap_test_data = pd.DataFrame(data=av_shap_test_data, columns = features)
shap.summary_plot(av_shap_values, features=av_shap_test_data,plot_type='bar',max_display=len(features),show=False)
plt.title('Pre Model',fontsize=25,fontweight='bold')
plt.xlabel('Average SHAP Value',fontsize=20)
plt.show()
        

Region_List_Morphine=['L PMC','L DLPFC','L IFG','L IFG',
                      'L DLPFC','L DLPFC','L DLPFC','L DLPFC',
                      'L FPA','L DLPFC','L FPA','R FPA',
                      'L FPA','R DLPFC','R FPA','R DLPFC',
                      'R PMC','R MI','R PMC','R PMC',
                      'R SI','R SMG','R PMC','R SMG']

## Regions are  L PMC Channels# 1
#               R PMC Channels# 17, 19,20,23
#               L DLPFC Channels# 2,5,6,7,8,10
#               R DLPFC Channels# 14,16
#               L FPA Channels# 9,11,13
#               R FPA Channels# 12,15
#               L IFG Channels# 3,4
#               R SMG Channels# 22, 24
#               R SI Channels# 21
#               R MI Channels# 18
### ------------ MORPHINE MODELS -------######

roi_list_morphine =['L PMC','R PMC','L DLPFC','R DLPFC','L FPA','R FPA','L IFG','R SMG','R SI','R MI']

features=[]

for i in range(0,len(roi_list_morphine)):
    
    features.append(roi_list_morphine[i])

# ## Averaging shap values for Post30-Morphine-Model

shap_av_val_pre=[]
shap_av_test_data_pre=[]

for i in range(0,n):
        
    shap_av_val_pre.append(np.abs(list_shap_values_post30morphine[i][0]).mean(axis=0))
    shap_av_test_data_pre.append(test_morphine_post30_data.mean(axis=0))
        

av_shap_values=np.mean(shap_av_val_pre,axis=0)
av_shap_test_data=np.mean(shap_av_test_data_pre,axis=0)

av_shap_values=average_shap_wrt_reg(av_shap_values,'morphine')
av_shap_test_data=average_shap_wrt_reg(av_shap_test_data,'morphine')
av_shap_values_post30_morphine=av_shap_values

shap.initjs()

av_shap_test_data = pd.DataFrame(data=av_shap_test_data, columns = features)
shap.summary_plot(av_shap_values, features=av_shap_test_data,plot_type='bar',max_display=len(features),show=False)
plt.title('Morphine 30 min Model',fontsize=25,fontweight='bold')
plt.xlabel('Average SHAP Value',fontsize=20)
plt.show()


# ## Averaging shap values for Post60-Morphine-Model

shap_av_val_pre=[]
shap_av_test_data_pre=[]

for i in range(0,n):
        
    shap_av_val_pre.append(np.abs(list_shap_values_post60morphine[i][0]).mean(axis=0))
    shap_av_test_data_pre.append(test_morphine_post60_data.mean(axis=0))
        

av_shap_values=np.mean(shap_av_val_pre,axis=0)
av_shap_test_data=np.mean(shap_av_test_data_pre,axis=0)

av_shap_values=average_shap_wrt_reg(av_shap_values,'morphine')
av_shap_test_data=average_shap_wrt_reg(av_shap_test_data,'morphine')
av_shap_values_post60_morphine=av_shap_values


shap.initjs()

av_shap_test_data = pd.DataFrame(data=av_shap_test_data, columns = features)
shap.summary_plot(av_shap_values, features=av_shap_test_data,plot_type='bar',max_display=len(features),show=False)
plt.title('Morphine 60 min Model',fontsize=25,fontweight='bold')
plt.xlabel('Average SHAP Value',fontsize=20)
plt.show()

# ## Averaging shap values for Post90-Morphine-Model

shap_av_val_pre=[]
shap_av_test_data_pre=[]

for i in range(0,n):
        
    shap_av_val_pre.append(np.abs(list_shap_values_post90morphine[i][0]).mean(axis=0))
    shap_av_test_data_pre.append(test_morphine_post90_data.mean(axis=0))
        

av_shap_values=np.mean(shap_av_val_pre,axis=0)
av_shap_test_data=np.mean(shap_av_test_data_pre,axis=0)

av_shap_values=average_shap_wrt_reg(av_shap_values,'morphine')
av_shap_test_data=average_shap_wrt_reg(av_shap_test_data,'morphine')
av_shap_values_post90_morphine=av_shap_values

shap.initjs()

av_shap_test_data = pd.DataFrame(data=av_shap_test_data, columns = features)
shap.summary_plot(av_shap_values, features=av_shap_test_data,plot_type='bar',max_display=len(features),show=False)
plt.title('Morphine 90 min Model',fontsize=25,fontweight='bold')
plt.xlabel('Average SHAP Value',fontsize=20)
plt.show()

### ------------ PLACEBO MODELS -------######
Region_List_Placebo=['L PMC','L PMC','L IFG','L IFG',
                      'L DLPFC','L DLPFC','L DLPFC','L DLPFC',
                      'L FPA','L DLPFC','L FPA','R FPA',
                      'L DLPFC','R DLPFC','R FPA','R DLPFC',
                      'R PMC','R SI','R PMC','R PMC',
                      'R SI','R SMG','R PMC','R SMG']

## Regions are  L PMC Channels# 1,2
#               R PMC Channels# 17, 19,20,23
#               L DLPFC Channels# 5,6,7,8,10,13
#               R DLPFC Channels# 14,16
#               L FPA Channels# 9,11
#               R FPA Channels# 12,15
#               L IFG Channels# 3,4
#               R SMG Channels# 22, 24
#               R SI Channels# 18, 21


roi_list_placebo =['L PMC','R PMC','L DLPFC','R DLPFC','L FPA','R FPA','L IFG','R SMG','R SI']

features=[]

for i in range(0,len(roi_list_placebo)):
    
    features.append(roi_list_placebo[i])

# ## Averaging shap values for Post30-Placebo-Model

shap_av_val_pre=[]
shap_av_test_data_pre=[]

for i in range(0,n):
        
    shap_av_val_pre.append(np.abs(list_shap_values_post30placebo[i][0]).mean(axis=0))
    shap_av_test_data_pre.append(test_placebo_post30_data.mean(axis=0))
        

av_shap_values=np.mean(shap_av_val_pre,axis=0)
av_shap_test_data=np.mean(shap_av_test_data_pre,axis=0)

av_shap_values=average_shap_wrt_reg(av_shap_values,'placebo')
av_shap_test_data=average_shap_wrt_reg(av_shap_test_data,'placebo')
av_shap_values_post30_placebo=av_shap_values

shap.initjs()

av_shap_test_data = pd.DataFrame(data=av_shap_test_data, columns = features)
shap.summary_plot(av_shap_values, features=av_shap_test_data,plot_type='bar',max_display=len(features),show=False)
plt.title('Placebo 30 min Model',fontsize=25,fontweight='bold')
plt.xlabel('Average SHAP Value',fontsize=20)
plt.show()


# ## Averaging shap values for Post60-Placebo-Model

shap_av_val_pre=[]
shap_av_test_data_pre=[]

for i in range(0,n):
        
    shap_av_val_pre.append(np.abs(list_shap_values_post60placebo[i][0]).mean(axis=0))
    shap_av_test_data_pre.append(test_placebo_post60_data.mean(axis=0))
        

av_shap_values=np.mean(shap_av_val_pre,axis=0)
av_shap_test_data=np.mean(shap_av_test_data_pre,axis=0)

av_shap_values=average_shap_wrt_reg(av_shap_values,'placebo')
av_shap_test_data=average_shap_wrt_reg(av_shap_test_data,'placebo')
av_shap_values_post60_placebo=av_shap_values

shap.initjs()

av_shap_test_data = pd.DataFrame(data=av_shap_test_data, columns = features)
shap.summary_plot(av_shap_values, features=av_shap_test_data,plot_type='bar',max_display=len(features),show=False)
plt.title('Placebo 60 min Model',fontsize=25,fontweight='bold')
plt.xlabel('Average SHAP Value',fontsize=20)
plt.show()

# ## Averaging shap values for Post90-Placebo-Model

shap_av_val_pre=[]
shap_av_test_data_pre=[]

for i in range(0,n):
        
    shap_av_val_pre.append(np.abs(list_shap_values_post90placebo[i][0]).mean(axis=0))
    shap_av_test_data_pre.append(test_placebo_post90_data.mean(axis=0))
        

av_shap_values=np.mean(shap_av_val_pre,axis=0)
av_shap_test_data=np.mean(shap_av_test_data_pre,axis=0)

av_shap_values=average_shap_wrt_reg(av_shap_values,'placebo')
av_shap_test_data=average_shap_wrt_reg(av_shap_test_data,'placebo')
av_shap_values_post90_placebo=av_shap_values

shap.initjs()
av_shap_test_data = pd.DataFrame(data=av_shap_test_data, columns = features)
shap.summary_plot(av_shap_values, features=av_shap_test_data,plot_type='bar',max_display=len(features),show=False)
plt.title('Placebo 90 min Model',fontsize=25,fontweight='bold')
plt.xlabel('Average SHAP Value',fontsize=20)
plt.show()


mean_pre_tpr = np.mean(pre_tpr,axis=0)
std_pre_tpr=np.std(pre_tpr,axis=0)
mean_pre_tpr[0]=0.0
mean_pre_fpr= np.mean(pre_fpr,axis=0)

mean_post30_morph_tpr = np.mean(post30_morph_tpr,axis=0)
std_post30_morph_tpr=np.std(post30_morph_tpr,axis=0)
mean_post30_morph_tpr[0]=0.0
mean_post30_morph_fpr= np.mean(post30_morph_fpr,axis=0)

mean_post60_morph_tpr = np.mean(post60_morph_tpr,axis=0)
std_post60_morph_tpr=np.std(post60_morph_tpr,axis=0)
mean_post60_morph_tpr[0]=0.0
mean_post60_morph_fpr= np.mean(post60_morph_fpr,axis=0)

mean_post90_morph_tpr = np.mean(post90_morph_tpr,axis=0)
std_post90_morph_tpr=np.std(post90_morph_tpr,axis=0)
mean_post90_morph_tpr[0]=0.0
mean_post90_morph_fpr= np.mean(post90_morph_fpr,axis=0)

mean_post30_place_tpr = np.mean(post30_place_tpr,axis=0)
std_post30_place_tpr=np.std(post30_place_tpr,axis=0)
mean_post30_place_tpr[0]=0.0
mean_post30_place_fpr= np.mean(post30_place_fpr,axis=0)

mean_post60_place_tpr = np.mean(post60_place_tpr,axis=0)
std_post60_place_tpr=np.std(post60_place_tpr,axis=0)
mean_post60_place_tpr[0]=0.0
mean_post60_place_fpr= np.mean(post60_place_fpr,axis=0)

mean_post90_place_tpr = np.mean(post90_place_tpr,axis=0)
std_post90_place_tpr=np.std(post90_place_tpr,axis=0)
mean_post90_place_tpr[0]=0.0
mean_post90_place_fpr= np.mean(post90_place_fpr,axis=0)

plt.title('ROC Curves',fontweight='bold')
plt.plot(mean_pre_fpr,mean_pre_tpr,'b',label='Pre Model, AUC: %.2f \u00B1 %.2f' % (np.mean(pre_auc),np.std(pre_auc)))
plt.plot(mean_post30_morph_fpr,mean_post30_morph_tpr,'k',label='Post 30 Morphine, AUC: %.2f \u00B1 %.2f' % (np.mean(post30_morph_auc),np.std(post30_morph_auc)))
plt.plot(mean_post60_morph_fpr,mean_post60_morph_tpr,'g',label='Post 60 Morphine, AUC: %.2f \u00B1 %.2f' % (np.mean(post60_morph_auc),np.std(post60_morph_auc)))
plt.plot(mean_post90_morph_fpr,mean_post90_morph_tpr,'r',label='Post 90 Morphine, AUC: %.2f \u00B1 %.2f' % (np.mean(post90_morph_auc),np.std(post90_morph_auc)))
plt.plot(mean_post30_place_fpr,mean_post30_place_tpr,'c',label='Post 30 Placebo, AUC: %.2f \u00B1 %.2f' % (np.mean(post30_place_auc),np.std(post30_place_auc)))
plt.plot(mean_post60_place_fpr,mean_post60_place_tpr,'y',label='Post 60 Placebo, AUC: %.2f \u00B1 %.2f' % (np.mean(post60_place_auc),np.std(post60_place_auc)))
plt.plot(mean_post90_place_fpr,mean_post90_place_tpr,'m',label='Post 90 Placebo, AUC: %.2f \u00B1 %.2f' % (np.mean(post90_place_auc),np.std(post90_place_auc)))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()



## Statistics of Accuracy Comparison



stat_pre_morph=sp.stats.kruskal(pre_acc,post30_morph_acc,post60_morph_acc,post90_morph_acc)
stat_pre_place=sp.stats.kruskal(pre_acc,post30_place_acc,post60_place_acc,post90_place_acc)

m_dict = {"pre_acc": pre_acc,
          "pre_sens": pre_sens,
          "pre_spe": pre_spe,
          "pre_auc":pre_auc,
          "post30_morph_acc": post30_morph_acc,
          "post30_morph_sens": post30_morph_sens,
          "post30_morph_spe": post30_morph_spe,
          "post30_morph_auc": post30_morph_auc,
          "post60_morph_acc": post60_morph_acc,
          "post60_morph_sens": post60_morph_sens,
          "post60_morph_spe": post60_morph_spe,
          "post60_morph_auc": post60_morph_auc,
          "post90_morph_acc": post90_morph_acc,
          "post90_morph_sens": post90_morph_sens,
          "post90_morph_spe": post90_morph_spe,
          "post90_morph_auc": post60_morph_auc,
          "post30_place_acc": post30_place_acc,
          "post30_place_sens": post30_place_sens,
          "post30_place_spe": post30_place_spe,
          "post30_place_auc": post30_place_auc,
          "post60_place_acc": post60_place_acc,
          "post60_place_sens": post60_place_sens,
          "post60_place_spe": post60_place_spe,
          "post60_place_auc": post60_place_auc,
          "post90_place_acc": post90_place_acc,
          "post90_place_sens": post90_place_sens,
          "post90_place_spe": post90_place_spe,
          "post90_place_auc": post90_place_auc}


io.savemat("pain_pre_morph_place_acc_sens_spe.mat", m_dict)
## Friedman test
d={}

d = {'Run':np.arange(1,n+1,1).tolist()*6,'Time':['30 min']*n+['60 min']*n+['90 min']*n+['30 min']*n+['60 min']*n+['90 min']*n, 'Drug': ['Morphine']*n*3+['Placebo']*n*3, 'Accuracy':post30_morph_acc+post60_morph_acc+post90_morph_acc+post30_place_acc+post60_place_acc+post90_place_acc}

df=pd.DataFrame(data=d)

aov=pg.friedman(data=df,dv='Accuracy',within='Drug',subject='Time',method='f')

shap_pre=[]
shap_post30morphine=[]
shap_post60morphine=[]
shap_post90morphine=[]
shap_post30placebo=[]
shap_post60placebo=[]
shap_post90placebo=[]

for ind in range (0,len(list_shap_values_pre)):
    
    shap_pre.append(np.mean(np.mean(list_shap_values_pre[ind][0],axis=0),axis=0))
    shap_post30morphine.append(np.mean(np.mean(list_shap_values_post30morphine[ind][0],axis=0),axis=0))
    shap_post60morphine.append(np.mean(np.mean(list_shap_values_post60morphine[ind][0],axis=0),axis=0))
    shap_post90morphine.append(np.mean(np.mean(list_shap_values_post90morphine[ind][0],axis=0),axis=0))
    shap_post30placebo.append(np.mean(np.mean(list_shap_values_post30placebo[ind][0],axis=0),axis=0))
    shap_post60placebo.append(np.mean(np.mean(list_shap_values_post60placebo[ind][0],axis=0),axis=0))
    shap_post90placebo.append(np.mean(np.mean(list_shap_values_post90placebo[ind][0],axis=0),axis=0))
    
    
shap_pre = average_shap_wrt_reg(np.array(shap_pre),'pre')
shap_post30morphine = average_shap_wrt_reg(np.array(shap_post30morphine),'morphine')
shap_post60morphine = average_shap_wrt_reg(np.array(shap_post60morphine),'morphine')
shap_post90morphine = average_shap_wrt_reg(np.array(shap_post90morphine),'morphine')
shap_post30placebo = average_shap_wrt_reg(np.array(shap_post30placebo),'placebo')
shap_post60placebo = average_shap_wrt_reg(np.array(shap_post60placebo),'placebo')
shap_post90placebo = average_shap_wrt_reg(np.array(shap_post90placebo),'placebo')

m_dict = {"shap_pre":shap_pre,
          "shap_post30morphine":shap_post30morphine,
          "shap_post60morphine":shap_post60morphine,
          "shap_post90morphine":shap_post90morphine,
          "shap_post30placebo":shap_post30placebo,
          "shap_post60placebo":shap_post60placebo,
          "shap_post90placebo":shap_post90placebo}
          
          
io.savemat("shapley_values_all.mat", m_dict)


plt.rcParams['lines.linewidth'] = 10
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['figure.figsize'] = [64,32]
plt.rcParams['axes.labelsize']=60
plt.rcParams['axes.labelweight']='bold'

plt.rcParams['font.size']=60
plt.rcParams['font.weight']='bold'
plt.rcParams['figure.dpi']=100
plt.rcParams['legend.loc'] = 'right'

cond = ['Pre','PM30','PM60','PM90','PP30','PP60','PP90']

wd=3
x =np.arange(0,300,30)

fig = plt.figure(figsize=(64,32))
fig.set_tight_layout(False)

for i in range (0,len(roi_list_pre)):
    if roi_list_pre[i]!='R MI':
        print(roi_list_pre[i])
        plt.bar(x[i]-(3*wd),np.mean(shap_pre[:,i]),yerr=np.std(shap_pre[:,i])/np.sqrt(30),color='blue',width=wd)
        plt.bar(x[i]-(2*wd),np.mean(shap_post30morphine[:,i]),yerr=np.std(shap_post30morphine[:,i])/np.sqrt(30),color='green',width=wd)
        plt.bar(x[i]-(wd),np.mean(shap_post60morphine[:,i]),yerr=np.std(shap_post60morphine[:,i])/np.sqrt(30),color='orange',width=wd)
        plt.bar(x[i],np.mean(shap_post90morphine[:,i]),yerr=np.std(shap_post90morphine[:,i])/np.sqrt(30),color='red',width=wd)
        plt.bar(x[i]+(wd),np.mean(shap_post30placebo[:,i]),yerr=np.std(shap_post30placebo[:,i])/np.sqrt(30),color='cyan',width=wd)
        plt.bar(x[i]+(2*wd),np.mean(shap_post60placebo[:,i]),yerr=np.std(shap_post60placebo[:,i])/np.sqrt(30),color='magenta',width=wd)
        plt.bar(x[i]+(3*wd),np.mean(shap_post90placebo[:,i]),yerr=np.std(shap_post90placebo[:,i])/np.sqrt(30),color='black',width=wd)

    else:
        plt.bar(x[i]-(2*wd),np.mean(shap_pre[:,i]),yerr=np.std(shap_pre[:,i])/np.sqrt(30),color='blue',width=wd)
        plt.bar(x[i]-wd,np.mean(shap_post30morphine[:,i]),yerr=np.std(shap_post30morphine[:,i])/np.sqrt(30),color='green',width=wd)
        plt.bar(x[i],np.mean(shap_post60morphine[:,i]),yerr=np.std(shap_post60morphine[:,i])/np.sqrt(30),color='orange',width=wd)
        plt.bar(x[i]+wd,np.mean(shap_post90morphine[:,i]),yerr=np.std(shap_post90morphine[:,i])/np.sqrt(30),color='red',width=wd)
plt.ylabel('Average Shapley Values')
plt.title('Average Shapley Values of 30 Run For All Regions and For All Conditions ',fontweight='bold')
plt.legend(cond,loc='upper center',ncol=7)
        
plt.ylim(-0.0003,0.00025)
plt.xticks(x,roi_list_pre)
padding = 1  # Adjust the value as needed

# Adjust the position of the x-axis and y-axis labels
fig.xaxis.set_label_coords(0.5, -padding)
fig.yaxis.set_label_coords(-padding, 0.5)

# Adjust the position of the axes
fig.set_position([padding, padding, 1 - (2 * padding), 1 - (2 * padding)])
        
        


