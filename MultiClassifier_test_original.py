# -*- coding: utf-8 -*-
"""

@author: SolveigCodes

url: https://github.com/SolveigCodes/AI_ensembles_for_AD
"""

# Depends on the resulting models from MultiClassifier_traincv.py


##### Libraries

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import GroupNormalization, BatchNormalization, Dropout
from keras.optimizers import SGD, Adam
from keras.models import load_model
import os
import pandas as pd
import nibabel as nib
import operator
from sklearn.model_selection import train_test_split
import csv
import time
import skimage
from skimage import measure
from classification_models_3D.tfkeras import Classifiers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras import backend as K
from keras.layers import Dropout, Dense, Activation, GlobalAveragePooling3D
from keras.models import Model
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from imblearn.metrics import specificity_score
from sklearn.utils.multiclass import type_of_target


##### Hyperparameters and global variables

dropoutrate = 0.4
epochs = 200
batch_size = 8
loss='sparse_categorical_crossentropy'
patience = 20
fold_no = 5 # Enter this value to specify which model to use. Here meant for models created from 5-fold cv training

# Set global seed
tf.random.set_seed(1)
rseed = 77

cropdim = 160
imlength = 160
imwidth = 160
imheight = 160

target_names = ['CN','MCI','AD']

# metrics
metrics = ['accuracy','bal_accuracy','recall_n','recall_w','spec_n','spec_w','prec_n','prec_w','F1_n','F1_w','MCC']


##### Variables for file names
# Create time string
timestr = time.strftime("%Y-%m-%d_%H%M")
print(timestr)
# Script version
scriptversion = "Multi1testorig"
print("Script: "+scriptversion)

# Common outputinfo
outputinfo = scriptversion+'_'+timestr+'_b'+str(batch_size)+'_e'+str(epochs)+'_f'+str(fold_no)


##### Load dataset and prepare data

# Location of folder that holds the images after linear processing (ref. t1-linear from Clinica)
# The images stored in a folder structure individual subject folders containing session folders
# https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/T1_Linear/
#
# Example: sub-ADNI016S6892 > ses-M00 > t1_linear > 
# The t1_linear folder contains three files - example:
#  i) sub-ADNI016S6892_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii'
#  ii) sub-ADNI016S6892_ses-M00_T1w_space-MNI152NLin2009cSym_res-1x1x1_affine
#  iii) sub-ADNI016S6892_ses-M00_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii

data_path = '22-12-19/subjects/' # Replace with actual folder name

# participants file created from Clinica adni-to-bids dataset converter
# https://aramislab.paris.inria.fr/clinica/docs/public/latest/BIDS/#an-overview-of-the-bids-structure

pfile = '/22-12-19/participants.tsv' # Replace with actual file location

# file from ADNI study data containing information about diagnosis per session
dfile = '/22-12-19/StudyData/DXSUM_PDXCONV_ADNIALL.csv' # Replace with actual file location


## Path to best model from training that will be used as weights for testing
path_bestmodel = '/Checkpoint/archname_fold5_best_model.hdf5' # Replace with actual file location
print("path best model: ",path_bestmodel) # for control purposes only

# Replace with actual file location. This file is produced by Multiclassifier_traincv.py
# Should be the file from the corresponding architecture
stratpath = '/stratlist_outputinfo-from-multiclassifier_traincv.csv' 



##### Functions


# Preprocessing

def crop(img, bounding):
    """
    Function that center crops image according to defined size (bounding)
    Returns image with defined size e.g. 164 x 164 x 164
    
    """
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def normalize(data):
    """
    Normalize element by element in array of image arrays
    Arguments: Object to be normalized
    Returns: Array of normalized image arrays
    
    """
    normvalue = []
    for i in range (len(data)):
        # print("length of data: ", len(data)) # 2023-12-29 debug
        n = (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]))
        normvalue.append(n)
        
    return np.array(normvalue)




##### Write files for statistical analysis
def stats(lst,fld,lststr,dlbls):
	"""
	Produce descriptive statistics for subjects in a dataset
	Receives list of paths, fold no, name of dataset, label dictionary
	Returns csv-file with descriptive statistics and csv-file with
		overview of subjects and their data
	"""
	p_df = pd.read_csv(pfile, sep= '\t', header=0)
	p_df = p_df.sort_values(by=['participant_id']) 
	sortlst = []
	flst = []
	dseslst = []
	for fname in range(len(lst)):
		pos = lst[fname].find('_ses-')
		sortlst.append(lst[fname][(pos-16):pos])
		flst.append(lst[fname])
	sortdf = p_df[p_df['participant_id'].isin(sortlst)] #retrieve a participant-file limited to participants in dataset
	sortdf = sortdf.sort_values(by=['participant_id']) 
	
	flst.sort() 
	# add a column to dataframe containing diagnosis at time of sessions (currently baseline)
	for f in range (len(flst)):
		dseslst.append(dlbls[flst[f]])
	dsesdf = pd.DataFrame(dseslst, columns=['diagnosis_ses'])
	combdf = sortdf.reset_index(drop=True)
	combdf = pd.concat([combdf, dsesdf], axis = 1)
	
	dstats = combdf.groupby(['diagnosis_ses', 'sex'], as_index=True).describe() #calculating descriptive statistics per diagnosis and sex category
	
	# paths for statistics files
	dstatsfile = '/MultiClassifier/Statistics/Descriptives/'+'dstats_'+outputinfo+'_'+lststr+'_fold'+str(fld)+'_.csv' # Replace with actual file location
	ostatsfile = '/MultiClassifier/Statistics/Overview/'+'overview_'+outputinfo+'_'+lststr+'_fold'+str(fld)+'_.csv' # Replace with actual file location
	
	dstats.to_csv(dstatsfile) #descriptive statistics
	combdf.to_csv(ostatsfile) #overview 
	return



##### Evaluation metrics	
def evalmetric(true, pred, metrics):
	"""
	Function that calculates evaluation metrics
	Arguments:
	Returns:
	
	"""
	
	for metric in metrics:
			
		if metric == 'accuracy':
			accuracy = accuracy_score(true, pred)
		elif metric == 'bal_accuracy':
			bal_acc = balanced_accuracy_score(true,pred)
		elif metric == 'recall_n':
			recall_n = recall_score(true, pred, labels=[0,1,2],average=None)
		elif metric == 'recall_w':
			recall_w = recall_score(true, pred, labels=[0,1,2], average='weighted')
		elif metric == 'spec_n':
			spec_n = specificity_score(true, pred, labels=[0,1,2], average=None)
		elif metric == 'spec_w':
			spec_w = specificity_score(true, pred, labels=[0,1,2], average='weighted')
		elif metric == 'prec_n':
			prec_n =  precision_score(true, pred, labels=[0,1,2], average=None)
		elif metric == 'prec_w':
			prec_w =  precision_score(true, pred, labels=[0,1,2], average='weighted')
		elif metric == 'F1_n':
			F1_n = f1_score(true, pred, labels=[0,1,2], average=None)
		elif metric == 'F1_w':
			F1_w = f1_score(true, pred, labels=[0,1,2], average='weighted')
		else: 
			MCC = matthews_corrcoef(true, pred)
			

	return accuracy, bal_acc, recall_n, recall_w, spec_n, spec_w, prec_n, prec_w, F1_n, F1_w, MCC


 
    
##### Classes

## Reference https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_paths, labels, batch_size=8, dim=(imlength,imwidth,imheight), n_channels=1,
                 n_classes=3, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_paths = list_paths
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_paths) / self.batch_size))
        

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
       
        # Find list of paths
        list_paths_temp = [self.list_paths[k] for k in indexes]
            
        # Generate data
        X, y = self.__data_generation(list_paths_temp)
        
        return X, y
        
                   
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_paths_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype="float32")

        # Generate data
        for i, paths in enumerate(list_paths_temp):
            T1_img = nib.load(paths)
            T1_img_data = T1_img.get_fdata()
            T1_img_data = crop(T1_img_data, (imlength,imwidth,imheight))
            T1_img_data = np.expand_dims(T1_img_data.astype("float32"), axis=3)
            X[i,] = T1_img_data
            y[i] = self.labels[paths]
        X = normalize(X)

        return X,y
        

##### Main script

##### Create data lists

stratimp = pd.read_csv(stratpath, header=0, index_col=0)


##### Label dictionaries

# string coded, i.e. 'CN', 'MCI','AD'
dlbls = {}
keys = stratimp['imagepath'].tolist()
values = stratimp['diag_let'].tolist()
for i in range(len(stratimp['imagepath'])):
	dlbls[keys[i]] = values[i]

# number coded, i.e. 0,1,2
labels = {}
keys = stratimp['imagepath'].tolist()
values = stratimp['diag_num'].tolist()
for i in range(len(stratimp['imagepath'])):
    labels[keys[i]] = values[i]

# Check for number of instances of each diagnosis
print("Number of diagnosis instances:")
print(stratimp['diag_let'].value_counts())


##### Split datasets
# Split the list of individual image paths and diagnosis list
# One part sent to cross validation (used in Multiclassifier_traincv.py)
# Keep a separate hold out set for external validation (x_hold, y_hold)

targ = stratimp.diag_let + '_' + stratimp.sex + '_' + stratimp.age
print("type of target: ",type_of_target(targ))
print("length of targ: ", len(targ))
print("shape of targ: ", np.shape(targ))
print("targ[0]: ", targ[0])
print("No of instances per class: ", pd.DataFrame(targ).value_counts(dropna=False))

x_train_test, x_hold, y_train_test, y_hold = train_test_split(stratimp['imagepath'], stratimp['diag_let'], test_size=0.20, random_state=rseed, stratify=targ)

x_train_test.to_csv('/MultiClassifier/xtraintest_'+scriptversion+'_'+timestr+'.csv') # Replace with actual file location
x_hold.to_csv('/MultiClassifier/xhold_'+scriptversion+'_'+timestr+'.csv') # Replace with actual file location


# datasets
print('length of x_train_test: ', len(x_train_test)) # Used for cv
print('length of x_hold: ', len(x_hold)) # Hold-outset used for testing
x_hold2=x_hold.tolist()
print('x_hold2[0]: ', x_hold2[0])

# write files for statistics
stats(x_train_test.tolist(), 'preCV', 'xtraintest_', dlbls)
stats(x_hold.tolist(), 'preCV', 'xhold_', dlbls)

# Dictionary for partition and labels for external validation set, x_hold
hold_partition = {'hold':x_hold2}


# Parameter dictionaries for generators
params = {'dim': (imlength,imwidth,imheight),
         'batch_size': batch_size,
         'n_classes': 3,
         'n_channels': 1,
         'shuffle': False}


# In[21]:


#print("hold_generator")
hold_generator = DataGenerator(hold_partition['hold'], labels, **params)


##### Define model architecture
### Choose between ResNet-18 and two VGG-based customized networks - uncomment relevant lines

### First model architecture
### ResNet
# ResNet18, preprocess_input = Classifiers.get('resnet18')
# input = layers.Input((cropdim, cropdim, cropdim, 1))
# cnn = ResNet18(weights=None, include_top = False, input_tensor=input, input_shape=(cropdim,cropdim,cropdim,1))
# x = cnn.output
# x = layers.AveragePooling3D(pool_size=(2, 2, 2))(x) 
# x = layers.Flatten()(x)
# normal_layer = tf.keras.layers.BatchNormalization()(x)
# output = layers.Dense(3, activation="softmax")(normal_layer) 
# model = keras.Model(input, output)
# model.summary()


### Second model architecture
### Based on customized architecture from article in Scientific Reports:
# # Liu, S., Masurkar, A.V., Rusinek, H. et al. 'Generalizable deep learning model for early Alzheimer’s disease detection from structural MRIs'
# # Sci Rep 12, 17106 (2022). https://doi.org/10.1038/s41598-022-20674-x
# # Git hub: https://github.com/NYUMedML/CNN_design_for_AD/blob/master/models/models.py
# # License: AGPL-3.0 License
# #
# # Modified part of the code by translating the neural network code from PyTorch to Keras
# # Keras Group Normalization - relation to Instance Normalization:
# # If the number of groups is set to the input dimension (number of groups is equal to number of channels),
# # then this operation becomes identical to Instance Normalization

# input = layers.Input((cropdim, cropdim, cropdim, 1))

# First convolution block                                                                                                                                                                                
x = layers.Conv3D(filters=16, kernel_size=1)(input)					
x = layers.GroupNormalization(groups=4)(x)						
x = layers.ReLU()(x)									
x = layers.MaxPooling3D(pool_size=3, strides=2)(x)				

# Second convolution block - input dim: 16, output dim: 128     
x = layers.Conv3D(filters=128, kernel_size=3, dilation_rate=2)(x)	
x = layers.GroupNormalization(groups=16)(x)						
x = layers.ReLU()(x)									
x = layers.MaxPooling3D(pool_size=3, strides=2)(x)				

# Third convolution block - input dim: 128, output dim: 256
x = layers.Conv3D(filters=256, kernel_size=5, padding='same', dilation_rate=2)(x)   
x = layers.GroupNormalization(groups=128)(x)                                        
x = layers.ReLU()(x)                                                                
x = layers.MaxPooling3D(pool_size=3, strides=2)(x)                                  

# Fourth convolution block - input dim: 256, output dim: 256
x = layers.Conv3D(filters=256, kernel_size=3, padding='same', dilation_rate=2)(x)	
x = layers.GroupNormalization(groups=256)(x)					
x = layers.ReLU()(x)                                                          
x = layers.MaxPooling3D(pool_size=3, strides=2)(x)				

# Fully connected layer 		
x = layers.Flatten()(x)
x = layers.Dense(128)(x) # feat_dim
output = layers.Dense(3, activation="softmax")(x)
model = keras.Model(input, output)
model.summary()


### Third model architecture
### Based on a customized published architecture:
# # Böhle, M., Eitel, F., Weygandt, M., & Ritter, K. (2019). Layer-Wise Relevance Propagation 
# # for Explaining Deep Neural Network Decisions in MRI-Based Alzheimer's Disease Classification. 
# # Frontiers in Aging Neuroscience, 11, 194–194. https://doi.org/10.3389/fnagi.2019.00194
# #
# # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6685087/pdf/fnagi-11-00194.pdf
# # https://github.com/moboehle/Pytorch-LRP/tree/master/ADNI Training.ipynb
# # BSD-3-Clause license, copyright 2019 Moritz Böhle
# # Modified parts of code by translating the neural network from Pytorch to Keras

# input = layers.Input((cropdim, cropdim, cropdim, 1))

# # First convolution block        
# x = layers.Conv3D(filters=8, kernel_size=3)(input)				
# x = layers.BatchNormalization()(x)
# x = layers.ReLU()(x)
# x = layers.MaxPooling3D(pool_size=2)(x)                 			          

# # Second convolution block - input dim: 8, output dim: 16     
# x = layers.Conv3D(filters=16, kernel_size=3)(x)         			
# x = layers.BatchNormalization()(x)						
# x = layers.ReLU()(x)									
# x = layers.MaxPooling3D(pool_size=3)(x)					

# # Third convolution block - input dim: 16, output dim: 32
# x = layers.Conv3D(filters=32, kernel_size=3)(x)         			
# x = layers.BatchNormalization()(x)
# x = layers.ReLU()(x)
# x = layers.MaxPooling3D(pool_size=2)(x)                 			

# # Fourth convolution block - input dim: 32, output dim: 64
# x = layers.Conv3D(filters=64, kernel_size=3)(x)	        		
# x = layers.BatchNormalization()(x)
# x = layers.ReLU()(x)
# x = layers.MaxPooling3D(pool_size=3)(x)                	 		

# # Fully connected layer 		
# x = layers.Flatten()(x)
# x = layers.Dropout(dropoutrate)(x)
# x = layers.Dense(128)(x) # feat_dim
# x = layers.Dropout(dropoutrate)(x)
# output = layers.Dense(3, activation="softmax")(x)
# model = keras.Model(input, output)
# model.summary()

##### Compile model 
opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(loss=loss, optimizer=opt, metrics=['sparse_categorical_accuracy'])



# Acquire labels
# Filter by subject ID acquired from file name
print('length x_hold2: ',len(x_hold2))

x_filt_hold = []
for i in range(len(x_hold2)):
	pos = x_hold2[i].find('_ses-')
	x_filt_hold.append(x_hold2[i][(pos-16):pos])
print('length x_filt_hold: ', len(x_filt_hold))
strat_hold = stratimp[(stratimp['subj'].isin(x_filt_hold))] # note: this list contains each subject only once


# Load best model
model.load_weights(path_bestmodel)
	

##### Evaluate the model
results = model.evaluate(hold_generator, verbose=2)
print("test loss: ",results[0])
print("test acc: ",results[1])

# #### Model prediction
prediction = model.predict(hold_generator, verbose=2)

predprob = []
	
for i in range(len(prediction)):
	predprob.append(list(prediction[i]))
	
## create list of true labels
true = []
count1 = 0
count2 = 0

#print("length of hold_generator: ", len(hold_generator))
	
for i in range (len(hold_generator)):
	for j in range(batch_size): # 2023-12-29 SJEKK DENNE
		true.append(hold_generator[i][1][j])
		count1+=1
	count2+=1

predlabel = list(np.argmax(predprob, axis=1))

## Remaining subjects not included in hold_generator due to sample size not divisible with batch size
## Dictionary for partition and labels for external validation set, x_hold
x_remain = x_hold2[count1:]
remain_partition = {'remain':x_remain}

remain_params = {'dim': (imlength,imwidth,imheight),
         'batch_size': len(x_remain),
         'n_classes': 3,
         'n_channels': 1,
         'shuffle': False}

remain_generator = DataGenerator(remain_partition['remain'], labels, **remain_params)

## Evaluate the model for remaining subjects
r_results = model.evaluate(remain_generator, verbose=2)
print("test loss remain: ",r_results[0])
print("test acc remain: ",r_results[1])

##### Model prediction
rprediction = model.predict(remain_generator, verbose=2)
rpredprob = []

for i in range(len(rprediction)):
	rpredprob.append(list(rprediction[i]))


## create list of true labels
rtrue = []
rcount1 = 0
rcount2 = 0
	
for i in range (len(remain_generator)):
	for j in range(len(x_remain)):
		rtrue.append(remain_generator[i][1][j])
		rcount1+=1
	rcount2+=1

rtrue = np.array(rtrue)
rpredlabel = list(np.argmax(rpredprob, axis=1))



# Creating list of criteria to sort subgroups	
subject = []
test_labels = []
gender = []
education = []
mmse = []
ethn = []
agegroup = []
	
	
for i in range(len(x_filt_hold)):
	subject.append(x_filt_hold[i])
	test_labels.append(strat_hold[strat_hold['subj']==x_filt_hold[i]].iloc[0]['diag_let'])
	gender.append(strat_hold[strat_hold['subj']==x_filt_hold[i]].iloc[0]['sex'])
	education.append(strat_hold[strat_hold['subj']==x_filt_hold[i]].iloc[0]['edu'])
	mmse.append(strat_hold[strat_hold['subj']==x_filt_hold[i]].iloc[0]['mmse'])
	ethn.append(strat_hold[strat_hold['subj']==x_filt_hold[i]].iloc[0]['ethn'])
	agegroup.append(strat_hold[strat_hold['subj']==x_filt_hold[i]].iloc[0]['age'])


test_predlabel = list(np.concatenate((predlabel,rpredlabel)).tolist())
print("test_pred: ")
print(test_predlabel)
test_true = list(np.concatenate((true,rtrue)).tolist())
print("test_true: ")
print(test_true)
test_predprob = list(np.concatenate((predprob,rpredprob)).tolist())
print("test_predprob: ")
print(test_predprob)
print("length test_pred: ", len(test_predlabel))
print("length test_true: ", len(test_true))
print("length test_predprob: ", len(test_predprob))

strattest = pd.DataFrame(list(zip(subject, test_labels, gender, agegroup, education, mmse, ethn, test_predlabel, test_true, test_predprob)), columns = ['subj','diag_let','sex', 'age','edu', 'mmse', 'ethn', 'pred', 'true', 'prob'])
print("length of strattest: ", len(strattest))
strattest.to_csv('MultiClassifier/Statistics/Strattest/'+outputinfo+'_strattest.csv') # Replace with actual file location


# # Global evaluation metrics

print("Global evaluation metrics:")
cm = confusion_matrix(test_true, test_predlabel)
print("confusion matrix global test set: ")
print(cm)


try:
	roc_auc_score(test_true, test_predprob, multi_class='ovr')
	
	g_acc,g_bal_acc,g_recall_n,g_recall_w,g_spec_n,g_spec_w,g_prec_n,g_prec_w,g_F1_n,g_F1_w,g_MCC = evalmetric(test_true, test_predlabel, metrics)
	
	g_rocauc_ovr = roc_auc_score(test_true, test_predprob, multi_class='ovr')
	g_rocauc_ovo = roc_auc_score(test_true, test_predprob, multi_class='ovo')
	
	print("accuracy: ", g_acc)
	print("balanced accuracy: ", g_bal_acc)
	print("recall (None): ", g_recall_n)
	print("recall (weighted): ", g_recall_w)
	print("specificity (None): ", g_spec_n)
	print("specificity (weighted): ", g_spec_w)
	print("precision (None): ", g_prec_n)
	print("precision (weighted): ", g_prec_w)
	print("F1 (None): ", g_F1_n)
	print("F1 (weighted): ", g_F1_w)
	print("MCC: ", g_MCC)
	print("ROC AUC, ovr: ", g_rocauc_ovr)
	print("ROC AUC, ovo: ", g_rocauc_ovo)

except ValueError:
	print("ValueError for global testset")
	pass



#### Subgroup filtering ###		
# filter per sex
is_female = strattest[(strattest['sex'] == 'F')]
print("length of is_female: ", len(is_female))
is_male = strattest[(strattest['sex'] == 'M')]
print("length of is_male: ", len(is_male))

## Not in use ## Future Exploration ##
# # filter per education
# edu_low = strattest[(strattest['edu'] == 'low')]
# print("length of edu_low: ", len(edu_low))
# edu_high = strattest[(strattest['edu'] == 'high')]
# print("length of edu_high: ", len(edu_high))
# # filter per mmse
# mmse_low = strattest[(strattest['mmse'] == 'low')]
# print("length of mmse_low: ", len(mmse_low))
# mmse_high = strattest[(stratsort['mmse'] == 'high')]
# print("length of mmse_high: ", len(mmse_high))
# # filter per ethnicity
# ethn_nonhisp = strattest[(strattest['ethn'] == 'Not Hisp/Latino')]
# print("length of ethn_nonhisp: ", len(ethn_nonhisp))
# ethn_hisp = strattest[(strattest['ethn'] == 'Hisp/Latino')]
# print("length of ethn_hisp: ", len(ethn_hisp))
# ethn_unknown = strattest[(strattest['ethn'] == 'Unknown')]
# print("length of ethn_unknown: ", len(ethn_unknown))
## ## ## ##

# Evaluation metrics sex
print("Evaluation metrics sex")

# Analysis gender = 'F'
print("Results for female:")
true_female = []
for i in range(len(is_female)):
	true_female.append(is_female.iloc[i]['true'])
print("true_female list: ", true_female)
pred_female = []
for i in range(len(is_female)):
	pred_female.append(is_female.iloc[i]['pred'])
print("pred_female list: ", pred_female)
predprob_female = []
for i in range(len(is_female)):
	predprob_female.append(is_female.iloc[i]['prob'])
print("predprob_female list: ", predprob_female)
female_cm = confusion_matrix(true_female, pred_female)
print("female confusion matrix: ")
print(female_cm)

try:
	roc_auc_score(true_female, predprob_female, multi_class='ovr')
	
	f_acc,f_bal_acc,f_recall_n,f_recall_w,f_spec_n,f_spec_w,f_prec_n,f_prec_w,f_F1_n,f_F1_w,f_MCC = evalmetric(true_female, pred_female, metrics)
	
	f_rocauc_ovr = roc_auc_score(true_female, predprob_female, multi_class='ovr')
	f_rocauc_ovo = roc_auc_score(true_female, predprob_female, multi_class='ovo')
	
	print("female accuracy: ", f_acc)
	print("female balanced accuracy: ", f_bal_acc)
	print("female recall (None): ", f_recall_n)
	print("female recall (weighted): ", f_recall_w)
	print("female specificity (None): ", f_spec_n)
	print("female specificity (weighted): ", f_spec_w)
	print("female precision (None): ", f_prec_n)
	print("female precision (weighted): ", f_prec_w)
	print("female F1 (None): ", f_F1_n)
	print("female F1 (weighted): ", f_F1_w)
	print("female MCC: ", f_MCC)
	print("female ROC AUC, ovr: ", f_rocauc_ovr)
	print("female ROC AUC, ovo: ", f_rocauc_ovo)

except ValueError:
	print("ValueError for female testset")
	pass

	
# Analysis sex = 'M'
print("Results for male:")
true_male = []
for i in range(len(is_male)):
	true_male.append(is_male.iloc[i]['true'])
print("true_male list: ", true_male)
pred_male = []
for i in range(len(is_male)):
	pred_male.append(is_male.iloc[i]['pred'])
print("pred_male list: ", pred_male)
predprob_male = []
for i in range(len(is_male)):
	predprob_male.append(is_male.iloc[i]['prob'])
print("predprob_male list: ", predprob_male)
male_cm = confusion_matrix(true_male, pred_male)
print("Male confusion matrix: ")
print(male_cm)

try:
	roc_auc_score(true_male, predprob_male, multi_class='ovr')
	
	m_acc,m_bal_acc,m_recall_n,m_recall_w,m_spec_n,m_spec_w,m_prec_n,m_prec_w,m_F1_n,m_F1_w,m_MCC = evalmetric(true_male, pred_male, metrics)
	
	m_rocauc_ovr = roc_auc_score(true_male, predprob_male, multi_class='ovr')
	m_rocauc_ovo = roc_auc_score(true_male, predprob_male, multi_class='ovo')
	
	print("male accuracy: ", m_acc)
	print("male balanced accuracy: ", m_bal_acc)
	print("male recall (None): ", m_recall_n)
	print("male recall (weighted): ", m_recall_w)
	print("male specificity (None): ", m_spec_n)
	print("male specificity (weighted): ", m_spec_w)
	print("male precision (None): ", m_prec_n)
	print("male precision (weighted): ", m_prec_w)
	print("male F1 (None): ", m_F1_n)
	print("male F1 (weighted): ", m_F1_w)
	print("male MCC: ", m_MCC)
	print("male ROC AUC, ovr: ", m_rocauc_ovr)
	print("male ROC AUC, ovo: ", m_rocauc_ovo)


except ValueError:
	print("ValueError for male testset")
	pass


print("Completed model evaluation testset: ", outputinfo)

