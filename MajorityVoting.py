# -*- coding: utf-8 -*-
"""

@author: SolveigCodes

url: https://github.com/SolveigCodes/AI_ensembles_for_AD
"""

# Depends on strattest files produced by MultiClassifier_test_original.py
# This script require that the strattest files contain the names of the architecture and model in the filename
# Results from five models per three architectures are used in this script
# All 15 strattest files need to be in the same folder
# Architectures: "Custom1", "Custom2", "RN18" (as in ResNet-18)
# Refer to script MultiClassifier_traincv.py for further details about architectures
#
# A solution for tie-breaks is included, which favorizes the "worst" diagnosis


##### Libraries

import os
import pandas as pd
import time

##### Global variables

timestr = time.strftime("%Y-%m-%d_%H%M")


##### File preparation

datapath = '/Results/MultiClassifier/Strattest/'

# Use one of the strattest files as a start for a base file 
basefile = '/Results/MultiClassifier/Strattest/Custom1_f1_strattest.csv'

majorityfile = '/Results/MultiClassifier/MajorityVote/majority_pred_'+timestr+'.csv'

columns = ['subj','diag_let','sex','edu','mmse','ethn','true']

basedf = pd.read_csv(basefile,index_col=0)
preddf = basedf[columns]


archs = ['Custom1', 'Custom2', 'RN18']
folds = ['f1', 'f2', 'f3', 'f4', 'f5'] # representing models that are saved after training with 5-fold crossvalidation

predcols=[]


# Naming prediction columns according using architecture and model names
# All strattest files must be in the same folder for this to work
for path, subdirs, files in os.walk(datapath):
    subdirs.sort() 
    for name in files:
        if ('strattest' in name):
            for arch in archs:
                if (arch in name):
                    for f in folds:
                        if(f in name):
                            tempdf=pd.read_csv(datapath+name,index_col=0)
                            preddf=pd.concat([preddf,tempdf[['pred']]], axis=1).rename(columns = {'pred': ('pred_'+arch+'_'+f)})               
                            predcols.append('pred_'+arch+'_'+f)


##### Enable filtering per architecture 
# Each archtiecture has 5 columns (folds 1-5) e.g. 'pred_RN18_f1'
Custom1cols=[]
Custom2cols=[]
RN18cols=[]

i=0
for i in range(len(predcols)):
    if ('Custom1' in predcols[i]):
        Custom1cols.append(predcols[i])
    elif('Custom2' in predcols[i]):
        Custom2cols.append(predcols[i])
    elif('RN18' in predcols[i]):
        RN18cols.append(predcols[i])
    else:
        pass


##### Majority vote predictions

#Custom1
# Add a column for writing majority vote results for architecture
preddf['Custom1pred']=pd.Series(dtype='int64')

# Create lists to be able to count the number of predictions belonging to each class
CN=[]
MCI=[]
AD=[]

# Determine what is the majority class among predictions from all five models
for i in range(0,len(preddf)):
    for j in range (len(Custom1cols)):
        if (preddf[Custom1cols[j]].iloc[i]==0):
            CN.append(preddf[Custom1cols[j]].iloc[i])
        elif(preddf[Custom1cols[j]].iloc[i]==1):
            MCI.append(preddf[Custom1cols[j]].iloc[i])
        else:
            AD.append(preddf[Custom1cols[j]].iloc[i])
    print("CN: ", CN)
    print(len(CN))
    print("MCI: ",MCI)
    print(len(MCI))
    print("AD: ", AD)
    print(len(AD))
    if (len(CN)>len(MCI)) and (len(CN)>len(AD)):
        #print("CN is the majority class")
        preddf.loc[i,'Custom1pred']=0
    elif(len(MCI)>len(CN)) and (len(MCI)>len(AD)):
       #print("MCI is the majority class")
        preddf.loc[i,'Custom1pred']=1
    elif(len(AD)>len(CN) and len(AD)>len(MCI)):
        #print("AD is the majority class")
        preddf.loc[i,'Custom1pred']=2
    elif(len(CN)==len(MCI) and len(AD)<len(CN)):
        print(i," Tie-break CN and MCI. MCI is the majority class")
        preddf.loc[i,'Custom1pred']=1
    elif(len(CN)==len(AD) and len(MCI)<len(CN)):
        print(i," Tie-break CN and AD. AD is the majority class")
        preddf.loc[i,'Custom1pred']=2
    elif(len(MCI)==len(AD) and len(CN)<len(MCI)):
        print(i," Tie-break MCI and AD. AD is the majority class")
        preddf.loc[i,'Custom1pred']=2
    else:
        pass
    
    CN=[]
    MCI=[]
    AD=[]



# Custom2
# Add a column for writing majority vote results for architecture
preddf['Custom2pred']=pd.Series(dtype='int64')

# Create lists to be able to count the number of predictions belonging to each class
CN=[]
MCI=[]
AD=[]

# Determine what is the majority class among predictions from all five models               
for i in range(0,len(preddf)):
    for j in range (len(Custom2cols)):
        if (preddf[Custom2cols[j]].iloc[i]==0):
            CN.append(preddf[Custom2cols[j]].iloc[i])
        elif(preddf[Custom2cols[j]].iloc[i]==1):
            MCI.append(preddf[Custom2cols[j]].iloc[i])
        else:
            AD.append(preddf[Custom2cols[j]].iloc[i])
    print("CN: ", CN)
    print(len(CN))
    print("MCI: ",MCI)
    print(len(MCI))
    print("AD: ", AD)
    print(len(AD))
    if (len(CN)>len(MCI)) and (len(CN)>len(AD)):
        #print("CN is the majority class")
        preddf.loc[i,'Custom2pred']=0
    elif(len(MCI)>len(CN)) and (len(MCI)>len(AD)):
       #print("MCI is the majority class")
        preddf.loc[i,'Custom2pred']=1
    elif(len(AD)>len(CN) and len(AD)>len(MCI)):
        #print("AD is the majority class")
        preddf.loc[i,'Custom2pred']=2
    elif(len(CN)==len(MCI) and len(AD)<len(CN)):
        print(i," Tie-break CN and MCI. MCI is the majority class") 
        preddf.loc[i,'Custom2pred']=1
    elif(len(CN)==len(AD) and len(MCI)<len(CN)):
        print(i," Tie-break CN and AD. AD is the majority class")
        preddf.loc[i,'Custom2pred']=2
    elif(len(MCI)==len(AD) and len(CN)<len(MCI)):
        print(i," Tie-break MCI and AD. AD is the majority class")
        preddf.loc[i,'Custom2pred']=2
    else:
        pass
    
    CN=[]
    MCI=[]
    AD=[]



# RN18
# Add a column for writing majority vote results for architecture
preddf['RN18pred']=pd.Series(dtype='int64')

# Create lists to be able to count the number of predictions belonging to each class
CN=[]
MCI=[]
AD=[]

# Determine what is the majority class among predictions from all five models               
for i in range(0,len(preddf)):
    for j in range (len(RN18cols)):
        if (preddf[RN18cols[j]].iloc[i]==0):
            CN.append(preddf[RN18cols[j]].iloc[i])
        elif(preddf[RN18cols[j]].iloc[i]==1):
            MCI.append(preddf[RN18cols[j]].iloc[i])
        else:
            AD.append(preddf[RN18cols[j]].iloc[i])
    print("CN: ", CN)
    print(len(CN))
    print("MCI: ",MCI)
    print(len(MCI))
    print("AD: ", AD)
    print(len(AD))
    if (len(CN)>len(MCI)) and (len(CN)>len(AD)):
        #print("CN is the majority class")
        preddf.loc[i,'RN18pred']=0
    elif(len(MCI)>len(CN)) and (len(MCI)>len(AD)):
       #print("MCI is the majority class")
        preddf.loc[i,'RN18pred']=1
    elif(len(AD)>len(CN) and len(AD)>len(MCI)):
        #print("AD is the majority class")
        preddf.loc[i,'RN18pred']=2
    elif(len(CN)==len(MCI) and len(AD)<len(CN)):
        print(i," Tie-break CN and MCI. MCI is the majority class")
        preddf.loc[i,'RN18pred']=1
    elif(len(CN)==len(AD) and len(MCI)<len(CN)):
        print(i," Tie-break CN and AD. AD is the majority class")
        preddf.loc[i,'RN18pred']=2
    elif(len(MCI)==len(AD) and len(CN)<len(MCI)):
        print(i," Tie-break MCI and AD. AD is the majority class")
        preddf.loc[i,'RN18pred']=2
    else:
        pass
    
    CN=[]
    MCI=[]
    AD=[]


   
# Write results to a file based on the basedf dataframe
preddf.to_csv(majorityfile)

