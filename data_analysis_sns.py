#breast cancer diagnostic data using seaborn 
#uci ML repository 
#train classification models -- detect if a given tumor is malginant or benign 

#FNA of a breast mass -- digitalized image 
#target variable, m = malignant, b = benign

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

data = pd.read_csv('')
data.head()
data.info() #33columns ,IDcolumn(unique) (may need to get rid of)
            #class labels are diagnosis (target variable)

columns = data.columns  
print(columns)

y = data.diagnosis
drop_column = ['id', 'diagnosis','Unnamed 32']
x = data.drop(drop_column, axis = 1)
x.head()

#look for class imbalance issues 
#observe dist. 

ax = sns.countplot(y, label='Count')
B, M = y.value_counts()
print('Number of B',B)
print('Number of M', M)

#B=357
#M=212
#more cases of b than m, clear issue with imbalance

x.describe()


#analyze feature matrix w/ violin plot
#violin plot used for numeric data (similar to boxplot)

#standardize data 
data  = x 
data_sx = (data-data.mean())/data.std()

data = pd.concat([y, data_std.iloc[:,0:10]]axis = 1)
data=pd.melt(data, id_vars = 'diagnosis',
            var_name = 'features',
            value_name = 'value'
         
plt.figure(figsize =(10,10))
sns.violinplot(x = 'features', y = 'value',
                hue = 'diagnosis', 
                data=data,
                split=True,
                inner='quartile')
                
plt.xticks(rotation=45);

sns.boxplot(x = 'features', y ='value', hue = 'diagnosis', 
            data=data)
            
plt.xticks(rotation=45);


#drop one of the correlated columns so we can avoid negative predictive accuracy 

sns.jointplot(x.loc[:,'concavity_worst'],
                x.loc[:,'concave points_worst'],
                kind = 'regg',
                color ='#ce1414');
                    
#swarmplots -- better for smaller plots vs violin (shows each data point)

sns.set(style = 'whitegrid', palette = 'muted')
#copy standardized data
data  = x 
data_sx = (data-data.mean())/data.std()

data = pd.concat([y, data_std.iloc[:,0:10]]axis = 1)
data=pd.melt(data, id_vars = 'diagnosis',
            var_name = 'features',
            value_name = 'value'
         
plt.figure(figsize =(10,10))
sns.swarplot(x = 'features', y = 'value', #changetoswarmplot
                hue = 'diagnosis', 
                data=data)
#only need color & x/y values and 'data'                
plt.xticks(rotation=45);

#observing all pair-wise conditions
#create heatmap

f, ax = plt.subplots(figsize = (18,18))
sns.heatmap(x.corr(), 
            annot=True, 
            linewidth = .5, 
            fmt='.1f',  
            ax = ax);
            
