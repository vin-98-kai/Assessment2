# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:43:14 2022

@author: Calvin
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import missingno as msno
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential,Input
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import TensorBoard
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization

#%% Functions
def cramers_corrected_stat(confusion_matrix):
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n    
    r,k = confusion_matrix.shape    
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% Static
CSV_PATH = os.path.join(os.getcwd(),'data','Train.csv')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'models','model.h5')
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(), 'logs',log_dir)
MMS_FNAME = os.path.join(os.getcwd(),'models','mms_fname.pkl')
OHE_PICKLE_PATH = os.path.join(os.getcwd(),'models','ohe_fname.pkl')
DOM_ENCODER_PATH = os.path.join(os.getcwd(),'models','dom_encoder.pkl')
EDUC_ENCODER_PATH = os.path.join(os.getcwd(),'models','edu_encoder.pkl')
MONTH_ENCODER_PATH = os.path.join(os.getcwd(),'models','month_encoder.pkl')
DEF_ENCODER_PATH = os.path.join(os.getcwd(),'models','default_encoder.pkl')
JOB_ENCODER_PATH = os.path.join(os.getcwd(),'models','job_type_encoder.pkl')
MARI_ENCODER_PATH = os.path.join(os.getcwd(),'models','marital_encoder.pkl')
COM_ENCODER_PATH = os.path.join(os.getcwd(),'models','com_type_encoder.pkl')
H_LOAN_ENCODER_PATH = os.path.join(os.getcwd(),'models','h_loan_encoder.pkl')
P_LOAN_ENCODER_PATH = os.path.join(os.getcwd(),'models','p_loan_encoder.pkl')
PREVCAMP_ENCODER_PATH = os.path.join(os.getcwd(),'models','p_camp_encoder.pkl')

# EDA
# Step 1: Data Loading
df = pd.read_csv(CSV_PATH)

# Step 2: Data Inspection
df.info() # can see several NaNs
df.isna().sum() # alot of Nans in days_since_prev_campaign_contact,customer_age
#,balance,last_contact_duration,marital,personal_loan,num_contacts_in_campaign
df.duplicated().sum() # no duplicates at this point
temp_view=df.describe().T

# removing id not useful month; lack of data against Target and 
#days_since_prev_campaign_contact because too many NaNs
df = df.drop(columns=['id','days_since_prev_campaign_contact'])

cat_column = df.columns[df.dtypes=='object']
cont_column = df.columns[(df.dtypes!='object')]

for i in cont_column: # continous
    plt.figure()
    sns.distplot(df[i])
    plt.show()

# for i in cat_column: # categorical
#     plt.figure(figsize=(14,12))
#     sns.countplot(df[i])
#     plt.show()

df.groupby(['job_type','education']).agg({'job_type':'count'})
df.groupby(['education','term_deposit_subscribed']).agg({'job_type':'count'})

for i in cat_column:
    plt.figure(figsize=(13,10))
    sns.countplot(df[i],hue=df['term_deposit_subscribed'])
    plt.show() # plot against Target

msno.bar(df) #to visualize the NaNs in the data 
msno.matrix(df) #to visualize the NaNs in the data 

#%% Regression Analysis
# Cramer's V Categorical VS Categorical
# to find correlation between the multiple categorical data
for i in cat_column:
    print(i)
    confusion_mat = pd.crosstab(df[i],df['term_deposit_subscribed']).to_numpy()
    print(cramers_corrected_stat(confusion_mat))

# Step 3: Data Cleaning
# removed three columns id,days_since_prev_campaign_contact,month
# convert categorical columns to integers

df_dummy = df.copy() # to duplicate data

le = LabelEncoder()

paths = [JOB_ENCODER_PATH,MARI_ENCODER_PATH,EDUC_ENCODER_PATH,
         DEF_ENCODER_PATH,H_LOAN_ENCODER_PATH,P_LOAN_ENCODER_PATH,
         COM_ENCODER_PATH,PREVCAMP_ENCODER_PATH,MONTH_ENCODER_PATH,
         DOM_ENCODER_PATH]

for index,i in enumerate(cat_column):
    temp = df_dummy[i]
    temp[temp.notnull()] = le.fit_transform(temp[temp.notnull()])
    df_dummy[i] = pd.to_numeric(temp,errors='coerce')
    with open(paths[index],'wb') as file:
        pickle.dump(le,file)

# Dealing with NaNs
knn_imp = KNNImputer()
df_dummy = knn_imp.fit_transform(df_dummy)
df_dummy = pd.DataFrame(df_dummy)
df_dummy.columns = df.columns

df_dummy.isna().sum() # check for NaNs
df_dummy.duplicated().sum() # check for duplicates

for i in df_dummy: # rounding decimals in dataframe
    df_dummy[i] = np.floor(df_dummy[i])

# Step 4: Feature Selection
for i in cat_column:
    print(i)
    confusion_mat=pd.crosstab(df_dummy[i],
    df_dummy['term_deposit_subscribed']).to_numpy()
    print(cramers_corrected_stat(confusion_mat))

for con in cont_column:
    print(con)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df_dummy[con],axis=-1),
           df_dummy['term_deposit_subscribed'])
    print(lr.score(np.expand_dims(df_dummy[con],axis=-1),
                   df_dummy['term_deposit_subscribed']))

# Selecting continous variables because they have higher lscore
X = df_dummy.loc[:,['customer_age','balance','last_contact_duration',
                 'num_contacts_in_campaign','num_contacts_prev_campaign',
                 'day_of_month']]
y = df_dummy['term_deposit_subscribed']
    
# Step 5: Data Preprocessing

# Features Scaling
mms = MinMaxScaler()
X = mms.fit_transform(X)
with open(MMS_FNAME,'wb') as file:
    pickle.dump(mms,file)

ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y,axis=-1))
with open(OHE_PICKLE_PATH,'wb') as file:
    pickle.dump(ohe,file)

#%% Model Development
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,
                                                 random_state=(123))

nb_features = np.shape(X)[1:]
nb_class = len(np.unique(y))

model = Sequential()
model.add(Input(shape=(nb_features)))
model.add(Dense(32, activation='relu',name='Hidden_Layer1'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu',name='Hidden_Layer2'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(nb_class,activation='softmax',name='output_layer'))
model.summary()

plot_model(model)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics ='acc')

# hist=model.fit(X_train,y_train,batch_size=64,
#                validation_data=(X_test,y_test),
#                epochs=10)

# callback
tensorboard_callback=TensorBoard(log_dir=LOG_FOLDER_PATH)
early_stopping_callback = EarlyStopping(monitor='loss',patience=3)

hist = model.fit(X_train,y_train,batch_size=64,validation_data=(X_test,y_test),
                  epochs=10,callbacks=early_stopping_callback)

# Model Training
hist.history.keys()
training_loss =  hist.history['loss']
training_acc = hist.history['acc']
validation_acc = hist.history['val_acc']
validation_loss = hist.history['val_loss']

plt.figure()
plt.plot(training_loss)
plt.plot(validation_loss)
plt.legend('train_loss','val_loss')
plt.show()

plt.figure()
plt.plot(training_acc)
plt.plot(validation_acc)
plt.legend(['train_acc','val_acc'])
plt.show()

#%% Model Evaluation
results = model.evaluate(X_test,y_test) # x_test = loss, y_test = accuracy
print(results)

pred_y = np.argmax(model.predict(X_test),axis=1)
true_y = np.argmax(y_test,axis=1)

cm = confusion_matrix(true_y,pred_y)
cr = classification_report(true_y,pred_y)
print(cm)
print(cr)

#Save model
model.save(MODEL_SAVE_PATH)

plot_model(model,show_shapes=True,show_layer_names=(True))

#%% Discussion
# distplot for cont_column:
# the customer age shows the estimate age ranging from 19 to 80+.
# the balance negative number but it will be remove 
# the day_of_month not equally distributed and lean more towards the 20th month
# the last_contacts_duration shows the estimate range of -1 to 1000+
# the num_contacts_in_campaign shows the estimate range of 0 to 10+
# the day_since_prev_campaign_contact shows the estimate range of -1 to 400+
# the num_contacts_in_prev_campaign shows the estimate range of 0 to 10+
# the term_deposit_subscribed two distribution is 0.0&1.0 because it was 
#converted from categorical can speculate that 1:recruited, 0:not recruited.

# countplot for cat_column:
# the jobtype for blue-collar is the highest
# married in marital status is the highest 
# in education the secondary educated is the highest
# in default the amount of registered customer is higher than those registered
# in housing loan there are more customer with housing loan
# in personal loan there are more customer without personal loan
# in communication_type there are more customer with cellular
# in month there seem to be no correlation with other columns but the highest 
#is May
# in the prev_campaign_outcome the highest is unknown

# removed multiple column such as id because its not useful data and 
#days_since_prev_campaign_contact it contain alot of NANs.
# Also dropping them at the BEGINNING to save time 
# accuracy/correlation of categorical columns is weak thats why I will be 
#selecting continous colmn. 