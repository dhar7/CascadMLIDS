#%%
import pandas as pd
from tensorflow.keras.utils import get_file
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

path1 =  "C:/Users/User/Desktop/Research/Akhand Sir/VANET-IDS/Car Hacking Dataset/DoS_dataset.csv"
path2 =  "C:/Users/User/Desktop/Research/Akhand Sir/VANET-IDS/Car Hacking Dataset/Fuzzy_dataset.csv"
path3 =  "C:/Users/User/Desktop/Research/Akhand Sir/VANET-IDS/Car Hacking Dataset/gear_dataset.csv"
path4 =  "C:/Users/User/Desktop/Research/Akhand Sir/VANET-IDS/Car Hacking Dataset/RPM_dataset.csv"
path5 =  "C:/Users/User/Desktop/Research/Akhand Sir/VANET-IDS/Car Hacking Dataset/normal_run_data.txt"


# READING THE CSV FILES AND NECESSARY ERRO HANDLING
df1 = pd.read_csv(path1,  header=None)

df2 = pd.read_csv(path2,  on_bad_lines='skip')
df2.columns = range(df2.shape[1])

df3 = pd.read_csv(path3, on_bad_lines='skip')
df3.columns = range(df3.shape[1])

df4 = pd.read_csv(path4, on_bad_lines='skip')
df4.columns = range(df4.shape[1])

df0 = pd.read_csv(path5, sep=' ', header=None)
df0.drop(df0.columns[[0,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,20,21,22]], axis=1, inplace=True)
df0.columns = range(df0.shape[1])
df0[11] = 0

#to see counts of unique values in a column ====>    print(df.iloc[:,11].value_counts())

# SEPARATING IN INDIVIDUAL CLASS
normal = df0.loc[df0.iloc[:,11] == 0]
dos = df1.loc[df1.iloc[:,11] == 'T']
fuzzy = df2.loc[df2.iloc[:,11] == 'T']
gear = df3.loc[df3.iloc[:,11] == 'T']
rpm = df4.loc[df4.iloc[:,11] == 'T']


 
# ASSIGING LABELS
dos[11] = 1
fuzzy[11] = 2
gear[11] = 3
rpm[11] = 4

dos_gear_rpm = pd.concat([dos,gear,rpm])




def preprocessing( df ):

    #DROPPING ROWS WITH NULL AND RESETTING INDEX
    df = df.dropna()  
    df = df.reset_index(drop=True)
    
    #DROPPING LABEL AND RESETTING INDEX
    label = df.iloc[:,11]
    df.drop(df.columns[[11]], axis=1, inplace=True) 
    df.columns = range(df.columns.size)
    
    #DROPPING IRRELEVENT COLUMN AND RESETTING INDEX
    df.drop(df.columns[[2]], axis=1, inplace=True)
    df.columns = range(df.columns.size)
    
    
    #CONVERTING HEX VALUES INTO DECIMAL
    for j in range(1,10):
        h = df.iloc[:,j]
        i = [int(x, 16) for x in h]
        df.drop(j, axis = 1, inplace = True)
        df.insert(j,'',i)
        df.columns = range(df.shape[1])
       
    
    ## NORMALIZATION 
    #normalized_df=(df-df.min())/(df.max()-df.min())
    #normalized_df[10] = label
    #normalized_df = normalized_df.fillna(0)
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())    
    #df[10] = label  
    df.insert(10,'',label)
    df.columns = range(df.shape[1])
    df = df.fillna(0)
    return df
    
    #if ((len(fuzzy.iloc[:,10].unique())) <= 1): 
    #    return df
    
    #dos = gear = rpm = pd.DataFrame()
    #if ((len(fuzzy.iloc[:,10].unique())) > 1):
        
    #,dos,gear,rpm
    
        

normal       = preprocessing(normal)
fuzzy        = preprocessing(fuzzy)
dos_gear_rpm  = preprocessing(dos_gear_rpm)
dos  = dos_gear_rpm.loc[dos_gear_rpm.iloc[:,10] == 1]
gear = dos_gear_rpm.loc[dos_gear_rpm.iloc[:,10] == 3]
rpm  = dos_gear_rpm.loc[dos_gear_rpm.iloc[:,10] == 4]


#gear   = preprocessing(gear, 1)
#rpm    = preprocessing(rpm, 1)


from sklearn.model_selection import train_test_split

#dos_gear_rpm = pd.concat( [dos,gear,rpm] )

normal_train, normal_test = train_test_split(normal, test_size=0.2, random_state=69)
dos_train, dos_test       = train_test_split(dos, test_size=0.2, random_state=69)
fuzzy_train, fuzzy_test   = train_test_split(fuzzy, test_size=0.2, random_state=18)
gear_train, gear_test     = train_test_split(gear, test_size=0.2, random_state=69)
rpm_train, rpm_test       = train_test_split(rpm, test_size=0.2, random_state=69)
#dos_gear_rpm_train,dos_gear_rpm_test = train_test_split(dos_gear_rpm, test_size=0.2)

train = pd.concat([normal_train,dos_train,fuzzy_train,gear_train,rpm_train])
test  = pd.concat([normal_test,dos_test,fuzzy_test,gear_test,rpm_test])

#train = pd.concat([normal_train,fuzzy_train,dos_gear_rpm_train])
#test  = pd.concat([normal_test,fuzzy_test,dos_gear_rpm_test])

#train = pd.concat([normal_train,fuzzy_train,dos_train])
#test  = pd.concat([normal_test,fuzzy_test,dos_test])

train = train.sample(frac = 1)
test = test.sample(frac = 1)
train = train.reset_index(drop=True)
test  = test.reset_index(drop=True)

#LABEL ENCODING
#from sklearn import preprocessing

#label_encoder = preprocessing.LabelEncoder()
#train[10] = label_encoder.fit_transform(train[10])
#test[10]  = label_encoder.fit_transform(test[10])


## DIVIDING X and Y
train_y = train.iloc[:,10]
test_y  = test.iloc[:,10]

train.drop(train.columns[[10]], axis=1, inplace=True) 
test.drop(test.columns[[10]], axis=1, inplace=True) 

train_x = train
test_x  = test
    
# ONE HOT ENCODING OF Y
train_y = pd.get_dummies(train_y)
test_y  = pd.get_dummies(test_y)

#DATA SPLITTING AND CONVERTING INTO ARRAY
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.20, random_state=42)      
#import numpy as np
train_x = np.asarray(train_x).astype('float32')
train_y = np.asarray(train_y).astype('float32')
test_x = np.asarray(test_x).astype('float32')
test_y = np.asarray(test_y).astype('float32')

#%%
#MODEL TRAINING AND EVALUATION


model = Sequential()
model.add(Dense(50, input_dim=train_x.shape[1], activation='relu'))
model.add(Dense(20, input_dim=train_x.shape[1], activation='relu'))
model.add(Dense(10, input_dim=train_x.shape[1], activation='relu'))
#model.add(Dense(1, kernel_initializer='normal'))
model.add(Dense(5,activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,  patience=5, verbose=1, mode='auto',restore_best_weights=True)
model.summary()

checkpoint_filepath = 'C:/Users/User/Desktop/Research/Akhand Sir/VANET-IDS/Car Hacking Dataset/best_model3'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(train_x,train_y,validation_data=(test_x,test_y), verbose=2,epochs=7, callbacks=[model_checkpoint_callback])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

pred = model.predict(test_x)
pred2 = np.argmax(pred,axis=1)
y_eval = np.argmax(test_y,axis=1)
score = metrics.accuracy_score(y_eval, pred2)
print("score: {}".format(score))     

#model.save("C:/Users/User/Desktop/Research/Akhand Sir/VANET-IDS/Car Hacking Dataset/100epochs")
hist_10_best_model3 = pd.DataFrame(history.history)
#%%
hist_10_best_model3.to_csv("C:/Users/User/Desktop/hist_10_best_model3.csv")
#%%
model = tf.keras.models.load_model("C:/Users/User/Desktop/Research/Akhand Sir/VANET-IDS/Car Hacking Dataset/best_model3")
pred = model.predict(test_x)
pred2 = np.argmax(pred,axis=1)
y_eval = np.argmax(test_y,axis=1)
score = metrics.accuracy_score(y_eval, pred2)
print("score: {}".format(score))  

index = []
confusion = [[],[],[],[],[]]
for i in range(test_x.shape[0]):
    if (y_eval[i] != pred2[i]):
        true = y_eval[i]
        predicted = pred2[i]
        index.append(i)
        confusion[true].append(predicted)
count = [{},{},{},{},{} ]     

for i in range(5):
    dictionary = count[i]
    for j in range(5):
        c = confusion[i].count(j)
        dictionary[j] = c
        


#%%

