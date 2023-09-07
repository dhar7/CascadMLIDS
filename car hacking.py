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

# checking unique values and min max avg
dfs = [dos]#,fuzzy,gear,rpm,normal]
name = ['dos']#,'fuzzy','gear','rpm','normal']
for i in range(len(dfs)):
    for j in range(11):
        #print( name[i] + "__column" + str(j) + "  unique Values :" + str(len(dfs[i].iloc[:,j].unique())))
        print( name[i] + "__column" + str(j)+"  min:" + str(dfs[i].iloc[:,j].min() ) + 
                                            "   max:" + str(dfs[i].iloc[:,j].max() ) +
                                            "   avg:" + str(dfs[i].iloc[:,j].mean()) +
                                       "   unique:"   + str(len(dfs[i].iloc[:,j].unique())) )
        #print(len(dfs[i].iloc[:,j].unique()))
        
#%%
np.save('C:/Users/User/Desktop/Research/Akhand Sir/VANET-IDS/Car Hacking Dataset/x_train.npy', x_train)    
np.save('C:/Users/User/Desktop/Research/Akhand Sir/VANET-IDS/Car Hacking Dataset/y_train.npy', y_train) 
np.save('C:/Users/User/Desktop/Research/Akhand Sir/VANET-IDS/Car Hacking Dataset/x_test.npy', x_test) 
np.save('C:/Users/User/Desktop/Research/Akhand Sir/VANET-IDS/Car Hacking Dataset/y_test.npy', y_test)   
#%%    
#df2 = pd.read_csv(path2, header=None)
#df2 = df2.iloc[: , :-1]

print("Read {} rows.".format(len(df)))
#print("Read {} rows.".format(len(df2)))
# df = df.sample(frac=0.1, replace=False) # Uncomment this line to 
# sample only 10% of the dataset
df.dropna(inplace=True,axis=1) # For now, just drop NA's 
df2.dropna(inplace=True,axis=1)
# (rows with missing values)

# The CSV file has no column heads, so add them
df.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]

df2.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]

pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 5)
# display 5 rows
print(df[0:5])
pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 5)
# display 5 rows
print(df2[0:5])
#%%


dummies = pd.get_dummies(df['outcome'])
dummies2 = pd.get_dummies(df2['outcome'])
i=0
i2=0
for x in dummies.columns:
    print(x)
    i += 1
print(i)    
for x in dummies2.columns:
    print(x)
    i2 += 1    
    
print(i2)  
#%%
normal = ['normal']
dos = ['neptune','pod','land','back','smurf','teardrop','apache2','mailbomb','worm','processtable','udpstorm']
probing = ['ipsweep','nmap','portsweep','satan','mscan','saint']
r2l = ['imap','ftp_write','warezclient','multihop','phf','spy','guess_passwd','warezmaster','httptunnel','named','sendmail','snmpgetattack','snmpguess','xlock','xsnoop']
u2r = ['loadmodule','buffer_overflow','rootkit','perl','ps','sqlattack','xterm']
nc = 0
dc = 0
pc = 0
rc = 0
uc = 0

nct = 0
dct = 0
pct = 0
rct = 0
uct = 0

#%%
for i in df.index:
    tmp = df['outcome'][i]
    #print(tmp)
    if (tmp in normal):
        #print('normal')
        nc += 1
print(nc) 
#%%       
for i in df.index:  
    tmp = df['outcome'][i]      
    if (tmp in dos):
        #print('dos') 
        dc += 1
        df.at[i,'outcome'] = 'dos'
print(dc)
#%%       
for i in df.index:  
    tmp = df['outcome'][i]      
    if (tmp in probing):
        #print('dos') 
        pc += 1
        df.at[i,'outcome'] = 'probing'
print(pc)
#%%       
for i in df.index:  
    tmp = df['outcome'][i]      
    if (tmp in r2l):
        #print('dos') 
        rc += 1
        df.at[i,'outcome'] = 'r2l'
print(rc)
#%%       
for i in df.index:  
    tmp = df['outcome'][i]      
    if (tmp in u2r):
        #print('dos') 
        uc += 1
        df.at[i,'outcome'] = 'u2r'
print(uc)

#%%
for i in df2.index:
    tmp = df2['outcome'][i]
    #print(tmp)
    if (tmp in normal):
        #print('normal')
        nct += 1
print(nct) 
#%%       
for i in df2.index:  
    tmp = df2['outcome'][i]      
    if (tmp in dos):
        #print('dos') 
        dct += 1
        df2.at[i,'outcome'] = 'dos'
print(dct)
#%%       
for i in df2.index:  
    tmp = df2['outcome'][i]      
    if (tmp in probing):
        #print('dos') 
        pct += 1
        df2.at[i,'outcome'] = 'probing'
print(pct)
#%%       
for i in df2.index:  
    tmp = df2['outcome'][i]      
    if (tmp in r2l):
        #print('dos') 
        rct += 1
        df2.at[i,'outcome'] = 'r2l'
print(rct)
#%%       
for i in df2.index:  
    tmp = df2['outcome'][i]      
    if (tmp in u2r):
        #print('dos') 
        uct += 1
        df2.at[i,'outcome'] = 'u2r'
print(uct)

#%%
import pandas as pd
import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore

def expand_categories(values):
    result = []
    s = values.value_counts()
    t = float(len(values))
    for v in s.index:
        result.append("{}:{}%".format(v,round(100*(s[v]/t),2)))
    return "[{}]".format(",".join(result))
        
def analyze(df):
    print()
    cols = df.columns.values
    total = float(len(df))

    print("{} rows".format(int(total)))
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count>100:
            print("** {}:{} ({}%)".format(col,unique_count,\
                int(((unique_count)/total)*100)))
        else:
            print("** {}:{}".format(col,expand_categories(df[col])))
            expand_categories(df[col])
#%%     
analyze(df)
analyze(df2)
#%%       
# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd
    
# Encode text values to dummy variables(i.e. [1,0,0],
# [0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
#%%
# Now encode the feature vector

encode_numeric_zscore(df, 'duration')
encode_text_dummy(df, 'protocol_type')
encode_text_dummy(df, 'service')
encode_text_dummy(df, 'flag')
encode_numeric_zscore(df, 'src_bytes')
encode_numeric_zscore(df, 'dst_bytes')
encode_text_dummy(df, 'land')
encode_numeric_zscore(df, 'wrong_fragment')
encode_numeric_zscore(df, 'urgent')
encode_numeric_zscore(df, 'hot')
encode_numeric_zscore(df, 'num_failed_logins')
encode_text_dummy(df, 'logged_in')
encode_numeric_zscore(df, 'num_compromised')
encode_numeric_zscore(df, 'root_shell')
encode_numeric_zscore(df, 'su_attempted')
encode_numeric_zscore(df, 'num_root')
encode_numeric_zscore(df, 'num_file_creations')
encode_numeric_zscore(df, 'num_shells')
encode_numeric_zscore(df, 'num_access_files')
encode_numeric_zscore(df, 'num_outbound_cmds')
encode_text_dummy(df, 'is_host_login')
encode_text_dummy(df, 'is_guest_login')
encode_numeric_zscore(df, 'count')
encode_numeric_zscore(df, 'srv_count')
encode_numeric_zscore(df, 'serror_rate')
encode_numeric_zscore(df, 'srv_serror_rate')
encode_numeric_zscore(df, 'rerror_rate')
encode_numeric_zscore(df, 'srv_rerror_rate')
encode_numeric_zscore(df, 'same_srv_rate')
encode_numeric_zscore(df, 'diff_srv_rate')
encode_numeric_zscore(df, 'srv_diff_host_rate')
encode_numeric_zscore(df, 'dst_host_count')
encode_numeric_zscore(df, 'dst_host_srv_count')
encode_numeric_zscore(df, 'dst_host_same_srv_rate')
encode_numeric_zscore(df, 'dst_host_diff_srv_rate')
encode_numeric_zscore(df, 'dst_host_same_src_port_rate')
encode_numeric_zscore(df, 'dst_host_srv_diff_host_rate')
encode_numeric_zscore(df, 'dst_host_serror_rate')
encode_numeric_zscore(df, 'dst_host_srv_serror_rate')
encode_numeric_zscore(df, 'dst_host_rerror_rate')
encode_numeric_zscore(df, 'dst_host_srv_rerror_rate')



encode_numeric_zscore(df2, 'duration')
encode_text_dummy(df2, 'protocol_type')
encode_text_dummy(df2, 'service')
encode_text_dummy(df2, 'flag')
encode_numeric_zscore(df2, 'src_bytes')
encode_numeric_zscore(df2, 'dst_bytes')
encode_text_dummy(df2, 'land')
encode_numeric_zscore(df2, 'wrong_fragment')
encode_numeric_zscore(df2, 'urgent')
encode_numeric_zscore(df2, 'hot')
encode_numeric_zscore(df2, 'num_failed_logins')
encode_text_dummy(df2, 'logged_in')
encode_numeric_zscore(df2, 'num_compromised')
encode_numeric_zscore(df2, 'root_shell')
encode_numeric_zscore(df2, 'su_attempted')
encode_numeric_zscore(df2, 'num_root')
encode_numeric_zscore(df2, 'num_file_creations')
encode_numeric_zscore(df2, 'num_shells')
encode_numeric_zscore(df2, 'num_access_files')
encode_numeric_zscore(df2, 'num_outbound_cmds')
encode_text_dummy(df2, 'is_host_login')
encode_text_dummy(df2, 'is_guest_login')
encode_numeric_zscore(df2, 'count')
encode_numeric_zscore(df2, 'srv_count')
encode_numeric_zscore(df2, 'serror_rate')
encode_numeric_zscore(df2, 'srv_serror_rate')
encode_numeric_zscore(df2, 'rerror_rate')
encode_numeric_zscore(df2, 'srv_rerror_rate')
encode_numeric_zscore(df2, 'same_srv_rate')
encode_numeric_zscore(df2, 'diff_srv_rate')
encode_numeric_zscore(df2, 'srv_diff_host_rate')
encode_numeric_zscore(df2, 'dst_host_count')
encode_numeric_zscore(df2, 'dst_host_srv_count')
encode_numeric_zscore(df2, 'dst_host_same_srv_rate')
encode_numeric_zscore(df2, 'dst_host_diff_srv_rate')
encode_numeric_zscore(df2, 'dst_host_same_src_port_rate')
encode_numeric_zscore(df2, 'dst_host_srv_diff_host_rate')
encode_numeric_zscore(df2, 'dst_host_serror_rate')
encode_numeric_zscore(df2, 'dst_host_srv_serror_rate')
encode_numeric_zscore(df2, 'dst_host_rerror_rate')
encode_numeric_zscore(df2, 'dst_host_srv_rerror_rate')


# display 5 rows

df.dropna(inplace=True,axis=1)
df[0:5]
df2.dropna(inplace=True,axis=1)
df2[0:5]
# This is the numeric feature vector, as it goes to the neural net


# Convert to numpy - Classification
x_columns = df.columns.drop('outcome')
x_train = df[x_columns].values
dummies = pd.get_dummies(df['outcome']) # Classification
outcomes = dummies.columns
num_classes = len(outcomes)
y_train = dummies.values  

x_columns = df2.columns.drop('outcome')
x_test = df2[x_columns].values
dummies = pd.get_dummies(df2['outcome']) # Classification
outcomes = dummies.columns
num_classes = len(outcomes)
y_test = dummies.values  
#%%
df.groupby('outcome')['outcome'].count()
#%%
import pandas as pd
import io
import requests
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
#%%

# Create a test/train split.  25% test
# Split into train/test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   x_train, y_train, test_size=0.25, random_state=42)

size = x_test.shape[0]
#%%

zero = 0
x_test_zero = []
y_test_zero = []
 
for i in range (size):
    if (y_test[i][0] == 1):
        zero += 1
        x_test_zero.append(x_test[i])
        y_test_zero.append(y_test[i])
y_test_zero = np.array(y_test_zero)
x_test_zero = np.array(x_test_zero)
#%%
one = 0
x_test_one = []
y_test_one = []
#size = x_test.shape[0] 
for i in range (size):
    if (y_test[i][1] == 1):
        one += 1
        x_test_one.append(x_test[i])
        y_test_one.append(y_test[i])
y_test_one = np.array(y_test_one)
x_test_one = np.array(x_test_one)
#%%
two = 0
x_test_two = []
y_test_two = []
#size = x_test.shape[0] 
for i in range (size):
    if (y_test[i][2] == 1):
        two += 1
        x_test_two.append(x_test[i])
        y_test_two.append(y_test[i])
x_test_two = np.array(x_test_two)
y_test_two = np.array(y_test_two)
#%%
three = 0
x_test_three = []
y_test_three = []
#size = x_test.shape[0] 
for i in range (size):
    if (y_test[i][3] == 1):
        three += 1
        x_test_three.append(x_test[i])
        y_test_three.append(y_test[i])
x_test_three = np.array(x_test_three)
y_test_three = np.array(y_test_three)
#%%
four = 0
x_test_four = []
y_test_four = []
#size = x_test.shape[0] 
for i in range (size):
    if (y_test[i][4] == 1):
        four += 1
        x_test_four.append(x_test[i])
        y_test_four.append(y_test[i])
x_test_four = np.array(x_test_four)
y_test_four = np.array(y_test_four)

#%%
# Create neural net
model = Sequential()
model.add(Dense(50, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(20, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(10, input_dim=x_train.shape[1], activation='relu'))
#model.add(Dense(1, kernel_initializer='normal'))
model.add(Dense(y_train.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                        patience=5, verbose=1, mode='auto',
                           restore_best_weights=True)
model.fit(x_train,y_train,validation_data=(x_test,y_test),     callbacks=[monitor],verbose=2,epochs=10)
#%%
model.summary()
#%%
pred = model.predict(x_test_four)
pred2 = np.argmax(pred,axis=1)
y_eval = np.argmax(y_test_four,axis=1)
score = metrics.accuracy_score(y_eval, pred2)
print("score: {}".format(score))