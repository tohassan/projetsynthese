import sys
sys.path.insert(1, '../a_data_processing')
from aa_load_data import *
from bb_eda import *


import pandas as pd
from sklearn.preprocessing import StandardScaler

print('========================== preprocessing train data ================================================')
f = ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']
to_drop = ['cycle','setting1','setting2','setting3','s1','s5','s6','s10','s16','s18','s19']
max_cycle.reset_index(level=0, inplace=True)
max_cycle.columns = ['id', 'last_cycle']

#print(max_cycle)

df_train = pd.merge(df_train, max_cycle, on='id')
df_train['RUL'] = df_train['last_cycle'] - df_train['cycle']
df_train.drop(to_drop+['last_cycle'], axis=1, inplace=True)
print(df_train)

# normalize df_train  science all sensos series have normal distribution as shown om notebook
train_data = df_train.copy()


# normalize train_data
train_data_n = train_data[f]
scaler = StandardScaler().fit(train_data_n.values)
train_data_n = scaler.transform(train_data_n.values)
train_data[f] = train_data_n
train_data_c = train_data.copy()
#train_data_cut = train_data.copy()

print(train_data)
print('========================== preprocessing test data ================================================')

t_max_cycle = pd.DataFrame(df_test[['id','cycle']].groupby('id').max())
t_max_cycle.reset_index(level=0, inplace=True)
t_max_cycle.columns = ['id', 'last_cycle']
print(t_max_cycle)

#Preparer / merger les données de test et truth
df_truth = pd.read_csv('../../data/RUL_FD001.txt', sep = ' ', header=None)
df_truth.drop([1], axis=1, inplace=True)
df_truth.columns=['more']
df_truth['id']=df_truth.index+1
df_truth['t_last']=df_truth['more']+t_max_cycle['last_cycle']
df_truth.drop(['more'], axis=1, inplace=True)
print(df_truth)

df_test = pd.merge(df_test, df_truth, on='id')
df_test['RUL'] = df_test['t_last'] - df_test['cycle']
df_test.drop(to_drop+['t_last'], axis=1, inplace=True)
print(df_test.round(3))

# normalize df_test science all sensos series have normal distribution as shown om notebook

test_data = df_test.copy()
test_data = test_data.groupby('id').last().reset_index() # test_data is 13096 and train_data 20631, so i regrouped by id here
#normalize test_data
test_data_n =test_data[f]
#scaler = StandardScaler().fit(test_data_n.values) # do not fit test data
test_data_n = scaler.transform(test_data_n.values)
test_data[f]=test_data_n
test_data_cut=test_data.copy()

print(test_data)

# ready data to use in modeles
X_train = train_data.drop(['id','RUL'],axis=1)
y_train = train_data.pop('RUL')
X_test = test_data.drop(['id','RUL'],axis=1)
y_test = test_data.pop('RUL')
print(X_test)

print('================== data cliped for new approch ================')
train_data_c['RUL'].clip(upper=125, inplace=True)
train_data_cut=train_data_c
X_train_cut = train_data_cut.drop(['id','RUL'],axis=1)
y_train_cut = train_data_cut.pop('RUL')
X_test_cut = test_data_cut.drop(['id','RUL'],axis=1)
y_test_cut = test_data_cut.pop('RUL')
