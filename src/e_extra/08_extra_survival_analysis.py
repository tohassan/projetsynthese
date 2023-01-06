import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from collections import Counter


#les noms de colonnes du dataset sont les suivants (s pour sensor):
col_names = ['engine_id','lifetime','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10',
             's11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

#Charger les données de training
df_train_list = ['df_train0','df_train1','df_train2','df_train3','df_train4']
train_eng_id  = [0,0,100,360,460,]
df_test_list = ['df_test0','df_test1','df_test2','df_test3','df_test4']
test_eng_id  = [0,709,809,1068,1168,]
for i in range(1,5):
    if i>0:
        df_train_list[i] = pd.read_csv('../../data/train_FD00'+str(i)+'.txt', sep = ' ', header=None)
        df_train_list[i].drop([26,27], axis=1, inplace=True)
        df_train_list[i].columns = col_names
        df_train_list[i].engine_id += train_eng_id[i]

df_train = pd.concat([df_train_list[1],df_train_list[2],df_train_list[3],df_train_list[4]])
df_train['broken']=1
print(df_train)

for i in range(1,5):
    if i>0:
        df_test_list[i] = pd.read_csv('../../data/test_FD00'+str(i)+'.txt', sep = ' ', header=None)
        df_test_list[i].drop([26,27], axis=1, inplace=True)
        df_test_list[i].columns = col_names
        df_test_list[i].engine_id += test_eng_id[i]

df_test = pd.concat([df_test_list[1],df_test_list[2],df_test_list[3],df_test_list[4]])
df_test['broken']=0
print(df_test)

dataset = pd.concat([df_train,df_test])
print(dataset.describe().transpose().round(3))
#normalize train_data
f = ['setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10',
             's11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
train_data_n =dataset[f]
scaler = StandardScaler().fit(train_data_n.values)
train_data_n = scaler.transform(train_data_n.values)
dataset[f]=train_data_n
print(dataset)
data_grouped = dataset.groupby('engine_id').last().reset_index()
print(data_grouped)
data_scrambled = data_grouped.sample(frac=1, random_state=2).reset_index(drop=True)
print(data_scrambled)


data_1 = Counter(data_scrambled['broken'].replace({0:'not broken yet', 1:'broken'}))
print(data_1)


# Creating an empty chart
fig, ((ax1, ax2)) = plt.subplots(1, 2,  figsize=(15, 4))

# Counting the number of occurrences for each category

category = list(data_1.keys())
counts = list(data_1.values())
idx = range(len(counts))
# Displaying the occurrences of the event/censoring
ax1.bar(idx, counts)
ax1.set_xticks(idx)
ax1.set_xticklabels(category)
ax1.set_title( 'Occurences of the event/censoring', fontsize=15)

bins = np.linspace(10, 400, 30)


# Showing the histogram of the survival times for the censoring
time_0 = data_scrambled.loc[ data_scrambled['broken'] == 0, 'lifetime']
# Showing the histogram of the survival times for the events
time_1 = data_scrambled.loc[ data_scrambled['broken'] == 1, 'lifetime']



ax2.hist(time_0, bins, alpha=0.5, color='blue', label = 'not broken yet')
ax2.hist(time_1, bins, alpha=0.3, color='black', label = 'broken')
ax2.set_title( 'Histogram - survival time', fontsize=15)

# Displaying everything side-by-side
plt.legend(fontsize=15)
plt.show()

#fs = ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']
#fs = ['s2','s4','s7','s11','s12','s13','s15','s20','s21']
fs=['s4','s15','s11','s14','s12','s9','s7','s3','s6','s2','s20','setting1','s17','s21','s13','setting2']

# Building training and testing sets
# Defining the time and event column
time_column = 'lifetime'
event_column = 'broken'
features = fs
N = data_scrambled.shape[0]
from sklearn.model_selection import train_test_split

index_train, index_test = train_test_split(range(N), test_size=0.25)
data_train = data_scrambled.loc[index_train].reset_index(drop=True)
data_test = data_scrambled.loc[index_test].reset_index(drop=True)

# Creating the X, T and E inputs
X_train, X_test = data_train[features], data_test[features]
T_train, T_test = data_train[time_column], data_test[time_column]
E_train, E_test = data_train[event_column], data_test[event_column]



# from pysurvival.models.multi_task import LinearMultiTaskModel
# #good:testSize:0.25 C-index:0.73, epoch:80000,lr:1e-5,optimizer:rmsprop , l2_reg :1,  l2_smooth = 5,
#
# # Initializing the MTLR with a time axis split into 300 intervals
# linear_mtlr = LinearMultiTaskModel(bins=200)
#
# # Fitting the model
# #linear_mtlr.fit(X_train, T_train, E_train, num_epochs = 50000,
# #                init_method = 'orthogonal', optimizer ='rmsprop',
# #                lr = 1e-5, l2_reg = 1,  l2_smooth = 5, )
