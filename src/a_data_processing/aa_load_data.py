import pandas as pd

versiondedonnees = "1.0"

#les noms de colonnes du dataset sont les suivants (s pour sensor):
col_names = ['id',
             'cycle',
             'setting1',
             'setting2',
             'setting3',
             's1',
             's2',
             's3',
             's4',
             's5',
             's6',
             's7',
             's8',
             's9',
             's10',
             's11',
             's12',
             's13',
             's14',
             's15',
             's16',
             's17',
             's18',
             's19',
             's20',
             's21',
             's22',
             's23']

#Charger les données de training
df_train = pd.read_csv('../../data/train_FD001.txt', sep = ' ', header=None)

#Ajouter les noms de colonnes
df_train.columns = col_names
#print(df_train.head())
    
#Charger les données de test
df_test = pd.read_csv('../../data/test_FD001.txt', sep = ' ', header=None)
    
#Ajouter les noms de colonnes
df_test.columns = col_names
    
#print(df_test.head())
