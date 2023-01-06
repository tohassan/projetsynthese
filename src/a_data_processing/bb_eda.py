from aa_load_data import *

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

print('========================== EDA training data ================================================')

# Verifier s'il y des valeurs manquantes
print('df_train: num row =  {} , num columns =  {}'.format(df_train.shape[0], df_train.shape[1]))
print(df_train.isnull().sum())

# Verifier les types de données (toutes est numérique)
print(df_train.dtypes)
print('df_train : tout les colonnes sont numerique. s22 et s23 sont nulles')

#supprimer les colonnes 26 et 27 (deux espaces de trop)
df_train.drop(['s22','s23'], axis=1, inplace=True)
print(df_train.head(5))
max_cycle = pd.DataFrame(df_train[['id','cycle']].groupby('id').max())
print(max_cycle)
sns.histplot(max_cycle, kde=True)
plt.show()
print('Peu de moteurs peuvent aller au dela de 300 cycles, la grande majorité atteint 200 cycles')

# Vérifier quelques  statistiques pour les settings
print(df_train[['setting1','setting2','setting3']].describe())
print("l'écart type de setting3 est à 0 et ceux de setting1 et setting2 sont faibles, devrions-nous supprimer ces settings?")

sensors = df_train[['s{}'.format(i) for i in range(1,22)]]
print(sensors.describe().transpose().round(2))
print ("Les écarts type des capteurs s1,s5,s6,s10,s16,s18 et s19 sont tous à 0, devrions-nous supprimer ces capteurs?")

z_corr = pd.DataFrame(df_train)
plt.figure(figsize=(15,8))
sns.heatmap(round(z_corr.corr(),1), vmin=-1, vmax=1, annot=True)
plt.show()
print('la matrice de correlation confirme notre constatation sur les capteurs! ')

#explore the engine i to see all selected sensors beheavior
t = df_train[df_train['id']==1].copy()
f=['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

#normalize
Z =t[f]
scaler = StandardScaler().fit(Z.values)
Z = scaler.transform(Z.values)
t[f]=Z

#plot
plt.figure(figsize=(10,5))
for i in f:
    p=sns.lineplot(x=t.cycle, y=t[i])

p.set(xlim=(0, 190))
plt.show()
print('========================== EDA test data ================================================')

# Vérfier s'il y des valeurs manquantes
print('df_test: num row =  {} , num columns =  {}'.format(df_test.shape[0], df_test.shape[1]))
print(df_test.isnull().sum())

# Vérifier les types de données (toutes est numérique)
print(df_test.dtypes)
print('df_test : tout les colonnes sont numerique. s22 et s23 sont nulles')

#supprimer les colonnes 26 et 27 (deux espaces de trop)
df_test.drop(['s22','s23'], axis=1, inplace=True)
print(df_test.head(5))


# Vérifier quelques  statistiques pour les settings
print(df_test[['setting1','setting2','setting3']].describe())
print("l'écart type de setting3 est à 0 et ceux de setting1 et setting2 sont faibles, devrions-nous supprimer ces settings?")

sensors = df_test[['s{}'.format(i) for i in range(1,22)]]
print(sensors.describe().transpose().round(2))
print ("Les écarts type des capteurs s1,s5,s6,s10,s16,s18 et s19 sont tous à 0, devrions-nous supprimer ces capteurs?")

z_corr_t = pd.DataFrame(df_test)
plt.figure(figsize=(15,8))
sns.heatmap(round(z_corr_t.corr(),1), vmin=-1, vmax=1, annot=True)
plt.show()
print('la matrice de correlation confirme notre constatation sur les capteurs! ')

print('================ new approch ============================')
cut_test=df_train.copy()
bb=cut_test.iloc[:191].copy()
plt.figure(figsize=(15,8))
sns.lineplot(x=bb.cycle,y=bb.s12)
plt.show()
