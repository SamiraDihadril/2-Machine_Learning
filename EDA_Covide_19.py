import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel('dataset.xlsx')
print(data.head())
df=data.copy()

pd.set_option('display.max_column',111)
pd.set_option('display.max_row',111)

#A) FORM ANALYSE #################################################################################
#1 id target var = sar-cov-result
#2 id nb l and nb cl  5644 111
print(df.shape)
#3 types of vars(qual/quan)
print(df.dtypes)
#4 count nb o var types 
print(df.dtypes.value_counts())
#5 count nb o var types / graph presentaion
print(df.dtypes.value_counts().plot.pie())

#6 analyse of nan val :
#6 show the entire nan values in the dataset using graph   black = val /  white= nan
plt.figure(figsize=(20,10))
sns.heatmap(df.isna(), cbar=False)
#6 mesurer le % des val nan
print(df.isna())
#6 la sum des val nan / nb of L
print((df.isna().sum()/df.shape[0] ).sort_values(ascending=True))
print('////////////////////////////////////////////////////////////////')
#6 coclus analyse de forme  : bqp de val nan / 2 grp of useful data 76$ test bacterie 89% taux sanguins 

#visualisation init  des useless val (boolean indexing <0.9 )
#injecter a notre dataframe juste les C qui on un % de nan < 0.9%
df= df[df.columns[df.isna().sum()/df.shape[0] <0.9 ]]
print(df)
sns.heatmap(df.isna(), cbar=False)
print('////////////////////////////////////////////////////////////////')
#drop C id
df= df.drop('Patient ID',axis=1)
print(df)




#B) ANALYSE de fond   #################################################################################
#Visualisation du target : SARS-Cov-2 exam ressult
#Examen de la C target : cont n and p cases en % (normalize = true()
print(df['SARS-Cov-2 exam result'].value_counts(normalize = True))
#cnclus: les 2 classes ne sont pas equilibrees  10% p / 90% n
# -> matrique ; squore f1 / sensibilite / precision
print('////////////////////////////////////////////////////////////////')

#segnification de var 
#tracer histogrames des variables continues : observation de la repartion des donnees 
#float : var continues standardisees , asymetriques
for col in df.select_dtypes('float'):
    plt.figure()
    sns.displot(df[col])  #distplot == distrubution plot (courbe)
#age  var quantite  nest pas un float : interpretation difficile
sns.displot(df['Patient age quantile'])
print(df['Patient age quantile'].value_counts())
#object : var qualitative  : * sont binaires  / Rhinovirus tres elevee
#1 visulisation de nb des categorie dans chaque var 
for col in df.select_dtypes('object'): 
    print(f'{col :-<50} {df[col].unique()}')
#2 count nb of val in each cat
for col in df.select_dtypes('object'): 
    plt.figure()
    print(df[col].value_counts())
    df[col].value_counts().plot.pie()
print('////////////////////////////////////////////////////////////////')

    
#visualisation de la relation entre nos var et target 
## creation des sous ensemble P/N
positive_df=df[df['SARS-Cov-2 exam result'] =='positive']
negative_df=df[df['SARS-Cov-2 exam result'] =='negative']

print('////////////////////////////////////////////////////////////////')
## creation des sous ensemble Blood/Viral
missing_rate=df.isna().sum()/df.shape[0]
blood_columns= df.columns[(missing_rate <0.9) & (missing_rate > 0.88)]
viral_columns= df.columns[(missing_rate <0.88) & (missing_rate > 0.75)]



#visualisation relation target/blood : le teaux de monocytes /platlettes leukocytes semplent liee au covide 19

for col in blood_columns:
    plt.figure()
    sns.distplot(positive_df[col], label='positive')
    sns.distplot(negative_df[col], label='negative')
    plt.legend()


#visualisation relation target/ age :
# countplot generate:genph count nb of apparition  de chaque Patient age quantile pour les ressultst p et n de SARS-Cov-2 exam result 

plt.figure()
sns.countplot(x='Patient age quantile',hue='SARS-Cov-2 exam result',data=df)
    
#visualisation relation target/ Viral
print(pd.crosstab(df['SARS-Cov-2 exam result'],df['Influenza A']))

for col in viral_columns:
    plt.figure()
    sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'],df[col]),annot=True,fmt='d')
    


#analyse var var 
##blood_data blood_data
####relation taux sanguin
plt.figure()
sns.pairplot(df[blood_columns])
plt.figure()
sns.heatmap(df[blood_columns].corr())
plt.figure()
sns.clustermap(df[blood_columns].corr())

####relation age/sang : tres faible corr entre age et TS max 0.28
for col in blood_columns:
    plt.figure()
    sns.lmplot(x='Patient age quantile', y=col,hue='SARS-Cov-2 exam result',data=df )
    
print(df.corr()['Patient age quantile'].sort_values())

print(pd.crosstab(df['Influenza A'], df['Influenza A, rapid test']))
print(pd.crosstab(df['Influenza B'], df['Influenza B, rapid test']))


####relation maladie/blood data : les teaux sanguins entre malades et C19 sont <>

df['est malade'] = np.sum(df[viral_columns[:-2]]=='detected',axis=1)>=1
print(df.head())

malade_df     =df[df['est malade'] == True]
non_malade_df =df[df['est malade'] == False]

for col in blood_columns:
    plt.figure()
    sns.distplot(malade_df[col], label='malade')
    sns.distplot(non_malade_df[col], label='non malade')
    plt.legend()


####relation hospitalisation / blood : intéressant dans le cas ou on voudrait prédire dans quelle service un patient devrait aller

def hospitalisation(df):
    if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
        return 'surveillance'
    elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
        return 'soins semi-intensives'
    elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
        return 'soins intensifs'
    else:
        return 'inconnu'
    
df['statut'] = df.apply(hospitalisation, axis=1)
print(df.head())



for col in blood_columns:
    plt.figure()
    for cat in df['statut'].unique():
        sns.distplot(df[df['statut']==cat][col], label=cat)
    plt.legend()



print(df[blood_columns].count())
print(df[viral_columns].count())



df1 = df[viral_columns[:-2]]
df1['covid'] = df['SARS-Cov-2 exam result']
df1.dropna()['covid'].value_counts(normalize=True)




df2 = df[blood_columns]
df2['covid'] = df['SARS-Cov-2 exam result']
df2.dropna()['covid'].value_counts(normalize=True)
