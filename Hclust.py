import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage 
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
def split_data(data):
    training= []
    test= []
    for row in data:
        if np.random.normal(0,1,1)>0.8:
            test.append(row)
        else:
           training.append(row)
    return (training, test)
def clean(df):
    new_df= df
    for i in range (0, df.shape[0]):
        string= str(new_df.at[i, 'Release Clause'])
        if string.find('M')>-1:
                n_string= string.replace('M', '')
                k_string= n_string.replace('€', '')
                if k_string=='':
                    k_string='0'
                k_string= float(k_string)
                k_string*= 1000*1000
        elif string.find('K')>-1:
            n_string= string.replace('K', '')
            k_string= n_string.replace('€', '')
            if k_string=='':
                k_string='0'
            k_string= float(k_string)
            k_string*= 1000
        else:        
            k_string= 0.0
        new_df.at[i, 'Release Clause']= k_string
    for i in range (1, df.shape[0]):
        string= str(new_df.at[i, 'Weight'])
        if string.find('lbs')>-1:
                n_string= string.replace('lbs', '')
                k_string= float(n_string)
        else:
            k_string='0'          
            k_string= float(k_string)
        new_df.at[i, 'Weight']= k_string
    return new_df
def clean_(df):
    new_df= df
    for i in range (1, df.shape[0]):
        string= str(new_df.at[i, 'Weight'])
        if string.find('lbs')>-1:
                n_string= string.replace('lbs', '')
                k_string= float(n_string)
        else:
            k_string='0'          
            k_string= float(k_string)
        new_df.at[i, 'Weight']= k_string
    return new_df
def normalize2D(data, data1):
    min_= float(my_min(data1))
    max_= float(my_max(data1))
    data_= []
    for i in range(len(data)):
        z= []
        for j in range(len(data[0])):
            #print(data[i][j])
            a= float(data[i][j])
            b= ((a-min_)/max_-min_)
            z.append(b)
        data_.append(z)
        #print(z)
            
    return data_
def normalize1D(data, data1):
    data_= []

    for i in range(len(data)):
        data_.append((float(data[i])-float(min(data1)))/(float(max(data1))-float(min(data1))))
    return data_

np.random.seed(123)  # for reproducibility

#print(general)
#print(value)

df = pd.read_csv('data.csv')
#print(df['Release Clause'])

#df= df.astype('category')
df.fillna("_na_").values
classnames, indices = np.unique(df['Nationality'], return_inverse=True)
df['Nationality']= indices
df['Club']= str(df['Club'])
classnames, indices = np.unique(df['Club'], return_inverse=True)
df['Club']= indices
df['Preferred Foot']= str(df['Preferred Foot'])
classnames, indices = np.unique(df['Preferred Foot'], return_inverse=True)
df['Preferred Foot']= indices
df['Work Rate']= str(df['Work Rate'])
classnames, indices = np.unique(df['Work Rate'], return_inverse=True)
df['Work Rate']= indices
df['Body Type']= str(df['Body Type'])
classnames, indices = np.unique(df['Body Type'], return_inverse=True)
df['Body Type']= indices
df['Position']= str(df['Position'])
classnames, indices = np.unique(df['Position'], return_inverse=True)
df['Position']= indices
df['Joined']= str(df['Joined'])
classnames, indices = np.unique(df['Joined'], return_inverse=True)
df['Joined']= indices
df['Height']= str(df['Height'])
classnames, indices = np.unique(df['Height'], return_inverse=True)
df['Height']= indices
df['Weight']= str(df['Weight'])
classnames, indices = np.unique(df['Weight'], return_inverse=True)
df['Weight']= indices
df= clean(df)
df.index= df['Name']

df= df[['Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Joined', 'Height', 'Weight', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']]
df['Release Clause']= (df['Release Clause'])/(df['Release Clause'].max()- df['Release Clause'].min())
#Train Test Validation split
train_df= df.head(100)
train_df.fillna(0, inplace = True)
linked = linkage(train_df, 'complete', 'canberra')

labelList = train_df.index

plt.figure(figsize=(10, 7))  
dendrogram(linked,  
            orientation='top',
	    labels= labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()  
cluster = AgglomerativeClustering(n_clusters= 6, affinity="cosine", linkage= "complete")
cluster.fit_predict(train_df)
train_df["Labels"]= cluster.labels_
print(train_df)
