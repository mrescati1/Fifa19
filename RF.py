import csv
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics


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
#df.fillna("_na_").values
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
df= clean(df)
#df= df[['Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Joined', 'Height', 'Weight', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']]
df= df[['Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Joined', 'Height', 'Weight', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']]
df['Release Clause']= (df['Release Clause'])/(df['Release Clause'].max()- df['Release Clause'].min())
#Train Test Validation split
train_df, test_df = train_test_split(df, test_size=0.1, random_state=2019)
train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=2019)
train_y= train_df['Release Clause']
train_df.drop('Release Clause', axis= 1, inplace=True)
test_y= test_df['Release Clause']
test_df.drop('Release Clause', axis= 1, inplace=True)
val_y= val_df['Release Clause']
val_df.drop('Release Clause', axis= 1, inplace=True)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
train_df.fillna(0, inplace = True)
print(train_df[train_df['Stamina'].isnull()])
train_df = pd.get_dummies(train_df)
train_y = pd.get_dummies(train_y)
rf = RandomForestRegressor(n_estimators = 2, random_state = 42, verbose= 1)

# Train the model on training data
rf.fit(train_df, train_y)
