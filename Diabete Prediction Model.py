# Importing Libraries
import numpy as np
import pandas as p
import matplotlib.pyplot as plt

# Loading Dataset
dataset=pd.read_csv("diabetes.csv")

# creating list of feature name which contains 0 values
zero_ValueUniqueFeatures=[features for features in dataset.columns if 0 in dataset[features].unique() and features not in ["Outcome"]]

# We have lot of 0 values across all this features ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']. We must replace it with Nan
def function_replace0ToNanValue(dataset,features):
    for feature in features:
        dataset[feature]=dataset[feature].replace(0,np.NaN)
        

# Calling replacing 0 value to Nan        
function_replace0ToNanValue(dataset,zero_ValueUniqueFeatures)

# Replacing Nan values with median
def function_replacingNanToMedian(dataset,features):
    for feature in features:
        dataset[feature]=dataset[feature].fillna(dataset[feature].median())
        

function_replacingNanToMedian(dataset,zero_ValueUniqueFeatures)


# Train and test Split
from sklearn.model_selection import train_test_split
train=dataset.drop(['Outcome'],axis=1)
# train=dataset.drop(columns='Outcome')
test=dataset['Outcome']
x_train,x_test,y_train,y_test=train_test_split(train,test,test_size=0.2,random_state=0)

# Diabete Model Prediction
from sklearn.ensemble import RandomForestClassifier
randomForest=RandomForestClassifier(n_estimators=25)
randomForest.fit(x_train,y_train)

#Creating pickel file for Diabete Model Predictions
import pickle

fileName="Diabete_Model_Prediction.pkl"
pickle.dump(randomForest,open(fileName,'wb'))
