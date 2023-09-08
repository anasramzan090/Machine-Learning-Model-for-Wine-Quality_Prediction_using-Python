#This program is written by M Anas Ramzan, as a project of internship supervised by Technohachs edu tceh
#importing important libraries to be used in program
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
#getting file containing data set
ds=pd.read_csv('D://ML Programs//Intern_Projects//Wine_Quality_Prediction//winequality-red.csv')
#visualizing the data
print(ds)
print(ds.head())
print(ds.isnull().sum()) #checking if some missing vale, if there is then we will take mean of that coilomn and put it thare
#Ploting the data
sns.catplot(x='quality',data=ds,kind='count')
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='citric acid',data=ds) #how coloumns are related to each other
cor=ds.corr()
plt.figure(figsize=(10,10))
#getting heat map of the data
sns.heatmap(cor, cbar=True,square=True,fmt='0.2f',annot=True, annot_kws={'size':8},cmap='Blues')
#Counting number of lables:
ds['quality'].value_counts()
#making and y variables
X=ds.drop('quality',axis=1)
Y=ds['quality'].apply(lambda y_value:1 if y_value>=7 else 0) #we are doing lable binerization ,coverting multiple value of lables into two specific binary value
#saperating the adat into test and train 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=3)
#getting the model trained
model=RandomForestClassifier().fit(X_train,Y_train)
#Reading input from the user
input_str = input("Enter 11 numbers from data saperated by comma i.e:- 7.8,0.58,0.02,2.0,0.073,9.0,18.0,0.9968,3.36,0.57,9.5: ")
#Spliting the input string into individual numbers
numbers_str = input_str.split(',')
#Converting the strings to floating-point values
numbers = [float(num_str) for num_str in numbers_str]
# Creating a list containing these numbers
input_list = numbers
#function for model prediction
def predict_with_model(in_data):
    id_as_np=np.asarray(in_data)
    inp_reshape=id_as_np.reshape(1,-1)
    prediction=model.predict(inp_reshape)
    return prediction
# Calling the model function with the input data
prediction_result = predict_with_model(input_list)
print("Input List:", input_list)
if prediction_result==0:
    print("Wine Quality is not Good!!")
elif prediction_result==1:
    print("Wine Quality is Good!!!")
print("Model Prediction Result:", prediction_result)
#Deploying machine learning trained model
filename='trained_model_for_wine_quality_prediction.sav'
pickle.dump(model,open(filename,'wb')) #dump to save model
loaded_model=pickle.load(open('trained_model_for_wine_quality_prediction.sav','rb'))
