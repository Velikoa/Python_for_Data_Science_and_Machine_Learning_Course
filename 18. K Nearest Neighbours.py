import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Classified Data', index_col=0)
desired_width = 450
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',15)

print(df.head())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()           #Create an instance just like every other ML algorithm
scaler.fit(df.drop('TARGET CLASS',axis=1))      #Fitting the scaler object to everything except the target class you will be categorising

#Scaling the dataset means that all the data columns have their values within the range of -1 and 1. This is to avoid having large
#variations in values eg: some columns have data with ranges going from 0 to 1000 while can just be between 1 and 20.

#now use feature to do a transformation - this performs a standardisation by centering and scaling.
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
print(scaled_features)

#Now need to place the standardised info in a dataframe.
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])     #Using the column names from the original df but in thee standardised dataset.
print(df_feat.head())

#Now that the data is ready, can plug it into a machine learning algorithm. But need to split the data into a training and testing datasete once again.
from sklearn.model_selection import train_test_split
X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Trying to determine whether someone will be in this Target CLass or not.
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)        #Need to fit the training data

#Now need to use the fitted data to obtain predictions from it.
pred = knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

#Need to determine what is the most appropriate k number to use. So below will be using from number 1 to 40 to see which one
#has the least number of errors in it. This will take a wjile to run the entire process/iteration.
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))        #this is the average error rate. Avg of where the predictions were not equal to the actual test values.

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

plt.show()

knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


##################################Project#####################################################################
project_df = pd.read_csv('KNN_Project_Data')

print(project_df.head())

#Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.
#sns.pairplot(data=project_df,hue='TARGET CLASS',palette='coolwarm')

#plt.show()

#** Create a StandardScaler() object called scaler.**
project_scaler = StandardScaler()

#** Fit scaler to the features.**
project_scaler.fit(project_df.drop('TARGET CLASS',axis=1))

#Use the .transform() method to transform the features to a scaled version.
project_scaled_features = project_scaler.transform(project_df.drop('TARGET CLASS',axis=1))

#Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.
project_scaled_df = pd.DataFrame(project_scaled_features,columns=project_df.columns[:-1])

print(project_scaled_df.head())

#Use train_test_split to split your data into a training set and a testing set.
X_project = project_scaled_df
y_project = project_df['TARGET CLASS']

X_train_project, X_test_project, y_train_project, y_test_project = train_test_split(X_project, y_project, test_size=0.3, random_state=101)

#Create a KNN model instance with n_neighbors=1
knn_project = KNeighborsClassifier(n_neighbors=1)

#Fit this KNN model to the training data.
knn_project.fit(X_train_project,y_train_project)

#Use the predict method to predict values using your KNN model and X_test.
predict_project = knn_project.predict(X_test_project)

#** Create a confusion matrix and classification report.**
print(confusion_matrix(y_test_project,predict_project))
print(classification_report(y_test_project,predict_project))

#Let's go ahead and use the elbow method to pick a good K Value!
#** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.**
errors = []

for i in range(1,40):
    knn_project = KNeighborsClassifier(n_neighbors=i)
    knn_project.fit(X_train_project,y_train_project)
    prediction_i = knn_project.predict(X_test_project)
    errors.append(np.mean(prediction_i != y_test_project))

#Now create the following plot using the information from your for loop.
plt.plot(range(1,40),errors,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title("Error Rate vs K Value")
plt.xlabel("K")
plt.ylabel("Error Rate")

plt.show()

#Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.
knn_project = KNeighborsClassifier(n_neighbors=31)
knn_project.fit(X_train_project,y_train_project)
predict_project = knn_project.predict(X_test_project)

print(classification_report(y_test_project,predict_project))
print("\n")
print(confusion_matrix(y_test_project,predict_project))






