import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("titanic_train.csv")
desired_width = 450
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',15)
print(train.head())

#Use seaborn to create a heatmap in order to see where the largest amount of data is missing so then you can clean it up afterwards.
#Use the 'isnull()' method to check whether there is data in a certain row or not
print(train.isnull())

#sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

#plt.show()

sns.set_style('whitegrid')

#simply check who survived and who did not. By adding the 'hue' you can further split the graph to show how many males vs females survived or didnt.
#sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')
#plt.show()

#sns.countplot(x='Survived', hue='Pclass', data=train)
#plt.show()


#To determine the ages of the passengers on the Titanic you can create a distribution plot.
#sns.distplot(train['Age'].dropna(), kde=False, bins=30)

#plt.show()

#Can also do the above using Pandas own visual tool instead of using seaborn.
#train['Age'].plot.hist(bins=30)

#plt.show()

#To keep track of the data column names just constantly call the pd.info() nethod
print(train.info())

#sns.countplot(x='SibSp', data=train)

#plt.show()

#train['Fare'].hist(bins=40, figsize=(10,4))

#plt.show()

########################################Second video######################################

#Create a boxplot and then impute the missing Age values using the average age - can use the average age by passenger class here.
#plt.figure(figsize=(10,7))
#sns.boxplot(x='Pclass', y='Age', data=train)

#plt.show()

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:         #if missing value is in 1st class - return average age of 37
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

#sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')     #now in the heatmap there wont be any yellow items indicating missing info

#plt.show()

#drop the Cabin column since simply too many missing values here to fix.
train.drop('Cabin',axis=1,inplace=True)

#drop any remaining missing values
train.dropna(inplace=True)

#now going to create dummy variable so the algo can read the letters/terms as numbers in order to better categorise them.
#the get_dummies meethod takes a column of data and converts it into a 1 or 0
sex = pd.get_dummies(train['Sex'], drop_first=True)           #to prevent multicollinearity have the drop_first=True

embark = pd.get_dummies(train['Embarked'],drop_first=True)

print(embark)

train = pd.concat([train,sex,embark],axis=1)            #axis=1 means you are adding the new info as a new column into the data

print(train.head(2))        #now you will see the 3 new columns at the end of the table

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)          #axis=1 since you are deleting columns here
train.drop(['PassengerId'],axis=1,inplace=True)

print(train.head(2))


###########################################Part 3########################################################

#first prepare the data into training data and test data
X = train.drop('Survived',axis=1)       #this is basically every other column in the dataset except for the Survived column.
y = train['Survived']                   #y is the data you are trying to predict

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()         #create an instance of a logistic regression model
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report           #tells you your precision, recall, etc.

print(classification_report(y_test,predictions))

#if want to use the confusion matrix instead
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)

################################################Exercise##########################################################

ad_data = pd.read_csv('advertising.csv')
print(ad_data.head())

print(ad_data.info())
print(ad_data.describe())

#** Create a histogram of the Age**
ad_data['Age'].hist(bins=30, figsize=(10,7))
plt.xlabel('Age')

plt.show()

#Create a jointplot showing Area Income versus Age.
sns.jointplot(x='Age',y='Area Income', data=ad_data)

plt.show()

#sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde',color='red')

#plt.show()

#sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')

#plt.show()


#** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**
#sns.pairplot(data=ad_data,hue='Clicked on Ad', palette='bwr')

#plt.show()

#** Split the data into training set and testing set using train_test_split**
X2 = ad_data.drop('Daily Time Spent on Site',axis=1)            #should consider dropping any columns which do NOT have numerical values!
y2 = ad_data['Daily Time Spent on Site']

X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size=0.3, random_state=101)      #30% of the total data is for testing and 70% for training the model

#** Train and fit a logistic regression model on the training set.**
logmodel2 = LogisticRegression()
logmodel2.fit(X2_train,y2_train)

#** Now predict values for the testing data.**
predictions2 = logmodel2.predict(X2_test)

#** Create a classification report for the model.**
print(classification_report(y2_test,predictions2))




