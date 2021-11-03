import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("kyphosis.csv")
print(df.head())

print(df.info())

sns.pairplot(data=df, hue='Kyphosis')
plt.show()

from sklearn.model_selection import train_test_split
X = df.drop("Kyphosis", axis=1)             #include everything from the df except the kyphosis column since want to know if thee patient still had the sickness or not.
y = df["Kyphosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Now going to test a single decision tree
from sklearn.tree import DecisionTreeClassifier

#Instantiate the model
dtree = DecisionTreeClassifier()

#Then fit the model to the training data (as with all other models)
dtree.fit(X_train,y_train)

#Now going to see how well the model is able to predict using the columns of Age, Number and Start whether the patient still had Kyphosis or not.
predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

#Now going to compare how these results are against a Random Forest model.
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train,y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test, rfc_pred))
print('\n')
print(classification_report(y_test, rfc_pred))


##################################Project###########################################################################

loans = pd.read_csv("loan_data.csv")
desired_width = 450
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',15)

print(loans.info())
print(loans.head())
print(loans.describe())

#** Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.**
#Tried using for loop but did not work since it would draw a graph for each and every instance!
loans[loans["credit.policy"] == 1]["fico"].plot.hist(bins=30, alpha=0.5, figsize=(10,4),color="blue", label="Credit.Policy=1")
#Basically saying - "I want the FICO column to be graphed when credit.policy column is equal to 1 and below is vice versa
#Alpha makes the histograms become see-through when they merge over each other.
loans[loans["credit.policy"] == 0]["fico"].plot.hist(bins=30, alpha=0.5, figsize=(10,4),color="red", label="Credit.Policy=0")
plt.legend()        #Can only call this if you have a 'label' in your histogram code above!
plt.xlabel("FICO")
plt.show()

#** Create a similar figure, except this time select by the not.fully.paid column.**
loans[loans["not.fully.paid"]==1]["fico"].plot.hist(bins=30, alpha=0.5, figsize=(10,4), color="blue", label="not.fully.paid=1")
loans[loans["not.fully.paid"]==0]["fico"].plot.hist(bins=30, alpha=0.5, figsize=(10,4), color="red", label="not.fully.paid=1")
plt.legend()        #This actually displays the legend on the graph
plt.xlabel("FICO")
plt.show()

#** Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid. **
plt.figure(figsize=(11,7))
sns.countplot(x="purpose", data=loans, hue="not.fully.paid")
#Note - if you do not indicate a hue, the couontplot will only count the number if instances each type of purpose happened.
#By indicating a further hue selection, you further show each purpose category broken down into whether it was fully paid or not.
plt.legend()
plt.xlabel("purpose")

plt.show()

#** Let's see the trend between FICO score and interest rate. Recreate the following jointplot.**
sns.jointplot(x="fico", y="int.rate", data=loans, color='purple')

plt.show()

#** Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy.
# Check the documentation for lmplot() if you can't figure out how to separate it into columns.**
sns.lmplot(x='fico', y='int.rate', data=loans, col='not.fully.paid', hue='credit.policy', palette='Set1')

plt.show()

#Notice that the purpose column as categorical
#That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies.
#Create a list of 1 element containing the string 'purpose'. Call this list cat_feats.
cat_feats = ['purpose']

#Now use pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe that
# has new feature columns with dummy variables. Set this dataframe as final_data.
final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)
print(final_data.info())

#** Use sklearn to split your data into a training set and a testing set as we've done in the past.**
X_project = final_data.drop('not.fully.paid', axis=1)       #No need to say loans['not.fully.paid'] in the round brackets since its already within final_data.drop!
y_project = final_data['not.fully.paid']                    #This is what we are trying to predict!
X_train, X_test, y_train, y_test = train_test_split(X_project, y_project, test_size=0.3, random_state=101)

#Let's start by training a single decision tree first!
#** Import DecisionTreeClassifier**
#Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.
project_dtree = DecisionTreeClassifier()

project_dtree.fit(X_train, y_train)

#Create predictions from the test set and create a classification report and a confusion matrix.
project_predictions = project_dtree.predict(X_test)

print(classification_report(y_test, project_predictions))
print('\n')
print(confusion_matrix(y_test, project_predictions))

#Now its time to train our model!
#Create an instance of the RandomForestClassifier class and fit it to our training data from the previous step.
project_random_forest = RandomForestClassifier(n_estimators=600)

project_random_forest.fit(X_train, y_train)

#Let's predict off the y_test values and evaluate our model.
#** Predict the class of not.fully.paid for the X_test data.**
project_predictions = project_random_forest.predict(X_test)

#Now create a classification report from the results. Do you get anything strange or some sort of warning?
print(classification_report(y_test, project_predictions))
print('\n')
print(confusion_matrix(y_test, project_predictions))






