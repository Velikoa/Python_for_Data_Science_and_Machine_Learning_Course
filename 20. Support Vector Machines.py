import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#sklearn comes with a few basic datasets built into the library so that you do not need to download your own ones.
from sklearn.datasets import load_breast_cancer

pd.set_option('display.width', 450)
pd.set_option('display.max_columns', 25)

#load an instance of the dataset
cancer = load_breast_cancer()

print(cancer.keys())

#if want a detailed descriptioin of the dataset use the below code
#print(cancer['DESCR'])

#going to set up a dataframe using these keys to grab the data
df_feats = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print(df_feats.head(2))

from sklearn.model_selection import train_test_split

print(cancer['target'])         #this just shows the data as zero or one - i.e. is it malignant or benign

X = df_feats
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.svm import SVC

#instantiate the model
model = SVC(gamma='auto')       #changed gamma to auto in order to get same results as in the lecture video

#fit the training set of data into the model
model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

#Grid search allows you to find what the optimal values for C and Gamma are for your SVC
from sklearn.model_selection import GridSearchCV

#NB!!! Need to do a grid search in order to find the best parameters!
#C - controls the cost of misclassification on the training data. Large C value give you low bias and high variance.
#Gamma - small gamma means a gausian with a large variance. Large gamma leads to high bias and low variance in the model.
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}               #in order to let sklearn choose the best values for you

grid = GridSearchCV(SVC(), param_grid, verbose=3)

grid.fit(X_train, y_train)

#Use the below to show what the best combination of parameters are for the cross validation score.
print(grid.best_params_)

print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test, grid_predictions))
print('\n')
print(classification_report(y_test, grid_predictions))


##############################Project#############################################################################

#*Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') *
iris = sns.load_dataset('iris')

#** Create a pairplot of the data set. Which flower species seems to be the most separable?**
sns.pairplot(iris, hue='species', palette='Dark2')

plt.show()

#Create a kde plot of sepal_length versus sepal width for setosa species of flower.
setosa = iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'], cmap='plasma', shade=True, shade_lowest=False)

plt.show()

#** Split your data into a training set and a testing set.**
X_project = iris.drop('species', axis=1)
y_project = iris['species']
X_train_project, X_test_project, y_train_project, y_test_project = train_test_split(X_project, y_project, test_size=0.3)

#Now its time to train a Support Vector Machine Classifier.
#Call the SVC() model from sklearn and fit the model to the training data.
project_model = SVC()

project_model.fit(X_train_project, y_train_project)

#Now get predictions from the model and create a confusion matrix and a classification report.
project_predictions = project_model.predict(X_test_project)

print(confusion_matrix(y_test_project, project_predictions))
print('\n')
print(classification_report(y_test_project, project_predictions))

#Create a dictionary called param_grid and fill out some parameters for C and gamma.
param_grid_project = {'C':[0.1, 1, 10, 100], 'gamma':[1, 0.1, 0.01, 0.001]}

#** Create a GridSearchCV object and fit it to the training data.**

grid_project = GridSearchCV(SVC(), param_grid_project, verbose=2)
grid_project.fit(X_train_project, y_train_project)

#** Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them. Were you able to improve?**
project_predictions = grid_project.predict(X_test_project)

print(confusion_matrix(y_test_project, project_predictions))
print('\n')
print(classification_report(y_test_project, project_predictions))





