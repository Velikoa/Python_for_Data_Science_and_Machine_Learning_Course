import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

yelp = pd.read_csv('yelp.csv')

pd.set_option('display.width', 320)
pd.set_option('display.max_columns',10)

print(yelp.head())
print(yelp.describe())
print(yelp.info())

#Create a new column called "text length" which is the number of words in the text column.
yelp['text_length'] = yelp['text'].apply(len)
print(yelp.head())

#Use FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings.
#Reference the seaborn documentation for hints on this
#Here first splitting up to be categorised according to how many stars are given to a rating.
#Then in the '.map. method you need to stipulate the type of graph as well as in the case of a hist the x-axis.
#If a 'row=' is also stipulated then it means you want to create a FURTHER combination of the row plus the column.
# For eg: if row='text_length', it will create a graph for each unique number of words against each 1 to 5 stars.
# This is why the pc states there is not enough memory to perform this.
g = sns.FacetGrid(data=yelp, col='stars')
g.map(plt.hist, 'text_length')

plt.show()

#Create a boxplot of text length for each star category.
sns.boxplot(x='stars', y='text_length', data=yelp, palette='rainbow')

plt.show()

#Create a countplot of the number of occurrences for each type of star rating.
sns.countplot(x='stars', data=yelp, palette='rainbow')

plt.show()

#** Use groupby to get the mean values of the numerical columns, you should be able to create this dataframe with the operation:**
stars_grouped = yelp.groupby(by='stars').mean()
print(stars_grouped.head())

#Use the corr() method on that groupby dataframe to produce this dataframe:
corr_group = stars_grouped.corr()
print(corr_group)

#Then use seaborn to create a heatmap based off that .corr() dataframe:
sns.heatmap(data=corr_group, cmap='coolwarm', annot=True)       #annot shows the numerical values on the graph.

plt.show()

#Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.
#Remember to use the '|' sign instead of using 'or' otherwise will get an error.
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
print(yelp_class.head())

#** Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class. (Your features and target/labels)**
X = yelp_class['text']
y = yelp_class['stars']

#Import CountVectorizer and create a CountVectorizer object.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

#** Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X.**
X = cv.fit_transform(X)

#Let's split our data into training and testing data.
#** Use train_test_split to split up the data into X_train, X_test, y_train, y_test. Use test_size=0.3 and random_state=101 **
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Time to train a model!
#** Import MultinomialNB and create an instance of the estimator and call is nb **
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

#Now fit nb using the training data.
nb.fit(X_train, y_train)

#Time to see how our model did!
#Use the predict method off of nb to predict labels from X_test.
predictions = nb.predict(X_test)

#** Create a confusion matrix and classification report using these predictions and y_test **
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print('\n')
print(confusion_matrix(y_test, predictions))

#Great! Let's see what happens if we try to include TF-IDF to this process using a pipeline.
#** Import TfidfTransformer from sklearn. **
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer()),         #'bow' is the name of the process and then after the comma is thee actual model you will pass it through.
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

#Redo the train test split on the yelp_class object.
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Now fit the pipeline to the training data.
# Remember you can't use the same training data as last time because that data has already been vectorized. We need to pass in just the text and labels
pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

#** Now use the pipeline to predict from the X_test and create a classification report and confusion matrix. You should notice strange results.**
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))



