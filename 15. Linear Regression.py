import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('USA_housing.csv')
desired_width = 450
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',10)
print(df.head())

#to get a list of the column names
print(df.columns)

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = df[['Price']]           #This is actually what we are trying to predict

#performing tuple unpacking here
#Why have 2 variables for X and 2 for y? Cos you need to show the train_test_split method that out of the total 'X_train' data, how much %
#will actually be used for testing? Therefore it makes sense to use 2 variables for each!
#test_size is the % of your data that you want to be allocated to the test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

#instantiate the model
lm = LinearRegression()

#fit the model ONLY to the test data - therefore specify that it must only fit to here. The fit takes only 2 inputs anyway.
lm.fit(X_train, y_train)
#finding the intercept of the linear regression
print(lm.intercept_)
#find the coefficients of the linear graph
print(lm.coef_)             #relate to the coefficients in X_train

############################################################################################################################
#Example of how to use the pd.DataFrame method:
#the first arguement is your data - note that it is a nested list - here I wanted to have 5 columns and 2 rows
#the index is the name of the row
#the column is obviously the name of the column - there must be the same number of names for rows and columns as there are data
df2 = pd.DataFrame([[1,2,3,4,5],[6,7,8]], index=['One','Two'], columns=['Col_1', 'Col_2', 'Col_3', 'Col_4', 'Col_5'])
print(df2.head())
#selecting a specific column
print(df2['Col_2'])
#selecting a specific row
print(df2.ix[1])

############################################################################################################################

cdf = pd.DataFrame(lm.coef_.transpose(), X.columns, columns=['Coeff'])
print(cdf)              #meaning - a 1 unit increase in Avg. Area income will result in an increase in price of the house by $21.52


#Obtaining predictions from the model
predictions = lm.predict(X_test)
print(predictions)          #prints predicted values of the prices of the houses

#Now you want to compare the actual prices of the houses (i.e. y_test) to the predicted values above from the model
print(y_test)

#plt.scatter(y_test, predictions)

plt.show()

#creating plot of the residuals - i.e. the actual values less the predicted
#sns.distplot((y_test-predictions))      #it should be normally distributed

#Mean absolute error
print(metrics.mean_absolute_error(y_test, predictions))
#Mean squared error
print(metrics.mean_squared_error(y_test, predictions))
#Root mean squared error
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))


plt.show()

###############################################################################################################################
#Project#
print('\n')

customers = pd.read_csv('Ecommerce Customers.txt')
#Convert the text file into a csv file
customers.to_csv(r'Ecommerce Customers.txt', index=None)

print(customers.head())
print(customers.describe())
print(customers.info())

#Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?
sns.jointplot(customers['Time on Website'], customers['Yearly Amount Spent'], data=customers, color='gray')

plt.show()

#Do the same but with the Time on App column instead.
sns.jointplot(customers['Time on App'], customers['Yearly Amount Spent'], data=customers, color='gray')

plt.show()

#Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.
sns.jointplot(customers['Time on App'], customers['Length of Membership'], data=customers, kind='hex')

plt.show()

#Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.(Don't worry about the the colors)
sns.pairplot(customers)

plt.show()

#Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)

plt.show()

#Training and testing the data
#Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# ** Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column. **
X2 = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y2 = customers['Yearly Amount Spent']

#** Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=101)

#Now its time to train our model on our training data!
#** Import LinearRegression from sklearn.linear_model **
#Create an instance of a LinearRegression() model named lm.
lm = LinearRegression()

#** Train/fit lm on the training data.**
lm.fit(X2_train, y2_train)

#Print out the coefficients of the model
print(lm.coef_)

#Now that we have fit our model, let's evaluate its performance by predicting off the test values!
#** Use lm.predict() to predict off the X_test set of the data.**
project_predictions = lm.predict(X2_test)
print(project_predictions)

sns.scatterplot(x=y2_test, y=project_predictions)

plt.show()

#Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
#** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**
print(metrics.mean_absolute_error(y2_test, project_predictions))
print(metrics.mean_squared_error(y2_test, project_predictions))
print(np.sqrt(metrics.mean_squared_error(y2_test, project_predictions)))

#You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data.
#Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().
#Plotting the difference between the actual data and the predicted data
sns.distplot(y2_test - project_predictions, bins=50)

plt.show()

#We still want to figure out the answer to the original question, do we focus our efforst on mobile app or
# website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.
# Let's see if we can interpret the coefficients at all to get an idea.
coefficients = pd.DataFrame(lm.coef_, index=X2.columns)
coefficients.columns = ['Coefficient']
print(coefficients)








