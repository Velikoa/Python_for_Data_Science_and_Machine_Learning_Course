import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('kc_house_data.csv')

pd.set_option('display.width', 350)
pd.set_option('display.max_columns', 15)

#Find out if there is any missing data.
#Use the df.isnull() to find out if the data is missing (will print as True)or it is not (will print False).
print(df.isnull().sum())

print(df.describe().transpose())

plt.figure(figsize=(10,6))
sns.distplot(df['price'])

plt.show()

#Below can use a countplot graph since the number of bedrooms is categorical.
sns.countplot(df['bedrooms'])

plt.show()

#When trying to predict price (i.e. your label), you should use some sort of feature which has a correlation to the label being predicted.
#Therefore, what can be done here is find the correlation for all these features/variables.
#Then sort the correlations to find the one most highly correlated and plot that.
print(df.corr()['price'].sort_values())

#Give graph bit larger.
plt.figure(figsize=(10,5))
sns.scatterplot(x='price', y='sqft_living', data=df)

plt.show()

#Can also do boxplots of various correlated features in order to visualise the distribution of the values.
plt.figure(figsize=(10,6))
sns.boxplot(x='bedrooms', y='price', data=df)

plt.show()

#Trying to understand whether the geographical location of a house will impact the price of the house below.
plt.figure(figsize=(12,8))
sns.histplot(x='price', y='long', data=df)

plt.show()

#Repeat the above distribution to see if latitude has any effect on house price - i.e. where are the most expensive houses.
plt.figure(figsize=(12,8))
sns.histplot(x='price', y='lat', data=df)

plt.show()

#Now will plot a scatterplot of longitude and latitude in order to find the exact locations/areas with most expensive houses.
#If 'hue='pricee'' is not included in the graph specs below, the graph will only print the long and lat - but when included it will
#highlight/bolden the coordinates where prices are highest witha different hue.
plt.figure(figsize=(12,8))
sns.scatterplot(x='long', y='lat', data=df, hue='price')

plt.show()

#The above graph still is not clearly showing the information needed as the outliers are causing large gaps to appear.
#Sorting the data according to price starting with most expensive and showing the top 20.
print(df.sort_values('price', ascending=False).head(20))

#Below is shpwing the top 1% of house prices.
print(len(df)*0.01)

#Need to remove the top valued houses.
#Basically saying, you want to select only houses from the 217th house onwards (after it has been sorted from highest to lowest).
#Only plotting the bottom 99% of houses in order to get a more clear colour distribution graph.
non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]

#Will use same code as above but now going to remove the white edge colour as well as since there are so many points over each
#other, the alpha will be placed as 0.2.
plt.figure(figsize=(12,8))
sns.scatterplot(x='long', y='lat', data=non_top_1_perc, edgecolor=None, alpha=0.2, palette='RdYlGn', hue='price')

plt.show()

#Create a boxplot on whether something is on the waterfront or not.
#The below shows that if you are on the waterfront the house will be more expensive which is right.
sns.boxplot(x='waterfront', y='price', data=df)

plt.show()

##############Part 2###########################

#Feature engineering here - in other words, removing features that you do not need from the dataframe.
#Below removing the 'id' column as it adds no value.
#Removing the entire column, therefore the axis is 1.
df = df.drop('id', axis=1)

#When you inspect the 'date' column, notice that it is in a string format.
#Below, we are chnaging it to a 'datetime' object.
#Why is this important? 'cos once it is in datetime format, you can extract the day or month, etc. from the data.
df['date'] = pd.to_datetime(df['date'])

#Now can extract the year or month component from the above adjusted dates.
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)

#The above lambda expression is the same as the function below:
# def year_extraction(date):
#     return date.year

#Trying to find out whether the month in which a house is sold has any impact on the price at which it was sold at.
plt.figure(figsize=(10,6))
sns.boxplot(x='month', y='price', data=df)
plt.show()

#The above boxplot is not clear to understand whether month has any impact. Therefore, going to use below code to analyse numbersmore closely.
#Finding the average price of a house in each month.
#Can even just use the .plot as a shortcut to very quickly graph the information.
df.groupby(by='year').mean()['price'].plot()
plt.show()

#Dropping the date column since already have the month and year columns now.
df = df.drop('date', axis=1)

#NB!!!! Remember, when the data is in the form of a number, pandas will assumee that it is ccontinuous and that for eg: zipcode
#98103 is greater or better than zipcode 98038 when in fact these are categorical pieces off data!
#Below finds how many values there are of each zipcode.
#print(df['zipcode'].value_counts())
#There are 70 different zipcodes which is too many and no value in mapping them all onto a map. However, in real world it would be good to consider this also.
df = df.drop('zipcode', axis=1)

#The year_renovated column has the majority of the values as 0. Meaning that there were not any renovations done actually.
#It would make more sense that as opposed to having the years as data and a 0 (where not renovated), can instead convert this column into
#a categorical data (of renovated or not renovated) using the custom apply function.
#In this case, however, it makes sense that the newer the renovation is, the more valuable it is. Therefore, actually as years/date increases, value is right to also go up.
#Can therefore leave the column as is.

#The square feet basement column also has th logic here - in that where there is a 0, it means there is no basement.
#Can also keep this column as is since makes sense that as the value of the square footage of the basement increases, the price of the
#house also increases. Therefore, the assumption that pandas follows (of assuming that as the digit value of the column is higher, it means it is more).
#this is all important as there are many instances when need to decide whether a column/feature needs to be changed to categorical from continuous.

#############Data Preprocessing and Creating of Model####################

#Separate our label from our features.
#Use .values (dont actually always have to) in order to make sure no issues between numpy arrays and pandas series. Therefore, it returns a numpy array.
X = df.drop('price', axis=1).values
y = df['price'].values

from sklearn.model_selection import train_test_split

#Do the test-train split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Perform the scaling.
# NB!!!! This is done AFTER the test-train split in order to ensure there is no data leakage.
# What is data leakage?
# Data leakage is when information from outside the training dataset is used to create the model.
# This additional information can allow the model to learn or know something that it otherwise would not know and
# in turn invalidate the estimated performance of the mode being constructed.
# If you normalize or standardize your entire dataset, then estimate the performance of your model using
# cross validation, you have committed the sin of data leakage.
# The data rescaling process that you performed had knowledge of the full distribution of data in the training
# dataset when calculating the scaling factors (like min and max or mean and standard deviation).
# This knowledge was stamped into the rescaled values and exploited by all algorithms in your cross validation test harness.

from sklearn.preprocessing import MinMaxScaler

# Create an instance of the scaler function.
scaler = MinMaxScaler()

# Previously the data was first fitted and then tranformed. However, there is a shortcut below that does both immediately.
X_train = scaler.fit_transform(X_train)

# Now transforming thee test set. NB!!!! We are NOT fitting to our test set since we do not want to assume prior info about the test set.
X_test = scaler.transform(X_test)

# Now creating the model.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# How do you know how many neurons should be in the model and how many layers?
print(X_train.shape)        #Shows that there are 19 features/columns - so use this as the neurons.

# What is an activation function?
# In deep learning, the process is that you take the weights of each neuron and multiply them by their feature or X variable.
# Then you add up the weighted values from each neuron in that layer and pass those weighted sums into an activation function which then
# transforms it into a number between a lower and upper limit.
# The popular activation functions are sigmoid, tanh and relu. Relu is the most accurate and most used now.
# How does relu work?
# It will take the weighted sum of the neurons and then say - "if the answers are less than 0, it will return 0. If the answers are
# more than 0, it will return the X value itself and feed that number on to the next layer of neurons."
# What is the logic here? Namely that if the number is negative, it means that the brains neuron is not firing. If it is more than 0
# and the higher the number is, the more powerful the neuron is firing. (think of it as a brain functioning).

# model = Sequential()
# model.add(Dense(19, activation='relu'))
# model.add(Dense(19, activation='relu'))
# model.add(Dense(19, activation='relu'))
# model.add(Dense(19, activation='relu'))

#model.add(Dense(1))         #The last layer has only 1 neuron since predicting only one value.

#Compile the model.
# Since it is a regression problem we use mean squared error (MSE) for our loss metric.
#model.compile(optimizer='adam', loss='mse')

# Now going to fit the model.
# Pass in 'validation data' as a criteria - after each epoch of training on the training data, the model will quickly rrun the test data and check the loss on the test data.
# This means can now see how the model is performing on thee training and test data.
# Keras will still only use the training data to update the weights and biases though (which is correct).
# Just make sure the data is suitable for tensorflow (i.e. it is a numpy array) - hence use .values above.
# Since this is a big data set, we  are going to feed the data into the model in batches.
# Usually batch sizes are in powers of 2.
# The smaller the batch size, the longer the training it takes but the less likley you will overfit to your data since you are not passing in all your data at once.
#model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=128, epochs=400)

# In order to not have to re-run the entire model in keras/tf all over again each time, the above line has been commented out
# and the result of the model are saved below as an H5 format file.
from tensorflow.keras.models import load_model

#model.save('house_pricing_model.h5')
model = load_model('house_pricing_model.h5')

# Can find out what the history of losses is for the model. Then pass it to be viewed as a dataframe.
# However, cannot print and plot it below i Pycharm since would then need to run the entire fitting again since history.history only works on the model.fit() instance.
#losses = pd.DataFrame(model.history.history)
# losses.plot()
# From this graph, you can see that the validation loss and the loss lines are decreasing with each epoch. This means that there is no overfitting risk here
# but the number of epochs can actually be increased. If however, there is a spike or increasse in the lines over epochs, its a sign of overfitting.

# Now going tot do an evaluation.
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

predictions = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, predictions)))

# Mean absolute error is what is your average absolute error across all your predictions.
print(mean_absolute_error(y_test, predictions))

# In this case we are out by around $100k.
# Is this a good or bad prediction? Refer to the mean of the original dataset and then compare it to the difference predicted here.
print(df['price'].describe())
# The mean here is around $540 296 which means we are out by around 20% which is not very good.

# This explains how much of the variance is explained by our model. 1 is the best result and anything lower is showing that the model is worse at predicting.
print(explained_variance_score(y_test, predictions))

# Compare the label to the predictions.
plt.figure(figsize=(12, 6))
plt.scatter(y_test, predictions)
# In a perfect model the y_test and the predictions will be a perfect straight line.
plt.plot(y_test, y_test, 'r')
plt.show()

# Below will try to predict the price of a house using the specs of the first house in the original dataset df.
single_house = df.drop('price', axis=1).iloc[0]

# Remember that the model has been trained on scaled features. Therefore cannot simply pass the above single_house features as they are.
# .reshape() takes a list/array and shapes it in a specific manner - splits the data into 19 columns and -1 means the number of rows is unknown.
single_house = scaler.transform(single_house.values.reshape(-1, 19))

# Now can actually use our model to predict a price for the features in single_house above.
print(model.predict(single_house))

# the house actually sold for $221 900 if you look at df.head(1) meaning we are predicting slightly higher prices than actual.
# To fix this, can re-train the model again (using more epochs) and even reduce the top 1 or 2% of expensive houses.

