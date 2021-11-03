import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('fake_reg.csv')

print(df.head())

#The df could be a gemstone which based on its 2 features it will have a selling price.
#Can prepare a regression here to predict what the future price of a gemstone will be should it have given features.
sns.pairplot(df)

plt.show()

#Create the usual train-test split.
from sklearn.model_selection import train_test_split

#First grab the features going to use.
#NB!!!!!! Tensorflow needs the data to be in numpy arrays and not in a pandas dataframes/series as it is now.
#Therefore need to convert the data into a numpy array - which is what '.values' does.
#Use capital X as per convention typically the feature matrix is 2 dimensional so indicate this with a capital.
X = df[['feature1', 'feature2']].values

#Use lower case y here as per convention since it is a 1-dimensional vector use lower case.
y = df['price'].values

#The 'test_size' can be made to be smaller if the dataset is really large.
#The below will grab random rows and split them between the train and test sets. The number for 'random_state' is arbitrary.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Can confirm that the 'X_train' and 'X_test' splits have been performed by their shapes below.
#Remember that the original data had 1000 rows - therefore now the training set has 700 now which is 70% while test data has 300 - the 30%.
print(X_train.shape)
print(X_test.shape)

#Need to normalise and scale the feature data.
#Its mainly the features that need to be scaled and not the label - i.e. the 'y' values.
from sklearn.preprocessing import MinMaxScaler
#print(help(MinMaxScaler))

#Create an instance of the scaler first.
scaler = MinMaxScaler()

#Scaler fits onto the data in order to calculate the parameters it needs to perform the scaling afterwards.
#Only run it on the Training set in order to prevent 'data leakage' and not look within the actual test data.
scaler.fit(X_train)

#This is the 2nd step. Namely, within 'fit' you allow the sklearn to determine by how much the data needs to be adjusted/scaled by.
#Then below the data is actually transformed.
#fit = calculate whats needed whereas transform actually performs the scaling.
X_train = scaler.transform(X_train)

#Now do it for the test set.
X_test = scaler.transform(X_test)

#If print out the max and min values will see they are 1 and 0. Namely, the original data has been scaled to lie between 0 and 1.
#Remember, only fitting onto the Training data so as not to cheat and use the test data.

#######Part 2################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#help(Sequential)
#This is the 1st way of creating a Keras based model.
#A densely connect NN means that its a normal feed-forward network where every neuron is connected to every other neuron in the next layer.
#'Units' is how many neurons are in the layer.
#'Activation'is what type of activation function these neurons should be using eg: sigmoid etc.
# model = Sequential([Dense(4, activation='relu'),
#                     Dense(2, activation='relu')
#                     Dense(1)])

#The 2nd way to create a Keras based model.
#Here you create an empty sequential model and then off it you add the layers in one at a time.
#This is more convenient - its more easy to comment out a layer that you do not want to include in the model.
#Here will use a rectified linear unit (the 'relu') part.
#The final layer has only 1 neuron since you are only trying to predict one value/variable - in this case the price of the gemstone.
model = Sequential()

model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

#Compiling the model.
#For the optimizer - how do you want to use the gradient descent? eg: Adam.
#The 'loss' - what are eyou trying to accomplish? if performing a multi-class classification proble - use 'categorical_crossentropy'.
#If a binary classification proble - use the 'binary_crossentropy' loss.
#In our case we are performing a regression problem since our label is continuous. Here use the mean-squared-error loss 'mse'.
#This makes sense as we are taking the mean-squared error of the predictions versus the truee values and we are trying to minimise this with the optimizer.
model.compile(optimizer='rmsprop', loss='mse')

#Now need to still fit the compiled model to the training data.
#'x=' the features we are trying to train on.
#'y=' the actual training labels that correspond to those feature points.
# epochs - 1 epoch means that you have gone over the entire dataset one time.
# verbose - how much output info is printed out on the console. The higher the verbose number the info will be displayed.
model.fit(x=X_train, y=y_train, epochs=250)

#Will notice that the 'loss' from each epoch is reduced with epoch. i.e. the model becomes better at predicting the result.
#Can keep track of the losses from each epoch using the 'history' method.
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
plt.show()

#How well will this mmodel perform on data it hasnt seen before.
#The below will return the loss that has been decided on. i.e. on data it hasnt seen before, the model will have a mean-squared error of 24.97.
print(model.evaluate(X_test,y_test, verbose=0))

#This can also be done on the training data as well.
print(model.evaluate(X_train,y_train, verbose=0))

#Pass in the test features and predict what the price should be of the gemstone.
#Below is a list of the prices that it predicted based on the X_test set.
test_predictions = model.predict(X_test)

#Now lets bring in the true values of the test set and compare them to the predicted values.
test_predictions = pd.Series(test_predictions.reshape(300,))      #Will now be a pandas series instead of a numpy array.

#Remember, you need to have it in a series shape in order to convert it into a DataFrame!
pred_df = pd.DataFrame(y_test, columns=['Test True Y'])     #This only shows a df with the true value of y.

#Now want to simply add onto this dataframe the predicted values/prices (i.e. the test_predictions dataframe).
#Ensure joining the df's along axis=1 - which is along the columns.
pred_df = pd.concat([pred_df, test_predictions], axis=1)

#Add the name of the 2 columns for the above dataframe.
pred_df.columns = ['Test True Y', 'Model Predictions']

print(pred_df)

#Now can also plot these 2 against themselves - i.e. to compare the actual pricess of gemstones against predicted price.
sns.scatterplot(x='Test True Y', y='Model Predictions', data=pred_df)

plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

#The below mean absolutee error is about 4. This means that the prerdicted prices of the gemstones are out by around $4 on average compared to actual.
#Is this good or bad? If compare to original data - the average price of the gemstones is around $498 - which means that if the difference is $4, it is less than
#1% of the average value by which it is out by. Meaning it is a very good model.
print(mean_absolute_error(pred_df['Test True Y'], pred_df['Model Predictions']))

print(mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions']))

#If want the root mean-squared error take it to the ppower of 0.5.
print(mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions'])**0.5)

#Now lets try on completely new data. i.e. pretend that find a new gemstone with the below features and want to find its price.
new_gem = [[998, 1000]]

#Remember that the model uses scaled data!!!!!
#Therefore need to scale the features of the new gemstone.
new_gem = scaler.transform(new_gem)

print(model.predict(new_gem))

#If running complex model, can save the model.
from tensorflow.keras.models import load_model

#Saving the model in a hdmf5 format.
model.save('my_gem_model.h5')

#If want to use this same model in a different notebook.
later_model = load_model('my_gem_model.h5')

#Can then simply use the same model but to predict new data.
print(later_model.predict(new_gem))


