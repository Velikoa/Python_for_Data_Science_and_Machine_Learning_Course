import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cancer_classification.csv')

pd.set_option('display.width', 350)
pd.set_option('display.max_columns', 15)

# CHeck if there are any null values.
print(df.info())
print(df.describe().transpose())

# For classification problems/predictions, it is good idea to start with a countplot to see how many instances there are for your label for each feature.
sns.countplot(x='benign_0__mal_1', data=df)
plt.show()

#Find the correlations between the features.
# Can also plot it with sns if want.
# Grabbing all the columns except the last one since perfectly correlated to itself.
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')
plt.show()

# Can also do a heatmap on the correlation if easier to understand.
plt.figure(figsize=(12,12))
sns.heatmap(df.corr())
plt.show()

# Preparing our test/train split.
# Remember to use .values so it is a numpy array.
X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

from sklearn.model_selection import train_test_split

# Since it is not a very big df, can decrease the test_size to be 25% only.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Fit the training data and transform/scale it.
X_train = scaler.fit_transform(X_train)

# Remember - no fitting done on the test data in order to prevent data leakage.
X_test = scaler.transform(X_test)

#########Dealing with Overfitting, building model and evaluation#############

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#Using .shape below, can see that the df has 426 rows and 30 columns/features.
print(X_train.shape)

#model = Sequential()

#Going to create layers with 30 neurons and using the relu (rectified linear unit) activation function.
# Note that using 30 neurons in the first layer due to there being 30 features.
# The 2nd layer has only 15 - this is arbitrary and can also have used 30 again. However, since the data is relatively simple/small, can reduce the
# Number of neurons slowly with each layer before it reaches 1 (the last neuron).
# model.add(Dense(30, activation='relu'))
# model.add(Dense(15, activation='relu'))

# NB!!!!!! The last layer will only have one neuron (as expected since only want one answer) and extremely important to note that
# the activation function is sigmoid! Why? 'cos it is a BINARY CLASSIFICATION PROBLEM.
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam')

# Now time to train the model.
# Running it for 600 epochs - this is too many but the point is to show what happens when you overtrain and overfit a model.
# model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test))

# model.save('cancer_classification_model.h5')

from tensorflow.keras.models import load_model
#model = load_model('cancer_classification_model.h5')

# Plotting out the loss.
# Cannot do it in our case in Pycharm.
# losses = pd.Dataframe(model.history.history.plot())
# losses.plot()
# plt.show()
# From the graph here, can see that in the first couple of epochs, BOTH the validation and training losses are decreasing. Meaning we have not overfitted yet.
# However, later the training loss is still going down but the validation loss is increasing - this means that the model is overfit.

# Going to use callbacks from tensorflow to stop the training beforer it gets overfitted.
# Need to create a NEW model - but can still use the same code as originally.
# model = Sequential()
#
# model.add(Dense(30, activation='relu'))
# model.add(Dense(15, activation='relu'))

# NB!!!!!! The last layer will only have one neuron (as expected since only want one answer) and extremely important to note that
# the activation function is sigmoid! Why? 'cos it is a BINARY CLASSIFICATION PROBLEM.
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='adam')

# How to use callbacks to stop the model from overtraining/overfitting.
from tensorflow.keras.callbacks import EarlyStopping

# Using the Earlystopping callback has 2 steps.
# First define EarlyStopping as a variable.
# Modes - what are you trying to do? Are you trying to maximize or minimize the thing you are tracking.
# In this case, since we are tracking losses - we want the loss to be minimized. If your metric is accuracy, this will need to be maximized.
# verbose - gives a report back of what has happened.
# patience - how many epochs will you wait/continue to train for even after it has been detected that there is variation between X_train and validation.
# Why is this? Due to there still being some 'noise' in the validation data which means do not want to immediately assume that the model is overfitting.
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

# The only difference now is that 'callbacks' is specified and passed as a list.
# Now the model will still attempt to run on 600 epochs unless it is triggered to stop by the callback.
# model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stop])
# Will note that the model stopped training after 83 epochs.
# Now if you view the graph of val loss against X_train losses, they are much closer to each other.

# model.save('cancer_classification_updated.h5')
#
# model = load_model('cancer_classification_updated.h5')

# The 3rd thing that can be done to prevent overfitting is to add dropout layers. This means that random neurons will be turned off.
# After every 'add' of a layer, will add the Dropout. rate - probablity that you will randomly turn a neuron off. If 0, you are taking 0% and turrning 0% off.
# The common rate for Dropout is between 0.2 and 0.5. Basically if 0.5, means that 50% of the neurons in each batch size will be turned off i.e. their
# weights and biases will not be getting updated.

from tensorflow.keras.layers import Dropout


# model = Sequential()
#
# model.add(Dense(30, activation='relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(15, activation='relu'))
# model.add(Dropout(0.5))

# NB!!!!!! The last layer will only have one neuron (as expected since only want one answer) and extremely important to note that
# the activation function is sigmoid! Why? 'cos it is a BINARY CLASSIFICATION PROBLEM.
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='adam')

# Note, we will still use EarlyStopping as defined above already.
# model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stop])

# This time the model trained for more epochs - makes sense since was removing neurons each time which means adjusting more effectively.

# model.save('cancer_classification_with_dropout.h5')

model = load_model('cancer_classification_with_dropout.h5')

# Evaluating the model.
# Since this is a classification model, no longer saying model.predict - now need to say model.predict_classes!
predictions = model.predict_classes(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))
# The confusion matrix that only 1 point has been incorrectly misclassified.
print(confusion_matrix(y_test, predictions))
