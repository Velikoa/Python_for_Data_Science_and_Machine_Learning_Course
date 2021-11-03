import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# The 'loan_status' column contains our label!

data_info = pd.read_csv('lending_club_info.csv', index_col='LoanStatNew')
print(data_info.head())
print(data_info.loc['revol_util']['Description'])

# Function that prints the description related to a specific column name when you specify the column name. Basically summary of above .loc call.
def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])

feat_info('mort_acc')


df = pd.read_csv('lending_club_loan_two.csv')
print(df.info())

##########Section 1 - Exploratory Data Analysis#################################

pd.set_option('display.width', 350)
pd.set_option('display.max_columns', 30)

# TASK: Since we will be attempting to predict loan_status, create a countplot as shown below.
# Remember - it is a good idea to always start off creating a countplot when have a classification problem.
# sns.countplot(x='loan_status', data=df)
# plt.show()

# TASK: Create a histogram of the loan_amnt column.
# plt.figure(figsize=(12, 4))
# sns.histplot(data=df, x='loan_amnt', bins=50, kde=False)
# plt.xlim(0, 45000)
# plt.show()

# TASK: Let's explore correlation between the continuous feature variables. Calculate the correlation between all
# continuous numeric variables using .corr() method.
corr = df.corr()
print(corr)

# TASK: Visualize this using a heatmap. Depending on your version of matplotlib, you may need to manually adjust the heatmap.
# The 'cmap' stands for 'colour mapping'.
# plt.figure(figsize=(12, 7))
# sns.heatmap(data=corr, annot=True, cmap='viridis')
# If not all the data is showing on the heatmap, can extend the y axis to be greater by playing around with the plt.ylim().
# plt.ylim(10, 0)
# plt.show()

# TASK: You should have noticed almost perfect correlation with the "installment" feature. Explore this feature
# further. Print out their descriptions and perform a scatterplot between them. Does this relationship make sense to you? Do you think there is duplicate information here?
# Why is it important to ensure these 2 columns are not duplicating info? 'cos if they are, when you create the predictive model,
# it will use itself/the answers to predict itself making it unreliable.
print(feat_info('installment'))
print(feat_info('loan_amnt'))

# plt.figure(figsize=(12, 7))
# sns.scatterplot(x='installment', y='loan_amnt', data=df)
# plt.show()

# TASK: Create a boxplot showing the relationship between the loan_status and the Loan Amount.
# sns.boxplot(x='loan_status', y='loan_amnt', data=df)
# plt.show()

# TASK: Calculate the summary statistics for the loan amount, grouped by the loan_status.
print(df.groupby('loan_status')['loan_amnt'].describe())

# TASK: Let's explore the Grade and SubGrade columns that LendingClub attributes to the loans. What are the unique possible grades and subgrades?
# NB!!!! Note that can use the 'sorted' method into which you pass the df and relevant columns if want to sort the data.
print(sorted(df['grade'].unique()))
print(sorted(df['sub_grade'].unique()))

# TASK: Create a countplot per grade. Set the hue to the loan_status label.
# sns.countplot(x='grade', hue='loan_status', data=df)
# plt.show()

# TASK: Display a count plot per subgrade. You may need to resize for this plot and reorder the x axis.
# Feel free to edit the color palette. Explore both all loans made per subgrade as well being separated based on the loan_status.
# After creating this plot, go ahead and create a similar plot, but set hue="loan_status"
# plt.figure(figsize=(12, 4))
# sub_grade_df = sorted(df['sub_grade'].unique())
# sns.countplot(x='sub_grade', data=df, order=sub_grade_df, palette='coolwarm')
# plt.show()

# plt.figure(figsize=(12, 4))
# sns.countplot(x='sub_grade', hue='loan_status', data=df, order=sub_grade_df, palette='coolwarm')
# plt.show()

# TASK: It looks like F and G subgrades don't get paid back that often. Isloate those and recreate the countplot just for those subgrades.
# f_and_g = df[(df['grade'] == 'G') | (df['grade'] == 'F')]
# plt.figure(figsize=(12, 4))
# subgrade_order = sorted(f_and_g['sub_grade'].unique())
# sns.countplot(x='sub_grade', data=f_and_g, order=subgrade_order, hue='loan_status')
# plt.show()

# TASK: Create a new column called 'loan_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".
def loan_repayment(repaid):
    if repaid == 'Fully Paid':
        return 1
    else:
        return 0


# Remember - when using .apply and have a function already made, do not need to run it as a function - i.e. do NOT use () on the function!
# The function above has the 'repaid' variable within it but it can be called anything. It basically is what the element is within the column
# the calculation is carried out upon - in the case the 'loan_status' column.
df['loan_repaid'] = df['loan_status'].apply(loan_repayment)
print(df[['loan_repaid', 'loan_status']])

# CHALLENGE TASK: (Note this is hard, but can be done in one line!) Create a bar plot showing the correlation of the
# numeric features to the new loan_repaid column. Helpful Link
# In this case, when dropping the 'loan_repaid' column, no need to specify the axis since the data is a series.
# df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
# plt.show()

##########Section 2 - Data PreProcessing###################

# Section Goals: Remove or fill any missing data. Remove unnecessary or repetitive features. Convert categorical string features to dummy variables.
df.head()
# Let's explore this missing data columns. We use a variety of factors to decide whether or not they would be useful, to see
# if we should keep, discard, or fill in the missing data.
# TASK: What is the length of the dataframe?
len(df)

# TASK: Create a Series that displays the total count of missing values per column.
# If you only state 'df.isnull()' this will show all the rows and all columns and have True or False at each element.
df.isnull().sum()

# TASK: Convert this Series to be in term of percentage of the total DataFrame
df.isnull().sum() / 396030 * 100

# TASK: Let's examine emp_title and emp_length to see whether it will be okay to drop them. Print out their feature
# information using the feat_info() function from the top of this notebook.
# NB!!!! If you want to know what the row/index numbers are where all the values of a specific column are NA or NaN use .index.to_list()
df[df['mort_acc'].isnull()].index.to_list()

print(feat_info('emp_title'))
print(feat_info('emp_length'))

# TASK: How many unique employment job titles are there?
# There are 173 105 unique job titles. It does not make sense to create dummy variables out of all these since then you
# would have 173 105 extra columns which is not logical.
df['emp_title'].nunique()
# The value_counts() function shows how many instances there are of each job title.
df['emp_title'].value_counts()

# TASK: Realistically there are too many unique job titles to try to convert this to a dummy variable feature. Let's remove that emp_title column.
df = df.drop('emp_title', axis=1)

# TASK: Create a count plot of the emp_length feature column. Challenge: Sort the order of the values.
# sorted(df['emp_length'].dropna().unique())
# emp_length_order = ['1 year', '10+ years', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '< 1 year']
# plt.figure(figsize=(12, 4))
# sns.countplot(x='emp_length', data=df, order=emp_length_order)
# plt.show()

# TASK: Plot out the countplot with a hue separating Fully Paid vs Charged Off
# sns.countplot(x='emp_length', data=df, order=emp_length_order, hue='loan_status')
# plt.show()

# CHALLENGE TASK: This still doesn't really inform us if there is a strong relationship between employment length and being charged off,
# what we want is the percentage of charge offs per category. Essentially informing us what percent of people per employment category
# didn't pay back their loan. There are a multitude of ways to create this Series. Once you've created it, see if visualize it with a bar plot.
# This may be tricky, refer to solutions if you get stuck on creating this Series.
df_by_emp_length = df.groupby('emp_length')['loan_status'].value_counts()
print(df_by_emp_length)

fully_paid = df[df['loan_status'] == 'Fully Paid'].groupby('emp_length').count()['loan_status']
charged_off = df[df['loan_status'] == 'Charged Off'].groupby('emp_length').count()['loan_status']
# This is the direct ratio between the two - otherwise would need to say charged off / (charged off + fully paid).
per_charged_off = charged_off / (fully_paid)
print(per_charged_off)

# per_charged_off.plot(kind='bar')
# plt.show()

# TASK: Charge off rates are extremely similar across all employment lengths. Go ahead and drop the emp_length column.
df = df.drop('emp_length', axis=1)

# TASK: Revisit the DataFrame to see what feature columns still have missing data.
# NB!!! Use .sum() instead of .count() at the end when adding up all the null values as .count() will add up BOTH the True and False instances
# of what we are asking - namely both when a value is null and it is not null. Whereas .sum() only adds up instances of what we are asking for.
print(df.isnull().sum())

# TASK: Review the title column vs the purpose column. Is this repeated information?
print(df['purpose'].head(10))
print(df['title'].head(10))

# TASK: The title column is simply a string subcategory/description of the purpose column. Go ahead and drop the title column.
df = df.drop('title', axis=1)

# NOTE: This is one of the hardest parts of the project! Refer to the solutions video if you need guidance, feel free to fill or drop the
# missing values of the mort_acc however you see fit! Here we're going with a very specific approach.
# TASK: Find out what the mort_acc feature represents
print(feat_info('mort_acc'))

# TASK: Create a value_counts of the mort_acc column.
print(df['mort_acc'].value_counts())

# TASK: There are many ways we could deal with this missing data. We could attempt to build a simple model to fill it in,
# such as a linear model, we could just fill it in based on the mean of the other columns, or you could even bin the columns
# into categories and then set NaN as its own category. There is no 100% correct approach!
# Let's review the other columsn to see which most highly correlates to mort_acc
# NB!!! Note difference between using sorted(df.corr()['mort_acc']) vs using .sort_values! sorted() created a list of
# the correlations whereas now its in a column.
# Cannot simply drop the mort_acc column since the missing values here are around 10% of the total population.
# We are trying to understand which of the other columns has a high correlation to mort_acc column in order to use it
# as a way in which the missing values for mort_acc to be filled using the other accounts it is correlated to.
mort_corr = df.corr()['mort_acc'].sort_values()
print(mort_corr)

# TASK: Looks like the total_acc feature correlates with the mort_acc , this makes sense! Let's try this fillna() approach.
# We will group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry. To get the result below:

# NB!!! Use the enumerate function to loop through the list of column headings which are created from df.columns!
for index, column_header in enumerate(df.columns):
    print(index, column_header)

print("\n Mean of mort_acc per total acc.")
print(df.groupby('total_acc')['mort_acc'].mean())

# CHALLENGE TASK: Let's fill in the missing mort_acc values based on their total_acc value. If the mort_acc is missing,
# then we will fill in that missing value with the mean value corresponding to its total_acc value from the Series we
# created above. This involves using an .apply() method with two columns. Check out the link below for more info, or review the solutions video/notebook.
avg_total_acc = df.groupby('total_acc')['mort_acc'].mean()
print(avg_total_acc[2.00])

# Function to check whether a value within the mort_acc column is NaN or not. If it is, it needs to return the average of
# the mort_acc's related to the total_acc's grouping that that row fits into.
def missing_mort(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return avg_total_acc[total_acc]
    else:
        return mort_acc

df['mort_acc'] = df.apply(lambda x: missing_mort(x['total_acc'], x['mort_acc']), axis= 1)
print(df.isnull().sum())

# TASK: revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the
# total data. Go ahead and remove the rows that are missing those values in those columns with dropna().
df = df.dropna()
print(df.isnull().sum())

# Categorical variables and dummy variables
# We're done working with the missing data! Now we just need to deal with the string values due to the categorical columns.
# TASK: List all the columns that are currently non-numeric.
non_numerics = df.select_dtypes(include=['object']).columns
print(non_numerics)

# TASK: Convert the term feature into either a 36 or 60 integer numeric data type using .apply() or .map().
print(df['term'].value_counts())
# For each element in the 'term' column, you only take the first 2 characters (since 3 is excluded in the range) and
# converting it into an integer.
df['term'] = df['term'].apply(lambda x: int(x[:3]))
print(df['term'])

# TASK: We already know grade is part of sub_grade, so just drop the grade feature.
df = df.drop('grade', axis=1)

# TASK: Convert the subgrade into dummy variables. Then concatenate these new columns to the original dataframe.
# Remember to drop the original subgrade column and to add drop_first=True to your get_dummies call.
subgrade_dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
df = pd.concat([df.drop('sub_grade', axis=1), subgrade_dummies], axis=1)
print(df.columns)
print(df.select_dtypes(['object']).columns)

# TASK: Convert these columns: ['verification_status', 'application_type','initial_list_status','purpose'] into dummy
# variables and concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns.
# Why use 'drop_first=True'? 'cos you do not want to duplicate information - for eg: if columns are Male and Female - there
# is no point in having the Male and Female column - can only show the Male column 'cos if it isn't a Male its obviously Female.
# These string variables in the columns are good to change to dummy variables since they are binary as well mainly.
remaining_dummies = pd.get_dummies(df[['verification_status', 'application_type', 'initial_list_status', 'purpose']], drop_first=True)
df = df.drop(['verification_status', 'application_type', 'initial_list_status', 'purpose'], axis=1)
df = pd.concat([df, remaining_dummies], axis=1)

# TASK:Review the value_counts for the home_ownership column.
print(df['home_ownership'].value_counts())

# TASK: Convert these to dummy variables, but replace NONE and ANY with OTHER, so that we end up with just 4
# categories, MORTGAGE, RENT, OWN, OTHER. Then concatenate them with the original dataframe. Remember to set
# drop_first=True and to drop the original columns.
# Can also replace the strings into OTHER using the .apply function or passing the changes into a dictionary.
df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
home_dummies = pd.get_dummies(df['home_ownership'], drop_first=True)
df = df.drop('home_ownership', axis=1)
df = pd.concat([df, home_dummies], axis=1)

# TASK: Let's feature engineer a zip code column from the address in the data set. Create a column called 'zip_code'
# that extracts the zip code from the address column.
df['zip_code'] = df['address'].apply(lambda x: x[-5:])

# TASK: Now make this zip_code column into dummy variables using pandas. Concatenate the result and drop the
# original zip_code column along with dropping the address column.
zip_dummies = pd.get_dummies(df['zip_code'], drop_first=True)
df = df.drop(['address', 'zip_code'], axis=1)
df = pd.concat([df, zip_dummies], axis=1)

# TASK: This would be data leakage, we wouldn't know beforehand whether or not a loan would be issued when using our
# model, so in theory we wouldn't have an issue_date, drop this feature.
df = df.drop('issue_d', axis=1)

# TASK: This appears to be a historical time stamp feature. Extract the year from this feature using a .apply
# function, then convert it to a numeric feature. Set this new data to a feature column called 'earliest_cr_year'.
# Then drop the earliest_cr_line feature.
# Instead of using the below code, can also convert the date to a datetime object and then request only for the year portion.
# Here we are starting from the 4th last character and selecting everything from there up until the end of the line.
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda x: int(x[-4:]))
df = df.drop('earliest_cr_line', axis=1)
print(df.select_dtypes(['object']).columns)

# Train Test Split
# TASK: Import train_test_split from sklearn.
from sklearn.model_selection import train_test_split

# TASK: drop the load_status column we created earlier, since its a duplicate of the loan_repaid column. We'll use
# the loan_repaid column since its already in 0s and 1s.
df = df.drop('loan_status', axis=1)

print('\n Starting Train Test Split', '\n', df.columns)

# TASK: Set X and y variables to the .values of the features and label.
X = df.drop('loan_repaid', axis=1).values
y = df['loan_repaid'].values

# TASK: Perform a train/test split with test_size=0.2 and a random_state of 101.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

# Normalising the data.
# TASK: Use a MinMaxScaler to normalize the feature data X_train and X_test. Recall we don't want data leakge from
# the test set so we only fit on the X_train data.
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating the model.
# TASK: Run the cell below to import the necessary Keras functions.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

# TASK: Build a sequential model to will be trained on the data. You have unlimited options here, but here is what the
# solution uses: a model that goes 78 --> 39 --> 19--> 1 output neuron. OPTIONAL: Explore adding Dropout layers 1 2
# It is a rule of thumb that the first layer should have the same amount of neurons as there are features in your dataset.
# model = Sequential()

# Input layer
# model.add(Dense(78, activation='relu'))
# model.add(Dropout(0.2))

# Hidden layer
# model.add(Dense(39, activation='relu'))
# model.add(Dropout(0.2))

# Hidden layer
# model.add(Dense(19, activation='relu'))
# model.add(Dropout(0.2))

# Output layer
# The sigmoid activation function pushes values to be between 0 and 1.
# model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
# model.compile(loss='binary_crossentropy', optimizer='adam')

# TASK: Fit the model to the training data for at least 25 epochs. Also add in the validation data for later plotting.
# Optional: add in a batch_size of 256.
# model.fit(x=X_train, y=y_train, epochs=25, batch_size=256, validation_data=(X_test, y_test))

# TASK: OPTIONAL: Save your model.
from tensorflow.keras.models import load_model

# model.save('keras_project_model.h5')

model = load_model('keras_project_model.h5')

# Evaluating model performance.
# TASK: Plot out the validation loss versus the training loss.
# Ignore since cannot perform in Pycharm without having to rerun entire testing/fitting.
# losses = pd.DataFrame(model.history.history)
# losses[['loss', 'val_loss']].plot()
# plt.show()

# TASK: Create predictions from the X_test set and display a classification report and confusion matrix for the X_test set.
from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict_classes(X_test)
print('\n Classification report results \n', classification_report(y_test, predictions))
print('\n Confusion matrix results \n', confusion_matrix(y_test, predictions))

# TASK: Given the customer below, would you offer this person a loan?
import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]

# Very important to change the data here from a Series to a Numpy array. Then reshape it as below.
model.predict_classes(new_customer.values.reshape(1,78))

# TASK: Now check, did this person actually end up paying back their loan?
print(df.iloc[random_ind]['loan_repaid'])


