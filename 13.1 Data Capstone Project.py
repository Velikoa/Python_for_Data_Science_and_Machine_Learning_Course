import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

#to show all columns in the dataframe
desired_width = 320
pd.set_option('display.width',desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)

#import the csv file as a dataframe
df = pd.read_csv('911.csv')

print(df.info())

print(df.head(3))

#Top 5 zip codes for 911 calls
print(df['zip'].value_counts().head(5))

#top 5 townships for 911 calls
print(df['twp'].value_counts().head(5))

#number of unique title codes
print(df['title'].nunique())

# In the titles column there are "Reasons/Departments" specified before the title code.
# These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column
# called "Reason" that contains this string value.
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])

#What is the most common Reason for a 911 call based off of this new column?
print(df['Reason'].value_counts())

#Now use seaborn to create a countplot of 911 calls by Reason.
sns.countplot(x='Reason',data=df,palette='viridis')

#plt.show()

# Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column?
print(type(df['timeStamp'].iloc[0]))            #iloc[0] simply selects the first element of the list to check what type of object it is.

# You should have seen that these timestamps are still strings. Use pd.to_datetime to convert the column from strings to DateTime objects.
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

#** You can now grab specific attributes from a Datetime object by calling them. For example:**
# time = df['timeStamp'].iloc[0]
# time.hour
#You can use Jupyter's tab method to explore the various attributes you can call. Now that the timestamp column
# are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week.
# You will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)

#Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week: **

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day of Week'] = df['Day of Week'].map(dmap)

#Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column
sns.countplot(x='Day of Week',data=df,hue=df['Reason'],palette='viridis')       #no 'y' coordinate needed since you are simply counting here
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)

#plt.show()

# Now do the same for Month
sns.countplot(x='Month',data=df,hue=df['Reason'],palette='viridis')
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)

#plt.show()

#Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use
# the count() method for aggregation. Use the head() method on this returned DataFrame.
byMonth = df.groupby('Month').count()
print(byMonth.head())

#Now create a simple plot off of the dataframe indicating the count of calls per month.
plt.plot(byMonth['twp'])

#plt.show()

#Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month.
# Keep in mind you may need to reset the index to a column.
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())                #Remember this is a scatter plot with a regression line through it
#plt.show()

#Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method
df['Date'] = df['timeStamp'].apply(lambda x: x.date())

#Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.
byDate = df.groupby('Date').count()
plt.plot(byDate['twp'])
#plt.tight_layout()
#plt.show()

#Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call
#Therefore still need to grouby Date but now seperating into Reason for the call as well.
# df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
# plt.title('Traffic')
# plt.tight_layout()
# plt.show()

#** Now let's move on to creating heatmaps with seaborn and our data. We'll first need to restructure the
# dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of
# ways to do this, but I would recommend trying to combine groupby with an unstack method. Reference the solutions
# if you get stuck on this!**

#Unstack method - turns the information into Matrix form - i.e. every row and column has an attributable item for it.

dayHour = df.groupby(['Day of Week','Hour']).count()['Reason'].unstack()
print(dayHour.head())

plt.figure(figsize=(12,6))                      #specify the size of the plot first but fo not have to do this.
sns.heatmap(data=dayHour,cmap='viridis')
#plt.show()



#** Now create a clustermap using this DataFrame. **
sns.clustermap(data=dayHour,cmap='viridis')
#plt.show()

#** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **
dayMonth = df.groupby(['Day of Week','Month']).count()['Reason'].unstack()
print(dayMonth.head())

plt.figure(figsize=(12,6))
sns.heatmap(data=dayMonth,cmap='viridis')
#plt.show()

sns.clustermap(data=dayMonth,cmap='viridis')
plt.show()

