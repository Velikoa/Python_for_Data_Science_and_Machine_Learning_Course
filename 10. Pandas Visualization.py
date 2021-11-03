import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv('df1',index_col=0)
print(df1.head())

df2 = pd.read_csv('df2')
print(df2.head())

#Create a histogram from a specific column of a dataframe
df1['A'].hist(bins=30)

plt.show()

#Another way to plot the graph is to use the 'kind' specification when calling a dataframe
df1['A'].plot(kind='hist',bins=30)

plt.show()

#Creating an area plot
df2.plot.area(alpha=0.4)        #Alpha shows transparency

plt.show()

#Creating a barplot
df2.plot.bar(stacked=True)

plt.show()

#Creating a line chart
df1.plot.line(x=df1.index,y='B',figsize=(12,3),lw=1)

plt.show()

#Creating a scatter plot in pandas
df1.plot.scatter(x='A',y='B')

plt.show()

#Creating box plots
df2.plot.box()

plt.show()

