import numpy as np
import pandas as pd

ser1 = pd.Series([1,2,3,4],['USA','Germany','USSR','Japan'])
print(ser1)         #Pandas series assigns the first element to a data and the second as its index

print(np.random.seed(101))             #Gives the same random numbers
df = pd.DataFrame(np.random.randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])
print(df)       #Creates a 5 by 4 matrix with random numbers

#Grab the W column
df['W']
#Grab more than one column in a df - then simply put all columns wanted in a list
print(df[['W','Z']])

#Adding a new column to the df
df['new'] = df['W'] + df['Y']
print(df)

#To drop the column use the .drop method and then specify axis=1
df.drop('new',axis=1,inplace=True)      #use inplace if you want the change to be permanent
print(df)

#when dropping a column use axis=1 and when drorpping rows use axis=1
#selecting rows
print(df.loc['A'])

#return the 3rd row
print(df.iloc[2])

print('#######################################')
###############################Next Video################################

print(df > 0)           #Print where values are greater than 0

#Print column W but only the rows where the amounts are greater than 0 - note that row C is removed here
print(df[df['W'] > 0])

#Using multiple conditions for a df
print(df[(df['W']>0) & (df['Y']>1)])        #Prints only column E

print('#######################################')
###########################Next Video####################################

outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)
df_2 = pd.DataFrame(np.random.randn(6,2),hier_index,['A','B'])

print(df_2)

#To call a set of numbers from within several indices
print(df_2.loc['G1'].loc[1])

df_2.index.names = ['Groups', 'Num']
print(df_2)

print('#########################################')
############################Next Video######################################

#Missing values
d = {'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]}
df_3 = pd.DataFrame(d)
print(df_3)
print(df_3.dropna())

#If you want to drop the columns with a NaN value, say axis=1
print(df_3.dropna(axis=1))
#Drop rows with a certain threshold of NaN values in them
print(df_3.dropna(thresh=2))        #Keeps only rows 0 and 1 but removes 2

#If you want to fill the NaN values and replace them with the mean value of that column
print(df_3['A'].fillna(value=df_3['A'].mean()))         #Column A's 3rd value now becomes 1.5'


print('###########################################################')
#########################New Video############################################

data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}

df_4 = pd.DataFrame(data)
print(df_4)

#Use the Groupby function
byComp = df_4.groupby('Company')
print(byComp.mean())            #Gives the mean sales of each company

#Can even call the sum or std dev or count of particular rows
print(df_4.groupby('Company').sum().loc['GOOG'])            #Return 320

print('#########################################################')
##########################New Video#######################################

#Concatenation joins df's together - need to call df.concat([pass the df's into a list])

########################New Video#########################################

df_5 = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
print(df_5.head())

#Print all the unique values in column 2
print(df_5['col2'].unique())
print(df_5['col2'].nunique())               #prints number of unique variables in the column

def times2(x):
       return x*2

print(df_5.apply(times2))                 #The apply function applied whatever function you pass into it to the entire df

#Can also be done in the form of a lambda expression
print(df_5['col2'].apply(lambda x: x*2))

#Sorting the information in the df by column
print(df_5.sort_values('col2'))           #Sorts the values whereby 444 and 444 are placed first

#find out whether an element in the df is a null value
print(df_5.isnull())               #In this case all values are false

print('\npivot tables\n')

data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df_6 = pd.DataFrame(data)
print(df_6)
print('\n')
print(df_6.pivot_table(values='D',index=['A','B'],columns=['C']))

print('########################################################\n')
#################################New video#######################################

df_7 = pd.read_excel('Excel_Sample.xlsx',sheet_name='Sheet1')
print(df_7)

#When writing a new file
df_7.to_excel('ExcelSample2.xlsx',sheet_name='NewSheet')


