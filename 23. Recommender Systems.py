import numpy as np
import pandas as pd

columns_name = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('u.data', sep='\t', names=columns_name)

print(df.head())

movie_titles = pd.read_csv('Movie_Id_Titles')

print(movie_titles.head())

#you will see there is a link between the item_id of the first dataframe and the title column in this movie_titles dataframe.
#What we can do now is to merge or replace the column of item_id in df to actually say what the title is from movie_titles dataframe.
#Can use the .merge function in pandas in order to merge the movie_titles dataframe within the original df along the item_id column.
#Since both dataframes have the item_id column this is what will be used to link the 2 dataframes when merging.
df = pd.merge(df, movie_titles, on='item_id')

print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

#Will create a ratings dataframe with average rating and number of ratings.
print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())     #Average of the rating column based on th title column. Now have movies with best ratings.

#Now lets see movies with most ratings.
print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())    #If do not have ascending=false the printout will be in alphabetical order of dataframe.

#Going to create the above but in a dataframe to be easier to call it in future.
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
print(ratings.head())

#This is not a fair rating though since might only have 1 person who voted for the movie wheras 1000 people for others.
#Therefore will add an additional column to our dataframe to include number of voters.
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
print(ratings.head())

ratings['num of ratings'].hist(bins=70)
plt.show()

ratings['rating'].hist(bins=70)
plt.show()

#Relationship between the average ratings and the number of ratings.
sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)  #The higher the rating the higher the number of votes for it.
plt.show()


#Part 2#####################

#Going to use pivot table to get the df into martix form.
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
print(moviemat.head())
#Will notice that there are many missing values - makes sense since not all people have watched every movie to give it a particular rating from 1 to 5.

#Printing most voted for movies.
print(ratings.sort_values('num of ratings', ascending=False).head(10))

#Going to grab the values for 2 movies - Star Wars and Liar Liar.
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

#This is a series with user_id and acatual rating that was given.
print(starwars_user_ratings.head())     #Simply shows the user number and the rating he/she gave to the movie.

#Use corrwith method which finds the correrlation between two objects in a dataframe.
similiar_to_starwars = moviemat.corrwith(starwars_user_ratings)     #Gives the correlation of bunch of movies with the rating that Star Wars has.
similiar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similiar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)

print(corr_starwars.head())

#Now going to sort the above correlation dataframe to find out the movies with ratings most correlated with the ratings given to Star Wars.

#There is an issue with this - namely, if you use the below code, you will see that there is a perfect correlation of 1 for the movies printed.
#This is likely due to the same individual who rated Star Wars a 5 would have also been the ONLY individual to watch and actually rate the movie with the
#perfect correlation below.
print(corr_starwars.sort_values('Correlation', ascending=False).head(10))

#To fix this, let us only consider movies with a certain amount of votes. In this case, if less than 100 votes, ignore.
corr_starwars = corr_starwars.join(ratings['num of ratings'])   #Using .join instead of .merge since already have the 'title' column as the index already which can be used to join the 2 dataframes along.
print(corr_starwars.head())

#Now filtering out movies that do not have 100 ratings.
print(corr_starwars[corr_starwars['num of ratings'] > 100].sort_values('Correlation', ascending=False).head())

#Now do same for Liar Liar movie.
corr_liarliar = pd.DataFrame(similiar_to_liarliar, columns=['Correlation'])
print(corr_liarliar.dropna(inplace=True))

corr_liarliar = corr_liarliar.join(ratings['num of ratings'])

#Now filter only for movies with more than 100 ratings.
print(corr_liarliar[corr_liarliar['num of ratings'] > 100].sort_values('Correlation', ascending=False).head())


















