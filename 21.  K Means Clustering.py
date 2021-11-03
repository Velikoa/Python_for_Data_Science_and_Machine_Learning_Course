import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#This is an unsupervised algorithm meaning that it tries to classify data based on their similarities.
#We are not trying to predict any sort of outcome - we are just trying to find patterns in the data.
#In K Means Clustering we need specify how many clusters we want the data to be clustered into.

#Going to generate artifical data using scikit-learn
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101)            #Just using this module to create data of your specification.

print(data)

plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')          #Using c=data[1] since the 2nd column in the dataset is for the categories

plt.show()

from sklearn.cluster import KMeans

#This algo will rnadomly assign each point to a cluster and will then find the centroid of that cluster.
#It will then iterate again and assign each data point to the centroid it is closest to.
#This is repeated until thee cluster variation cannot be reduced any further.
kmeans = KMeans(n_clusters=4)           #Instantiate the model and state how many clusters you expect or want to explore
kmeans.fit(data[0])

#If want the centers of the clusters print below
print(kmeans.cluster_centers_)

#If want to show the labels that the algo believes to be true for the clusters.
print(kmeans.labels_)

#Below am plotting the original labels (how they were grouped) versus which cluster the algo believes they should be labelled under.
fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(10,6))        #They are sharing a y-axis.
ax1.set_title('K Means')
ax1.scatter(data[0][:,0], data[0][:,1], c=kmeans.labels_, cmap='rainbow')

ax2.set_title('Original')
ax2.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')

plt.show()

#########################################Project#############################################################

pd.set_option('display.width', 450)
pd.set_option('display.max_columns', 25)

#** Read in the College_Data file using read_csv. Figure out how to set the first column as the index.**
college_data = pd.read_csv('College_Data', index_col=0)

print(college_data.head())

#** Check the info() and describe() methods on the data.**
print(college_data.info())
print(college_data.describe())

#** Create a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column. **
sns.lmplot(x='Grad.Rate', y='Room.Board', data=college_data, hue='Private', palette='coolwarm',size=6, aspect=1, fit_reg=False)
plt.show()

#Create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.
sns.lmplot(x='Outstate', y='F.Undergrad', data=college_data, hue='Private', palette='coolwarm', size=6, aspect=1, fit_reg=False)
plt.show()

#** Create a stacked histogram showing Out of State Tuition based on the Private column. Try doing this using sns.FacetGrid.
# If that is too tricky, see if you can do it just by using two instances of pandas.plot(kind='hist'). **
sns.set_style('darkgrid')
g = sns.FacetGrid(college_data, hue='Private', palette='coolwarm', size=6, aspect=2)
g.map(plt.hist, 'Outstate', bins=20, alpha=0.7)
plt.show()

#Create a similar histogram for the Grad.Rate column.
sns.set_style('darkgrid')
p = sns.FacetGrid(college_data, hue='Private', palette='coolwarm', size=6, aspect=2)
p.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)
plt.show()

#** Notice how there seems to be a private school with a graduation rate of higher than 100%.What is the name of that school?**
print(college_data[college_data['Grad.Rate'] > 100])

#** Set that school's graduation rate to 100 so it makes sense. You may get a warning not an error) when doing this operation,
# so use dataframe operations or just re-do the histogram visualization to make sure it actually went through.**
college_data['Grad.Rate']['Cazenovia College'] = 100

print(college_data[college_data['Grad.Rate'] > 100])

sns.set_style('darkgrid')
p = sns.FacetGrid(college_data, hue='Private', palette='coolwarm', size=6, aspect=2)
p.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)
plt.show()

#** Create an instance of a K Means model with 2 clusters.**
kmeans_instance = KMeans(n_clusters=2)

#Fit the model to all the data except for the Private label.
kmeans_instance.fit(college_data.drop('Private', axis=1))

#** What are the cluster center vectors?**
print(kmeans_instance.cluster_centers_)

#** Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.**
def converter(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0

college_data['Cluster'] = college_data['Private'].apply(converter)

print(college_data.head())

#** Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.**
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(college_data['Cluster'], kmeans_instance.labels_))
print(classification_report(college_data['Cluster'], kmeans_instance.labels_))



