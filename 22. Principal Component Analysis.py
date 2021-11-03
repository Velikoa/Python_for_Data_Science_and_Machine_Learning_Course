import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

#Here using a dataset that already exists within the sklearn library.
cancer = load_breast_cancer()

#The below shows the name of each category in the dictionary.
print(cancer.keys())

#print(cancer['DESCR'])

#set it up in order to use it in panddas - i.e. in a dataframe format.
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

print(df.head())

from sklearn.preprocessing import StandardScaler

#Need to scale the data - i.e. to ensure that all the variables in the columns are of the same range. Each feature has a single unit variance.
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

#Actually performing the PCA
from sklearn.decomposition import PCA


#Here going to try to visualise the entire 30 component dataframe using only 2 principle components.
pca = PCA(n_components=2)
pca.fit(scaled_data)
#Then transform it to its first principle components.
x_pca = pca.transform(scaled_data)

#Below shows how originally the shape/size of the data had 30 columns and after it was scaled and transformed, it now has 2.
print(scaled_data.shape)
print(x_pca.shape)

#Now that the 30 dimensions have been reduced to only 2, we can simply plot this info using matplotlib.
plt.figure(figsize=(8,6))
#Plotting all the rows from column zero and plot against all rows from column 1. 'c' is for colour - i.e. you are colouring the malignant vs benign.
plt.scatter(x_pca[:,0], x_pca[:,1], c=cancer['target'], cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

plt.show()

pca.components_
df_comp = pd.DataFrame(pca.components_, columns=cancer['feature_names'])
#Now creating heatmap in order to see how these components are related to each other.
plt.figure(figsize=(12,6))
sns.heatmap(df_comp, cmap='plasma')

plt.show()
