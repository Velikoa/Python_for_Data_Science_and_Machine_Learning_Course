import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chart_studio.plotly as py
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()             #allows you to use cufflinks offline

#Data
df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())
print(df.head())

df2 = pd.DataFrame({'Category':['A','B','C'],'Values':[32,43,50]})
print(df2)

df.iplot()                  #Note that iplot does not seem to work in Pycharm!!!!

df.iplot(kind='scatter',x='A',y='B',mode='markers')

df2.iplot(kind='bar',x='Category',y='Values')







plt.show()