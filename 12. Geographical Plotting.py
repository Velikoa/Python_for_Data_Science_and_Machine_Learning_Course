import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import pandas as pd

data = dict(type='choropleth',
            locations=['AZ','CA','NY'],
            locationmode='USA-states',
            colorscale='Portland',
            text=['text 1','text 2', 'text 3'],
            z=[1.0,2.0,3.0],
            colorbar={'title':'Colorbar Title Goes Here'})

layout = dict(geo={'scope':'usa'})

choromap = go.Figure(data=[data], layout=layout)

#print(plot(choromap))                          #Shows as a map on your browser in HTML format

df = pd.read_csv('2011_US_AGRI_Exports')

data = dict(type='choropleth',
            colorscale='YlOrRd',
            locations=df['code'],
            locationmode='USA-states',
            z=df['total exports'],
            text=df['text'],
            marker=dict(line=dict(color='rgb(255,255,255)',width=2)),
            colorbar={'title':'Millions USD'}
            )

layout = dict(title='2011 US Agriculture Exports by State',
              geo=dict(scope='usa',showlakes=True,lakecolor='rgb(85,173,240)'))

choromap2 = go.Figure(data=[data], layout=layout)

#print(plot(choromap2))


###############International Plotting####################################

df = pd.read_csv('2014_World_GDP')

data = dict(
    type='choropleth',
    locations=df['CODE'],
    z=df['GDP (BILLIONS)'],
    text=df["COUNTRY"],
    colorbar={'title':'GDP in Billions USD'}
)

layout = dict(
    title='2014 Global GDP',
    geo=dict(showframe=False,
             projection={'type':'mercator'})
)

choromap3 = go.Figure(data=[data], layout=layout)

print(plot(choromap3))