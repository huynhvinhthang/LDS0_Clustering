import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pandas.io import pickle
import seaborn as sns
import squarify
import plotly.express as px
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import scipy
import pickle
import datetime

import project2_func

df = pd.read_csv('C:\\Users\DELL\Desktop\LDS0\Week4\Project2\OnlineRetail.csv', header = 0, encoding= 'cp1252')

#create a list of condition
conditions = [
    df['Country'].isin(['United Kingdom', 'EIRE', 'Channel Islands']),
    df['Country'].isin(['Canada', 'Brazil', 'USA']),
    df['Country'].isin(['Japan', 'Cyprus', 'Israel', 'Bahrain', 'Hong Kong', 'Singapore', 'Lebanon', 'United Arab Emirates', 'Saudi Arabia']),
    df['Country'].isin(['France', 'Australia', 'Netherlands', 'Germany', 'Norway', 'Switzerland', 'Spain', 'Poland', 'Portugal', 'Italy', 'Belgium', 'Lithuania', 'Iceland', 'Denmark', 'Sweden', 'Austria', 'Finland', 'Greece', 'Czech Republic', 'European Community', 'Malta']),
    df['Country'].isin(['RSA'])
    
]

#create a list of values
values = ['UK', 'America', 'Asia', 'Europe', 'Africa' ]
df['Continent'] = np.select(conditions, values)

#Create total sale features
df['Total_Sales'] = df['Quantity'] * df['UnitPrice']

#Convert InvoiceDate into Date type because there are some of value are string.
string_to_date = lambda x: datetime.datetime.strptime(x,"%d-%m-%Y %H:%M").date()
df['InvoiceDate'] = df['InvoiceDate'].apply(string_to_date)
df['InvoiceDate'] = df['InvoiceDate'].astype('datetime64[ns]')

#create Month and Year feature from InvoiceDate feature
df['Month'] = pd.DatetimeIndex(df['InvoiceDate']).month
df['Year'] = pd.DatetimeIndex(df['InvoiceDate']).year

#missing value dataframe
missing = project2_func.missing_values_table(df)

df = df[df['CustomerID'].isna() != True]
#delete junk stockcode
list_of_junk_StockCode = ['D', 'S', 'M', 'Post', 'DOT', 'BANK CHARGE', 'AMAZONFEE', 'CRUK']
for i in list_of_junk_StockCode:
    df = df[df['StockCode'] != i]
#delete junk description
list_of_junk_description = ['20713', '?', '??', '???', '?display?', '?lost', '?sold as sets?', '?? missing', '???lost',
                            '????missing', '????damages????', '*Boombox Ipod Classic', '*USB Office Mirror Ball']
for i in list_of_junk_description:
    df = df[df['Description'] != i]

#Take only the successfull Invoice where Quantity >0/
df = df[df['Quantity'] >0]

#Delete outliers
Q1 = np.percentile(df['Quantity'], 25)
Q3 = np.percentile(df['Quantity'], 75)
upper_outlier = df[df['Quantity'] > (Q3 + 1.5*(Q3 - Q1))].shape[0]
lower_outlier = df[df['Quantity'] < (Q1 - 1.5*(Q3 - Q1))].shape[0]
df = df[(df['Quantity'] < (Q3 + 1.5*(Q3 - Q1))) & (df['Quantity'] > (Q1 - 1.5*(Q3 - Q1)))]

Q1 = np.percentile(df['UnitPrice'], 25)
Q3 = np.percentile(df['UnitPrice'], 75)
upper_outlier = df[df['UnitPrice'] > (Q3 + 1.5*(Q3 - Q1))].shape[0]
lower_outlier = df[df['UnitPrice'] < (Q1 - 1.5*(Q3 - Q1))].shape[0]
df = df[(df['UnitPrice'] < (Q3 + 1.5*(Q3 - Q1))) & (df['UnitPrice'] > (Q1 - 1.5*(Q3 - Q1)))]

Q1 = np.percentile(df['Total_Sales'], 25)
Q3 = np.percentile(df['Total_Sales'], 75)
upper_outlier = df[df['Total_Sales'] > (Q3 + 1.5*(Q3 - Q1))].shape[0]
lower_outlier = df[df['Total_Sales'] < (Q1 - 1.5*(Q3 - Q1))].shape[0]
df = df[(df['Total_Sales'] < (Q3 + 1.5*(Q3 - Q1))) & (df['Total_Sales'] > (Q1 - 1.5*(Q3 - Q1)))]

max_date = df['InvoiceDate'].max().date()

Recency = lambda x: (max_date - x.max().date()).days
Frequency = lambda x: len(x.unique())
Monetary = lambda x: round(sum(x),2)

df_RFM = df.groupby('CustomerID').agg({'InvoiceDate':Recency,
                                       'InvoiceNo':Frequency,
                                       'Total_Sales':Monetary})

df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
df_RFM = df_RFM.sort_values('Monetary', ascending = False)

#create rfm label
r_labels = range(4,0,-1)
f_labels = range(1,5)
m_labels = range(1,5)

#assign these labels to 4 equal percentile gr
r_groups = pd.qcut(df_RFM['Recency'].rank(method = 'first'), q = 4, labels = r_labels)

f_groups = pd.qcut(df_RFM['Frequency'].rank(method = 'first'), q = 4, labels = f_labels)

m_groups = pd.qcut(df_RFM['Monetary'].rank(method = 'first'), q = 4, labels = m_labels)

#create new columns R, F, M
df_RFM  =df_RFM.assign(R = r_groups.values, F = f_groups.values, M = m_groups.values)

#concat rfm quartile
def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis = 1)

#calculate RFM score
df_RFM['RFM_Score'] = df_RFM[['R', 'F', 'M']].sum(axis = 1).astype('int')
df_RFM.head()
#create RFM_group with function
df_RFM['RFM_Group'] = df_RFM.apply(project2_func.RFM_Group, axis = 1)

# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_agg = df_RFM.groupby('RFM_Group').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(0)

rfm_agg.columns = rfm_agg.columns.droplevel()
rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

# Reset the index
rfm_agg = rfm_agg.reset_index()

#RFM squarify chart
figv1 = plt.gcf()
ax = figv1.add_subplot()
figv1.set_size_inches(14, 10)
squarify.plot(sizes=rfm_agg['Count'],
              text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
              label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
                      for i in range(0, len(rfm_agg))], alpha=0.5, ax= ax)
ax.set_title("Customers Segments",fontsize=26,fontweight="bold")
plt.axis('off')
#RFM group chart
figv2 = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="RFM_Group",
           hover_name="RFM_Group", size_max=100, title = "The group Segmentation with Monetary and Recency")
#RFM 3d group chart
figv3 = px.scatter_3d(df_RFM, x='Recency', y='Frequency', z='Monetary',
                    color = 'RFM_Group', opacity=0.5, title = "RFM Group Segmentation")
figv3.update_traces(marker=dict(size=5),
                  
                  selector=dict(mode='markers'))


#create minmax dataframe 
df_now = df_RFM[['Recency', 'Frequency', 'Monetary']]
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(df_now)
df_minmax = pd.DataFrame(X_minmax, columns = ['Recency', 'Frequency', 'Monetary'])

kmeans_minmax = KMeans(n_clusters=4)
kmeans_minmax.fit(df_minmax)
labels_minmax = kmeans_minmax.labels_ 
df_now['Group_minmax'] = labels_minmax

#kmeans minmax 3d
figv4 = px.scatter_3d(df_now, x='Recency', y='Frequency', z='Monetary',
                    color = 'Group_minmax', opacity=0.5, title = "KMeans Group Clustering with RFM minmax data")
figv4.update_traces(marker=dict(size=5),
                  
                  selector=dict(mode='marker'))

# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_agg_minmax = df_now.groupby('Group_minmax').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(0)

rfm_agg_minmax.columns = rfm_agg_minmax.columns.droplevel()
rfm_agg_minmax.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
rfm_agg_minmax['Percent'] = round((rfm_agg_minmax['Count']/rfm_agg_minmax.Count.sum())*100, 2)
# Reset the index
rfm_agg_minmax = rfm_agg_minmax.reset_index()

figv5 = px.scatter(rfm_agg_minmax, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Group_minmax",
           hover_name="Group_minmax", size_max=100, title = "KMeans Group Clustering with RFM minmax data")


#GUI
#main page
st.title("Data Science Project")
st.write("## Customer Segmentation Clustering")

#Show result
menu = ["EDA", "Models Comparision", 'Kmeans Model']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'EDA':
    st.subheader('EDA')
    st.write('### Dataframe head')
    st.dataframe(df.head(5))
    st.write('### Dataframe Tail')
    st.dataframe(df.tail(5))
    st.write("### Features Outliers")
    st.image("Project2/boxplot.png")
    st.write("Total quantity of Countries")
    st.image("project2/total_quantity.png")
    st.write("### Missing value in data")
    st.dataframe(missing)
    st.write("#### List of invalid StockCode")
    st.write(list_of_junk_StockCode)
    st.write("#### List of invalid Description")
    st.write(list_of_junk_description)
    st.write("#### Data with RFM cluster method")
    st.dataframe(df_RFM)
    st.write("#### RFM Group after Calculating")
    st.dataframe(rfm_agg)
    st.write("#### RFM Group with Squarify plot")
    st.pyplot(figv1)
    st.write("#### The RFM Group with Monetary and Recency")
    st.plotly_chart(figv2)
    st.write("#### The 3d plot for RFM Group")
    st.plotly_chart(figv3)
elif choice == 'Models Comparision':
    st.subheader("Models Comparision")
    st.write("### KMeans Model")
    st.write("#### KMeans Model with RFM data")
    st.dataframe(pd.read_csv("Project2/kmeans_agg.csv", index_col = 0))
    st.write("#### The 3d Kmeans Group")
    st.image("Project2/kmeans_3d.png")
    st.write("#### The Kmeans group information")
    st.write("####Kmeans model with RFM minmax data")
    st.write("The 3d KMeans Minmax Group")
    st.image("Project2/kmean_3d_1.png")
    st.write("#### The KMeans minmax group with monetary and recency")
    st.image("Project2/kmeans_group_1.png")
    st.write("#### The Kmeans minmax group information")
    st.dataframe(pd.read_csv("Project2/kmeans_agg_minmax.csv", index_col = 0))
    st.write("### Hierarchical Model")
    st.write("#### Hiererchical with RFM data")
    st.write("#### RFM data - Dendogram")
    st.image("project2/hrc.png")
    st.write("#### The 3d RFM Hiererchical group")
    st.image("project2/hrc_3d.png")
    st.write("#### The Hierarchical group information")
    st.dataframe(pd.read_csv("Project2/hrc.csv", index_col = 0))
    st.write("#### Hierarchical group with Monetary and Recency")
    st.image("project2/hrc_group.png")
    st.write("#### Hierarchical with RFM minmax data")
    st.write("#### RFM minmax data - Dendogram")
    st.image("project2/hrc_minmax.png")
    st.write("#### The 3d Hierarchical minmax group")
    st.image("project2/hrc_3d_1.png")
    st.write("The Hierarchical minmax group information")
    st.dataframe(pd.read_csv("Project2/hrc_agg_minmax.csv", index_col = 0))
    st.write("#### The Hierarchical group with Monetary and Recency")
    st.image("project2/hrc_group1.png")
    st.write("#### Gaussian Mixture Model")
    st.write("#### Gaussian Mixture with RFM data")
    st.write("#### The 3d GMM Group")
    st.image("project2/gmm_3d.png")
    st.write("#### The GMM group information")
    st.dataframe(pd.read_csv("Project2/gmm_agg.csv", index_col = 0))
    st.write("#### Gaussian Mixture with RFM minmax data")
    st.write("#### The 3d GMM minmax group")
    st.image("project2/gmm_3d_1.png")
    st.write("The GMM minmax group information")
    st.dataframe(pd.read_csv("Project2/gmm_agg_minmax.csv", index_col = 0))
    st.write("#### The GMM minmax group with Monetary and Recency")
    st.image("project2/gmm_group_1.png")
elif choice == "Kmeans Model":
    st.write("### KMeans Minmax Model")
    st.dataframe(df_minmax.head())
    st.dataframe(df_now.head())
    #st.write(labels_minmax)
    st.plotly_chart(figv4)
    st.dataframe(rfm_agg_minmax)
    st.plotly_chart(figv5)

