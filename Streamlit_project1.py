# from secrets import choice
import pandas as pd
import numpy as np
import squarify
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

from sklearn import metrics
import import_ipynb
# import pickle
import streamlit as st
from sklearn.mixture import GaussianMixture
import Lib

# 1. Read data
df = pd.read_csv("../Project1_CustomerSegmentation/data/OnlineRetail.csv", encoding='unicode_escape')

#--------------
# GUI
st.title("Data Science Projects")
st.write("PROJECT 1 - Customer Segmentation")

# Upload file
uploaded_file = st.file_uploader('Choose a file', type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='latin-1')
    data.to_csv('OnlineRetail_new.csv', index=False)

# 2. Data pre-processing
# drop rows where Quantity < 0
df = df[df.Quantity >= 0]
# drop rows where CustomerID == null
df = df[df.CustomerID.notnull()]
# drop rows where UnitPrice < 0
df = df[df.UnitPrice >= 0]
# drop duplicated rows
df = df.drop_duplicates()
# Convert column 'InvoiceDate' to datetime datatype
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
# convert InvoiceNo data type to integer
df.InvoiceNo = df.InvoiceNo.astype(int)
# Create new column 'Amount'
df['Amount'] = df['Quantity'] * df['UnitPrice']

# Get max date of dataframe
max_date = df['InvoiceDate'].max().date()
# Calculate R, F, M
Recency = lambda x: (max_date - x.max().date()).days
Frequency = lambda x: x.nunique()
Monetary = lambda x: round(sum(x),2)

df_RFM = df.groupby('CustomerID').agg({'InvoiceDate' : Recency,
                                        'InvoiceNo' : Frequency,
                                        'Amount' : Monetary,
                                        })
# Rename column names
df_RFM.columns = ['Recency', 'Frequency', 'Monetary']

# Create labels for Recency, Frequency, Monetary & Assign them to 5 equal percentile groups and convert their labels from categorical to integer
r_groups = pd.qcut(df_RFM['Recency'].rank(method='first'), q=5, labels=range(5, 0, -1)).astype(int)
f_groups = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=5, labels=range(1,6)).astype(int)
m_groups = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=5, labels=range(1,6)).astype(int)

# Create new columns R, F, M
df_RFM = df_RFM.assign(R=r_groups, F=f_groups, M=m_groups)

# 3. Build model
gmm = GaussianMixture(n_components=5, covariance_type='spherical', random_state=0)
gmm.fit(df_RFM[['Recency','Frequency','Monetary']])

gmm_segment = gmm.predict(df_RFM[['Recency','Frequency','Monetary']])
df_RFM['GMM_segment'] = gmm_segment

# Calculate average values for each GMM_segment, and return a size of each segment 
gmm_agg = df_RFM.groupby('GMM_segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'R': 'mean',
    'F': 'mean',
    'M': ['mean', 'count']}).round(0)

gmm_agg.columns = gmm_agg.columns.droplevel()
gmm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'RMean','FMean','MMean','Count']
gmm_agg['Percent'] = round((gmm_agg['Count']/gmm_agg.Count.sum())*100, 2)

# Reset the index
gmm_agg = gmm_agg.reset_index()

# #4. Save models
# pkl_filename = "Customer_Segmentation_GMM.pkl"  
# with open(pkl_filename, 'wb') as file:  
#     pickle.dump(gmm, file)

# #5. Load models 
# with open(pkl_filename, 'rb') as file:  
#     gmm_model = pickle.load(file)

# GUI
menu = ['Business Objective', 'Build Project', 'New Prediction']
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':
    st.subheader('Business Objective')
    st.write("""
    - Công ty X chủ yếu bán các sản phẩm là quà tặng dành cho những dịp đặc biệt. Nhiều khách hàng của công ty là khách hàng bán buôn.
    - Công ty X mong muốn có thể bán được nhiều sản phẩm hơn cũng như giới thiệu sản phẩm đến đúng đối tượng khách hàng, chăm sóc và làm hài lòng khách hàng.
    """)  
    st.write("""###### => Problem/ Requirement: Sử dụng Machine Learning để phân loại các nhóm khách hàng.""")

elif choice == 'Build Project':
    st.subheader('Build Project')
    st.write('### 1. Raw data')
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3))

    st.write('### 2. RFM data')
    st.dataframe(df_RFM[['Recency','Frequency','Monetary']].head(3))
    st.dataframe(df_RFM[['Recency','Frequency','Monetary']].tail(3))

    st.write('### 3. Visualize R, F, M')
    fig = plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    sns.distplot(df_RFM['Recency'])
    plt.subplot(1, 3, 2)
    sns.distplot(df_RFM['Frequency'])
    plt.subplot(1, 3, 3)
    sns.distplot(df_RFM['Monetary'])
    plt.suptitle('Distplot of R, F, M')
    st.pyplot(fig)

    fig = plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.boxplot(df_RFM['Recency'])
    plt.subplot(1, 3, 2)
    plt.boxplot(df_RFM['Frequency'])
    plt.subplot(1, 3, 3)
    plt.boxplot(df_RFM['Monetary'])
    plt.suptitle('Boxplot of R, F, M')
    st.pyplot(fig)

    st.write('### 4. Build model...')
    st.write('### 5. Report')

    # Visualize results
    fig = plt.figure(figsize=(8, 5))
    count = df_RFM.groupby(df_RFM['GMM_segment']).size()
    plt.bar(count.index, count.values, color='olivedrab')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.dataframe(gmm_agg)
    st.write("""
    Cluster description:
    - Cluster 4 - SVIP: Nhóm khách hàng đặc biệt (chỉ có 2 người) chi tiêu rất khủng, mua hàng thường xuyên nhất và vẫn đang mua hàng.
    - Cluster 1 - VIP1: Nhóm khách hàng chi tiêu cao, mua hàng thường xuyên và mới mua hàng gần đây.
    - Cluster 3 - VIP2: Nhóm khách hàng chi tiêu cao nhưng ít hơn VIP1, mua hàng thường xuyên và mới mua hàng gần đây.
    - Cluster 2 - Loyal/Regular: Khách hàng mua hàng đều đặn, chi tiêu ở mức khá, thời gian mua hàng gần nhất không xa.
    - Cluster 0 - Member: Khách hàng mua hàng không thường xuyên, chi tiêu không cao và đã khá lâu không mua hàng.
    """)  

    # Visualize results
    fig = plt.figure(figsize=(10, 5))
    Lib.treemap_customer_segmentation(gmm_agg,font_size=14)
    st.pyplot(fig)

    # Visualization - 2D Scatter
    fig = px.scatter(gmm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="GMM_segment",
           hover_name="GMM_segment", size_max=80)
    st.plotly_chart(fig)

    # Visualization - 3D scatter
    fig = px.scatter_3d(df_RFM, x='Recency', y='Frequency', z='Monetary',
                        color = 'GMM_segment', opacity=0.5)
    fig.update_traces(marker=dict(size=5),selector=dict(mode='markers'))
    st.plotly_chart(fig)

elif choice == 'New Prediction':
    st.write('Upload data')
    flag = False
    uploaded_file2 = st.file_uploader('Choose a csv file', type=['csv'])
    if uploaded_file2 is not None:
        data = pd.read_csv(uploaded_file2, encoding='latin-1')
        st.dataframe(data)
        flag=True

    if flag:
        st.write('Content:')
        if data.shape[0]>0:
            #Preprocessing 
            data_RFM = Lib.df_RMF_preprocessing(data)
            gmm_segment = gmm.predict(data_RFM[['Recency','Frequency','Monetary']])
            data_RFM['GMM_segment'] = gmm_segment
            st.dataframe(data_RFM.head())