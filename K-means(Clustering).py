import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df=pd.read_csv("D:/Data Science/income.csv")


#plt.scatter(df["Age"],df['Income($)'])

km=KMeans(n_clusters=3)
y_pred=km.fit_predict(df[['Age','Income($)']])
#y_pred

# df['Cluster']=y_pred
# df.head()


# df1=df[df.Cluster==0]
# df2=df[df.Cluster==1]
# df3=df[df.Cluster==2]
# plt.scatter(df1['Age'],df1['Income($)'],color='green')
# plt.scatter(df2['Age'],df2['Income($)'],color='red')
# plt.scatter(df3['Age'],df3['Income($)'],color='blue')
# plt.show()



scaler=MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[['Income($)']])
# df.head()


scaler=MinMaxScaler()
scaler.fit(df[['Age']])
df['Age']=scaler.transform(df[['Age']])
# df

km=KMeans(n_clusters=3)
y_pred=km.fit_predict(df[['Age','Income($)']])


df['Cluster']=y_pred

print(df.head(20))


df1=df[df.Cluster==0]
df2=df[df.Cluster==1]
df3=df[df.Cluster==2]
plt.scatter(df1['Age'],df1['Income($)'],color='green')
plt.scatter(df2['Age'],df2['Income($)'],color='red')
plt.scatter(df3['Age'],df3['Income($)'],color='blue')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()