import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Get The Data
data = pd.read_csv('College_Data',index_col=0)
print(data)

sns.scatterplot(x="Grad.Rate",y="Room.Board",data=data,hue="Private")
plt.show()
sns.scatterplot(x="F.Undergrad",y="Outstate",data=data,hue="Private")
plt.show()
sns.set_style('darkgrid')
g=sns.FacetGrid(data,hue="Private",palette='coolwarm',height=6,aspect=2)
g.map(plt.hist,"Outstate",bins=20,alpha=0.7)
plt.show()

sns.set_style('darkgrid')
g= sns.FacetGrid(data,hue="Private",palette="coolwarm",height=6,aspect=2)
g.map(plt.hist,"Grad.Rate",bins=20,alpha=0.8)
plt.show()

print(data[data["Grad.Rate"]>100])
data.loc['Cazenovia College','Grad.Rate']=90


sns.set_style('darkgrid')
g= sns.FacetGrid(data,hue="Private",palette="coolwarm",height=6,aspect=2)
g.map(plt.hist,"Grad.Rate",bins=20,alpha=0.8)
plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(data.drop("Private",axis=1))
predict=kmeans.labels_
#print(predict)


def filter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0
    
data["Cluster"]=data["Private"].apply(filter)
print(data["Cluster"])

# df=[]
# for i in data["Private"]:
#     if i=="Yes":
#         df.append(1)
#     else:
#         df.append(0)

# data["Cluster"]=df

f,(ax1,ax2)=plt.subplots(2,1,sharey=True,figsize=(10,6))
ax1.scatter(data["PhD"],data["Outstate"],c=predict,cmap='rainbow')
ax2.scatter(data["Accept"],data["Enroll"],c=data["Cluster"],cmap='rainbow')
ax1.set_title("Predicted")
ax2.set_title("Original")
plt.show()

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(data["Cluster"],predict))
print(classification_report(data["Cluster"],predict))