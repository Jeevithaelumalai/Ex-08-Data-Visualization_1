# Ex-08-Data-Visualization-

## AIM
To Perform Data Visualization on a complex dataset and save the data to a file. 

# Explanation
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature generation and selection techniques to all the features of the data set
### STEP 4
Apply data visualization techniques to identify the patterns of the data.


# CODE:
DEVELOPED BY : JEEVITHA E
REG NO : 212222230054
# Loading the dataset :
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("/content/Superstore (3).csv",encoding='unicode_escape')
df
```
# Removing unnecessary data variables :
```
df.drop('Row ID',axis=1,inplace=True)
df.drop('Order ID',axis=1,inplace=True)
df.drop('Customer ID',axis=1,inplace=True)
df.drop('Customer Name',axis=1,inplace=True)
df.drop('Country',axis=1,inplace=True)
df.drop('Postal Code',axis=1,inplace=True)
df.drop('Product ID',axis=1,inplace=True)
df.drop('Product Name',axis=1,inplace=True)
df.drop('Order Date',axis=1,inplace=True)
df.drop('Ship Date',axis=1,inplace=True)
print("Updated dataset")
df

df.isnull().sum()
Detecting and removing outliers in current numeric data :
plt.figure(figsize=(12,10))
plt.title("Data with outliers")
df.boxplot()
plt.show()

plt.figure(figsize=(12,10))
cols = ['Sales','Quantity','Discount','Profit']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
data visualization :
line plots :
import seaborn as sns
sns.lineplot(x="Sub-Category",y="Sales",data=df,marker='o')
plt.title("Sub Categories vs Sales")
plt.xticks(rotation = 90)
plt.show()

sns.lineplot(x="Category",y="Profit",data=df,marker='o')
plt.xticks(rotation = 90)
plt.title("Categories vs Profit")
plt.show()

sns.lineplot(x="Region",y="Sales",data=df,marker='o')
plt.xticks(rotation = 90)
plt.title("Region area vs Sales")
plt.show()

sns.lineplot(x="Category",y="Discount",data=df,marker='o')
plt.title("Categories vs Discount")
plt.show()

sns.lineplot(x="Sub-Category",y="Quantity",data=df,marker='o')
plt.xticks(rotation = 90)
plt.title("Sub Categories vs Quantity")
plt.show()
```
# bar plots:
```
sns.barplot(x="Sub-Category",y="Sales",data=df)
plt.title("Sub Categories vs Sales")
plt.xticks(rotation = 90)
plt.show()

sns.barplot(x="Category",y="Profit",data=df)
plt.title("Categories vs Profit")
plt.show()

sns.barplot(x="Sub-Category",y="Quantity",data=df)
plt.title("Sub Categories vs Quantity")
plt.xticks(rotation = 90)
plt.show()

sns.barplot(x="Category",y="Discount",data=df)
plt.title("Categories vs Discount")
plt.show()

plt.figure(figsize=(12,7))
sns.barplot(x="State",y="Sales",data=df)
plt.title("States vs Sales")
plt.xticks(rotation = 90)
plt.show()

plt.figure(figsize=(25,8))
sns.barplot(x="State",y="Sales",hue="Region",data=df)
plt.title("State vs Sales based on Region")
plt.xticks(rotation = 90)
plt.show()
Histogram :
sns.histplot(data = df,x = 'Region',hue='Ship Mode')
sns.histplot(data = df,x = 'Category',hue='Quantity')
sns.histplot(data = df,x = 'Sub-Category',hue='Category')
plt.xticks(rotation = 90)
plt.show()
sns.histplot(data = df,x = 'Quantity',hue='Segment')
plt.hist(data = df,x = 'Profit')
plt.show()
count plot :
plt.figure(figsize=(10,7))
sns.countplot(x ='Segment', data = df,hue = 'Sub-Category')
sns.countplot(x ='Region', data = df,hue = 'Segment')
sns.countplot(x ='Category', data = df,hue='Discount')
sns.countplot(x ='Ship Mode', data = df,hue = 'Quantity')
Barplot :
sns.boxplot(x="Sub-Category",y="Discount",data=df)
plt.xticks(rotation = 90)
plt.show()
sns.boxplot( x="Profit", y="Category",data=df)
plt.xticks(rotation = 90)
plt.show()
plt.figure(figsize=(10,7))
sns.boxplot(x="Sub-Category",y="Sales",data=df)
plt.xticks(rotation = 90)
plt.show()
sns.boxplot(x="Category",y="Profit",data=df)
sns.boxplot(x="Region",y="Sales",data=df)
plt.figure(figsize=(10,7))
sns.boxplot(x="Sub-Category",y="Quantity",data=df)
plt.xticks(rotation = 90)
plt.show()
sns.boxplot(x="Category",y="Discount",data=df)
plt.figure(figsize=(15,7))
sns.boxplot(x="State",y="Sales",data=df)
plt.xticks(rotation = 90)
plt.show()
```
# KDE plot :
```
sns.kdeplot(x="Profit", data = df,hue='Category')
sns.kdeplot(x="Sales", data = df,hue='Region')
sns.kdeplot(x="Quantity", data = df,hue='Segment')
sns.kdeplot(x="Discount", data = df,hue='Segment')
#violin plot
sns.violinplot(x="Profit",data=df)
sns.violinplot(x="Discount",y="Ship Mode",data=df)
sns.violinplot(x="Quantity",y="Ship Mode",data=df)
Point plot :
sns.pointplot(x=df["Quantity"],y=df["Discount"])
sns.pointplot(x=df["Quantity"],y=df["Category"])
sns.pointplot(x=df["Sales"],y=df["Sub-Category"])
Pie Chart :
df.groupby(['Category']).sum().plot(kind='pie', y='Discount',figsize=(6,10),pctdistance=1.7,labeldistance=1.2)
df.groupby(['Sub-Category']).sum().plot(kind='pie', y='Sales',figsize=(10,10),pctdistance=1.7,labeldistance=1.2)
df.groupby(['Region']).sum().plot(kind='pie', y='Profit',figsize=(6,9),pctdistance=1.7,labeldistance=1.2)
df.groupby(['Ship Mode']).sum().plot(kind='pie', y='Quantity',figsize=(8,11),pctdistance=1.7,labeldistance=1.2)

df1=df.groupby(by=["Category"]).sum()
labels=[]
for i in df1.index:
    labels.append(i)  
plt.figure(figsize=(8,8))
colors = sns.color_palette('pastel')
plt.pie(df1["Profit"],colors = colors,labels=labels, autopct = '%0.0f%%')
plt.show()

df1=df.groupby(by=["Ship Mode"]).sum()
labels=[]
for i in df1.index:
    labels.append(i)
colors=sns.color_palette("bright")
plt.pie(df1["Sales"],labels=labels,autopct="%0.0f%%")
plt.show()
#HeatMap
df4=df.copy()
encoding :
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
le=LabelEncoder()
ohe=OneHotEncoder
oe=OrdinalEncoder()

df4["Ship Mode"]=oe.fit_transform(df[["Ship Mode"]])
df4["Segment"]=oe.fit_transform(df[["Segment"]])
df4["City"]=le.fit_transform(df[["City"]])
df4["State"]=le.fit_transform(df[["State"]])
df4['Region'] = oe.fit_transform(df[['Region']])
df4["Category"]=oe.fit_transform(df[["Category"]])
df4["Sub-Category"]=le.fit_transform(df[["Sub-Category"]])
Scaling :
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df5=pd.DataFrame(sc.fit_transform(df4),columns=['Ship Mode', 'Segment', 'City', 'State','Region',
                                               'Category','Sub-Category','Sales','Quantity','Discount','Profit'])
```
# Heatmap :
plt.subplots(figsize=(12,7))
sns.heatmap(df5.corr(),cmap="PuBu",annot=True)
plt.show()

# OUPUT:
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/f4a8b75b-83a4-46bf-9d56-53ff9bce7241)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/429e9d12-74c3-4991-bb99-f072e6c458e3)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/d104a843-be2b-4cb6-a27f-22e2ff0ba752)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/38be81c4-eaaf-49e7-870c-dd8a65598940)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/2f2c4cfe-a5ce-4081-ac6f-10c15e19ad28)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/a8ece1b8-038d-4170-b30c-b4ef959307c7)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/cc9291a3-4a37-4d40-bc11-fb8b3d2e41d5)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/d8804b0c-1253-4229-845e-22b635912eb4)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/998f8681-be57-4d0e-b765-090826bfdfdb)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/54ff8de4-4a6f-4c57-9f9d-b71f918c2544)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/093a9abf-6371-4ece-a6ac-5a44ecc299cf)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/58c4fd1b-a817-4ec3-83bb-5d9f95903e9d)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/766fa6fe-5575-48e7-98b2-74a715d98167)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/a4b37d73-2b86-4b77-b09a-c855faee43ae)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/64aeeac1-44d4-47a2-b4c2-01ca766483dd)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/f0cc50fa-8361-4315-b091-0d598ea88507)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/62e5ffd5-da1d-4492-b6e8-cdba9aa1e392)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/13941a20-0ca1-49f9-ba31-56886f379181)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/df2b5ff2-5648-4aab-80fe-3d72789185c4)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/5ef04f87-3c5b-4c63-b256-67fe7d318e41)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/fed64800-31d4-43a6-9a2e-bddac5281c7f)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/4da80c8b-ecdf-4180-b125-dfa273f131d0)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/b3f78818-d394-4902-97f2-26551932ad8a)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/355d37b4-97a6-4bf9-8282-1137c62608a5)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/38ddac6c-d5da-44eb-9b58-6495de9e57d0)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/264a300c-2838-42e4-b5e1-0f92be8d14ba)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/31d9d150-e3a0-4022-ac50-3de18e4234f9)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/09065d19-c3bb-4cfa-8395-5d93e90f7109)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/25823a33-e779-48cc-922d-fd654ffa0d37)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/88b06dd0-ee7d-4913-aafb-06b1789a0b24)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/5a657384-8361-4845-a1fe-a263c4d9f6cd)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/96fbb402-f3c3-468b-8d78-b3864395193a)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/8828c1d4-74d5-4dc6-b55d-44f18cd99d97)
![image](https://github.com/Jeevithaelumalai/Ex-08-Data-Visualization_1/assets/118708245/df613422-ab60-42b3-84c9-a99af7c9748b)

# RESULT:
Hence,Data Visualization is applied on the complex dataset using libraries like Seaborn and Matplotlib successfully and the data is saved to file
