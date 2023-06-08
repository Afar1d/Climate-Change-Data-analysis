import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


### nlp generate question
### task1 analyze and ask question notebook. missing values & outlier
### https://www.kaggle.com/code/sevgisarac/climate-change/notebook
# Guiding Questions¶

# What are the ten most countries that suffer from temperature change mostly in the last ten years? Done
# What are the ten countries that suffer from temperature change at the very least in the last ten years? Done
# what is 10 most year that Temperature change in all country? Done
# What is the trend of temperature change in the world?

data = pd.read_csv('Environment_Temperature.csv', encoding='cp1252')
# ============================================ #
# Check for the Duplicates
Dup_Rows = data[data.duplicated()]
# Dup_Rows.info() There is no duplicated columns

# ============================================ #
# irrelevant observations
# The 'Area Code', 'Months','Element Code' columns are a unique identifier of each row which does not provide any statistical information.
data.drop(columns=['Area Code'], inplace=True)
data.drop(columns=['Months Code'], inplace=True)
data.drop(columns=['Element Code'], inplace=True)
# unit columns all C
data.drop(columns=['Unit'], inplace=True)
data.drop(columns=['Element'], inplace=True)

# ============================================ #

# Spilling and rename
#  Area into Country Name
data['Country Name'] = data['Area']
data.drop(columns=['Area'], inplace=True)
# Deal with months column
print(data['Months'].describe()) # 7 uniques
print(data['Months'].unique())

# I changed seasons names in the months   'Dec\x96Jan\x96Feb' Winter , 'Mar\x96Apr\x96May' Spring, 'Jun\x96Jul\x96Aug' Summer , 'Sep\x96Oct\x96Nov' Fall
data['Months'].replace(to_replace='Dec–Jan–Feb', value='Winter', inplace=True)
data['Months'].replace(to_replace='Mar–Apr–May', value='Spring', inplace=True)
data['Months'].replace(to_replace='Jun–Jul–Aug', value='Summer', inplace=True)
data['Months'].replace(to_replace='Sep–Oct–Nov', value='Fall', inplace=True)

# ============================================ #
# Manipulating data frame and new table
data.drop(data.iloc[:, 1:60], inplace=True, axis=1)
data2 = pd.read_csv('Temperature_change_Data.csv')

# Deal with missing values
print(data.isna().sum())
# Drop the rows that contain missing values
data.dropna(how='any', inplace=True)
data2.dropna(how='any', inplace=True)
# ============================================ #
data2.info()
data.info()

# Analysis
# What are the ten most countries that suffer from temperature change mostly in the last ten years?

# sort data by year
data2.sort_values(['year'],axis=0, ascending=False,inplace=True,na_position='first')
print(data2.head(10))

# take data for last 10 years
year10 = data2.loc[(data2['year'] >= 2010)]

# calculate mean of tem_change for each country name
gg = year10.groupby('Country Code')['tem_change'].agg(['count', 'mean']).reset_index()
# sort by the mean form top to down
gg.sort_values(['mean'],axis=0, ascending=False,inplace=True,na_position='first')
gg.info()

# draw the graph
X = gg.iloc[:10, 0]
Y = gg.iloc[:10, 2]
plt.bar(X, Y)
plt.xlabel('Country Code', fontsize=20)
plt.ylabel('Temperature', fontsize=20)
plt.show()

# What are the ten countries that suffer from temperature change at the very least in the last ten years?
# sort by the mean form down to top
gg.sort_values(['mean'],axis=0, ascending=True,inplace=True,na_position='first')
# draw the graph
X = gg.iloc[:10, 0]
Y = gg.iloc[:10, 2]
plt.bar(X, Y)
plt.xlabel('Country Code', fontsize=20)
plt.ylabel('Temperature', fontsize=20)
plt.show()

# what is 10 most year that Temperature change in all country?

World = data2.groupby('year')['tem_change'].agg(['count', 'mean']).reset_index()

World.sort_values(['mean'],axis=0, ascending=True,inplace=True,na_position='first')

X = World['year']
Y = World['mean']
plt.plot(X, Y)
plt.xlabel('Year', fontsize=10)
plt.ylabel('Temperature', fontsize=10)
plt.show()