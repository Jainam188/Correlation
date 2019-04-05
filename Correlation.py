#!/usr/bin/env python
# coding: utf-8

# Imported Necessary Libraries

# In[1]:


import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Taken Dataset of Forest Fire from the UCI and I am also putting the link off Dataset to directly download or you can download the dataset from my github repository from given link.
# Printing the Head part of Dataset.

# In[15]:


data = pd.read_csv("forestfires.csv")
print("Head Part of the Dataset ")
print(data.head())


# For understanding of the given data we are printing information of the dataset.

# In[16]:


print("Information about all columns presented in Dataset ")
print(data.info())


# As we can see that month and day columns are object so we cannot find correlation between them so we are converting them by giving the values to it as shown below.

# In[17]:


data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
data.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)


# described data which will tell us the mean value, Standard Deviation, minimum and maximum values of the each column of the dataset.

# In[18]:


print(data.describe())


# Finding Correlationship between all columns. Correlation help to find the relationship between feature columns and target column. 
# There are 3 types of correlation.
# 
# 1.Pearson Correlation
# 2.Spearman's Correlation
# 3.Kendall's Tau Correlation

# In[19]:


# Pearson Correlation
print("Pearson Correlation with All Values")
pearson_corr = data.corr(method='pearson')
print(pearson_corr)


# In[20]:


# Spearman's Correlation
print("Spearman's Correlation with All Values")
spear_corr = data.corr(method='spearman')
print(spear_corr)


# In[21]:


# Kendall's Tau Correlation
print("Kendall's Tau Correlation with All Values")
kendall_corr = data.corr(method='kendall')
print(kendall_corr)


# We are using seaborn library to visualize the correlation by Heatmap. When it's showing heated red area then it shows the positive correlation and if it shows the cooler blue area then it shows the negative correlation.

# In[22]:


# To show correlation as Coolwarm
sns.heatmap(spear_corr, cmap='coolwarm')
plt.title("Correlation Between All Columns")
plt.show()


# We can find the correlation with only targeted column.

# In[23]:


# Correlation with only Target Value
print("Correlation with Only Taget Value")
corr = data.corr()['area'][:-1]
print(corr)


# We can also find correlation directly between two variable.

# In[26]:


one_corr = data['temp'].corr(data['area'])
print(one_corr)


# After correlation we can say that the temperature has the highest positive relationship with the burned area. and DMC, months has also positive correlation and rain has the lowest and negative correlation.
# So we are plotting the month and the area using Bar chart for visualize the realtionship between them.

# In[9]:


y = data['month']
x = data['area']
plt.bar(y, x, align='center', alpha=0.5)
plt.xlabel('Months')
plt.ylabel('Area')
plt.title('Burned Area By Months')
plt.show()


# we are plotting the temperature and the area using Bar chart for visualize the realtionship between them.

# In[10]:


y = data['temp']
x = data['area']
plt.bar(y, x, align='center', alpha=0.5)
plt.xlabel('Temp')
plt.ylabel('Area')
plt.title('Burned Area affected by Temp')
plt.show()

