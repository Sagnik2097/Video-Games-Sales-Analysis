#!/usr/bin/env python
# coding: utf-8

# #                                     Video Games Sales Analysis 

# The “Video Games Sales Analysis And Visualization” project is an in-depth exploration of the global video game industry. Leveraging a comprehensive dataset of video game sales, this project aims to uncover patterns, trends, and insights that can help stakeholders make informed decisions.
# 
# The project involves the use of advanced data analysis techniques and visualization tools to interpret the data effectively. It covers various aspects such as sales by region, platform preference, genre popularity, and the impact of critics’ ratings on sales.
# 
# By visualizing these data points, the project provides a clear and concise view of the video game market’s dynamics. It serves as a valuable resource for game developers, marketers, and strategists in the gaming industry, helping them understand market trends and consumer preferences.
# 
# Whether you’re a game enthusiast curious about market trends or a professional seeking actionable insights, this project offers a fascinating look into the world of video games. Enjoy the journey through data, and happy gaming!

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as st
pd.set_option('display.max_columns', None)

import math

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set_style('whitegrid')

from sklearn.preprocessing import StandardScaler
from scipy import stats


# In[2]:


data = pd.read_csv('Video_game_Dataset.csv')
data.head()


# In[3]:


data.shape


# Deleted some incomplete data. You can see, from 2016 the data we have those are not fully completed. It will help us to do better analysis. Incomplete data always hamper in analysis

# In[4]:


drop_row_index = data[data['Year'] > 2015].index
data = data.drop(drop_row_index)


# In[5]:


data.shape


# In[6]:


data.info()


# # Their fields and data types are:
# 
# Rank - Ranking of overall sales, integer
# 
# Name - The games name
# 
# Platform - Platform of the games release (i.e. PC,PS4, etc.), object
# 
# Year - Year of the game's release, float
# 
# Genre - Genre of the game ,object
# 
# Publisher - Publisher of the game, object
# 
# NA_Sales - Sales in North America (in millions), float
# 
# EU_Sales - Sales in Europe (in millions), float
# 
# JP_Sales - Sales in Japan (in millions), float
# 
# Other_Sales - Sales in the rest of the world (in millions), float
# 
# Global_Sales - Total worldwide sales, float

# In[7]:


data.describe()


# In[8]:


data.describe(include=['object', 'bool'])


# In[9]:


data.isnull().sum()


# we don't have much missing values

# In[ ]:





# # 1. What genre games have been made the most?

# In[10]:


data['Genre'].value_counts()


# In[11]:


plt.figure(figsize=(15,10))
sns.countplot(x='Genre',data=data, order=data['Genre'].value_counts().index)


# Findings= Most of the people love action and sports game. in action 3316 and in sports 2346 games release.

# # 2. Which year had the most game release?

# In[12]:


plt.figure(figsize=(15, 10))
sns.countplot(x="Year", data=data, order = data.groupby(by=['Year'])['Name'].count().sort_values(ascending=False).index)
plt.xticks(rotation=90)


# Findings=
# - **2009**: 1431 games
# - **2008**: 1428 games
# - **2010**: 1259 games
# - **2007**: 1202 games
# - **2011**: 1139 games
# 
# 2008 to 2010 was most game released

# # 3. Top 5 years games release by genre.

# In[13]:


plt.figure(figsize=(20, 10))
sns.countplot(x="Year", data=data, hue='Genre',order=data.Year.value_counts().iloc[:5].index)


# # 4. Which year had the highest sales worldwide?

# In[14]:


data_year = data.groupby(by=['Year'])['Global_Sales'].sum()
data_year = data_year.reset_index()
data_year.sort_values(by=['Global_Sales'], ascending=False)


# In[15]:


plt.figure(figsize=(15, 10))
sns.barplot(x='Year', y='Global_Sales', data=data_year)
plt.xticks(rotation=90)


# Findings=
# 
# - **2008**: 678.90 
# - **2009**: 667.30 
# - **2007**: 611.13 
# - **2010**: 600.45 
# - **2006**: 521.04 
# 
# **2006** was not among the top 5 years in terms of game releases, it ranks in the top 5 for highest global sales.

# # 5. Which genre game has been released the most in a single year?

# In[16]:


year_max_df = data.groupby(['Year','Genre']).size().reset_index(name='count')
year_max_idx = year_max_df.groupby(['Year'])['count'].transform(max) == year_max_df['count']
year_max_genre = year_max_df[year_max_idx].reset_index(drop=True)
year_max_genre = year_max_genre.drop_duplicates(subset=['Year','count'],keep='last').reset_index(drop=True)


# In[17]:


genre = year_max_genre['Genre'].values


# In[18]:


plt.figure(figsize=(25, 10))
g = sns.barplot(x='Year', y='count', data=year_max_genre)
index = 0

for value in year_max_genre['count'].values:
    g.text(index, value + 5, str(genre[index] + '---' +str(value)), color='#000', size=14, rotation= 90, ha="center")
    index += 1
    
plt.xticks(rotation=90)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.show()


# Findings=
# 
# - **2009**: 272 
# - **2012**: 266 
# 

# # 6. Which genre game has sold the most in a single year?

# In[19]:


year_sale_df = data.groupby(['Year','Genre'])['Global_Sales'].sum().reset_index()
year_sale = year_sale_df.groupby(by=['Year'])['Global_Sales'].transform(max) == year_sale_df['Global_Sales']
year_sale_max = year_sale_df[year_sale].reset_index(drop=True)
#year_sale_max


# In[20]:


genre = year_sale_max['Genre']


# In[21]:


plt.figure(figsize=(25, 10))
g = sns.barplot(x='Year', y='Global_Sales', data=year_sale_max)
index = 0
for value in year_sale_max['Global_Sales']:
    g.text(index, value + 1, str(genre[index] + '----' +str(round(value, 2))), color='#000', size=14, rotation= 90, ha="center")
    index += 1

plt.xticks(rotation=90)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Count Sales', fontsize=20)
plt.show()


# Findings=
# 
# - **2009 Action**: 139.36 million 
# - **2008 Action**: 136.39 miliion

# # 7. Which genre game have the highest sale price globally

# In[22]:


data_genre = data.groupby(by=['Genre'])['Global_Sales'].sum()
data_genre = data_genre.reset_index()
data_genre = data_genre.sort_values(by=['Global_Sales'], ascending=False)


# In[23]:


plt.figure(figsize=(20, 15))
sns.barplot(x='Genre',y='Global_Sales', data=data_genre)


# Action and Sports are most highest sale price globally

# # 8. Which platfrom have the highest sale price globally

# In[24]:


data_platform = data.groupby(by=['Platform'])['Global_Sales'].sum()
data_platform = data_platform.reset_index()
data_platform = data_platform.sort_values(by=['Global_Sales'], ascending=False)


# In[25]:


plt.figure(figsize=(15, 10))
sns.barplot(x="Platform", y="Global_Sales", data=data_platform)


# Finding = 
# **PS2**

# # 9. Which individual game have the highest sale price globally?

# In[26]:


top_game_sale = data.head(20)
top_game_sale = top_game_sale[['Name', 'Year', 'Genre', 'Global_Sales']]
top_game_sale = top_game_sale.sort_values(by=['Global_Sales'], ascending=False)
#top_game_sale


# In[27]:


name = top_game_sale['Name']
year = top_game_sale['Year']


# In[28]:


plt.figure(figsize=(30, 18))
g = sns.barplot(x='Name', y='Global_Sales', data=top_game_sale)
index = 0
for value in top_game_sale['Global_Sales']:
    g.text(index, value - 18, name[index], color='#000', size=14, rotation= 90, ha="center")
    index += 1

plt.xticks(rotation=90)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Count Sales', fontsize=20)
plt.show()


# Finding = 
# **Wii Sports** is the individual game have the highest sale price globally

# # 10. Sales compearison by genre

# In[29]:


comp_genre = data[['Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
#comp_genre
comp_map = comp_genre.groupby(by=['Genre']).sum()
#comp_map


# In[30]:


plt.figure(figsize=(15, 10))
sns.set(font_scale=1)

sns.heatmap(comp_map, annot=True, fmt='.1f', cmap='coolwarm')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[31]:


comp_table = comp_map.reset_index()
comp_table = pd.melt(comp_table, id_vars=['Genre'], value_vars=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], var_name='Sale_Area', value_name='Sale_Price')
comp_table.head()


# In[32]:


plt.figure(figsize=(15, 10))
sns.set_style('white')


sns.barplot(x='Genre', y='Sale_Price', hue='Sale_Area', data=comp_table, palette="Blues")


# Finding=
# 
# **Action** , **Sports** and **Shooter** have much sales compare with others. and North America (NA_sales) have heights sales all the time.

# # 11. Sales compearison by platform

# In[33]:


comp_platform = data[['Platform', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
comp_platform.head()


# In[34]:


comp_platform = comp_platform.groupby(by=['Platform']).sum().reset_index()


# In[35]:


#comp_table = comp_map.reset_index()
comp_table = pd.melt(comp_platform, id_vars=['Platform'], value_vars=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], var_name='Sale_Area', value_name='Sale_Price')
comp_table.head()


# In[36]:


plt.figure(figsize=(30, 15))
sns.set_style('white')

sns.barplot(x='Platform', y='Sale_Price', hue='Sale_Area', data=comp_table, palette="Blues")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# **X360**, **Wii**, and **PS** series are market leader and North Americas Sales is performing good

# # 12. Top 20 Publisher

# In[37]:


top_publishers = data.groupby(by=['Publisher'])['Year'].count().sort_values(ascending=False)
top_publishers = pd.DataFrame(top_publishers).reset_index()
#top_publishers


# In[38]:


plt.figure(figsize=(15, 10))
sns.countplot(x="Publisher", data=data, order = data.groupby(by=['Publisher'])['Year'].count().sort_values(ascending=False).iloc[:20].index)
plt.xticks(rotation=90)


# Finding= 
# 
# **Electronic Arts(EA Sports)** : 1339

# # 13. Top global sales by publisher

# In[39]:


sale_publisher = data[['Publisher', 'Global_Sales']]
sale_publisher = sale_publisher.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(20)
sale_publisher = pd.DataFrame(sale_publisher).reset_index()
#sale_publisher


# In[40]:


plt.figure(figsize=(15, 10))
sns.set_style('white')

sns.barplot(x = 'Publisher', y = 'Global_Sales', data = sale_publisher, palette="Blues")


plt.xticks(rotation=90)


# Despite not being a top 5 publisher, **Nintendo’s** focus on quality over quantity has led to their financial success. With **696** high-quality games

# # 14. publisher comperison

# In[41]:


comp_publisher = data[['Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]
comp_publisher.head()


# In[42]:


comp_publisher = comp_publisher.groupby(by = ['Publisher']).sum().reset_index().sort_values(by = ['Global_Sales'], ascending= False)
comp_publisher = comp_publisher.head(20)
#comp_publisher


# In[43]:


comp_publisher = pd.melt(comp_publisher, id_vars=['Publisher'], value_vars=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], 
                         var_name='Sale_Area', value_name='Sale_Price')
comp_publisher


# In[44]:


plt.figure(figsize=(30, 15))
sns.set_style('white')

sns.barplot(x='Publisher', y='Sale_Price', hue='Sale_Area', data=comp_publisher,palette="Blues")
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.xlabel('Publisher', fontsize=20)
plt.ylabel('Sele Price', fontsize=20)
plt.show()


# # 15. Top publisher by Count each year

# In[45]:


top_publisher = data[['Year', 'Publisher']]
top_publisher_df = top_publisher.groupby(by=['Year','Publisher']).size().reset_index(name='Count')
top_publisher_idx = top_publisher_df.groupby(by=['Year'])['Count'].transform(max) == top_publisher_df['Count']
top_publisher_count = top_publisher_df[top_publisher_idx].reset_index(drop=True)
top_publisher_count = top_publisher_count.drop_duplicates(subset=['Year','Count'],keep='last').reset_index(drop=True)
#top_publisher_count


# In[46]:


publisher= top_publisher_count['Publisher']


# In[47]:


plt.figure(figsize=(35, 20))
sns.set_style('white')

g = sns.barplot(x='Year', y='Count', data=top_publisher_count, palette="Blues")
index = 0
for value in top_publisher_count['Count'].values:
    g.text(index, value + 5, str(publisher[index] + '----' +str(value)), color='#000', size=20, rotation= 90, ha="center")
    index += 1


plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.show()


# # 16. Total revenue by region

# In[48]:


top_sale_reg = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
top_sale_reg = top_sale_reg.sum().reset_index()
top_sale_reg = top_sale_reg.rename(columns={"index": "region", 0: "sale"})
top_sale_reg


# In[49]:


plt.figure(figsize=(12, 8))
sns.set_style('white')

sns.barplot(x='region', y='sale', data = top_sale_reg, palette="Blues")


# In[50]:


labels = top_sale_reg['region']
sizes = top_sale_reg['sale']

# Get the 'Blues' color palette
colors = sns.color_palette('Blues')

plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors)
plt.show()


# Findings=
# 
# **North America** almost Cover **50%** of sales

# # 17. Sales Histogram

# In[51]:


plt.figure(figsize=(25,30))
sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
for i, column in enumerate(sales_columns):
    plt.subplot(3,2,i+1)
    sns.distplot(data[column], bins=20, kde=False, fit=stats.gamma)


# # ------------------------------Thank you for staying with us -----------------------------------

# In[ ]:




