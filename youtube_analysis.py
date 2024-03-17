#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


comments= pd.read_csv(r'C:\Users\lata6\Downloads\Youtube\Dataset\UScomments.csv', on_bad_lines='warn')


# In[3]:


comments.head()


# In[4]:


comments.isnull().sum()


# In[5]:


comments.dropna(inplace=True)


# In[6]:


comments.isnull().sum()


# In[7]:


get_ipython().system('pip install textblob')


# In[8]:


from textblob import TextBlob


# In[9]:


comments.head(6)


# In[10]:


TextBlob("Logan Paul it's yo big day ‼️‼️‼️").sentiment.polarity


# In[11]:


polarity = []

for comment in comments ['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)


# In[12]:


len(polarity)


# In[13]:


comments ['polarity'] = polarity


# In[14]:


comments.head(5)


# In[15]:


filter1 = comments['polarity']==1


# In[16]:


comments_positive = comments[filter1]


# In[17]:


filter2 = comments['polarity']==-1


# In[18]:


comments_negative = comments[filter2]


# In[19]:


get_ipython().system('pip install wordcloud')


# In[20]:


from wordcloud import WordCloud, STOPWORDS


# In[21]:


set(STOPWORDS)


# In[22]:


total_comments_psotive = ' '.join(comments_positive['comment_text'])


# In[23]:


wordcloud= WordCloud(stopwords= set(STOPWORDS)).generate(total_comments_psotive)


# In[24]:


plt.imshow(wordcloud)
plt.axis('off')


# In[25]:


total_comments_negative = ' '.join(comments_negative['comment_text'])


# In[26]:


wordcloud2= WordCloud(stopwords= set(STOPWORDS)).generate(total_comments_negative)


# In[27]:


plt.imshow(wordcloud2)
plt.axis('off')


# In[28]:


get_ipython().system('pip install emoji==2.2.0')


# In[29]:


import emoji 


# In[30]:


comments['comment_text'].head(6)


# In[31]:


comment = ' trending 😉'


# In[32]:


[char for char in comment if char in emoji.EMOJI_DATA]


# In[33]:


all_emoji_list= [char for char in comments['comment_text'].dropna() if char in emoji.EMOJI_DATA]


# In[34]:


all_emoji_list[0:10]


# In[35]:


for comment in comments['comment_text'].dropna():  # Iterate through each comment
    emojis_in_comment = [char for char in comment if char in emoji.EMOJI_DATA]  # Extract emojis from the current comment
    all_emoji_list.extend(emojis_in_comment)


# In[ ]:





# In[36]:


from collections import Counter


# In[37]:


emojis = [Counter(all_emoji_list).most_common(10)[i][0] for i in range (10)]


# In[38]:


freqs = [Counter(all_emoji_list).most_common(10)[i][1] for i in range (10)]


# In[39]:


get_ipython().system('pip install plotly')


# In[40]:


import plotly.graph_objs as go 
from plotly.offline import iplot


# In[41]:


trace = go.Bar(x=emojis, y=freqs)


# In[42]:


iplot([trace])


# In[ ]:


#Exporting data in csv, json and sql


# In[43]:


import os


# In[44]:


files= os.listdir(r'C:\Users\lata6\Downloads\Youtube\additional_data')


# In[45]:


files


# In[46]:


files_csv= [file for file in files if '.csv' in file]


# In[47]:


files_csv


# In[48]:


import warnings 
from warnings import filterwarnings
filterwarnings('ignore')


# In[49]:


full_df = pd.DataFrame()
path = r'C:\Users\lata6\Downloads\Youtube\additional_data'

for file in files_csv:
    current_df = pd.read_csv(path+'/'+file, encoding='iso-8859-1')
    
    full_df = pd.concat([full_df, current_df], ignore_index = True)


# In[50]:


full_df.shape #raw data 


# In[51]:


full_df[full_df.duplicated()].shape


# In[52]:


full_df= full_df.drop_duplicates()


# In[53]:


full_df.shape


# In[54]:


full_df[0:1000].to_csv(r'C:\Users\lata6\Downloads\Youtube\export_data/youtube_sample.csv', index= False)


# In[55]:


full_df[0:1000].to_json(r'C:\Users\lata6\Downloads\Youtube\export_data/youtube_sample.json')


# In[56]:


get_ipython().system('pip install sqlalchemy')


# In[57]:


get_ipython().system('pip install --upgrade sqlalchemy typing_extensions')


# In[58]:


#creating an engine -> allow us to connect with the database
from sqlalchemy import create_engine 


# In[61]:


engine=create_engine(r'sqlite:///C:\Users\lata6\Downloads\Youtube\export_data/youtube_sample.sqlite')


# In[62]:


full_df[0:1000].to_sql('Users', con=engine, if_exists='append')


# In[ ]:


#Analysing the most liked category!!


# In[63]:


full_df.head(5)


# In[64]:


#dictionary 
full_df['category_id'].unique()


# In[66]:


json_df= pd.read_json(r'C:\Users\lata6\Downloads\Youtube\additional_data/US_category_id.json')


# In[67]:


json_df


# In[69]:


json_df['items']


# In[70]:


json_df['items'][0] # kind=dictionary, snippet=inner-dictionary


# In[71]:


json_df['items'][1] #will retrieve title and category id for the dictionary of most liked category dictionary


# In[73]:


cat_dict = {} #creating a dictionary of categories

#retrieving the values of item dictionary -> have to access data like 'id':'title'
for item in json_df['items'].values:
    cat_dict[int(item['id'])]=item['snippet']['title']



# In[74]:


cat_dict


# In[75]:


full_df['category_name']= full_df['category_id'].map(cat_dict)


# In[76]:


plt.figure(figsize=(12,8))
sns.boxplot(x='category_name', y='likes', data=full_df)
plt.xticks(rotation='vertical')


# In[ ]:


#whether audience is engaged or not 


# In[79]:


full_df['like_rate']=(full_df['likes']/full_df['views'])*100
full_df['dislike_rate']=(full_df['dislikes']/full_df['views'])*100
full_df['comment_count_rate']=(full_df['comment_count']/full_df['views'])*100


# In[80]:


full_df.columns


# In[83]:


plt.figure(figsize=(8,6))
sns.boxplot(x='category_name', y='like_rate', data=full_df)
plt.xticks(rotation='vertical')
plt.show()


# In[84]:


sns.regplot(x='views', y='likes', data= full_df)
#if views will increase by one factor or unit, then by unit the likes will increase


# In[86]:


full_df[['views', 'likes', 'dislikes']].corr()


# In[88]:


sns.heatmap(full_df[['views', 'likes', 'dislikes']].corr(), annot= True)


# In[ ]:


#which channels have the largest number of trending videos? 
#EDA


# In[89]:


#using value_counts()
full_df['channel_title'].value_counts()


# In[97]:


#using grouby
cdf = full_df.groupby(['channel_title']).size().sort_values(ascending=False).reset_index()


# In[98]:


cdf= cdf.rename(columns={0:'total_videos'})


# In[99]:


cdf


# In[100]:


import plotly.express as px


# In[101]:


px.bar(data_frame= cdf[0:20], x='channel_title', y= 'total_videos')


# In[ ]:


#does the punctuations have an impact on views, likes, dislikes?


# In[102]:


full_df['title'][0]


# In[103]:


import string 


# In[104]:


string.punctuation


# In[107]:


len([char for char in full_df['title'][0] if char in string.punctuation])


# In[116]:


def punc_count(text):
    return len([char for char in text if char in string.punctuation])


# In[117]:


sample = full_df[0:10000]


# In[118]:


sample['count_punc'] = sample['title'].apply(punc_count)


# In[119]:


sample['count_punc']


# In[120]:


#boxplot
plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc', y='views', data=sample)
plt.show()


# In[121]:


plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc', y='likes', data=sample)
plt.show()


# In[ ]:




