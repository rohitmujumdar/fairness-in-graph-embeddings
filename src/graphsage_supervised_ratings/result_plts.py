#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


regular_df = pd.read_csv('./results/reco_group.csv', index_col=0)
# regular_df = pd.read_csv('./results_fair/reco_group.csv', index_col=0)

# fair_df = pd.read_csv('../../results/fairwalk/reco_group.csv', index_col=0)


# In[3]:


regular_df.head()


# In[4]:


sf = regular_df[regular_df['gender'] == 'F']['count'].sum()
sm = regular_df[regular_df['gender'] == 'M']['count'].sum()


# In[5]:


assert sf + sm == regular_df['count'].sum()


# In[6]:


def get_prob(row):
    if row['gender'] == 'F':
        return row['count']*100/sf
    return row['count']*100/sm


# In[7]:


regular_df['fraction'] = regular_df.apply(lambda x : get_prob(x), axis=1)


# In[8]:


regular_df.head()


# In[9]:


male_df = regular_df[regular_df.gender == "M"].sort_values(by='genre')
female_df = regular_df[regular_df.gender == "F"].sort_values(by='genre')

print(round(np.std(male_df['fraction'].values - female_df['fraction'].values),2))


# In[ ]:





# In[10]:


# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np


# labels = list(regular_df['genre'].unique())#['G1', 'G2', 'G3', 'G4', 'G5']
# men_means = regular_df[regular_df['gender']=='M']['fraction']#[20, 34, 30, 35, 27]
# women_means = regular_df[regular_df['gender']=='F']['fraction']#[25, 32, 34, 20, 25]

# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots(figsize=(10,5))
# rects1 = ax.bar(x - width/2, men_means, width, label='Male')
# rects2 = ax.bar(x + width/2, women_means, width, label='Female')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('% of recommendations')
# ax.set_title('% of recommendations from each genre for Females')
# ax.set_xticks(x)
# ax.set_xticklabels(labels, rotation = 45)
# ax.legend()


# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')


# # autolabel(rects1)
# # autolabel(rects2)

# fig.tight_layout()
# plt.savefig('../../results/node2vec.png')
# plt.show()


# In[ ]:




