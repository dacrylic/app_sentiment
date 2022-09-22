# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import spacy
import numpy as np
from spacy import displacy
import pickle as pk
from os import getcwd
import pandas as pd
import json
from google_play_scraper import app
import datetime
import seaborn as sns
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from google_play_scraper import Sort, reviews_all ,reviews
import numpy as np
from scipy.special import softmax
import helper as tr

nlp = spacy.load('en_core_web_sm')
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

from app_store_scraper import AppStore
import random
import pandas as pd
import datetime as dt
import numpy as np

import json

applist = [('Singtel', 'com.singtel.mysingtel'), ('Starhub', 'com.starhub.csselfhelp')]

df_msta = []
for name, apps in applist:
    all_reviews, _ = reviews(apps,
                        # sleep_milliseconds=0,  # defaults to 0
                          lang='en',  # defaults to 'en'
                          country='sg',  # defaults to 'us'
                          sort=Sort.NEWEST,
                          count = 10000
                         )

    df = pd.DataFrame(np.array(all_reviews), columns=['review'])
    df = df.join(pd.DataFrame(df.pop('review').tolist()))
    df['app'] = name
    df['platform'] = 'android'
    columns = list(df)
    appendlist = df.values.tolist()
    for entry in appendlist:
        df_msta.append(entry)

df_msta = pd.DataFrame(df_msta)
df_msta.columns = columns


df_msta = df_msta.loc[df_msta['at'].between('2020-01-01', '2022-12-31')]
#removing NAs
df_msta = df_msta.dropna(subset=['content']).reset_index()
df_msta.isnull().sum()
'''
df_msta = pd.read_csv('app_scores.csv')
df_msta = df_msta.loc[df_msta['at'].between('2022-06-01', '2022-12-31')]
df_msta = df_msta.reset_index(drop = True)
'''
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

reviews = {}

for i in range(len(df_msta["content"])):
    if i % 50 == 0:
      print('i = {}'.format(i))
    doc = nlp(df_msta["content"][i])
    review_sents =[]
    ex = i
    for idx, sentence in enumerate(doc.sents):
        #print (sentence)
        text = str(sentence)
        text = tr.preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        res = (scores[0], scores[1], scores [2])
        review_sents.append({"scores":res, "text":sentence,  "date" : df_msta['at'][ex], 'app' : df_msta['app'][ex], 'rating' : df_msta['score'][ex], 'platform' : df_msta['platform'][ex]})
    reviews[i]=review_sents

my_list = []
for items in reviews.items():
  joke = items[1]
  for item in joke:
    my_list.append([v for k,v in item.items()])

final_df = pd.DataFrame(my_list)
loc = pd.DataFrame.from_records(final_df[0], columns=['neg','neu','pos'])
final_df = final_df.drop(0, axis=1).join(loc)
final_df.columns = ['text','date','app','rating','platform', 'neg','neu','pos']

final_df['sentiment'] = final_df[['neg','neu','pos']].idxmax(axis=1)

from sklearn.preprocessing import OneHotEncoder

#creating instance of one-hot-encoder
encoder = OneHotEncoder(handle_unknown='ignore')

#perform one-hot encoding on 'team' column
encoder_df = pd.DataFrame(encoder.fit_transform(final_df[['sentiment']]).toarray())

#merge one-hot encoded columns back with original DataFrame
df_analysis = final_df.join(encoder_df)


df_analysis.columns = ['text', 'date', 'app', 'rating', 'platform', 'neg_p', 'neu_p','pos_p','sentiment', 'neg', 'neu', 'pos']

trying = df_analysis.groupby(['date','app','platform']).agg({'neg': 'sum',
                          'neu': 'sum',
                          'pos': 'sum'})

trying = trying.reset_index()
trying['date'] = pd.to_datetime(trying['date'])
trying.set_index('date', inplace=True)
#trying = trying.groupby(['app','category','platform']).resample('D').sum()
trying = trying.groupby(['app','platform']).resample('D').sum()
trying['score'] = (trying['neg'] * 0 + trying['neu'] * 0.5 + trying['pos'] * 1) / (trying['neg'] + trying['neu'] + trying['pos'])
#trying['score'] = (trying['neg'] * 0 + trying['pos'] * 1) / (trying['neg'] + trying['pos'])
trying = trying.fillna(method ='ffill')
#trying = trying.dropna()
trying['rolling'] = trying['score'].rolling(30).mean()
trying = trying.fillna(method ='bfill')

print (trying)

import seaborn as sns
sns.set(rc={'figure.figsize':(20 ,12)})

sns.lineplot(data=trying, x='date', y= 'rolling',
             hue = 'app',
             color = 'black',
             linestyle='--',
             ci = None
             )
import matplotlib.pyplot as plt
plt.show()

# Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/