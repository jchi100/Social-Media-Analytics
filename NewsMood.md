

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import tweepy
import time
import seaborn as sns
```


```python
# Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
consumer_key = "JeOpBYvN38K6jwVKoQTuKhofj"
consumer_secret = "qNKYIpKRxVR95w3BzfD9CITJ2xhq7MM3I36yEqh2U6WsJAfB9q"
access_token = "975022209948413954-FxRMca7HQaLq01HP9YoXj3tk6HrHx2w"
access_token_secret = "gt4ZvTpOfSb69flVrWfLcEtpVmZCxhL0ly4BUq8n8GfUf"


# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python

# Target Search Term
target_terms = ("@BBC", "@CBS", "@CNN", "@FoxNews", "@nytimes")

# Array to hold sentiment
sentiment_array = []

# Counter
counter = 1

# Variables for holding sentiments
date_list= []
compound_list = []
positive_list = []
negative_list = []
neutral_list = []
tweetago_list = []
target_list = []

for target in target_terms:

# Loop through 5 pages of tweets (total 100 tweets)
    counter=0
  #  for x in range(5):
    for x in range(5):
    # Get all tweets from home feed
        public_tweets = api.user_timeline(target,count=20, result_type="recent")

    # Loop through all tweets 
        for tweet in public_tweets:
       
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]
            tweets_ago = counter
        
            compound_list.append(compound)
            positive_list.append(pos)
            negative_list.append(neg)
            neutral_list.append(neu)
            date_list.append(tweet["created_at"])
            tweetago_list.append(tweets_ago)
            target_list.append(target)
            # Add to counter 
            counter = counter+1
```


```python
sentiment_dict = {"Target": target_list,
                 "Date": date_list,
                 "Compound":compound_list,
                 "Postitive":positive_list,
                 "Negative":negative_list,
                  "Nuetral":neutral_list,
                 "Tweet Ago":tweetago_list}

sentiment_df = pd.DataFrame(sentiment_dict)
sentiment_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Nuetral</th>
      <th>Postitive</th>
      <th>Target</th>
      <th>Tweet Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.6249</td>
      <td>Mon Mar 26 19:03:02 +0000 2018</td>
      <td>0.194</td>
      <td>0.806</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.7783</td>
      <td>Mon Mar 26 18:24:04 +0000 2018</td>
      <td>0.000</td>
      <td>0.688</td>
      <td>0.312</td>
      <td>@BBC</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>Mon Mar 26 17:30:04 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.5859</td>
      <td>Mon Mar 26 16:25:02 +0000 2018</td>
      <td>0.000</td>
      <td>0.787</td>
      <td>0.213</td>
      <td>@BBC</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.3818</td>
      <td>Mon Mar 26 15:45:04 +0000 2018</td>
      <td>0.161</td>
      <td>0.744</td>
      <td>0.095</td>
      <td>@BBC</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
bbc_df = sentiment_df.loc[sentiment_df['Target']=='@BBC']
bbc_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Nuetral</th>
      <th>Postitive</th>
      <th>Target</th>
      <th>Tweet Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.6249</td>
      <td>Mon Mar 26 19:03:02 +0000 2018</td>
      <td>0.194</td>
      <td>0.806</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.7783</td>
      <td>Mon Mar 26 18:24:04 +0000 2018</td>
      <td>0.000</td>
      <td>0.688</td>
      <td>0.312</td>
      <td>@BBC</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>Mon Mar 26 17:30:04 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.5859</td>
      <td>Mon Mar 26 16:25:02 +0000 2018</td>
      <td>0.000</td>
      <td>0.787</td>
      <td>0.213</td>
      <td>@BBC</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.3818</td>
      <td>Mon Mar 26 15:45:04 +0000 2018</td>
      <td>0.161</td>
      <td>0.744</td>
      <td>0.095</td>
      <td>@BBC</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
cbs_df = sentiment_df.loc[sentiment_df['Target']=='@CBS']
cbs_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Nuetral</th>
      <th>Postitive</th>
      <th>Target</th>
      <th>Tweet Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>0.5080</td>
      <td>Mon Mar 26 22:14:25 +0000 2018</td>
      <td>0.0</td>
      <td>0.880</td>
      <td>0.120</td>
      <td>@CBS</td>
      <td>0</td>
    </tr>
    <tr>
      <th>101</th>
      <td>0.0000</td>
      <td>Mon Mar 26 20:42:48 +0000 2018</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@CBS</td>
      <td>1</td>
    </tr>
    <tr>
      <th>102</th>
      <td>0.7959</td>
      <td>Mon Mar 26 16:59:19 +0000 2018</td>
      <td>0.0</td>
      <td>0.786</td>
      <td>0.214</td>
      <td>@CBS</td>
      <td>2</td>
    </tr>
    <tr>
      <th>103</th>
      <td>0.8070</td>
      <td>Mon Mar 26 16:05:18 +0000 2018</td>
      <td>0.0</td>
      <td>0.622</td>
      <td>0.378</td>
      <td>@CBS</td>
      <td>3</td>
    </tr>
    <tr>
      <th>104</th>
      <td>0.0000</td>
      <td>Sun Mar 25 23:57:34 +0000 2018</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@CBS</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
cnn_df = sentiment_df.loc[sentiment_df['Target']=='@CNN']
cnn_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Nuetral</th>
      <th>Postitive</th>
      <th>Target</th>
      <th>Tweet Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>200</th>
      <td>-0.8658</td>
      <td>Mon Mar 26 23:45:04 +0000 2018</td>
      <td>0.447</td>
      <td>0.553</td>
      <td>0.0</td>
      <td>@CNN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>201</th>
      <td>-0.5859</td>
      <td>Mon Mar 26 23:38:28 +0000 2018</td>
      <td>0.202</td>
      <td>0.798</td>
      <td>0.0</td>
      <td>@CNN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>202</th>
      <td>0.0000</td>
      <td>Mon Mar 26 23:30:14 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>@CNN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>203</th>
      <td>0.0000</td>
      <td>Mon Mar 26 23:15:03 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>@CNN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>204</th>
      <td>0.0000</td>
      <td>Mon Mar 26 23:10:09 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>@CNN</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
fox_df = sentiment_df.loc[sentiment_df['Target']=='@FoxNews']
fox_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Nuetral</th>
      <th>Postitive</th>
      <th>Target</th>
      <th>Tweet Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>300</th>
      <td>0.2023</td>
      <td>Mon Mar 26 23:47:52 +0000 2018</td>
      <td>0.000</td>
      <td>0.904</td>
      <td>0.096</td>
      <td>@FoxNews</td>
      <td>0</td>
    </tr>
    <tr>
      <th>301</th>
      <td>-0.4019</td>
      <td>Mon Mar 26 23:41:55 +0000 2018</td>
      <td>0.119</td>
      <td>0.881</td>
      <td>0.000</td>
      <td>@FoxNews</td>
      <td>1</td>
    </tr>
    <tr>
      <th>302</th>
      <td>-0.5267</td>
      <td>Mon Mar 26 23:24:30 +0000 2018</td>
      <td>0.206</td>
      <td>0.794</td>
      <td>0.000</td>
      <td>@FoxNews</td>
      <td>2</td>
    </tr>
    <tr>
      <th>303</th>
      <td>-0.3182</td>
      <td>Mon Mar 26 23:17:19 +0000 2018</td>
      <td>0.179</td>
      <td>0.699</td>
      <td>0.122</td>
      <td>@FoxNews</td>
      <td>3</td>
    </tr>
    <tr>
      <th>304</th>
      <td>-0.6486</td>
      <td>Mon Mar 26 23:09:14 +0000 2018</td>
      <td>0.281</td>
      <td>0.719</td>
      <td>0.000</td>
      <td>@FoxNews</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
nytimes_df = sentiment_df.loc[sentiment_df['Target']=='@nytimes']
nytimes_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Nuetral</th>
      <th>Postitive</th>
      <th>Target</th>
      <th>Tweet Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>400</th>
      <td>0.5106</td>
      <td>Mon Mar 26 23:47:06 +0000 2018</td>
      <td>0.000</td>
      <td>0.645</td>
      <td>0.355</td>
      <td>@nytimes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>401</th>
      <td>-0.8957</td>
      <td>Mon Mar 26 23:32:04 +0000 2018</td>
      <td>0.435</td>
      <td>0.565</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>402</th>
      <td>-0.5574</td>
      <td>Mon Mar 26 23:17:04 +0000 2018</td>
      <td>0.167</td>
      <td>0.833</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>2</td>
    </tr>
    <tr>
      <th>403</th>
      <td>0.0000</td>
      <td>Mon Mar 26 23:02:04 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>3</td>
    </tr>
    <tr>
      <th>404</th>
      <td>0.0000</td>
      <td>Mon Mar 26 22:45:06 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create plot
ax = bbc_df.plot.scatter(x="Tweet Ago", y='Compound', s = 200, color='LightBlue',alpha=0.70, edgecolor='Black',label='BBC',figsize=(10, 6))
cbs_df.plot.scatter(x="Tweet Ago", y='Compound',s = 200,color='Green', alpha=0.70,edgecolor='Black',label='CBS',ax=ax)
cnn_df.plot.scatter(x="Tweet Ago", y='Compound',s = 200,color='Red',alpha=0.70, edgecolor='Black',label='CNN', ax=ax)
fox_df.plot.scatter(x="Tweet Ago", y='Compound',s = 200,color='DarkBlue',alpha=0.70, edgecolor='Black',label='Fox', ax=ax)
nytimes_df.plot.scatter(x="Tweet Ago", y='Compound',s = 200,color='Yellow',alpha=0.70, edgecolor='Black',label='New York Times', ax=ax)

sns.set()

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.,title='Media Sources')
plt.title("Sentiment Analysis of Media Tweets (%s)" % (time.strftime("%x")))
plt.xlabel('Tweets Ago')
plt.ylabel('Tweet Polarity')

plt.savefig('SentimentAnalysisofMediaTweets.png',bbox_inches='tight')
plt.show()
```


![png](output_9_0.png)



```python
sentiment_summary= sentiment_df.groupby(['Target'], as_index=False).agg({'Compound': 'mean'})
sentiment_summary
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>Compound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBC</td>
      <td>0.111025</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@CBS</td>
      <td>0.432190</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@CNN</td>
      <td>-0.124475</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@FoxNews</td>
      <td>-0.189065</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@nytimes</td>
      <td>-0.020270</td>
    </tr>
  </tbody>
</table>
</div>




```python
names = ['BBC','CBS','CNN','Fox','NYT']
ax =sentiment_summary.plot(x='Target', y='Compound', kind='bar',alpha=0.5, linewidth =3, align="edge", width = 1.0,facecolor = 'red',edgecolor = 'black',figsize=(10,6))

x_axis = np.arange(len(names))
tick_locations = [value+0.2 for value in x_axis]

plt.xticks(tick_locations, names,rotation=0 )
plt.grid(linestyle='dotted')
plt.title("Overall Media Sentiment based on Twitter (%s)" % (time.strftime("%x")))
plt.ylabel("Tweet Polarity")


ax.patches[0].set_facecolor('lightblue')
ax.patches[1].set_facecolor('green')
ax.patches[2].set_facecolor('red')
ax.patches[3].set_facecolor('darkblue')
ax.patches[4].set_facecolor('yellow')

for p in ax.patches:
    ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.savefig('OverallSentimentonTwitter.png')
plt.show()
```


![png](output_11_0.png)



```python
sentiment_df.to_csv("SentimentAnalysisofMediaTweets.csv", sep=',', encoding='utf-8')
```
