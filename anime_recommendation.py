# =============== Data Loading ===============

# Import Library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading up the data
df = pd.read_csv('anime.csv')
df.head()
df.shape

# =============== Exploratory Data Analysis ===============

# Univariate EDA
type_count = df['type'].value_counts()
sns.barplot(x=type_count.values, y=type_count.index,
            palette='muted').set_title('Anime Types')

plt.tight_layout()
plt.show()

# Multivariate EDA
df[['name', 'rating', 'members', 'type']].sort_values(by='rating', ascending=False).query('members>500000')[:5]

# =============== Data Preparation ===============

# Missing Values
df.info()
df.isnull().sum()

# Missing values on rating feature
df[df['rating'].isnull()]

df['rating'] = df['rating'].astype(float)
df['rating'].fillna(df['rating'].median(),inplace = True)
df['rating'].isnull().any()

# Missing values on type feature
df.type.unique()
df[df['type'].isnull()]

df.loc[(df['name'] == "Steins;Gate 0"), 'type'] = 'TV'
df.loc[(df['name'] == "Steins;Gate 0"), 'episodes'] = '23'
df.loc[(df['name'] == "Violet Evergarden"), 'type'] = 'TV'
df.loc[(df['name'] == "Violet Evergarden"), 'episodes'] = '13'
df.loc[(df['name'] == "Code Geass: Fukkatsu no Lelouch"), 'type'] = 'TV'
df.loc[(df['name'] == "Code Geass: Fukkatsu no Lelouch"), 'episodes'] = '25'
df.loc[(df['name'] == "K: Seven Stories"), 'type'] = 'Movie'
df.loc[(df['name'] == "K: Seven Stories"), 'episodes'] = '6'
df.loc[(df['name'] == "Free! (Shinsaku)"), 'type'] = 'TV'
df.loc[(df['name'] == "Free! (Shinsaku)"), 'episodes'] = '12'
df.loc[(df['name'] == "Busou Shoujo Machiavellianism"), 'type'] = 'TV'
df.loc[(df['name'] == "Busou Shoujo Machiavellianism"), 'episodes'] = '12'
df.loc[(df['name'] == "Code:Realize: Sousei no Himegimi"), 'type'] = 'TV'
df.loc[(df['name'] == "Code:Realize: Sousei no Himegimi"), 'episodes'] = '12'
df.loc[(df['name'] == "Flying Babies"), 'type'] = 'TV'
df.loc[(df['name'] == "Flying Babies"), 'episodes'] = '12'
df.loc[(df['name'] == "Gamers!"), 'type'] = 'TV'
df.loc[(df['name'] == "Gamers!"), 'episodes'] = '12'
df.loc[(df['name'] == "Ganko-chan"), 'type'] = 'TV'
df.loc[(df['name'] == "Ganko-chan"), 'episodes'] = '10'
df.loc[(df['name'] == "Gekidol"), 'type'] = 'TV'
df.loc[(df['name'] == "Gekidol"), 'episodes'] = '12'
df.loc[(df['name'] == "Ginga Eiyuu Densetsu (2017)"), 'type'] = 'OVA'
df.loc[(df['name'] == "Ginga Eiyuu Densetsu (2017)"), 'episodes'] = '110'
df.loc[(df['name'] == "Grancrest Senki"), 'type'] = 'TV'
df.loc[(df['name'] == "Grancrest Senki"), 'episodes'] = '24'
df.loc[(df['name'] == "IDOLiSH7"), 'type'] = 'TV'
df.loc[(df['name'] == "IDOLiSH7"), 'episodes'] = '17'
df.loc[(df['name'] == "Isekai Shokudou"), 'type'] = 'TV'
df.loc[(df['name'] == "Isekai Shokudou"), 'episodes'] = '12'
df.loc[(df['name'] == "Oushitsu Kyoushi Haine"), 'type'] = 'TV'
df.loc[(df['name'] == "Oushitsu Kyoushi Haine"), 'episodes'] = '12'
df.loc[(df['name'] == "Peace Maker Kurogane (Shinsaku)"), 'type'] = 'TV'
df.loc[(df['name'] == "Peace Maker Kurogane (Shinsaku)"), 'episodes'] = '24'
df.loc[(df['name'] == "Seikaisuru Kado"), 'type'] = 'TV'
df.loc[(df['name'] == "Seikaisuru Kado"), 'episodes'] = '12'
df.loc[(df['name'] == "UQ Holder!"), 'type'] = 'TV'
df.loc[(df['name'] == "UQ Holder!"), 'episodes'] = '12'
df.loc[(df['name'] == "Citrus"), 'type'] = 'TV'
df.loc[(df['name'] == "Citrus"), 'episodes'] = '12'
df.loc[(df['name'] == "Hitorijime My Hero"), 'type'] = 'TV'
df.loc[(df['name'] == "Hitorijime My Hero"), 'episodes'] = '12'

df.isnull().sum()
df.dropna(subset=['type'], inplace=True)
df['type'].isnull().any()

# Missing values on genre feature

df[df['genre'].isnull()]

df['genre'].fillna('Unknown', inplace=True)
df['genre'].isnull().any()
df.isnull().sum()

all_genres = defaultdict(int)
for genres in df['genre']:
  for genre in genres.split(','):
    all_genres[genre.strip()] += 1

genres_cloud = WordCloud(width=800, height=400, background_color='white',
                        colormap='gnuplot').generate_from_frequencies(all_genres)

plt.imshow(genres_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Text Cleaning
def text_cleaning(text):
  text = re.sub(r'&quot;', '', text)
  text = re.sub(r'.hack//', '', text)
  text = re.sub(r'&#039;', '', text)
  text = re.sub(r'A&#039;s', '', text)
  text = re.sub(r'I&#039;', 'I\'', text)
  text = re.sub(r'&amp;', 'and', text)
  return text

df['name'] = df['name'].apply(text_cleaning)

# =============== Modelling and Result ===============

# Content Based Filtering

# TF-IDF Vectorizer
tf = TfidfVectorizer()
tf.fit(df['genre']) 
tf.get_feature_names()

tfidf_matrix = tf.fit_transform(df['genre'])
tfidf_matrix.shape

tfidf_matrix.todense()

pd.DataFrame(tfidf_matrix.todense(), columns=tf.get_feature_names(),
            index=df['name']).sample(48, axis=1).sample(10, axis=0)

# Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix) 
cosine_sim

cosine_sim_df = pd.DataFrame(cosine_sim, index=df['name'], columns=df['name'])
print('Shape:', cosine_sim_df.shape)

cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

# Get Recommendations
def genre_recommendations(anime_name, similarity_data=cosine_sim_df,
                          items=df[['name', 'genre', 'type', 'rating']], k=5):
  index = similarity_data.loc[:,anime_name].to_numpy().argpartition(range(-1, -k, -1))
  closest = similarity_data.columns[index[-1:-(k+2):-1]]
  closest = closest.drop(anime_name, errors='ignore')
  return pd.DataFrame(closest).merge(items).head(k)

# Result
df.loc[df['name']=="Kimi no Na wa."]
genre_recommendations('Kimi no Na wa.')

# =============== Evaluation ===============

# Precision
def precision(anime_name, similarity_data=cosine_sim_df, k=5):
  act_set = set(similarity_data.loc[:,anime_name])
  pred_set = set(similarity_data.loc[:,anime_name][:k])
  result = len(act_set & pred_set) / float(k)
  return result

precision('Kimi no Na wa.')