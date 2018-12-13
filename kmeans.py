import json
from pprint import pprint
import numpy as np
import pandas as pd 
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

cv = CountVectorizer(stop_words='english')
tf = TfidfVectorizer(stop_words='english')

corpus = []
sections = []
limit = 30000

with open('newyorktimes_filtered.jsonl') as f:
    for i, line in enumerate(f):
        if i > limit: break
        meta = json.loads(line)
        if meta['lead_paragraph']:
            text = meta['lead_paragraph']
            corpus.append(text)
            section = meta['section']
            sections.append(section)
            
'''
X = cv.fit_transform(corpus)
print(cv.get_feature_names())
print(X.toarray())
'''

matrix = tf.fit_transform(corpus)
idf_df = pd.DataFrame(matrix.toarray(), columns=tf.get_feature_names())
print(idf_df)

x_train, x_test, y_train, y_test = train_test_split(matrix, sections, test_size = .4, random_state= 42)

true_k = 5
clf = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
clf.fit(x_train, y_train)

print("Top terms per cluster:")
order_centroids = clf.cluster_centers_.argsort()[:, ::-1]
terms = tf.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

Y = tf.transform(["The stock market contains money and stocks about finance stuff.  Investing is fun if you don't lose money to bearish stocks."])
prediction = clf.predict(Y)
print(prediction)