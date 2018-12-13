import json
from pprint import pprint
import numpy as np
import pandas as pd 
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

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

matrix = tf.fit_transform(corpus)
idf_df = pd.DataFrame(matrix.toarray(), columns=tf.get_feature_names())
print(idf_df)

x_train, x_test, y_train, y_test = train_test_split(matrix, sections, test_size = .4, random_state= 42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

print("\n")
print("Prediction")

Y = tf.transform(["The stock market contains money and stocks about finance stuff.  Investing is fun if you don't lose money to bearish stocks."])
prediction = clf.predict(Y)
print(prediction)

#Calculating Accuracy Score
predicted = clf.predict(x_test)
print('Accuracy Score:')
print(accuracy_score(y_test,predicted))