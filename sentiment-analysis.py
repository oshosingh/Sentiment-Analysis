
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#data = pd.read_csv('../input/train.tsv',delimiter='\t')


# In[ ]:


# sentiment = data['Sentiment']
# feature = data['Phrase']


# ### Libraries for processing text data

# In[2]:


import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemma = WordNetLemmatizer()


def preprop(text):
    text = text.lower()
    text = re.sub('[^a-z]',' ',text)
    text = text.split()
    text = [lemma.lemmatize(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)


# ### Vectoriser function

# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,3),max_features=6000,stop_words='english')

clean_feature=[]


# ### Train Function

# In[ ]:


# def train(feature,sentiment,classifier):
#     for i in feature:
#         clean_feature.append(preprop(i))
#     feat_vector = tfidf.fit_transform(clean_feature).toarray()
#     classifier.fit(feat_vector,sentiment)
#     return classifier,feat_vector


# In[ ]:


# from sklearn.naive_bayes import GaussianNB
# nb = GaussianNB()

# classifier,feat_vector = train(feature,sentiment,nb) # Training the claasifier


import pickle


# In[ ]:


# classifiers = 'hello/sentiment_analyser.sav'
# tfidfs = 'hello/tfidf.sav'
# pickle.dump(classifier, open(classifiers, 'wb'))
# pickle.dump(tfidf, open(tfidfs, 'wb'))


# In[6]:


def preprop_for_inference(text,tfidf):
    lis = []
    text = preprop(text)
    lis.append(text)
    text_feat_vector = tfidf.transform(lis).toarray()
    return text_feat_vector


# In[7]:


def load_model_for_inference():
    sentiment_model = pickle.load(open('sentiment_analyser.sav','rb'))
    tfidf = pickle.load(open('tfidf.sav','rb'))
    return sentiment_model,tfidf


# In[8]:


def inference():
    print("Enter sentiment : ")
    s = input()
    sentiment_model,tfidf = load_model_for_inference()
    s = preprop_for_inference(s,tfidf)
    predict = sentiment_model.predict(s)[0]
    if predict==0 or predict==1:
        print('Sentiment Predicted by model is : Negative')
    else:
        print('Sentiment Predicted by model is : Positive')


# In[19]:


inference()

