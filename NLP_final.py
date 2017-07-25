# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 20:28:26 2017

@author: Shaurya Rawat
"""
import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import argparse
import os
import json
from collections import Counter
from nltk.corpus import stopwords
from nltk import bigrams
from nltk.tokenize import TweetTokenizer
import string
from collections import defaultdict
import operator
import math
from datetime import datetime
import numpy as np
from textblob import TextBlob
import re
import vincent

# from text_preprocessor import preprocess
import opinion_lexicon
consumer_key='cG14rhsMIVvWiAOWduGLtpgio'
consumer_secret='Sj9rQQm2NXtiVPSXzgrSyvOYD9ZFe8hI6p3bqlV5YzjfT92pgX'
access_token='879316971761934336-OuB2uOazQpIG310F4K2Ka7dAbS56edK'
access_secret='rnWduE1zZgukOzrqsj7rbdAiWczf8eG6RS3TRgSwjCEoS'

def get_parser():
       parser=argparse.ArgumentParser(description="Twitter Downloader")
       parser.add_argument("-q",
                           "--query",
                           dest="query",
                           help="Query/Filter",
                           default='-')
       parser.add_argument("-d",
                           "--data-dir",
                           dest="data_dir",
                           help="Output/Data Directory")
       return parser

class MyListener(StreamListener):
       def __init__(self,data_dir,query):
              query_fname=format_filename(query)
              self.outfile="%s/ stream_%s.json"%(data_dir,query_fname)

       def on_data(self,data):
              try:
                     with open(self.outfile,'a') as f:
                            f.write(data)
                            print(data)
                            return True
              except BaseException as e:
                     print("Error on_data: %s"% str(e))
                     time.sleep(5)
              return True

       def on_error(self,status):
              print(status)
              return True

def format_filename(fname):
       return ''.join(convert_valid(one_char) for one_char in fname)

def convert_valid(one_char):
       valid_chars="-_.%s%s"% (string.ascii_letters,string.digits)
       if one_char in valid_chars:
              return one_char
       else:
              return '_'

@classmethod
def parse(cls,api,raw):
       status=cls.first_parse(api,raw)
       setattr(status,'json',json.dumps(raw))
       return status



if __name__=='__main__':
       parser=get_parser()
       args=parser.parse_args()
       auth=OAuthHandler(consumer_key,consumer_secret)
       auth.set_access_token(access_token,access_secret)
       api=tweepy.API(auth)

       twitter_stream=Stream(auth,MyListener(args.data_dir,args.query))
       twitter_stream.filter(track=[args.query])

tokenizer=TweetTokenizer()
def tokenize(tweet):
       return tokenizer.tokenize(tweet)

def preprocess(tweet,lowercase=False):
       tokens=tokenize(tweet)
       if lowercase:
              tokens=[token.lower() for token in tokens if not token.startswith('http')]
       return tokens

fpath='D:/IE MBD 2016/NLP/stream_trump.json'

punctuation=list(string.punctuation)
ignore=['rt','via','RT','_','...','..','"']
stop=stopwords.words('english')+punctuation+ignore

def terms_single(tweet,lower=True):
       return set(terms_all(tweet,lower=True))

def terms_hash(tweet,lower=True):
       return [term for term in preprocess(tweet.get('text',''),lowercase=lower) if term.startswith('#')]

def terms_only(tweet,lower=True):
       tokenised_tweet=preprocess(tweet.get('text',''),lowercase=lower)
       return [term for term in tokenised_tweet if term not in stop and not term.startswith(('#','@'))]

def terms_all(tweet,lower=True):
       tokenised_tweet=preprocess(tweet.get('text',''),lowercase=lower)
       return [term for term in tokenised_tweet if term not in stop]

def terms_bigrams(tweet):
       return list(bigrams(terms_all(tweet,lower=True)))

def extract_frequencies(filtering_method,filepath,bigram=False):
       term_frequency=Counter()
       number_of_tweets=0
       with open(fpath,'r') as f:

              for line in f:
                  number_of_tweets +=1
                  tweet= json.loads(line)
                  terms=filtering_method(tweet)
                  term_frequency.update(terms)
       return (term_frequency,number_of_tweets)

def __extend_coocurrences(matrix,terms_list,bigrams_list=[]):
       if bigrams_list:
              for i in range(len(bigrams_list)-1):
                     for j in range(i+1,len(terms_list)):
                            bigram,term=[bigrams_list[i],terms_list[j]]
                            matrix[bigram][term] +=1
       else:
              for i in range(len(terms_list)-1):
                     for j in range(i+1,len(terms_list)):
                            term1,term2=[terms_list[i],terms_list[j]]
                            matrix[term1][term2]+=1
       return matrix

def build_coocurrences(filepath,bigram=False):
       term_frequency=defaultdict(lambda:defaultdict(int))
       with open(fpath,'r') as f:
              for line in f:
                     tweet=json.loads(line)
                     terms=terms_only(tweet)
                     bigrams_list=[]
                     if bigram:
                            bigrams_list=terms_bigrams(tweet)
                     term_frequency=__extend_coocurrences(term_frequency,terms,bigrams_list)
       return term_frequency

vocab=opinion_lexicon.vocab

def compute_probabilities(frequencies,cooccur_matrix,number_of_tweets):
       prob_term={}
       prob_cooccur=defaultdict(lambda: defaultdict(int))

       for term1,occurances in frequencies.items():
              prob_term[term1]=float(occurances)/float(number_of_tweets)
              for term2 in cooccur_matrix[term1]:
                     prob_cooccur[term1][term2]=float(cooccur_matrix[term1][term2])/float(number_of_tweets)
       return (prob_term,prob_cooccur)

def compute_pmi(cooccur_matrix,prob_cooccur,prob_single,prob_bigram={}):
       pmi=defaultdict(lambda: defaultdict(int))
       for term1 in prob_bigram or prob_single:
              for term2 in cooccur_matrix[term1]:
                     if prob_bigram:
                            denom=prob_bigram[term1]*prob_single[term2]
                     else:
                            denom=prob_single[term1]*prob_single[term2]
#                     try:
                     pmi[term1][term2]=math.log(float(prob_cooccur[term1][term2])/float(denom),2)
#                     except ZeroDivisionError:
#                            pmi[term1][term2]= 0
       return pmi
   

def compute_semantic_orientation(prob_term,pmi,vocab):
       semantic_orientation={}
       for term,_ in prob_term.items():
              positive_assoc=sum(pmi[term][positive_word] for positive_word in vocab['positive_vocab'])
              negative_assoc=sum(pmi[term][negative_word] for negative_word in vocab['negative_vocab'])
              semantic_orientation[term]=positive_assoc-negative_assoc
       return semantic_orientation

#filepath='D:/IE MBD 2016/NLP/sentiment_log.txt'

#with open(filepath,"a") as f:
print(str(datetime.now())+'\n\n')

print('Extracting features from tweets..')
common_bigrams=extract_frequencies(terms_bigrams,'D:/IE MBD 2016/NLP/stream_trump.json')[0].most_common(20)
common_terms=extract_frequencies(terms_only,'D:/IE MBD 2016/NLP/stream_trump.json')[0].most_common(20)

print('Calculating frequency of single terms..')
frequency_single,number_of_tweets=extract_frequencies(terms_single,'D:/IE MBD 2016/NLP/stream_trump.json')

print('Calculating frequency of bigrams..')
frequency_bigrams,number_of_bigrams=extract_frequencies(terms_bigrams,'D:/IE MBD 2016/NLP/stream_trump.json',bigram=True)

print('Calculating frequency of co-occurence matrix for single terms..')
cooccur_single=build_coocurrences('D:/IE MBD 2016/NLP/stream_trump.json')

print('Calculating frequency of co-occurrence matrix for bigrams..')
cooccur_bigrams=build_coocurrences('D:/IE MBD 2016/NLP/stream_trump.json',bigram=True)

print('Calculating probabilities for single terms..')
prob_single,prob_cooccur_single=compute_probabilities(frequency_single,cooccur_single,number_of_tweets)

print('Calculating probabilities for bigrams..')
prob_bigrams,prob_cooccur_bigrams=compute_probabilities(frequency_bigrams,cooccur_bigrams,number_of_tweets)

print('Calculating pointwise mutual information for single terms..')
single_pmi=compute_pmi(cooccur_single,prob_cooccur_single,prob_single)

print('Calculating pointwise mutual information for bigrams..')
bigrams_pmi=compute_pmi(cooccur_bigrams,prob_cooccur_bigrams,prob_single,prob_bigrams)

print('Calculating semantic orientation for single terms..')
semantic_orientation_single=compute_semantic_orientation(prob_single,single_pmi,vocab)

print('Calculating semantic orientation for bigrams..')
semantic_orientation_bigrams=compute_semantic_orientation(prob_bigrams,bigrams_pmi,vocab)

print('Sorting semantic orientation for single terms..')
semantic_sorted_single=sorted(semantic_orientation_single.items(),key=operator.itemgetter(1),reverse=True)

print('Sorting semantic orientation for bigrams..')
semantic_sorted_bigrams=sorted(semantic_orientation_bigrams.items(),key=operator.itemgetter(1),reverse=True)

print('Extracting most positive single terms..')
top_pos_single=[b for b in semantic_sorted_single if b[0] not in opinion_lexicon.vocab['positive_vocab'] and b[0] not in opinion_lexicon.vocab['negative_vocab']][:20]

print('Extracting most negative single terms..')
top_neg_single=[b for b in semantic_sorted_single if b[0] not in opinion_lexicon.vocab['positive_vocab'] and b[0] not in opinion_lexicon.vocab['negative_vocab']][:-21:-1]

print('Extracting most positive bigrams...')
top_pos_bigrams = [b for b in semantic_sorted_bigrams
                   if b[0][0] not in opinion_lexicon.vocab['positive_vocab']
                   and b[0][1] not in opinion_lexicon.vocab['positive_vocab']
                   and b[0][0] not in opinion_lexicon.vocab['negative_vocab']
                   and b[0][1] not in opinion_lexicon.vocab['negative_vocab']
                   ][:20]

print('Extracting most negative bigrams...')
top_neg_bigrams = [b for b in semantic_sorted_bigrams
                   if b[0][0] not in opinion_lexicon.vocab['positive_vocab']
                   and b[0][1] not in opinion_lexicon.vocab['positive_vocab']
                   and b[0][0] not in opinion_lexicon.vocab['negative_vocab']
                   and b[0][1] not in opinion_lexicon.vocab['negative_vocab']
                   ][:-21:-1]

#print('Writing to {0}..'.format(filepath))

print('Number of tweets analysed: {0}\n\n'.format(number_of_tweets).encode('utf-8'))

print('\n')

print('Most positive single terms\n')
print (top_pos_single)
print('\n')

print('Most negative single terms\n')
print(top_neg_single)
print('\n')

print('Most positive bigrams\n')
print(top_pos_bigrams)
print('\n')

print('Most negative bigrams\n')
print(top_neg_bigrams)
print('\n')

print('Most common single terms\n')
print(common_terms)
print('\n')

print('Most common bigrams\n')
print(common_bigrams)
print('\n**********\n\n')



############################ Get Sentiment Percentages for a given Query########################
class TwitterAnalysis(object):

    def __init__(self):

        # Twitter app token and keys
        consumer_key         = "cG14rhsMIVvWiAOWduGLtpgio"
        consumer_secret     = "Sj9rQQm2NXtiVPSXzgrSyvOYD9ZFe8hI6p3bqlV5YzjfT92pgX"
        access_token         = "879316971761934336-OuB2uOazQpIG310F4K2Ka7dAbS56edK"
        access_token_secret     = "rnWduE1zZgukOzrqsj7rbdAiWczf8eG6RS3TRgSwjCEoS"

        try:
            self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tweepy.API(self.auth)
        except:
            print("Error authenticating app.")


    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):

        tweet_analysis = TextBlob(self.clean_tweet(tweet))

        if tweet_analysis.sentiment.polarity > 0:
            return 'positive'
        elif tweet_analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def fetch_tweets(self, query, count=10):

        tweets = []

        try:
            fetched_tweets = self.api.search(q=query, count=count)

            for tweet in fetched_tweets:
                parsed_tweet = {}
                parsed_tweet["text"] = tweet.text
                parsed_tweet["sentiment"] = self.get_tweet_sentiment(parsed_tweet["text"])

                if tweet.retweet_count > 0:
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
            return tweets
        except tweepy.TweepError as e:
            print("Error: " + str(e))


def main():

    twitterApi = TwitterAnalysis()
    get_tweets = twitterApi.fetch_tweets(query="Trump", count=200)
    
    print (str(datetime.now())+'\n\n')
    positive_tweets = [tweet for tweet in get_tweets if tweet['sentiment'] == 'positive']
    print("Positive tweets percentage: {} %".format(100*len(positive_tweets)/len(get_tweets)))

    negative_tweets = [tweet for tweet in get_tweets if tweet['sentiment'] == 'negative']
    print("Negative tweets percentage: {} %".format(100*len(negative_tweets)/len(get_tweets)))

    print("Neutral tweets percentage: {} %".format(100*(len(get_tweets) - len(negative_tweets) - len(positive_tweets))/len(get_tweets)))


if __name__ == "__main__":
    main()


######################### Bar chart visualization using Vincent with D3.js in backend ##############
def prepare_bar_chart_data(filter_method,filepath):
    word_freq=extract_frequencies(filter_method,filepath)[0].most_common(20)
    labels,freq=zip(*word_freq)
    data={'data':freq,'x':labels}
    bar=vincent.Bar(data,iter_idx='x')
    bar.to_json('term_freq.json',html_out=True,html_path='chart.html')
    
prepare_bar_chart_data(terms_only,fpath)
prepare_bar_chart_data(terms_hash,fpath)
prepare_bar_chart_data(terms_bigrams,fpath)

##############################################################################




from nltk.tokenize import word_tokenize
import nltk
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

documents = []

for r in short_pos.split('\n'):
    documents.append( (r, "pos") )

for r in short_neg.split('\n'):
    documents.append( (r, "neg") )


all_words = []

short_pos_words = word_tokenize(short_pos.decode('latin-1'))
short_neg_words = word_tokenize(short_neg.decode('latin-1'))

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document.decode('latin-1'))
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set =  featuresets[10000:]

################ INITIAL RUN####################################
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


voted_classifier = VoteClassifier(
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)



############### SAVING CLASSIFIERS TO PICKLE TO IMPROVE RUNTIME SPEED ###################################
all_words = []
documents = []
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p.decode('latin-1'))
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
            
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p.decode('latin-1'))
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]


save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document.decode('latin-1'))
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[10000:]
training_set = featuresets[:10000]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)

save_classifier = open("pickled_algos/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()

########################### Machine Learning Sentiment Analysis ############################
documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()




word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document.decode('latin-1'))
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features



featuresets_f =[(find_features(rev), category) for (rev, category) in documents]


random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[10000:]
training_set = featuresets[:10000]



open_file = open("pickled_algos/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()



open_file = open("pickled_algos/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()




voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)




def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

################################################################################################

## Test the sentiment module
print(sentiment("I really love NLP. It is a very interesting and exciting field."))
print(sentiment("I hate the fact that the course is going to end. Still much to learn."))

################################################################################################
import pandas as pd
testdata=pd.read_csv("D:/IE MBD 2016/NLP/testData.tsv",sep='\t')
testdata.head()
testdata=testdata[:5000]
sent=[]
for i in range(len(testdata)):
    _=sentiment(testdata.review[i])
    sent.append(_)

import matplotlib.pyplot as plt   
import matplotlib.animation as animation
from matplotlib import style
import time
style.use("ggplot")

fig=plt.figure()
ax1=fig.add_subplot(1,1,1)

def animate(i):
    xar=[]
    yar=[]
    x=0
    y=0
    for l in sent:
        x+=1
        if "pos" in l:
            y+=1
        elif "neg" in l:
            y-=1
        xar.append(x)
        yar.append(y)
    ax1.clear()
    ax1.plot(xar,yar)
ani=animation.FuncAnimation(fig,animate,interval=100)
plt.show()

###################################################################################################























































