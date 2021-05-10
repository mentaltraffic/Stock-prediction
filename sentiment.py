# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:30:45 2021

@author: HP

GENERATES PROBABILITY AND SCORE FOR EVERY TEXT FILE
"""


import os
import io
import flair
sentiment_model = flair.models.TextClassifier.load('en-sentiment')
mainfolder=os.getcwd()
D=mainfolder+"/dataset2/dataset"
os.chdir(D)
lof=os.listdir(os.getcwd())
for fi in lof:
     file=io.open(fi,encoding="utf-8")
     fr=file.read()
     # print(fr)
     sentence = flair.data.Sentence(fr)
     sentiment_model.predict(sentence)
     # print(sentence)
     probability = sentence.labels[0].score  # numerical value 0-1
     sentiment = sentence.labels[0].value  # 'POSITIVE' or 'NEGATIVE'
     #if(sentiment=='POSITIVE'):
     print(fi,probability,sentiment)
         #print(probability)
         #print(sentiment)
     # break


