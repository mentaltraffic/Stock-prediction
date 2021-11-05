# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 23:39:52 2021

@author: monal

REMOVES  PUNCTUATION AND SPECIAL CHARACTERS
LEMMATIZES TEXT
"""


     
    
            
def remove_punctuation_special_chars(sentence):
 sentence = nlp(sentence)
 processed_sentence = ' '.join([token.text for token in sentence 
  if token.is_punct != True and 
     token.is_quote != True and 
     token.is_bracket != True and 
     token.is_currency != True and 
     token.is_digit != True])
 return processed_sentence  

def lemmatize_text(sentence):
    sentence = nlp(sentence)
    processed_sentence = ' '.join([word.lemma_ for word in 
    sentence])
    processed_sentence = processed_sentence.replace("â€ ™ ", "")
    processed_sentence = processed_sentence.replace("\n", "")
    processed_sentence = processed_sentence.replace(" 's", "")
     
    return processed_sentence

import io
import os  
import spacy
from spacy.lang.en import English
nlp = spacy.load("en_core_web_sm") 

#read input textfiles
mainfolder=os.getcwd()
D=mainfolder+"/dataset2/dataset"
os.chdir(D)
lof=os.listdir(os.getcwd())
for fi in lof:
     file=io.open(fi,encoding="utf-8")
     file1=file.read()
     open(fi, 'w').close()          #erases text data
     doc = nlp(file1)   
     sentences = list(doc.sents)    
     for sentence in sentences:
        sentence = remove_punctuation_special_chars(sentence.text)
        sentence = lemmatize_text(sentence)
        if len(sentence) >= 2:
            f = open(fi, "a",encoding="utf-8")
            f.write(sentence)               #rewrites text data
        #print(sentence)
     
