#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:28:11 2019

@author: samialsayadi
"""

import json
import os
import codecs
import glob
import pickle
import chardet

from nltk.corpus import stopwords,wordnet
from nltk import pos_tag,word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.stem.isri import ISRIStemmer

def words():
    allWords = None
    with open('arabic-wordlist.txt', 'r') as infile:
        allWords = [line.strip() for line in infile]

    return set(allWords)


# Extract a list of tokens from a cleaned string.
def tokenize(s):
    stopWords = set(stopwords.words('arabic'))
    wordsToKeep = words() - stopWords

    return [x for x in word_tokenize(s)
            if x in wordsToKeep and len(x) >= 3]
    
def wordnetPos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
documentDict=dict()
for filename in os.listdir('Aji-News-corpus'):
    if filename[-3:] == 'txt':
        with open(os.path.join('Aji-News-corpus',filename),'r',encoding="utf-16-le") as infile:
            documentDict[filename]=infile.read()

print ("Cleaning....")
documents=[]
for filename,docutext in documentDict.items():
    tokens=tokenize(docutext)
    tagged_tokens=pos_tag(tokens)
    lemma=WordNetLemmatizer()
    stemmedTokens = [lemma.lemmatize(word, wordnetPos(tag))
                     for word, tag in tagged_tokens]
    documents.append({
        'filename': filename,
        'text': docutext,
        'words': stemmedTokens,
    })
with open( 'Aji-News-corpus.json', 'w') as outfile:

    outfile.write(json.dumps(documents,ensure_ascii=False))
print ('Cleaning is done!')
