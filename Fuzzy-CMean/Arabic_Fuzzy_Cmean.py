#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:02:45 2019

@author: samialsayadi
"""

'''
    A simple topic model using singular value decomposition
    applied to a corpus of TAC-2011.
'''
import json
import csv

import numpy as np
import pandas as pd
import sys
from collections import Counter
from numpy.linalg import svd
from shutil import copyfile
from skcmeans.algorithms import Probabilistic
from skcmeans.algorithms import GustafsonKesselMixin
from skcmeans.algorithms import Fuzzy
from skcmeans.algorithms import CMeans
from scipy.cluster.vq import kmeans2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics

np.set_printoptions(threshold=sys.maxsize)

def low_rank_approx(matrix, k=6):
    """
    Computes an k-rank approximation of a matrix
    """
    U,sigma,V= np.linalg.svd(matrix, full_matrices=False)
    Ar = np.zeros((len(U), len(V)))
    for i in range(k):
        Ar += sigma[i] * np.outer(U.T[i], V[i])

    return U[:,:k],Ar, V[:k,:]

def normalize(matrix):
    '''
        Normalize a document term matrix according to a local
        and global normalization factor. For this we chose a
        simple logarithmic local normalization with a global
        normalization based on entropy.
    '''
    numWords, numDocs = matrix.shape
    localFactors = np.log(np.ones(matrix.shape) + matrix.copy())    
    '''
    localFactors tfij is local weigth for term i to document j 
    in this phase we calculate term weight base on document     
    '''    
    probabilities = matrix.copy()    
    rowSums = np.sum(matrix, axis=1)
    # divide each column items by the row sums
    assert all(x > 0 for x in rowSums)
    probabilities = (probabilities.T / rowSums).T
#    np.savetxt('ID.csv', probabilities, fmt="%d", delimiter=",")
    '''
    golbalfactors Gij is global weigth for term i to documents N j 
    in this phase we calculate term weight base on corpus 
    
    '''
    entropies = (probabilities * np.ma.log(probabilities).filled(0) /
                 np.log(numDocs))
    # matrix is -1 
#    s=np.ones(numWords)+np.sum(entropies, axis=1)
    globalFactors = np.ones(numWords) + np.sum(entropies, axis=1)
#    np.savetxt('globalFactors.csv', globalFactors, fmt="%10.5f", delimiter=",")
    # multiply each column by the global factors for the rows
    normalizedMatrix = (localFactors.T * globalFactors).T  
    
    return normalizedMatrix

def makeDocumentTermMatrix(data):
    
    '''    
        Return the document-term matrix for the given list of stories.
        stories is a list of dictionaries {string: string|[string]}
        of the form
        
            {
                'filename': string
                'words': [string]
                'text': string
            }
            
        The list of words include repetition, and the output document-
        term matrix contains as entry [i,j] the count of word i in story j
    
    '''
    
    words = allWords(data)
    wordToIndex = dict((word, i) for i, word in enumerate(words))
    indexToWord = dict(enumerate(words))
    indexToDocument = dict(enumerate(data))
    matrix = np.zeros((len(words), len(data)))

    
    """
    (matrix.shape) (rownumber,colnumber)
    """
    '''
    size of matrix in TAC-2011 is length of words is 7019 
    and the length of documents is 100    
    '''
    for docID, document in enumerate(data):
        docWords = Counter(document['words'])
        #repeate of words in each doc
        for word, count in docWords.items():
            # count is repeat no of words 
            matrix[wordToIndex[word], docID] = count

    return matrix, (indexToWord, indexToDocument)


def FCMcluster(vectors):
    
    """
    Does a simple Fuzzy  Cmeans clustering
#    """    
    model =Fuzzy(n_clusters=5, n_init=10 , max_iter=300,tol=0.000001)
    model.fit(vectors)
    return model.predict(vectors)
#    return kmeans2(vectors, k=len(vectors[0]))
def allWords(data):
    words = set()
    for entry in data:
        words |= set(entry['words'])
    return list(sorted(words))
def load():
    with open('Aji-News-corpus.json', 'r') as infile:
        data = json.loads(infile.read())
    return data
data = load()
matrix, (indexToWord, indexToDocument) = makeDocumentTermMatrix(data)
matrix = normalize(matrix)
print ('Classifying the TAC data into 10 parts')
U,sigma,V = low_rank_approx(matrix,k=7)
#np.savetxt('globalFactors.csv', V, fmt="%10.5f", delimiter=",")

projectedDocuments = np.dot(matrix.T, U)
#np.savetxt('globalFactors.csv', projectedDocuments, fmt="%10.5f", delimiter=",")
documentClustering = FCMcluster(projectedDocuments)
documentClusters = [
    [indexToDocument[i]['filename']
     for (i, x) in enumerate(documentClustering) if x == j]
    for j in range(len(set(documentClustering)))
]
    
clusterNumber=1
for clusters in documentClusters:
    for clusteredDocumentName in clusters:
        srcfile='Aji-News-corpus/'+clusteredDocumentName
        dstfile='clusteredDocuments/'+str(clusterNumber)+'/'+clusteredDocumentName
        copyfile(srcfile,dstfile)
    clusterNumber+=1
