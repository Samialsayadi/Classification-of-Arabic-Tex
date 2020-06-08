# Classification of Arabic Text Using Singular Value Decomposition and Fuzzy C-Means Algorithms [here](https://link.springer.com/chapter/10.1007%2F978-981-15-3357-0_8).

The proposed system uses to classify Arabic documents, comments on social media, and also classify the opinion about the product. and it validated by two common datasets (Alj-News 5 and CNN News).

The algorithms use Singular Value Decompisiotn to reduce the high dimension and make the classification based semantic, then apply Fuzzy C-Means as Classifier Algorithm. 

The  proposed system consists of two main scripts:  `cleaning_arabic.py` and  `Arabic_Fuzzy_Cmean.py`. 

# Pre-Processing steps:
<ol>
<li> Get all the Arabic words [here:](https://sourceforge.net/projects/arabic-wordlist).

```python
def words():
allWords = None
with open('arabic-wordlist.txt', 'r') as infile:
    allWords = [line.strip() for line in infile]

return set(allWords)
```
<li> Tokenization and Removal of stopwords by using the NLTK library.

```python
# Extract a list of tokens from a cleaned string.
def tokenize(token):
    stopWords = set(stopwords.words('arabic'))
    wordsToKeep = words() - stopWords

    return [x for x in word_tokenize(token)
            if x in wordsToKeep and len(x) >= 3]
```
<li> Create Json file from:  'filename', 'text', and 'words'.

```python
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
```
</ol>

Decomposition Method(SVD):
> before applied the SVD algorithm, you must computes an k-rank approximation of a matrix.
```python
from numpy.linalg import svd

def low_rank_approx(matrix, k=6):
    U,sigma,V= np.linalg.svd(matrix, full_matrices=False)
    Ar = np.zeros((len(U), len(V)))
    for i in range(k):
        Ar += sigma[i] * np.outer(U.T[i], V[i])

    return U[:,:k],Ar, V[:k,:]
```
> Then, you can apply Decomposition Method(SVD) as follow by using left-singular vectors.

```python
from numpy.linalg import svd
U,sigma,V = low_rank_approx(matrix,k=7)

projectedDocuments = np.dot(matrix.T, U)

 ```
 
Document Term Matrix:
>Return the document-term matrix for the given list of stories. stories is a list of dictionaries {string: string|[string]} of the form
>            {
>                'filename': string
>                 'words': [string]
>                 'text': string
>             }
>             The list of words include repetition, and the output document-term matrix contains as entry [i,j] the count of word i in story j.
 ```python
 
 def makeDocumentTermMatrix(data):
 
 words = allWords(data)
 wordToIndex = dict((word, i) for i, word in enumerate(words))
 indexToWord = dict(enumerate(words))
 indexToDocument = dict(enumerate(data))
 matrix = np.zeros((len(words), len(data)))
 
 for docID, document in enumerate(data):
     docWords = Counter(document['words'])
     #repeate of words in each doc
     for word, count in docWords.items():
         # count is repeat no of words 
         matrix[wordToIndex[word], docID] = count
         
 return matrix, (indexToWord, indexToDocument)
 
 ```
 Normalize a Document-term Matrix:
 
>  Normalize a document term matrix according to a local and global normalization factor. 
> For this we chose a simple logarithmic local normalization with a global normalization based on entropy.
 
 ```python
 
 def normalize(matrix):
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
     '''
     golbalfactors Gij is global weigth for term i to documents N j 
     in this phase we calculate term weight base on corpus 
 
     '''
     entropies = (probabilities * np.ma.log(probabilities).filled(0) /
                  np.log(numDocs))
     # matrix is -1 
     globalFactors = np.ones(numWords) + np.sum(entropies, axis=1)
     # multiply each column by the global factors for the rows
     normalizedMatrix = (localFactors.T * globalFactors).T  
     
     return normalizedMatrix
 ```
 
Classification Method (Fuzzy C-Mean):


```python

from skcmeans.algorithms import Fuzzy

def FCMcluster(vectors):
 
    """
    Does a simple Fuzzy  Cmeans clustering
#    """    
    model =Fuzzy(n_clusters=5, n_init=10 , max_iter=300,tol=0.000001)
    model.fit(vectors)
    return model.predict(vectors)
documentClustering = FCMcluster(projectedDocuments)
 ```
 # download the full paper
You can download the full paper from. [here](https://link.springer.com/chapter/10.1007%2F978-981-15-3357-0_8).


# Dependencies

* [NLTK:](https://anaconda.org/anaconda/nltk) `conda install nltk` 
* [numpy:](https://anaconda.org/anaconda/numpy) `conda install -c anaconda numpy `.
* [scikit-cmeans:](https://pypi.org/project/scikit-cmeans) `pip install scikit-cmeans`.
