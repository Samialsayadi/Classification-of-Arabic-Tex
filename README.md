# Classification of Arabic Text Using Singular Value Decomposition and Fuzzy C-Means Algorithms

The proposed system use to classify Arabic documents, comments on social media, and also classify the opinion about the product. and it validated by two common datasets (Alj-News 5 and CNN News).

The algorithm uses Singular Value Decompisiotn to reduce the high dimension and make the classification based semantic, then apply Fuzzy C-Means as Classifier Algorithm. 

The  proposed system consists of two main scripts:  `cleaning_arabic.py` and  `Arabic_Fuzzy_Cmean.py`. 

# Pre-Processing steps:
<ol>
<li> Get all the Arabic words.[here:](https://sourceforge.net/projects/arabic-wordlist)

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
```python
from numpy.linalg import svd
U,sigma,V = low_rank_approx(matrix,k=7)

projectedDocuments = np.dot(matrix.T, U)

 ```
Classification Method (Fuzzy C-Mean):

```python
 
def FCMcluster(vectors):
 
    """
    Does a simple Fuzzy  Cmeans clustering
#    """    
    model =Fuzzy(n_clusters=5, n_init=10 , max_iter=300,tol=0.000001)
    model.fit(vectors)
    return model.predict(vectors)
#    return kmeans2(vectors, k=len(vectors[0]))
documentClustering = FCMcluster(projectedDocuments)
 ```
 # download the full paper
You can download the full paper from. [here](https://link.springer.com/chapter/10.1007%2F978-981-15-3357-0_8).


# Dependencies

* [NLTK:](https://anaconda.org/anaconda/nltk) `conda install nltk` 
* [numpy:](https://anaconda.org/anaconda/numpy) `conda install -c anaconda numpy `.
* [scikit-cmeans:](https://pypi.org/project/scikit-cmeans) `pip install scikit-cmeans`.
