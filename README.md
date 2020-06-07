# Classification of Arabic Text Using Singular Value Decomposition and Fuzzy C-Means Algorithms

The proposed system has generated Arabic Classification and it may use to classify Arabic documents, comments on social media, and also classify the opinion about the product. 

The algorithm uses Singular Value Decompisiotn to reduce high dimension and execute the semantic classification, then apply Fuzzy C-Means as Classifier Algorithm. 

The  proposed system consists of two main scripts:  `cleaning_arabic.py` and  `Arabic_Fuzzy_Cmean.py`. 

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

You can download the full paper from. [here](https://link.springer.com/chapter/10.1007%2F978-981-15-3357-0_8).


# Dependencies

* [numpy:](https://anaconda.org/anaconda/numpy) `conda install -c anaconda numpy `.
