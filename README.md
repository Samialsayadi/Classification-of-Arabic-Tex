# Classification of Arabic Text Using Singular Value Decomposition and Fuzzy C-Means Algorithms

The proposed system have generated Arabic Classification and it may use to classify arabic documents, comments on social madie, and also classify the opionion about product. 

The algorithm uses Singular Value Decompisiotn to reduce high dimention and execute the smantic classification, then apply Fuzzy C-Means as Classifier Algorithm. 


The  proposed system consists of two main scripts:  `cleaning_arabic.py` and  `Arabic_Fuzzy_Cmean.py`. 

Decomposition Method(SVD):
``` 
python 
from numpy.linalg import svd
U,sigma,V = low_rank_approx(matrix,k=7)

projectedDocuments = np.dot(matrix.T, U)

``` 
Classification Method (Fuzzy C-Mean):

``` 
 python 
 
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



# Dependencies

* [numpy:](https://anaconda.org/anaconda/numpy) `conda install -c anaconda numpy `.
