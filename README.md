# RecSys

The RecSys library implements different RecSys algorithms described in research papers including mine. The library is written in C# for the .NET platform (https://github.com/Microsoft/dotnet), and runs on Windows, Linux (Mono), and MacOS X (Mono).

Implemented Algorithms
----------------------
  - **Matrix Factorization**[[1]](#1): 
    - the standard and popular matrix factorization approach to collaborative filtering.
  - **UserKNN**[[2]](#2): 
    - the standard and popular K-Nearest Neighbors approach to collaborative filtering.
  - **PrefUserKNN**[[3]](#3):
    - UserKNN plus accepting *preference relations* as input to perofrm top-N recommendation.
  - **PrefNMF Matrix Factorization**[[4]](#4):
    - MatrixFactorization plus accepting *preference relations* as input to perofrm top-N recommendation.
  - **Ordinal Random Fields**[[5]](#5):
    - a combination of Markov Random Fields and Ordinal Matrix Factorization techniques.
    - exploits both *Global* and *Local* structures in the rating matrix.
    - exploits *ordinal properties* of ratings.
  - **Preference Random Fields**[[6]](#6):
    - exploits both *Global* and *Local* structures in the rating matrix.
    - takes *preference relations* as input to perofrm top-N recommendation.
  
  
Installation
-------------
Coming soon.



References
----------
  
  - <a name="1"></a>[1]: Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, (8), 30-37.
    
  - <a name="3"></a>[3]: Brun, A., Hamad, A., Buffet, O., & Boyer, A. (2010, September). Towards preference relations in recommender systems. In Workshop on Preference Learning, European Conference on Machine Learning and Principle and Practice of Knowledge Discovery in Databases (ECML-PKDD 2010).

  


Copyright & Licensing
---------------------
Coming soon.
