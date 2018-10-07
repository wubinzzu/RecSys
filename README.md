# RecSys

The RecSys library implements different RecSys algorithms described in research papers including mine. The library is written in C# for the .NET platform (https://github.com/Microsoft/dotnet), and runs on Windows, Linux, and MacOS X.

What is special about this library?
-------------------------
Most libraries are trying to put all algorithms into a unified framework with deep class inheritance and highly modularized. This will make the libraries very "Object Oriented" and but difficult to read by new users as you will need to jump between multiple files to understand one single algorithm.

The good news is, the RecSys tries to put each algorithm into a single self-contained source file and even implemented in single function when possible. In other words, you only need to open the source file of the algorithm you are interested in and read it from top to bottom in a linear manner.

Implemented Algorithms
----------------------
These algorithms are aimed to be implemented in the same way they are described in the authors' papers, however, differences may still exist.
  - **Matrix Factorization** [1]: 
    - the standard and popular matrix factorization approach to collaborative filtering.
  - **UserKNN** [2]: 
    - the standard and popular K-Nearest Neighbors approach to collaborative filtering.
  - **PrefUserKNN** [3]:
    - UserKNN plus accepting *preference relations* as input to perofrm top-N recommendation.
  - **PrefNMF Matrix Factorization** [4]:
    - Matrix Factorization plus accepting *preference relations* as input to perofrm top-N recommendation.
  - **Ordinal Matrix Factorization** [5]:
    - exploits ordinal properties of ratings.
    - produce a full distributino over the ratings instead of a point estimate.
  - **Ordinal Random Fields** [6]:
    - a combination of Markov Random Fields and Ordinal Matrix Factorization techniques.
    - exploits both *Global* and *Local* structures in the rating matrix.
    - exploits *ordinal properties* of ratings.
  - **Preference Random Fields** [7]:
    - exploits both *Global* and *Local* structures in the rating matrix.
    - takes *preference relations* as input to perofrm top-N recommendation.
  
  
Installation
-------------
Coming soon.



References
----------
  
 1. Koren, Y., Bell, R., & Volinsky, C. (2009). [Matrix factorization techniques for recommender systems](http://dx.doi.org/10.1109/MC.2009.263). Computer, (8), 30-37. [[PDF]](http://www2.research.att.com/~volinsky/papers/ieeecomputer.pdf)
 2. Konstan, J. A., Miller, B. N., Maltz, D., Herlocker, J. L., Gordon, L. R., & Riedl, J. (1997). [GroupLens: applying collaborative filtering to Usenet news](http://dx.doi.org/10.1145/245108.245126). Communications of the ACM, 40(3), 77-87. [[PDF]](https://www.ischool.utexas.edu/~i385q/readings/konstan_1997.pdf)
 3. Brun, A., Hamad, A., Buffet, O., & Boyer, A. (2010, September). [Towards preference relations in recommender systems](http://www.ke.tu-darmstadt.de/events/PL-10/papers/1-Brun.pdf). In Workshop on Preference Learning, European Conference on Machine Learning and Principle and Practice of Knowledge Discovery in Databases (ECML-PKDD 2010). [[PDF]](http://www.ke.tu-darmstadt.de/events/PL-10/papers/1-Brun.pdf)
 4. Desarkar, M. S., Saxena, R., & Sarkar, S. (2012). [Preference relation based matrix factorization for recommender systems](http://dx.doi.org/10.1007/978-3-642-31454-4_6). In User Modeling, Adaptation, and Personalization (pp. 63-75). Springer Berlin Heidelberg. [[PDF]](http://www.researchgate.net/profile/Sudeshna_Sarkar2/publication/241770977_Preference_relation_based_matrix_factorization_for_recommender_systems/links/0deec53606e7ad7334000000.pdf)
 5. Koren, Y., & Sill, J. (2011, October). [OrdRec: an ordinal model for predicting personalized item rating distributions](http://dx.doi.org/10.1145/2043932.2043956). RecSys 2011 (pp. 117-124). ACM. [[PDF]](http://labs.yahoo.com/files/paper.pdf)
 6. S.-W. Liu, T. Tran, G. Li and Y, Jiang. [Ordinal Random Fields for Recommender Systems](http://www.jmlr.org/proceedings/papers/v39/liu14.html). In Proceedings of the Sixth Asian Conference on Machine Learning (ACML 2014), pp. 283–298, 2014. [[PDF]](http://prada-research.net/~truyen/papers/liu2014ordinal.pdf)

  
