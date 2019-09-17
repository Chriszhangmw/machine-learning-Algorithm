Artificial Intelligence Notebook
===========================
some artificial intelligence algorithm code and demo 

****
	
|Author|Chris Zhang|
|---|---
|E-mail|zhangmw_play@163.com


****
## content
* [xgboost](#xgboost)
* [EM_algorithm](#EM_algorithm)
* [AutoSummarization](#AutoSummarization)
* [News_extract](#News_extract)
* [sentiment_analysis](#sentiment_analysis)
* [decision_tree](#decision_tree)
* [LR_KNN_SVM](#LR_KNN_SVM)
* [HMM](#HMM)
* [CRF](#CRF)
* [Skip_Gram](#Skip_Gram)



xgboost
-----------
### features:
	1.regularization, xgboost add the regularization to control the model,regularization contains the total of leaf number and each leaf's score's L2
	2.parallel processing, xgboost parallel processing is not based on trees, but based on feature. 
    reference:https://www.jianshu.com/p/7467e616f227 and 
              https://blog.csdn.net/szm21c11u68n04vdclmj/article/details/78410212
	3.flexibility:support user defined objective function and evaluation function, as long as the function has second derivative
	4.missing value processing,with a missing sample of the value of the feature, xgboost can automatically learn its splitting direction
	5.pruning, comapred with GBM, it is not easy to fall into local optimal solution.
### Usage:
	1.xgboost support two class API, one is the original, another is from sklearn
	2.xgboost can solve classifiers and regression problem, the code in the git take the iris as classification example, and take Boston house price as regression example
EM_algorithm
------
### used in IBM model 1:
![9](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/EM_algorithm/lan1.png)
![10](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/EM_algorithm/lan2.png)
![11](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/EM_algorithm/ibmmodel1.png)

AutoSummarization
------
### algorithm:
	1.sentence embedding with SIF, calculate the  whole text embedding and compared with each sentence, get importance by similarity
	2.use windows=3 to skip the whole text and build the sentence graph
### results:
before:
![Image text](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/AutoSummarization/result1.png)
after:
![Image text](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/AutoSummarization/result2.png)

News_extract
------
### steps(use LTP from Harbin Institute of Technology):
	1.extract "说" and the similar words（Word2vector and graph search）
	2.use LTP to tagging, ner and dependency analysis to obtain the central word
	3.looking for the ner which is fit central word
	4.extract the speaking sentence, use "。 ？ ！ "as the sentence splitting symbol 
	5.use smooth inverse frequency (SIF) and PCA to handle the word vector, then calculate the similarity by Cosine
![](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/News_extract/algorithm1.png)
### results:
![](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/News_extract/newsextractresults1.png)

sentiment_analysis
------
### algorithm:
1.textCNN
![](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/sentiment_analysis/textCNN.png)
2.GRU
![](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/sentiment_analysis/gru.png)

decision_tree
------
### usage:
change the values of the categorical variables;For the categorical varibles, we need to create dummy variables. 
### draw the tree:
![4](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/decision_tree/tree.png)

LR_KNN_SVM
------
### introduction:
	1.use the heart dataset to display the usage of logic regression，KNN and SVM algorithm by sklearn
	2.how to use Grid Search CV to find the best parameters
	3.how to do basic data analysis and data visualization(sample)
### sample data visualization:
![5](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/LR_KNN_SVM/distribution1.png)
![6](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/LR_KNN_SVM/correlation.png)
![7](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/LR_KNN_SVM/box.png)
![8](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/LR_KNN_SVM/multi_dis.png)

HMM
------
### introduction:

CRF
------
### introduction:

Skip_Gram
------
### training word2vector with skip-gram model,for example, the training text are :
KTH Royal Institute of Technology is the largest and most respected technical university in Sweden—ranked top 100 in the 2020 QS World University Rankings. By choosing KTH, you gain access to a vibrant student life and a prestigious academic environment.
### results of the vectors:
![9](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/Skip_Gram/skip1.png)





