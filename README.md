AI recording
===========================
some project or demo for AI

****
	
|Author|Chris Zhang|
|---|---
|E-mail|zhangmw_play@163.com


****
## content
* [xgboost](#xgboost)
* [dynamic_programming](#dynamic_programming)
* [AutoSummarization](#AutoSummarization)
* [News_extract](#News_extract)




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
dynamic_programming
------
### used in IBM model 1

AutoSummarization
------
### TextRank

News_extract
------
### steps(use LTP from Harbin Institute of Technology):
	1.extract "说" and the similar words（Word2vector and graph search）
	2.use LTP to tagging, ner and dependency analysis to obtain the central word
	3.looking for the ner which is fit central word
	4.extract the speaking sentence, use "。 ？ ！ "as the sentence splitting symbol 
	5.use smooth inverse frequency (SIF) and PCA to handle the word vector, then calculate the similarity by Cosine
![Image text](https://raw.github.com/Chriszhangmw/machine-learning-and-demo/master/news_extract/algorithm1.png)




