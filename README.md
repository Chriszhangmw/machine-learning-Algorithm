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
* [sentiment_analysis](#sentiment_analysis)
* [decision_tree](#decision_tree)



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
