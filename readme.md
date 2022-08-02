# Shared Task on Bias, Threat and Aggression Identification in Context(BiTAg-Con)

![](https://github.com/shaurysrivastav27/ASOC-2022/blob/main/images/chart.png)

## Abstract
Aggression and its manifestations in different forms have taken unprecedented proportions with the tremendous growth of the internet and social media. In this model we are working on different aspects of aggressive and offensive language usage online and its automatic identification. We have classified each sample at seven different levels namely aggression level, aggression intensity, discursive role, gender bias, religious bias, caste/class bias and ethnicity/racial bias. The sample can be given any combinations of the tags and the occurrence of one is not dependent on the other.



## Introduction

The task that we have taken for this research project is to detect whether a given text is offensive or not. Further we have divided the text into 7 different categories. The main motivation for choosing this topic is to develop an efficient system to detect this offensive content on the social media platform which can be removed afterwards. This will prevent  the unease and negativity that may spread in society, people or any community group’s mind. 
Methodology
In this section a detailed description of the dataset used is given along with the detailed steps employed for the pre-processing of the text data and the details of the proposed Model. 
Dataset 
Sub-tasks
(a) Aggression, Gender Bias, Racial Bias, Religious Intolerance and Bias and Casteist Bias on social media and (b) the "discursive role" of a given comment in the context of the previous comment(s).
Gender Bias : It has three subclasses Gender (GEN), Gender Threat (GENT) and Non-Gender (NGEN).
 Ethnicity/Racial Bias. Ethnic bias can be defined as engaging in discriminatory behavior, holding negative attitudes toward, or otherwise having less favorable reactions toward people based on their ethnicity. Such bias has become an important focus of research in the social sciences. It has three subclasses Ethnic/Racial comments (ETH), Ethnic/Racial Threat(ETHT), Non Ethnic/Racial comments(NCOM). 
Communal bias :  It has three subclasses Communal (COM), Communal Threat (COMT), Non-Communal (NCOM).
Caste/class bias :  It has three subclasses Casteist/Classist comments (CAS), Casteist/classist Threat (CAST), Non-Casteist/Classist comments (NCAS).
Aggression Level : It has three subclasses ‘Overtly Aggressive’(OAG), ‘Covertly Aggressive’(CAG) and ‘Non-aggressive’(NAG) text data.
Aggression Intensity: This level gives a 4-way classification in between ‘Physical Threat’(PTH), ‘Sexual Threat’(STH), ‘Non-threatening Aggression’(NtAG) and ‘Curse/Abuse’(CuAG).
Religious Bias : At the level E, the task is to develop a 3-way classifier for classifying the text as ‘communal’ (COM), ‘Communal Threat’(COMT) and ‘non-communal’(NCOM).
The dataset is multilingual with a total of over 140,000 samples (over 60,000 unique samples) for training and development and over 15,000 unique samples for testing in four Indian languages Meitei, Bangla(Indian variety), Hindi and English. The dataset consists of comments from a total of 158 videos i.e., it has a comment thread in total.
All the data is collected from YouTube.

## Pre-processing
Removing noise: The twitter handles and URL links are the main source of noise in the given dataset. Simple regular expressions were created to delete anything which was followed by @to remove the twitter handles and anything starting with HTTP to remove the URL links.
Removing Punctuation marks and special symbols: All kinds of punctuation marks and special symbols appearing in the dataset were cleared since these characters add no value to text-understanding and in return induce noise into the system.
Lowering of dataset: we have lowered the case of the sentences of the dataset to form uniformity in the dataset and we don't have to care about the case of the dataset then.
One Hot Encoding: Categorical offensive values were label encoded as 0,1,2 to negative and positive words respectively. This was done to give a numeric representation to the categorical data.
We have considered Logistic Regression, Naïve Bayes and support vector machine (SVM) for text classifications. We have used training data on each model by performing GridSearchCV for all the combinations of feature parameters. We have analyzed performance on the basis of average score of the cross validation.


### SVM(Support vector machine) 
The first model with which we started out was SVM as its state of the art for classification tasks. The parameters we used were 
C = 1 and the kernel was linear but it didn’t perform well on the training dataset as the data wasn’t linearly separable. 
The next change we made was : kernel = ‘rbf’  C = 1 , this performed well on train data but didn’t perform well on the train dataset even with regularization .
### MNB(Multinomial Bayes)
The next model we used was Multinomial Bayes which is commonly used for text classification. But due to a sparse dataset from TfidfVectorizer it had some limitations and even with hyper parameter tuning the performance didn’t improve much. 
The parameters used were  alpha = 1e-03 . 
Decision Trees and Random Forest
We used tree methods as well for classification but random forest was overfitting on the given dataset, hence it was not able to capture a general trend in the dataset. 
The parameters used were : 
Criterion : entropy , max depth = 4. 
### Logistic regression
We finally moved on logistic regression for classification purposes and this model out performed all the above models on a test dataset . 
Although the train micro f1 was lesser as compared to the svm model, it performed well on the test dataset. 
We used different solvers and C values but got the best result on : C =5, solver = ‘newton-cg’.
The possible reason was that the dataset in sparse form was linearly separable, but the svm model being hard margin was not a generalized model for our case , but logistic regression generalized the metrics and hence performed really well in our case. 
Reasons for not moving towards deep learning techniques
When we talk about NLP we directly take into account the state of the art models such as BERT(MuRIL) , transformers . But since in our case the simpler model (Logistic Regression) was performing well, hence we didn’t move on to the deep learning model. 
The existing models had tokenizers which use sentence piece methods and hence while our dataset was codemixed it would break the useful words into irrelevant tokens . 
For e.g : BERT’s tokenizer doesn’t have word ‘ANNA’ (brother)
So , it breaks down the word into ‘AN’ ,’NA’ , which isn’t even close to brother. 
The other reason was, these were simple classification tasks : i.e: whether a sentence is aggressive or not . 
So the presence of certain words were the only parameters we had to take care of . 
Hence, in our case logistic regression performed really well. 

![](https://github.com/shaurysrivastav27/ASOC-2022/blob/main/images/Screenshot%20from%202022-08-02%2010-25-35.png)

# Shared task on dravidian Languages (Ongoing)

sentiment
```
macro avg			best result reported		our result
kannada  :	        0.59 XLM,BERT               0.66    SVC(C=2,gamma='scale',degree=1,kernel='rbf',class_weight=weights)
tamil	 :          0.61 DistilBERT             0.61    SVC(C=2,gamma=5,kernel='rbf',class_weight=weights)
malayalam:	        0.63 BERT 					0.66	SVC(C=2,gamma=2,kernel='rbf',class_weight=weights)
```

offensive 
```
macro avg
tamil    :			0.74 DistilBERT             0.80    SVC(C=100,gamma='scale',degree=2,kernel='poly',coef=1.0)
malayalam:          0.94 XLM					0.73	LR
````
