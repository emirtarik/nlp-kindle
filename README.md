# NLP Text Classification - Kindle book reviews

#### Author: *Emir Tarık DAKIN*

## Introduction
This project aims to classify book reviews from Amazon Kindle Store category based on their ratings. The project starts by giving a brief description of the data and then moves on to some Machine Learning algorithms. After taking a general overlook with the help of Lazy Classifier, a selection of these algorithms are optimized. Finally, an LSTM model is implemented and the results are compared.

## Dataset
The data is a small subset of 12,000 book reviews from Amazon's Kindle Store category, which can be found [here](https://www.kaggle.com/meetnagadia/amazon-kindle-book-review-for-sentiment-analysis). The dataset originally consists of a body of text `reviewText`, a summary (or the title) of the review `summary` and a `rating` variable that ranges from 1 to 5 (stars). The `rating` variable is used to create the `score`, which ranges from 1 to 3 (respectively: *negative*, *neutral* and *positive*). A sample of the processed dataset can be found below.


  |       | reviewText                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |   rating |   score |
  |------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------:|--------:|
  |  2573 | Not to bad but I expected a little more                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |        3 |       2 |
  | 10615 | I chose 5 stars because it's readable and very thorough.  Interesting that crimes don't change much, just the consequences.  The hammurabi code certainly decreased crimes rates permanently.  Something to think about.                                                                                                                                                                                                                                                                                                                                                                                                     |        5 |       3 |
  |  3294 | This is an erotic sci-fi romance written in third person. The heroine comes from Earth, and she has developed a way to manipulate DNA. Her expertise is in high demand, especially from two warring nations who want her to cure their citizens of a deadly virus. She is kidnapped and thus begins her adventure.This story might skirt close to forced sex for some readers with the heroine's desire artificially high. There are two lovers for the heroine and quite a few love scenes amongst the plot action.I found this easy to read with the writing and plot smooth. It's the first in a series. 3.5 stars for me |        3 |       2 |
  |  1486 | This book was not my cup of tea.  I stopped reading at about page 50.  I found it very slow-moving with too much emphasis on minute details that weren't necessary for plot or character building.  Yes, I wanted to know what was in that darn box,  but not badly enough for me to finish reading it.* I received a complimentary copy from the author through LibraryThing in exchange for an honest review.                                                                                                                                                                                                              |        1 |       1 |
  |  5276 | The prose went purple on the first page, so I skipped the aeons of italicized print to go to straight text.  Uhm.  The self-referential opening made me just slap my forehead and swear (yet again) never to buy a book unless I can make it through the first paragraph.                                                                                                                                                                                                                                                                                                                                                    |        1 |       1 |

We can see that the number of reviews are more or less evenly distributed. Of course this will change when the ratings are grouped under the `score`, leaving less reviews in the *neutral* category.

|    |   rating |   reviewText |
|---:|---------:|-------------:|
|  4 |        5 |         3000 |
|  3 |        4 |         3000 |
|  2 |        3 |         2000 |
|  1 |        2 |         2000 |
|  0 |        1 |         2000 |

## Text cleaning
The review texts are cleaned by:

- converting to lowercase
- removing punctuations
- removing new line characters
- removing stopwords
- keeping a stemmed and a non-stemmed version

Below is an example of a random review, which is unexpectedly pious.

> The Old Testament has always been a bit of a mystery to me so I've  not studied it as much as other scriptures, until now. This part of David Ridges study series ties it in with gospel principles that are in all the scriptures and we see how the Gospel of Jesus Christ is eternal and was preached and practiced from the beginning.

Here is the same review after cleaning the text.

> old testament always bit mystery studied much scriptures part david ridges study series ties gospel principles scriptures see gospel jesus christ eternal preached practiced beginning

And here it is again, stemmed this time, still very much pious.

> old testament alway bit mysteri studi much scriptur part david ridg studi seri tie gospel principl scriptur see gospel jesus christ etern preach practic begin

For classification purposes, TF-IDF representation is used for each review using 1 and 2-gram words.

## ML algorithms
Starting off, a convenient package called [Lazy Predict](https://github.com/shankarpandala/lazypredict) helps to get insights on how the most-known algorithms work without any tuning. Below is a list of algorithms applied on the Kindle book reviews dataset and basic respective metrics. SVC methods are the best (they also take the longest).

| Model                         |   Accuracy |   Balanced Accuracy | ROC AUC   |   F1 Score |   Time Taken |
|:------------------------------|-----------:|--------------------:|:----------|-----------:|-------------:|
| NearestCentroid               |   0.695    |            0.662475 |           |   0.712826 |     0.711752 |
| LinearDiscriminantAnalysis    |   0.7175   |            0.629369 |           |   0.71278  |     8.89857  |
| RidgeClassifierCV             |   0.734167 |            0.619075 |           |   0.712769 |     8.29491  |
| RidgeClassifier               |   0.73375  |            0.618671 |           |   0.712429 |     1.61702  |
| LogisticRegression            |   0.690417 |            0.614999 |           |   0.691442 |     7.19013  |
| LinearSVC                     |   0.6975   |            0.614471 |           |   0.693572 |   103.08     |
| LGBMClassifier                |   0.732917 |            0.612181 |           |   0.708803 |    37.4257   |
| BernoulliNB                   |   0.68625  |            0.608822 |           |   0.691231 |     0.827498 |
| SVC                           |   0.749167 |            0.607702 |           |   0.709223 |   286.954    |
| PassiveAggressiveClassifier   |   0.69125  |            0.605003 |           |   0.6872   |     2.61749  |
| SGDClassifier                 |   0.691667 |            0.602834 |           |   0.686716 |    15.9865   |
| Perceptron                    |   0.684583 |            0.600952 |           |   0.680754 |     1.6244   |
| XGBClassifier                 |   0.718333 |            0.599195 |           |   0.694547 |   142.149    |
| NuSVC                         |   0.745    |            0.596642 |           |   0.699707 |   328.084    |
| GaussianNB                    |   0.630833 |            0.592256 |           |   0.645817 |     0.902328 |
| ExtraTreesClassifier          |   0.73     |            0.574199 |           |   0.672454 |    22.4783   |
| CalibratedClassifierCV        |   0.720417 |            0.568919 |           |   0.667968 |   367.867    |
| RandomForestClassifier        |   0.710417 |            0.557347 |           |   0.654228 |    12.9318   |
| AdaBoostClassifier            |   0.665417 |            0.538349 |           |   0.633779 |    16.8121   |
| BaggingClassifier             |   0.655    |            0.534288 |           |   0.627617 |    29.848    |
| QuadraticDiscriminantAnalysis |   0.691667 |            0.532706 |           |   0.629626 |    11.9611   |
| DecisionTreeClassifier        |   0.557917 |            0.484086 |           |   0.557636 |     5.69524  |
| ExtraTreeClassifier           |   0.55375  |            0.474102 |           |   0.555026 |     0.854397 |
| KNeighborsClassifier          |   0.498333 |            0.420522 |           |   0.490383 |    80.8093   |
| DummyClassifier               |   0.395833 |            0.33672  |           |   0.3962   |     0.601491 |
| LabelSpreading                |   0.345    |            0.334168 |           |   0.17853  |    10.8347   |
| LabelPropagation              |   0.345    |            0.334168 |           |   0.17853  |     9.50772  |
  
The same table but with the stemmed dataset this time is shown below.

| Model                         |   Accuracy |   Balanced Accuracy | ROC AUC   |   F1 Score |   Time Taken |
|:------------------------------|-----------:|--------------------:|:----------|-----------:|-------------:|
| NearestCentroid               |   0.68875  |            0.647887 |           |   0.704496 |     0.664193 |
| BernoulliNB                   |   0.701667 |            0.620169 |           |   0.69929  |     0.804801 |
| LGBMClassifier                |   0.722083 |            0.601189 |           |   0.69212  |    39.2781   |
| LinearDiscriminantAnalysis    |   0.6875   |            0.599636 |           |   0.6824   |     8.64746  |
| RidgeClassifierCV             |   0.706667 |            0.591745 |           |   0.682084 |     8.10922  |
| RidgeClassifier               |   0.70625  |            0.590918 |           |   0.681435 |     1.59814  |
| SVC                           |   0.73     |            0.589862 |           |   0.682854 |   288.484    |
| XGBClassifier                 |   0.710833 |            0.588596 |           |   0.679463 |   143.082    |
| LogisticRegression            |   0.670417 |            0.586767 |           |   0.667628 |     7.30668  |
| SGDClassifier                 |   0.668333 |            0.574314 |           |   0.658131 |    18.6974   |
| ExtraTreesClassifier          |   0.7225   |            0.57309  |           |   0.661486 |    21.1623   |
| PassiveAggressiveClassifier   |   0.6575   |            0.571532 |           |   0.65304  |     2.81159  |
| Perceptron                    |   0.659167 |            0.570956 |           |   0.655097 |     1.49383  |
| LinearSVC                     |   0.65875  |            0.569086 |           |   0.65186  |   100.346    |
| GaussianNB                    |   0.6075   |            0.568074 |           |   0.626221 |     0.838924 |
| CalibratedClassifierCV        |   0.703333 |            0.560547 |           |   0.650221 |   351.726    |
| RandomForestClassifier        |   0.701667 |            0.551876 |           |   0.638835 |    10.0304   |
| AdaBoostClassifier            |   0.667917 |            0.543395 |           |   0.63305  |    17.0222   |
| BaggingClassifier             |   0.648333 |            0.532293 |           |   0.616795 |    32.1461   |
| QuadraticDiscriminantAnalysis |   0.67875  |            0.53189  |           |   0.614887 |    11.6421   |
| DecisionTreeClassifier        |   0.56625  |            0.489024 |           |   0.562785 |     6.1668   |
| ExtraTreeClassifier           |   0.54875  |            0.467413 |           |   0.542463 |     0.849395 |
| KNeighborsClassifier          |   0.514167 |            0.415929 |           |   0.481629 |    79.9365   |
| LabelSpreading                |   0.33     |            0.33361  |           |   0.164281 |    10.5021   |
| LabelPropagation              |   0.33     |            0.33361  |           |   0.164281 |     9.33287  |
| DummyClassifier               |   0.391667 |            0.332009 |           |   0.390989 |     0.588089 |

It is clear that almost all algorithms make more sense of the unstemmed text, as the accuracies for the models on the stemmed data are lower in most cases. Therefore, the rest of the project uses the unstemmed version of the dataset. Moving forward, the following models are optimized:

- Logistic regression
  - Lasso optimization
  - Ridge optimization
- Random forest
- Gradient boosting
- Naive Bayes

The models are optimized t

Tuned Logistic Regression with Lasso optimization:

|   row_0 |   0 |   1 |    2 |
|--------:|----:|----:|-----:|
|       0 | 657 | 133 |   99 |
|       1 |  23 |  63 |   24 |
|       2 | 145 | 181 | 1075 |

``Test accuracy - Logistic regression Lasso = 0.747917``

Tuned Logistic Regression with Ridge optimization:

|   row_0 |   0 |   1 |    2 |
|--------:|----:|----:|-----:|
|       0 | 657 | 120 |   84 |
|       1 |  20 |  58 |   19 |
|       2 | 148 | 199 | 1095 |

``Test accuracy - Logistic regression Ridge = 0.754167``

Tuned Random Forest:

|   row_0 |   0 |   1 |    2 |
|--------:|----:|----:|-----:|
|       0 | 548 | 111 |   55 |
|       1 |   1 |   1 |    0 |
|       2 | 276 | 265 | 1143 |

``Test accuracy - Random forest = 0.705000``

Tuned SVC:

|   row_0 |   0 |   1 |    2 |
|--------:|----:|----:|-----:|
|       0 | 630 | 125 |  103 |
|       1 |  59 |  98 |   74 |
|       2 | 136 | 154 | 1021 |

``Test accuracy - SVC = 0.728750``

Tuned Gradient boosting:

|   row_0 |   0 |   1 |    2 |
|--------:|----:|----:|-----:|
|       0 | 622 | 118 |  101 |
|       1 |  45 |  70 |   39 |
|       2 | 158 | 189 | 1058 |

``Test accuracy - Gradient Boosting = 0.729167``

Tuned Naive Gaussian Bayes:

Number of mislabeled points out of a total 2400 points : 694
|   row_0 |   0 |   1 |   2 |
|--------:|----:|----:|----:|
|       0 | 649 | 131 | 133 |
|       1 |  82 | 146 | 154 |
|       2 |  94 | 100 | 911 |

``Test accuracy - Gaussian Naive Bayes = 0.710833``

## Neural networks

### Crude models

The application of neural networks start with two crude sequential models. As expected, the most basic model that uses stochastic gradient descent is not that good and performs worse than the tuned ML algorithms in the previous part. However, there is room to improve.

![Crude neural network with SGD](/images/crude.png "Crude neural network with SGD")

This time, the same model is run with mini batch gradient descent and the inclusion of a momentum variable. This is significantly better than the initial model but still not as good as the logistic regression with the ridge optimization.

![Crude neural network with mini batch + momentum](/images/crude_mini.png "Crude neural network with mini batch + momentum")


|                             |   accuracy |     f1 |   precision |   recall |
|:----------------------------|-----------:|-------:|------------:|---------:|
| Crude SGD                   |     0.7079 | 0.7018 |      0.6968 |   0.7079 |
| Crude mini-batch + Momentum |     0.7496 | 0.7076 |      0.7234 |   0.7496 |

### CNN models


![Convolutional neural network](/images/c1.png "Convolutional neural network")



![Convolutional neural network with L1 optimization](/images/c2.png "Convolutional neural network with L1 optimization")



![Convolutional neural network with L2 optimization](/images/c3.png "Convolutional neural network with L2 optimization")



|          |   accuracy |     f1 |   precision |   recall |
|:---------|-----------:|-------:|------------:|---------:|
| CNN      |     0.6975 | 0.6993 |      0.7014 |   0.6975 |
| CNN + l1 |     0.7238 | 0.6902 |      0.6783 |   0.7238 |
| CNN + l2 |     0.7125 | 0.6817 |      0.6701 |   0.7125 |

