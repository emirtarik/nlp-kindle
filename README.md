# NLP Text Classification - Kindle book reviews

#### Author: Emir Tarık DAKIN

## Introduction
This project aims to classify book reviews from Amazon Kindle Store category based on their ratings. The project starts by giving a brief description of the data and then moves on to some Machine Learning algorithms. After taking a general overlook with the help of Lazy Classifier, a selection of these algorithms are optimized. Finally, an LSTM model is implemented and the results are compared.

## Dataset
The data is a small subset of 12,000 book reviews from Amazon's Kindle Store category, which can be found [here](https://www.kaggle.com/meetnagadia/amazon-kindle-book-review-for-sentiment-analysis). The dataset originally consists of a body of text `reviewText`, a summary (or the title) of the review `summary` and a `rating` variable that ranges from 1 to 5 (stars). The `rating` variable is used to create the `score`, which ranges from 1 to 3 (respectively: negative, neutral and positive). A sample of the processed dataset can be found below.


  |       | reviewText                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |   rating |   score |
  |------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------:|--------:|
  |  2573 | Not to bad but I expected a little more                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |        3 |       2 |
  | 10615 | I chose 5 stars because it's readable and very thorough.  Interesting that crimes don't change much, just the consequences.  The hammurabi code certainly decreased crimes rates permanently.  Something to think about.                                                                                                                                                                                                                                                                                                                                                                                                     |        5 |       3 |
  |  3294 | This is an erotic sci-fi romance written in third person. The heroine comes from Earth, and she has developed a way to manipulate DNA. Her expertise is in high demand, especially from two warring nations who want her to cure their citizens of a deadly virus. She is kidnapped and thus begins her adventure.This story might skirt close to forced sex for some readers with the heroine's desire artificially high. There are two lovers for the heroine and quite a few love scenes amongst the plot action.I found this easy to read with the writing and plot smooth. It's the first in a series. 3.5 stars for me |        3 |       2 |
  |  1486 | This book was not my cup of tea.  I stopped reading at about page 50.  I found it very slow-moving with too much emphasis on minute details that weren't necessary for plot or character building.  Yes, I wanted to know what was in that darn box,  but not badly enough for me to finish reading it.* I received a complimentary copy from the author through LibraryThing in exchange for an honest review.                                                                                                                                                                                                              |        1 |       1 |
  |  5276 | The prose went purple on the first page, so I skipped the aeons of italicized print to go to straight text.  Uhm.  The self-referential opening made me just slap my forehead and swear (yet again) never to buy a book unless I can make it through the first paragraph.                                                                                                                                                                                                                                                                                                                                                    |        1 |       1 |


|    |   rating |   reviewText |
|---:|---------:|-------------:|
|  4 |        5 |         3000 |
|  3 |        4 |         3000 |
|  2 |        3 |         2000 |
|  1 |        2 |         2000 |
|  0 |        1 |         2000 |








  | Model                         |   Accuracy |   Balanced Accuracy | ROC AUC   |   F1 Score |   Time Taken |
  |:------------------------------|-----------:|--------------------:|:----------|-----------:|-------------:|
  | NearestCentroid               |   0.695    |            0.662475 |           |   0.712826 |     0.639638 |
  | LinearDiscriminantAnalysis    |   0.7175   |            0.629369 |           |   0.71278  |     8.62887  |
  | RidgeClassifierCV             |   0.734167 |            0.619075 |           |   0.712769 |     7.98943  |
  | RidgeClassifier               |   0.73375  |            0.618671 |           |   0.712429 |     1.56434  |
  | LogisticRegression            |   0.690417 |            0.614999 |           |   0.691442 |     7.33513  |
  | LinearSVC                     |   0.6975   |            0.614471 |           |   0.693572 |   101.602    |
  | LGBMClassifier                |   0.732917 |            0.612181 |           |   0.708803 |    36.8064   |
  | BernoulliNB                   |   0.68625  |            0.608822 |           |   0.691231 |     0.758016 |
  | SVC                           |   0.749167 |            0.607702 |           |   0.709223 |   286.186    |
  | PassiveAggressiveClassifier   |   0.69125  |            0.605003 |           |   0.6872   |     2.53631  |
  | SGDClassifier                 |   0.691667 |            0.602834 |           |   0.686716 |    15.9707   |
  | Perceptron                    |   0.684583 |            0.600952 |           |   0.680754 |     1.56892  |
  | XGBClassifier                 |   0.718333 |            0.599195 |           |   0.694547 |   141.005    |
  | NuSVC                         |   0.745    |            0.596642 |           |   0.699707 |   325.632    |
  | GaussianNB                    |   0.630833 |            0.592256 |           |   0.645817 |     0.794095 |
  | ExtraTreesClassifier          |   0.73     |            0.574199 |           |   0.672454 |    14.8279   |
  | CalibratedClassifierCV        |   0.720417 |            0.568919 |           |   0.667968 |   357.226    |
  | RandomForestClassifier        |   0.710417 |            0.557347 |           |   0.654228 |    10.0703   |
  | AdaBoostClassifier            |   0.665417 |            0.538349 |           |   0.633779 |    12.3546   |
  | BaggingClassifier             |   0.655    |            0.534288 |           |   0.627617 |    27.5029   |
  | QuadraticDiscriminantAnalysis |   0.691667 |            0.532706 |           |   0.629626 |    11.4495   |
  | DecisionTreeClassifier        |   0.557917 |            0.484086 |           |   0.557636 |     4.66952  |
  | ExtraTreeClassifier           |   0.55375  |            0.474102 |           |   0.555026 |     0.773879 |
  | KNeighborsClassifier          |   0.498333 |            0.420522 |           |   0.490383 |    80.0082   |
  | DummyClassifier               |   0.395833 |            0.33672  |           |   0.3962   |     0.588086 |
  | LabelSpreading                |   0.345    |            0.334168 |           |   0.17853  |    10.8557   |
  | LabelPropagation              |   0.345    |            0.334168 |           |   0.17853  |     9.43163  |
  
  
  
  
  
  
  
  
  
  
  
  
  
