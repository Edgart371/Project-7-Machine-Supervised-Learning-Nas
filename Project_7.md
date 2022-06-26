<img src="https://bit.ly/2VnXWr2" alt="Ironhack Logo" width="100"/>

# Project 7 - Machine/Supervised learning - 

*Ferdi,Edgar*
*[DA, Paris & 06/22]*

## Description 

In Module 3 we started with a new topic: machine learning. The first week focused on supervised learning models and using these to predict and classify given data correctly. The challenge in this weeks project was to use data about different comets, created by NASA, including ultiple measurement details such as velocity, mean radius, perimeter etc. and create a supervised learning Model that can determine if a comet should be seen as a danger to earth (going to breach the atmosphere and hit earths surface) or not.

We begn by cleaning and preping the data to be ready for model firring and application. We dealt with missing values, performed exploratory data analysis, checked for outliers, and encoded if necessary. Next step on te way was feature selection. Using multiple tests such as RFE, SFM etc. (for details see slides) we selected different features being categorized as most important to have the opportunity of testing multiple feature combinations and seeing which will give us the best results from the models. The last step before stating with testing different models we went through the hyper parameter tuning process to see which parameters for the different models will give us the best results.

With all of the necessary information and our cleaned, prepared data we thenn applied following eight supervised learning models.
- Logistic regression (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- (https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html)
- BernoulliNB (https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)
- AdaBoostClassifier (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- PassiveAggressiveClassifier (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)
- SGDClassifier (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- ComplementNB (https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html)
- RandomForestClassifier (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- Hyper Tunning
- RandomizedSearchCV (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

To recieve the best possible outcome of our model (also regarding the precious business in this case --> importanc f finding all dangerous comets in stead of pure accuracy) we tried out different sampling methods to allow the models to perform different classification processes.

## Plan
- Create Trello plan 
- Create repo on GitHub
- Data cleaning and preparation in spider
- EDA and encoding of data
- Feature selection
- Hyper parameter tuning
- Application of our eight models
- Experimenting with different sampling methods to recieve best possible result
- Prepare presentation

## Deliverables

- Original data in csv format
- Cleaning code
- Clean data in cvv format
- Slides for presentation

## Links to deliverables and additional links

[Repository](https://github.com/Edgart371/Project7)  
[Trello] (https://leubef.atlassian.net/jira/software/projects/TVP6/boards/3)





