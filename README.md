# Machine_learning_projects

## Overview:-
* This repository contains projects on ML Classification and Regression
>* [Classification](https://en.wikipedia.org/wiki/Statistical_classification):-  classification is the problem of identifying to which of a set of categories (sub-populations) a new observation belongs, 
on the basis of a training set of data containing observations (or instances) whose category membership is known.
>>* Ex. Prediction of heart disease( Yes or No)

![classification](https://user-images.githubusercontent.com/75840165/109419751-04d5c900-79f5-11eb-93b6-004d2875116b.png)
------------------------------------------------------------------

>* [Regression](https://en.wikipedia.org/wiki/Regression_analysis):- In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships between a dependent 
variable (often called the 'outcome variable') and one or more independent variables (often called 'predictors', 'covariates', or 'features').
>>* Ex. Prediction of number (price prediction)

![Regression](https://user-images.githubusercontent.com/75840165/109419745-ff787e80-79f4-11eb-9f1d-55646eb0dce8.png)
--------------------------------------------------

## Repository Guide:-
Here is list for every project in this repository and link to introduction of project

--------------------------------------------------
### Classification:-
* <a href="#Titanic">Titanic Survival Prediction</a>


## Tools And Workflow:-

![machine-learning-life-cycle](https://user-images.githubusercontent.com/75840165/114572787-6dbb8c80-9c95-11eb-8865-d8bf0d677f68.png)

* Collect Data
* Exploratory Data Analysis:- (with visualization)
>>* Pandas, Matplotlib, [Seaborn](https://seaborn.pydata.org/)
* Feature Engineering:-
>* Handle missing values and Categorical Features - sklearn
>* Outlier's (depending on which model to use)
>* Feature Scaling (depending on model)
* Split data
* Feature selection:-
>* Correlation matrix, [VariationThreshold](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html)
* Building model 
>* [Selecting models](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
>* Train and Evaluate model
>* If more than one depending on initial accuracy take for Hyperparameter Tuning - [RandomSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV), [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
* Predict on test set

<img src="https://user-images.githubusercontent.com/75840165/109416819-79553b80-79e6-11eb-9b75-a0861894af1c.png" height=500, width=700/>

## Project Introduction's:-(for detail discription visit respective project)

### Classification:-
-----------------------------------------------

<h4 id='Titanic'> Titanic Survial Predictor</h4>

* This project is to predict the passengers in titanic will survive or not on the basis of given data.
* In this notebook we've used Logistic Regression, Random Forest, KNN and GradientBoost classifier and got better accuracy on these top 2 models i.e.GradientBoostClassifier and Logistic Regression.
* After Hypertunning got an accuracy 84.21 % for Logistic Regression.

* Get source code [Visit][Titanic]

<h4 id='Diabetes'> Diabetes Predictor</h4>

* The objective of this project is to predict whether a patient has diabetes or not on the basis of certain diagnostic measurements given in dataset.
* In this notebook we've used Random Forest, KNN and Decision Tree classifier and got better accuracy on the Random Forest Classifier.
* After Hypertunning got an accuracy 81.30 % for Random Forest Classifier.

* Get source code [Visit][Diabetes]
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------







<!-- Classification Links-->
[Titanic]: https://github.com/ShreyasKadam77/Machine_Learning_Projects/tree/master/Classification_problems/Titanic_survival_predictions
[Diabetes]:
https://github.com/ShreyasKadam77/Machine_Learning_Projects/tree/master/Classification_problems/Diabetes_predictor




