# Loan Approval Prediction

#### Problem Statement:
   • Company wants to automate the loan eligibility process based on customer details provided while filling online application form.<br/>
   • These detail are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, etc.

## Project Overview 
• The objective of this project is to create a machine learning model that **predicts whether loan of customer would be approved or not based on the features like Income, Number of Dependents, etc.** <br/>
• This project **helps to automate the loan eligibility process.** <br/>
• The dataset for the project is taken from Kaggle.


## Resources Used
• Packages: **pandas, numpy, sklearn, matplotlib, seaborn.**<br/>
• Dataset: https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset <br/>


## Exploratory Data Analysis (EDA)
• **Removed unwanted columns**: 'Loan_ID'<br/>
• **Plotted bargraphs and countplots** for categorical and numerical features respectively to analyze data distrubution<br/>
• In some numerical features, data distribution was **right skewed**. Used **Log transformation** to make those features to **normally distributed**.

![applicant_income](data/applicant_income.png) ![applicant_income_log](data/applicant_income_log.png)<br/>

• For **Numerical Features** : **Replaced NaN values with mean**<br/>
• For **Categorical Feature** (Credit_History): **Replaced NaN values with mode**<br/>
• **Handling ordinal and nominal categorical features**<br/>
• Feature Selection using **correlation matrix**<br/>

![correlation](data/correlation.png)<br/>


## Model Building and Evaluation
Metrics: Accuracy, Confusion Matrix and ROC Curve <br/>
• Got best accuracy of 77.27 for Ada Boost Classifier.

![confusion_matrix](data/confusion_matrix.png)<br/>

![ROC](data/ROC_curve.png)
