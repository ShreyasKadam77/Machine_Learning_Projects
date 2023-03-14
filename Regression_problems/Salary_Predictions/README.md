# Data Science Salary Prediction

## Project Overview
• Created a machine learning model that **estimates salary of data scientist based on the features like rating, company_founded, etc.**<br/>
• This project **helps data scientist/analyst to negotiate their income for an existing or a new job**<br/>
• Built a flask API endpoint that was hosted on a local webserver. The API endpoint takes in a request with features and returns an estimated salary.

## Resources Used
• Packages: **pandas, numpy, sklearn, matplotlib, seaborn, Flask.**<br/>
• Dataset: https://github.com/PlayingNumbers/ds_salary_proj/blob/master/glassdoor_jobs.csv <br/>


## Exploratory Data Analysis (EDA)
• **Removed unwanted columns**: 'Unnamed: 0'<br/>
• **Plotted bargraphs and countplots** for numerical and categorical features respectively for EDA<br/>
• **Numerical Features** (Rating, Founded,Salary): **Replaced NaN or -1 values with mean or meadian based on their distribution**<br/>

![salary1](data/salary1.png) ![salary2](data/salary2.png)<br/>

• **Categorical Features: Replaced NaN or -1 values with 'Other'/'Unknown' category**<br/>
• **Removed unwanted alphabet/special characters from Salary feature**<br/>
• **Converted the Salary column into one scale** i.e from (per hour, per annum, employer provided salary) to (per annum)


## Feature Engineering
• **Creating new features** from existing features e.g. **job_in_headquaters from (job_location, headquarters)**, etc.<br/>
• Trimming columns i.e. **Trimming features having more than 10 categories to reduce the dimensionality**<br/>
• **Handling ordinal and nominal categorical features**<br/>
• Feature Selection using **correlation matrix**<br/>

![correlation](data/correlation.png)<br/>

• Feature Scaling using **StandardScalar**

## Model Building and Evaluation
Metric: Root Mean Squared Error (RMSE)<br/>
• Multiple Linear Regression: 25.911<br/>
• Lasso Regression: 26.379<br/>
• **Random Forest: 19.050**<br/>

## Feature Importance
![feature_importances](data/feature_importances.png)

## Model Prediction

![Prediction](data/predictions.gif)
