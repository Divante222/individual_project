# Project Goal

Find Drivers of Heart Disease or Attack and create a model that can accurately predict Heart Disease or Attack while performing better than baseline.

# Project Description

Using the dataset for heart disease prediction from Kaggle, look for any features that have statistical significance to be used in a model for predicting heart disease or attack.

# Initial Questions/Thoughts

I believe that a majority of the feature columns will be good predictors of heart disease or attack.

# Data Dictionary

Coming soon

# Steps to Reproduce

Data was acquired from: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
Clone the repo and run through the final report.

# Project Plan

## Acquire

- Acquire dataset from link
- Cache full df for future use
- View data .info, .describe, .shape

## Prepare
- There are 253680 rows and 11 columns
- View/correct datatypes
- changed age from integers to correct age bins
- There were no null values
- Visualize full dataset for univariate exploration (histograms and boxplots)
    - Handle outliers
- I got rid of the outliers in BMI by getting rid of the top 1 percent
- verified datatypes
- made all of the column names lower case
- split the data on the target variable heart disease or attack

### Pre-processing

- Scaling data on train
- Encoding any necessary columns 
- Document how I'm changing the data

## Exploration

- Use unscaled data for multivariate exploration
    - Hypothesize
    - Visualize
    - Run stats tests
        - Run chi-squared test on catagorical vs. target
        - Run comparison of means test on continuous vs target
    - Summarize

### Exploration Summary
> - All of the features were statistically significant towards heard disease or attack

> - More people with high blood pressure had heart disease or a heart attack than those that did not
> - More people with high cholestreol had heart disease or a heart attack than those that did not
> - More people that smoked had heart disease or a heart attack than those that did not
> - People without diabeties had heart attacks 8.6% more than those that did have diabetes
> - You are 5% more likely to have a heart attack if you are a female
> - People who consumed heavy amounts of alcohol had heart disease or heart attacks 6.81% more than those that did not consume heavy amounts of alcohol

> - People in higher age brackets have a higher percentage of people who have had a heart disease or attack
> - People who have had heart attacks had more days of bad physical health
> - People who have had heart attacks had more days of bad mental health
> - People who have had heart attack and people who didn't have heart attacks have a similar bmi mean

### Initial Questions/Thoughts

# Modeling

- Use scaled/encoded data
- Split into X_variables and y_variables
- Determine evaluation metrics
    - Establishing baseline
- Run different models on train/validate
- Pick best model and evaluate on test

# Conclusion
> * My top model performed beat baseline by .02 %

# Recommendations
> * I would not recommend using the model because it did not beat baseline by a significant result