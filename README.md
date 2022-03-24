# Predicting Customer Churn at Telco

## About the Project 

We will conduct an in depth analysis of Zillow property data from 2017. We will use exploratory analysis techniques to identify the key drivers of the assessed tax value for those properties, then use machine learning algorithms to create a model capable of predicting tax values based on features of the property. 

### Project Description

Property values have skyrocketed over the last two years. With such rapid changes in home values, predicting those values has become even harder to predict than before. Since Zillow's estimate of home value is one of the primary drivers of website traffic, having a reliable estimate is paramount. Any improvement we can make on the previous model will help us out-estimate our competitors and keep us at the top as the most trusted name in real estate technology. 

This project will analyze property attributes in relation to their 2017 assessed tax value, develop a model for predicting that value based on those attributes, and leave with recommendations for how to improve future predictions. 


### Project Goals

By improving upon the previous model, Zillow can enhance it's reputation for reliable property value estimates and better position itself in the real estate technology marketplace. 

### Initial Questions

- Which has a greater effect on home values, number of bedrooms, number of bathrooms, or square footage? 

- A "3/2" home is often considered ideal for many buyers. Could this be a useful categorical feature? 

### Data Dictionary

| Variable          | Meaning                                                               | values          |
| -----------       | -----------                                                           | -----------     |
| tax_value         | The total tax assessed value of the parcel (target variable)          | 1,000 - 1,038,547 |
| bedrooms          |  Number of bedrooms in home                                           | 2 - 5 |
| bathrooms         | Number of bathrooms in home including fractional bathrooms            | 1 - 4 |
| sqft              |  Calculated total finished living area of the home (in square feet)   | 152 - 3,566 |


### Steps to Reproduce

1. You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the zillow database. The env.py should also contain a function named get_db_url() that establishes the string value of the database url. Store that env file locally in the repository. 
2. clone my repo (including the acquire.py, prepare.py, explore.py, and model.py modules) (confirm .gitignore is hiding your env.py file)
3. libraries used are pandas, matplotlib, seaborn, numpy, sklearn. 

### The Plan

1. Acquisition
- In this stage, I obtained Zillow 2017 property data by querying the Codeup MySQL database hosted at data.codeup.com.
2. Preparation
- I cleaned and prepped the data by:

<!-- TODO: add preparation steps here -->

3. Exploration
- I conducted an initial exploration of the data
- then explored further, to answer the initial questions posed above
4. Modeling 
- Using varying parameter values and combinations of features, I tested over [_____________] different models of varying types, including:

<!-- TODO: add model types here -->

- I then chose the model which performed with the smallest error on unseen data.

### How did we do?

 <!-- TODO: add information about model performance and goals here  -->

### Key Findings

<!-- add information about drivers of property value here -->

### Recommendations

<!-- add recommendations for model improvement here -->

### Next Steps

<!-- add information about next steps here -->