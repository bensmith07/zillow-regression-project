# Predicting Home Value using Zillow data

## About the Project 

We will conduct an in depth analysis of Zillow property data from 2017. We will use exploratory analysis techniques to identify the key drivers of the assessed tax value for those properties, then use machine learning algorithms to create a model capable of predicting tax values based on features of the property. 

### Project Description

Property values have skyrocketed over the last two years. With such rapid changes in home values, predicting those values has become even harder to predict than before. Since Zillow's estimate of home value is one of the primary drivers of website traffic, having a reliable estimate is paramount. Any improvement we can make on the previous model will help us out-estimate our competitors and keep us at the top as the most trusted name in real estate technology. 

This project will analyze property attributes in relation to their 2017 assessed tax value, develop a model for predicting that value based on those attributes, and leave with recommendations for how to improve future predictions. 


### Project Goals

By creating a reliable model for predicting property values, Zillow can enhance it's reputation for reliable property value estimates and better position itself in the real estate technology marketplace. 

### Initial Questions

- Is there a significant correlation between the number of bathrooms in a home and it's value?

- Is there a significant correlation between the number of bedrooms in a home and it's value?

- Is there a significant correlation between the square footage of a home and it's value?

- Which has a greater effect on home values, number of bedrooms, number of bathrooms, or square footage? 

- A "3/2" home (one with three bedrooms and two bathrooms) is often considered ideal for many buyers. Could this be a useful categorical feature? 


### Data Dictionary

| Variable          | Meaning                                                                   | values          |
| -----------       | -----------                                                               | -----------     |
| tax_value         | The total tax assessed value of the parcel (target variable)              | 1,000 - 1,038,547 |
| bedrooms          | Number of bedrooms in home                                                | 2 - 5 |
| bathrooms         | Number of bathrooms in home including fractional bathrooms                | 1 - 4 |
| sqft              | Calculated total finished living area of the home (in square feet)        | 152 - 3,566 |
| age               | Age of the structure (in years) at the time the data was collected (2017) | 1 - 137 |
| fips              |  Federal Information Processing Standard code (county of the property)    | '06059', '06037', '06111' |

### Steps to Reproduce

1. You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the zillow database. The env.py should also contain a function named get_db_url() that establishes the string value of the database url. Store that env file locally in the repository. 
2. clone my repo (including the acquire.py, prepare.py, explore.py, and model.py modules) (confirm .gitignore is hiding your env.py file)
3. libraries used are pandas, matplotlib, seaborn, numpy, sklearn, math. 

### The Plan

1. Acquisition
- In this stage, I obtained Zillow 2017 property data by querying the Codeup MySQL database hosted at data.codeup.com. The original source of this data was the Zillow competition hosted by Kaggle.com
2. Preparation
- I cleaned and prepped the data by:
    - removing all observations that included null values
    - renaming columns for readability
    - changing data types where appropriate
    - adding a feature: age (represents age of the property in years)
3. Exploration
- I conducted an initial exploration of the data by examing relationships between each of the potential features and the target
- then explored further, to answer the initial questions posed above
4. Modeling 
- Using varying combinations of features, I tested multiple Ordinary Least Squares (OLS) Regression models. 
- I then chose the model which performed with the smallest error on unseen data.

### How did we do?

We expect that our model will typically predict a value that is within approximately \$549,000 of the actual value of the property. This large of an error makes this not an exceptionally useful model. However, this is 175,000 better than our baseline error of approximately 724,000. This indicates that we have successfully identified drivers of home value, and gives us a jumping off point for further analysis and modeling. 

### Key Findings:

We determined that the following factors are significant drivers of home value:
- number of bedrooms
- number of bathrooms
- square footage (finished area)

### Recommendations:

Zillow should continue to collect data regarding the number of bedrooms and bathrooms in a home, as well as the home's area in square feet. This data has been conclusively shown to assist in predicting home values. If using this analysis to decide which homes are worth investing in, an investor should lean towards homes with higher values in these categories, all other considerations being equal. 

### Next Steps: 

Given more time, I would examine additional features as drivers of home value. Some factors that I would expect to have significant influence include:

    - whether or not a property has a pool or spa
    - whether or not there is a garage on the property
    - the size of the garage
    - more granular information about location, such as zip code or neighborhood
    
These features could be explored directly, through visualization and statistical testing, or they could be identified through automated features selection techniques such as Recursive Feature Elimination. 

Additionally, since real estate markets are based heavily on location, I might expect models to perform better which individually focus on a distinct geographic area. 

The goal would be to produce a model that has an error small enough to be useful to someone intending to sell or purchase a single family home. 