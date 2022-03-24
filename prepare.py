import os
import pandas as pd
import env
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


def remove_outliers_v2(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
    df.drop(columns=['outlier'], inplace=True)
    print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df

def prep_zillow_1(df):
    
    # check for null values
    total_nulls = df.isnull().sum().sum()
    
    # if the total number of nulls is less than 5% of the number of observations in the df
    if total_nulls / len(df) < .05:
        # drop all rows containing null values
        df = df.dropna()
    else:
        print('Number of null values > 5% length of df. Evaluate further before dropping nulls.')
    
    # renaming columns for readability
    df = df.rename(columns = {'bedroomcnt': 'bedrooms',
                              'bathroomcnt': 'bathrooms', 
                              'calculatedfinishedsquarefeet': 'sqft', 
                              'taxvaluedollarcnt': 'tax_value',
                              'yearbuilt': 'year_built',
                              'taxamount': 'tax_amount'})
    # changing data types
    
    # changing year from float to int
    df['year_built'] = df.year_built.apply(lambda year: int(year))
    # adding a feature: age 
    df['age'] = 2017 - df.year_built
    # drop original year_built_column
    df = df.drop(columns='year_built')
    
    # changing fips codes to strings
    df['fips'] = df.fips.apply(lambda fips: '0' + str(int(fips)))
    
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'sqft', 'age', 'tax_amount', 'tax_value'])
    
    return df

def train_test_validate_split(df, test_size=.2, validate_size=.3, random_state=42):
    '''
    This function takes in a dataframe, then splits that dataframe into three separate samples
    called train, test, and validate, for use in machine learning modeling.

    Three dataframes are returned in the following order: train, test, validate. 
    
    The function also prints the size of each sample.
    '''
    train, test = train_test_split(df, test_size=.2, random_state=42)
    train, validate = train_test_split(train, test_size=.3, random_state=42)
    
    print(f'train\t n = {train.shape[0]}')
    print(f'test\t n = {test.shape[0]}')
    print(f'validate n = {validate.shape[0]}')
    
    return train, test, validate

def scale_zillow(train, validate, test, scaler_type=MinMaxScaler()):    
    features_to_scale = ['bedrooms', 'bathrooms', 'sqft', 'age', 'tax_amount']
    other_features = ['fips']
    target = 'tax_value'

    # establish empty dataframes for storing scaled dataset
    train_scaled = pd.DataFrame(index=train.index)
    validate_scaled = pd.DataFrame(index=validate.index)
    test_scaled = pd.DataFrame(index=test.index)

    # screate and fit the scaler
    scaler = scaler_type.fit(train[features_to_scale])

    # adding scaled features to scaled dataframes
    train_scaled[features_to_scale] = scaler.transform(train[features_to_scale])
    validate_scaled[features_to_scale] = scaler.transform(validate[features_to_scale])
    test_scaled[features_to_scale] = scaler.transform(test[features_to_scale])

    # adding other features (no scaling) to scaled dataframes
    train_scaled[other_features] = train[other_features]
    validate_scaled[other_features] = validate[other_features]
    test_scaled[other_features] = test[other_features]

    # adding target variable (no scaling) to scaled dataframes
    train_scaled[target] = train[target]
    validate_scaled[target] = validate[target]
    test_scaled[target] = test[target]
    
    return train_scaled, validate_scaled, test_scaled