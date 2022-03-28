import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

def prep_zillow_1(df):
    '''
    This function takes in a dataframe of zillow data obtained using the acquire.zillow_2017_data function. 

    It checks for null values and removes all observations containing null values if the number of null values is 
    less than 5% the total number of observations. 

    It renames feature columns for readability and adherence to snake_case conventions. 

    It creates a feature, 'age', by subtracting year_built from the year of the transactions (2017), then
    drops the original year_built column. 

    It changes fips codes from number types to a string type. 

    The cleaned dataframe is returned. 
    '''
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
                              'yearbuilt': 'year_built'})
    # changing data types:
    # changing year from float to int
    df['year_built'] = df.year_built.apply(lambda year: int(year))
    # adding a feature: age 
    df['age'] = 2017 - df.year_built
    # drop original year_built_column
    df = df.drop(columns='year_built')
    # changing fips codes to strings
    df['fips'] = df.fips.apply(lambda fips: '0' + str(int(fips)))
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

def remove_outliers(train, validate, test, k, col_list):
    ''' 
    This function takes in a dataset split into three sample dataframes: train, validate and test.
    It calculates an outlier range based on a given value for k, using the interquartile range 
    from the train sample. It then applies that outlier range to each of the three samples, removing
    outliers from a given list of feature columns. The train, validate, and test dataframes 
    are returned, in that order. 
    '''
    for col in col_list:
        q1, q3 = train[col].quantile([.25, .75])  # get quartiles
        iqr = q3 - q1   # calculate interquartile range
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound
        # remove outliers from each of the three samples
        train = train[(train[col] > lower_bound) & (train[col] < upper_bound)]
        validate = validate[(validate[col] > lower_bound) & (validate[col] < upper_bound)]
        test = test[(test[col] > lower_bound) & (test[col] < upper_bound)]
    #return sample dataframes without outliers
    return train, validate, test

def scale_zillow(train, validate, test, target, scaler_type=MinMaxScaler()):
    '''
    This takes in the train, validate, and test dataframes, as well as the target label. 

    It then fits a scaler object to the train sample based on the given sample_type, applies that
    scaler to the train, validate, and test samples, and appends the new scaled data to the 
    dataframes as additional columns with the prefix 'scaled_'. 

    train, validate, and test dataframes are returned, in that order. 
    '''
    # identify quantitative features to scale
    quant_features = [col for col in train.columns if (train[col].dtype != 'object') & (col != target)]
    # establish empty dataframes for storing scaled dataset
    train_scaled = pd.DataFrame(index=train.index)
    validate_scaled = pd.DataFrame(index=validate.index)
    test_scaled = pd.DataFrame(index=test.index)
    # screate and fit the scaler
    scaler = scaler_type.fit(train[quant_features])
    # adding scaled features to scaled dataframes
    train_scaled[quant_features] = scaler.transform(train[quant_features])
    validate_scaled[quant_features] = scaler.transform(validate[quant_features])
    test_scaled[quant_features] = scaler.transform(test[quant_features])
    # add 'scaled' prefix to columns
    for feature in quant_features:
        train_scaled = train_scaled.rename(columns={feature: f'scaled_{feature}'})
        validate_scaled = validate_scaled.rename(columns={feature: f'scaled_{feature}'})
        test_scaled = test_scaled.rename(columns={feature: f'scaled_{feature}'})
    # concat scaled feature columns to original train, validate, test df's
    train = pd.concat([train, train_scaled], axis=1)
    validate = pd.concat([validate, validate_scaled], axis=1)
    test = pd.concat([test, test_scaled], axis=1)
    #identify scaled features
    scaled_features = [col for col in train.columns if col.startswith('scaled_')]

    return train, validate, test

def encode_zillow(train, validate, test, target):
    '''
    This function takes in the train, validate, and test samples, as well as a label for the target variable. 

    It then encodes each of the categorical variables using one-hot encoding with dummy variables and appends 
    the new encoded variables to the original dataframes as new columns with the prefix 'enc_{variable_name}'.

    train, validate and test dataframes are returned (in that order)
    '''
    
    features_to_encode = [col for col in train.columns if (train[col].dtype == 'object') & (col != target)]
                        
    for feature in features_to_encode:
        dummy_df = pd.get_dummies(train[feature],
                                  prefix=f'enc_{train[feature].name}',
                                  drop_first=True)
        train = pd.concat([train, dummy_df], axis=1)
        
    for feature in features_to_encode:
        dummy_df = pd.get_dummies(validate[feature],
                                  prefix=f'enc_{validate[feature].name}',
                                  drop_first=True)
        validate = pd.concat([validate, dummy_df], axis=1)
        
    for feature in features_to_encode:
        dummy_df = pd.get_dummies(test[feature],
                                  prefix=f'enc_{test[feature].name}',
                                  drop_first=True)
        test = pd.concat([test, dummy_df], axis=1)
    
    return train, validate, test