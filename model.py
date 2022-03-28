import pandas as pd
import sklearn as sk
from math import sqrt
from sklearn.linear_model import LinearRegression 

def determine_regression_baseline(train, target, return_results_df=False):
    
    # create empty dataframe for storing prediction results
    results = pd.DataFrame(index=train.index)
    # assign actual values for the target variable
    results['actual'] = train[target]
    # assign a baseline using mean
    results['baseline_mean'] = train[target].mean()
    # assign a baseline using median
    results['baseline_median']= train[target].median()
    
    # get RMSE values for each potential baseline
    RMSE_baseline_mean = sqrt(sk.metrics.mean_squared_error(results.actual, results.baseline_mean))
    RMSE_baseline_median = sqrt(sk.metrics.mean_squared_error(results.actual, results.baseline_median))
    
    # compare the two RMSE values; drop the lowest performer and assign the highest performer to baseline variable
    if RMSE_baseline_median < RMSE_baseline_mean:
        results = results.drop(columns='baseline_mean')
        results['RMSE_baseline'] = RMSE_baseline_median
        baseline = 'median'
    else:
        results = results.drop(columns='baseline_median')
        results['RMSE_baseline'] = RMSE_baseline_mean
        baseline = 'mean'

    print(f'The highest performing baseline is the {baseline} target value.')
    
    # return
    if return_results_df:
        return results

def run_baseline(train,
                 validate,
                 target,
                 model_number,
                 model_info,
                 model_results):

    y_train = train[target]
    y_validate = validate[target]

    # identify model number
    model_number = 'baseline'
    #identify model type
    model_type = 'baseline'

    # store info about the model

    # create a dictionary containing model number and model type
    dct = {'model_number': model_number,
           'model_type': model_type}
    # append that dictionary to the model_info dataframe
    model_info = model_info.append(dct, ignore_index=True)


    # establish baseline predictions for train sample
    y_pred = baseline_pred = pd.Series(train[target].mean()).repeat(len(train))

    # get metrics
    dct = {'model_number': model_number, 
           'sample_type': 'train', 
           'metric_type': 'RMSE',
           'score': sqrt(sk.metrics.mean_squared_error(y_train, y_pred))}
    model_results = model_results.append(dct, ignore_index=True)


    # establish baseline predictions for validate sample
    y_pred = baseline_pred = pd.Series(validate[target].mean()).repeat(len(validate))

    # get metrics
    dct = {'model_number': model_number, 
           'sample_type': 'validate', 
           'metric_type': 'RMSE',
           'score': sqrt(sk.metrics.mean_squared_error(y_validate, y_pred))}
    model_results = model_results.append(dct, ignore_index=True)
    
    model_number = 0
    
    return model_number, model_info, model_results

def run_OLS(train, validate, target, model_number, model_info, model_results):

    features1 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft']
    features2 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age']
    features3 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 'enc_fips_06059', 'enc_fips_06111']
    feature_combos = [features1, features2, features3]

    for features in feature_combos:

        # establish model number
        model_number += 1

        #establsh model type
        model_type = 'OLS linear regression'

        # store info about the model

        # create a dictionary containing the features and hyperparamters used in this model instance
        dct = {'model_number': model_number,
               'model_type': model_type,
               'features': features}
        # append that dictionary to the model_info dataframe
        model_info = model_info.append(dct, ignore_index=True)

        #split the samples into x and y
        x_train = train[features]
        y_train = train[target]

        x_validate = validate[features]
        y_validate = validate[target]

        # create the model object and fit to the training sample
        linreg = LinearRegression(normalize=True).fit(x_train, y_train)

        # create the model object and fit to the training sample
        linreg = LinearRegression(normalize=True).fit(x_train, y_train)

        # make predictions for the training sample
        y_pred = linreg.predict(x_train)
        sample_type = 'train'

        # store information about model performance
        # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
        dct = {'model_number': model_number, 
               'sample_type': sample_type, 
               'metric_type': 'RMSE',
               'score': sqrt(sk.metrics.mean_squared_error(y_train, y_pred))}
        model_results = model_results.append(dct, ignore_index=True)

        # make predictions for the validate sample
        y_pred = linreg.predict(x_validate)
        sample_type = 'validate'

        # store information about model performance
        # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
        dct = {'model_number': model_number, 
               'sample_type': sample_type, 
               'metric_type': 'RMSE',
               'score': sqrt(sk.metrics.mean_squared_error(y_validate, y_pred))}
        model_results = model_results.append(dct, ignore_index=True)
        
    return model_number, model_info, model_results

def final_test_model1(train, test):

    x_train = train[['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft']]
    y_train = train[target]
    
    x_test = test[['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft']]
    y_test = test[target]
    
    linreg = LinearRegression(normalize=True).fit(x_train, y_train)
    y_pred = linreg.predict(x_test)
    
    RMSE = sqrt(mean_squared_error(y_test, y_pred))
    
    print('Model 1 RMSE: ', '${:,.2f}'.format(RMSE))
