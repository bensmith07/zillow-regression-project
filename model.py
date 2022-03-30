import pandas as pd
import sklearn as sk
from math import sqrt
from sklearn.linear_model import LinearRegression, LassoLars 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def determine_regression_baseline(train, target):
    '''
    This function takes in a train sample and a continuous target variable label and 
    determines whether the mean or median performs better as a baseline prediction. 
    '''
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
    # print the results
    print(f'The highest performing baseline is the {baseline} target value.')

def run_baseline(train,
                 validate,
                 target,
                 model_number,
                 model_info,
                 model_results):
    '''
    This function performs the operations required for storing information about baseline performance for
    a regression model.
    '''

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

    # create a dictionary containing information about the baseline's performance on train
    dct = {'model_number': model_number, 
           'sample_type': 'train', 
           'metric_type': 'RMSE',
           'score': sqrt(sk.metrics.mean_squared_error(y_train, y_pred))}
    # append that dictionary to the model_results dataframe
    model_results = model_results.append(dct, ignore_index=True)


    # establish baseline predictions for validate sample
    y_pred = baseline_pred = pd.Series(validate[target].mean()).repeat(len(validate))

    # create a dictionary containing information about the baseline's performance on validate
    dct = {'model_number': model_number, 
           'sample_type': 'validate', 
           'metric_type': 'RMSE',
           'score': sqrt(sk.metrics.mean_squared_error(y_validate, y_pred))}
    # append that dictionary to the model results dataframe
    model_results = model_results.append(dct, ignore_index=True)
    
    # reset the model_number to 0 to be changed in each subsequent modeling iteration
    model_number = 0
    
    return model_number, model_info, model_results

def run_OLS(train, validate, target, model_number, model_info, model_results):
    '''
    This function creates various OLS regression models and stores infomation about their performance
    for later evaluation.
    '''

    features1 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft']
    features2 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age']
    features3 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111']
    features4 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111',
                 'scaled_garage_sqft']
    features5 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111',
                 'scaled_garage_sqft', 'scaled_pools']
    features6 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111',
                 'scaled_garage_sqft', 'scaled_pools', 'scaled_lot_sqft']
    feature_combos = [features1, features2, features3, features4, features5, features6]

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

def run_LassoLars(train, validate, target, model_number, model_info, model_results):
    '''
    This function creates various LASSO + LARS regression models and stores infomation about their performance
    for later evaluation.
    '''

    features1 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft']
    features2 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age']
    features3 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111']
    features4 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111',
                 'garage_sqft']
    features5 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111',
                 'scaled_garage_sqft', 'scaled_pools']
    features6 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111',
                 'scaled_garage_sqft', 'scaled_pools', 'scaled_lot_sqft']
    feature_combos = [features1, features2, features3, features4, features5, features6]

    # set alpha hyperparameter
    alpha = 1

    for features in feature_combos:


        # establish model number
        model_number += 1

        #establsh model type
        model_type = 'LASSO + LARS'

        # store info about the model

        # create a dictionary containing the features and hyperparameters used in this model instance
        dct = {'model_number': model_number,
               'model_type': model_type,
               'features': features,
               'alpha': alpha}
        # append that dictionary to the model_info dataframe
        model_info = model_info.append(dct, ignore_index=True)

        #split the samples into x and y
        x_train = train[features]
        y_train = train[target]

        x_validate = validate[features]
        y_validate = validate[target]

        # create the model object and fit to the training sample
        linreg = LassoLars(alpha=alpha).fit(x_train, y_train)

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

def run_PolyReg(train, validate, target, model_number, model_info, model_results):
    '''
    This function creates various Polynomial Regression models and stores infomation about their performance
    for later evaluation.
    '''

    features1 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft']
    features2 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111', 'scaled_garage_sqft', 'scaled_pools', 'scaled_lot_sqft']
    feature_combos = [features1, features2]

    for features in feature_combos:
        for degree in range(2,6):

            # establish model number
            model_number += 1

            #establsh model type
            model_type = 'Polynomial Regression'

            # store info about the model

            # create a dictionary containing the features and hyperparameters used in this model instance
            dct = {'model_number': model_number,
                   'model_type': model_type,
                   'features': features,
                   'degree': degree}
            # append that dictionary to the model_info dataframe
            model_info = model_info.append(dct, ignore_index=True)

            #split the samples into x and y
            x_train = train[features]
            y_train = train[target]

            x_validate = validate[features]
            y_validate = validate[target]

            # create a polynomial features object
            pf = PolynomialFeatures(degree=degree)

            # fit and transform the data
            x_train_poly = pf.fit_transform(x_train)
            x_validate_poly = pf.fit_transform(x_validate)

            # create the model object and fit to the training sample
            linreg = LinearRegression().fit(x_train_poly, y_train)

            # make predictions for the training sample
            y_pred = linreg.predict(x_train_poly)
            sample_type = 'train'

            # store information about model performance
            # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
            dct = {'model_number': model_number, 
                'sample_type': sample_type, 
                'metric_type': 'RMSE',
                'score': sqrt(sk.metrics.mean_squared_error(y_train, y_pred))}
            model_results = model_results.append(dct, ignore_index=True)

            # make predictions for the validate sample
            y_pred = linreg.predict(x_validate_poly)
            sample_type = 'validate'

            # store information about model performance
            # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
            dct = {'model_number': model_number, 
                'sample_type': sample_type, 
                'metric_type': 'RMSE',
                'score': sqrt(sk.metrics.mean_squared_error(y_validate, y_pred))}
            model_results = model_results.append(dct, ignore_index=True)
        
    return model_number, model_info, model_results


def final_test_model18(train, test, target):
    '''
    This function recreates the regression model previously found to perform with the smallest error, then 
    evaluates that model on the test sample and prints the resulting RMSE.
    '''
    # establish x-train with the appropriate set of features
    x_train = train[['scaled_bedrooms',
                    'scaled_bathrooms',
                    'scaled_sqft',
                    'scaled_age',
                    'enc_fips_06059',
                    'enc_fips_06111',
                    'scaled_garage_sqft',
                    'scaled_pools',
                    'scaled_lot_sqft']]
    # establish y train as the target values
    y_train = train[target]
    
    # establish x-test with the appropriate set of features
    x_test = test[['scaled_bedrooms',
                    'scaled_bathrooms',
                    'scaled_sqft',
                    'scaled_age',
                    'enc_fips_06059',
                    'enc_fips_06111',
                    'scaled_garage_sqft',
                    'scaled_pools',
                    'scaled_lot_sqft']]
    # establish y_test as the target values
    y_test = test[target]

    # create a polynomial features object
    pf = PolynomialFeatures(degree=4)

    # fit and transform x_train and x_test
    x_train_poly = pf.fit_transform(x_train)
    x_test_poly = pf.fit_transform(x_test)
    
    # create and fit the model on the training data
    linreg = LinearRegression(normalize=True).fit(x_train_poly, y_train)
    # create predictions on the test sample
    y_pred = linreg.predict(x_test_poly)
    # compute the rmse performance metric
    RMSE = sqrt(mean_squared_error(y_test, y_pred))
    #display the results
    print('Model 3 RMSE: ', '${:,.2f}'.format(RMSE))

def display_model_results(model_results):
    '''
    This function takes in the model_results dataframe created in the Model stage of the 
    Zillow Regression analysis project. This is a dataframe in tidy data format containing the following
    data for each model created in the project:
    - model number
    - metric type (accuracy, precision, recall, f1 score)
    - sample type (train, validate)
    - score (the score for the given metric and sample types)
    The function returns a pivot table of those values for easy comparison of models, metrics, and samples. 
    '''
    # create a pivot table of the model_results dataframe
    # establish columns as the model_number, with index grouped by metric_type then sample_type, and values as score
    # the aggfunc uses a lambda to return each individual score without any aggregation applied
    return model_results.pivot_table(columns='model_number', 
                                     index=('metric_type', 'sample_type'), 
                                     values='score',
                                     aggfunc=lambda x: x)

def get_best_model_results(model_results, n_models=3):
    '''
    This function takes in the model_results dataframe created in the Modeling stage of the 
    TelCo Churn analysis project. This is a dataframe in tidy data format containing the following
    data for each model created in the project:
    - model number
    - metric type (accuracy, precision, recall, f1 score)
    - sample type (train, validate)
    - score (the score for the given metric and sample types)

    The function identifies the {n_models} models with the highest scores for the given metric
    type, as measured on the validate sample.

    It returns a dataframe of information about those models' performance in the tidy data format
    (as described above). 

    The resulting dataframe can be fed into the display_model_results function for convenient display formatting.
    '''
    # create an array of model numbers for the best performing models
    # by filtering the model_results dataframe for only validate scores
    best_models = (model_results[(model_results.sample_type == 'validate')]
                                                 # sort by score value in ascending order
                                                 .sort_values(by='score', 
                                                              ascending=True)
                                                 # take only the model number for the top n_models
                                                 .head(n_models).model_number
                                                 # and take only the values from the resulting dataframe as an array
                                                 .values)
    # create a dataframe of model_results for the models identified above
    # by filtering the model_results dataframe for only the model_numbers in the best_models array
    # TODO: make this so that it will return n_models, rather than only 3 models
    best_model_results = model_results[(model_results.model_number == best_models[0]) 
                                     | (model_results.model_number == best_models[1]) 
                                     | (model_results.model_number == best_models[2])]

    return best_model_results

