import pandas as pd
import sklearn as sk
from math import sqrt

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
    
    # return
    if return_results_df:
        return results, baseline
    else:
        return baseline