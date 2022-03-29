import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
 
def correlation_test(data_for_category_1, data_for_category_2, alpha=.05):
    '''
    This function takes in data for two variables and performs a pearsons r statistitical test for correlation. 
    It outputs to the console values for r and p and compares them to a given alpha value, then outputs to the 
    console whether or not to reject the null hypothesis based on that comparison. 
    '''
    # display hypotheses
    print(f'H0: There is no linear correlation between {data_for_category_1.name} and {data_for_category_2.name}.')
    print(f'H1: There is a linear correlation between {data_for_category_1.name} and {data_for_category_2.name}.')
    # conduct the stats test and store values for p and r
    r, p = stats.pearsonr(data_for_category_1, data_for_category_2)
    # display the p and r values
    print('\nr = ', round(r, 2))
    print('p = ', round(p, 3))
    # compare p to alpha, display whether to reject the null hypothesis
    if p < alpha:
        print('\nReject H0')
    else:
        print('\nFail to Reject H0')

def value_by_bathrooms(train):
    '''
    This function takes in the zillow train sample and uses seaborn to create box plots of 
    tax_value for each number of bathrooms that exists in the sample. 
    '''
    # establish figure size
    plt.figure(figsize=(10,8))
    # create the plot
    sns.boxplot(data=train,
                  x='bathrooms',
                  y='tax_value')
    # establish title
    plt.title('Value by Number of Bathrooms')
    # display the plot
    plt.show()

def sqft_vs_value(train):
    '''
    This function takes in the zillow train sample and uses seaborn to create a scatter plot
    of tax_value vs square feet, with a best-fit regression line. 
    '''
    # create the plot
    sns.lmplot(x='sqft', 
               y='tax_value', 
               data=train.sample(1000, random_state=42), 
               line_kws={'color': 'red'})
    # establish the title
    plt.title('Value by Square Footage')
    # display the plot
    plt.show()

def value_by_bedrooms(train):
    '''
    This function takes in the zillow train sample and uses seaborn to create box plots of the 
    distribution of tax_value for each number of bedrooms that exists in the sample. 
    '''
    # establish the figure size
    plt.figure(figsize=(10,8))
    # create the plot
    sns.boxplot(data=train,
                  x='bedrooms',
                  y='tax_value')
    # establish plot title
    plt.title('Value by Number of Bedrooms')
    # display the plot
    plt.show() 

def value_correlations(train):
    '''
    This functino takes in the zillow train sample and uses pandas and seaborn to create a
    ordered list and heatmap of the correlations between the various quantitative feeatures and the target. 
    '''
    # create a dataframe of correlation values, sorted in descending order
    corr = pd.DataFrame(train.corr().abs().tax_value).sort_values(by='tax_value', ascending=False)
    # rename the correlation column
    corr.columns = ['correlation (abs)']
    # establish figure size
    plt.figure(figsize=(10,8))
    # creat the heatmap using the correlation dataframe created above
    sns.heatmap(corr, annot=True)
    # establish a plot title
    plt.title('Features\' Correlation with Value')
    # display the plot
    plt.show()