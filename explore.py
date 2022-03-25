import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_variable_pairs(train):
    quant_features = [col for col in train.columns if train[col].dtype != object]
    feature_combos = list(itertools.combinations(quant_features, 2))
    for combo in feature_combos:
        sns.lmplot(x=combo[0], y=combo[1], data=train, line_kws={'color': 'red'})
        plt.show()
        
def months_to_years(df):
    df['tenure_years'] = df.tenure_months // 12
    return df

def plot_categorical_and_continuous_vars(train, categ_vars, cont_vars):    
    for cont_var in cont_vars:
        for categ_var in categ_vars:

            plt.figure(figsize=(30,10))
            
            # barplot of average values
            plt.subplot(131)
            sns.barplot(data=train,
                        x=categ_var,
                        y=cont_var)
            plt.axhline(train[cont_var].mean(), 
                        ls='--', 
                        color='black')
            plt.title(f'{cont_var} by {categ_var}', fontsize=14)
            
            # box plot of distributions
            plt.subplot(132)
            sns.boxplot(data=train,
                          x=categ_var,
                          y=cont_var)
            
            # swarmplot of distributions
            
            # for larger datasets, use a sample of n=1000
            if len(train) > 1000:
                train_sample = train.sample(1000)

                plt.subplot(133)
                sns.swarmplot(x=categ_var,
                              y=cont_var,
                              data=train_sample)
            
            # for smaller datasets, plot all data
            else:
                plt.subplot(133)
                sns.swarmplot(x=categ_var,
                              y=cont_var,
                              data=train)
            plt.show()

def correlation_test(data_for_category_1, data_for_category_2, alpha=.05):
    print(f'H0: There is no linear correlation between {data_for_category_1.name} and {data_for_category_2.name}.')
    print(f'H1: There is a linear correlation between {data_for_category_1.name} and {data_for_category_2.name}.')
    r, p = stats.pearsonr(data_for_category_1, data_for_category_2)
    print('\nr = ', round(r, 2))
    print('p = ', round(p, 3))
    if p < alpha:
        print('\nReject H0')
    else:
        print('\nFail to Reject H0')
