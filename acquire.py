import os
import pandas as pd
import env

def zillow_2017_data():
    '''
    This function uses a SQL query to access the Codeup MySQL database and join 
    together all the relevant data using the following tables:
      - properties_2017
      - propertylandusetype
      - predictions_2017
    The data obtained includes all properties in the dataset which had a transaction in 2017.
    The function caches a csv in the local directory for later use. 
    '''
    # establish a filename for the local csv
    filename = 'zillow.csv'
    # check to see if a local copy already exists. 
    if os.path.exists(filename):
        print('Reading from local CSV...')
        # if so, return the local csv
        return pd.read_csv(filename)
    # otherwise, pull the data from the database:
    # establish database url
    url = env.get_db_url('zillow')
    # establish query
    sql = '''
            SELECT bedroomcnt, 
                    bathroomcnt, 
                    calculatedfinishedsquarefeet, 
                    taxvaluedollarcnt, 
                    yearbuilt, 
                    fips,
                    garagetotalsqft,
                    poolcnt, 
                    lotsizesquarefeet 
              FROM properties_2017
                JOIN propertylandusetype USING (propertylandusetypeid)
                JOIN predictions_2017 USING(parcelid)
              WHERE propertylandusedesc IN ("Single Family Residential", 
                                            "Inferred Single Family Residential")
                AND transactiondate LIKE "2017%%";
            '''
    print('No local file exists\nReading from SQL database...')
    # query the database and return the resulting table as a pandas dataframe
    df = pd.read_sql(sql, url)
    # save the dataframe to the local directory as a csv
    print('Saving to local CSV... ')
    df.to_csv(filename, index=False)
    # return the resulting dataframe
    return df