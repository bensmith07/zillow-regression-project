import os
import pandas as pd
import env

def zillow_2017_data():

#TODO: adjust SQL query to acquire only properties with a TRANSACTION in 2017
    
    filename = 'zillow.csv'
    
    if os.path.exists(filename):
        print('Reading from local CSV...')
        return pd.read_csv(filename)
    
    url = env.get_db_url('zillow')
    sql = '''
            SELECT bedroomcnt, 
                   bathroomcnt, 
                   calculatedfinishedsquarefeet, 
                   taxvaluedollarcnt, 
                   yearbuilt, 
                   taxamount,
                   fips 
              FROM properties_2017
                LEFT JOIN propertylandusetype USING (propertylandusetypeid)
              WHERE propertylandusedesc IN ("Single Family Residential", 
                                            "Inferred Single Family Residential")
                  AND parcelid IN (
                                   SELECT properties_2017.parcelid
                                     FROM properties_2017
                                       JOIN predictions_2017 USING(parcelid)
                                     WHERE transactiondate LIKE "2017%%"
                                  );
            '''
    
    print('No local file exists\nReading from SQL database...')
    df = pd.read_sql(sql, url)

    print('Saving to local CSV... ')
    df.to_csv(filename, index=False)
    
    return df