import os
import pandas as pd
import env

def zillow_2017_data():
    
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
                    fips 
              FROM properties_2017
                JOIN propertylandusetype USING (propertylandusetypeid)
                JOIN predictions_2017 USING(parcelid)
              WHERE propertylandusedesc IN ("Single Family Residential", 
                                            "Inferred Single Family Residential")
                AND transactiondate LIKE "2017%%";
            '''
    
    print('No local file exists\nReading from SQL database...')
    df = pd.read_sql(sql, url)

    print('Saving to local CSV... ')
    df.to_csv(filename, index=False)
    
    return df