


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from env import host, user, password

###################### Acquire Telco Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_data():
    '''
    This function reads the telco data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
                FROM properties_2017
                WHERE propertylandusetypeid = "261";
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df


def get_zillow_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_df.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('zillow_df.csv')
        
    return df

def clean_zillow_data(df):
    '''
    This function takes in the acquired df, from the new_zillow_data or get_zillow_data functions
    and cleans up the data by dropping necessary rows and/or columns,
    encoding and/or renaming columns, changing datatypes where necessary, etc. 
    '''
    #drop rows with null values
    df = df.dropna()
    
    #Convert the fips columns from float to object since they are not numbers to be calculated, and the year and calculatedsquarefeet column
    # from float to integer since it is not necessary to show decimal places
    df = df.astype({"yearbuilt": int, "fips": object, "calculatedfinishedsquarefeet": int})
    
    # Create dummy columns for fips since each code represents a county
    dummy_df = pd.get_dummies(df[['fips']], dummy_na=False)
    
    #concat on to the original df
    df = pd.concat([df, dummy_df], axis=1)
    
    #Change the column names to be the county names 
    #According to FIPS codes, 06037 is Los Angeles County, 06059 is Orange County, and 06111 is Ventura County
    #Rename columns for better readability and simplicity
    df = df.rename(columns={"fips_6037.0": "la_county", "fips_6059.0": "orange_county", "fips_6111.0": "ventura_county", "bedroomcnt": "bedrooms", "bathroomcnt": "bathrooms", "calculatedfinishedsquarefeet": "total_sqft", "taxvaluedollarcnt": "tax_value", "yearbuilt": "year_built"})
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test

#Handle outliers to see if shape of data changes at all
#Using John Salas function:

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

    