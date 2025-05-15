import pandas as pd
import os

pd.set_option('display.max_columns', None)


def german_data_clean(data_folder, data_name, save_folder, save_name):

    # load data file
    df = pd.read_csv(data_folder + data_name + '.csv', header=0)
    
    print(df.head())

    
    df['sex'] = [1 if (x == 'male') else 0 for x in df['sex']]
    
    df['age'] = [0 if (x <= 25) else 1 for x in df['age']]

    df['housing'] = [1 if (x == 'own') else 0 for x in df['housing']]
    df['employment-since'] = [1 if (x in [">=7years", "4<= <7 years"]) else 0 for x in df['employment-since']]
    
    df = df[['sex', 'age', 'housing', 'class-label', 'employment-since']]
    
    print(df.head())

    
    df.to_csv(save_folder + save_name + '.csv')
    
    
def adult_data_clean(data_folder, data_name, save_folder, save_name):

    # load data file
    dataframe = pd.read_csv(data_folder + data_name + '.csv', header=0, na_values='?')
                
    # drop rows with missing
    dataframe = dataframe.dropna()
    dataframe.reset_index(drop=True, inplace=True)
    
    dataframe['marital-status'] = [1 if x in ('Married-AF-spouse','Married-civ-spouse','Married-spouse-absent') 
                                    else 0 for x in dataframe['marital-status']]
    dataframe['relationship'] = [1 if x in ('Husband', 'Wife') 
                                    else 0 for x in dataframe['relationship']]
    dataframe['race'] = [1 if (x == 'White') else 0 for x in dataframe['race']]
    dataframe['gender'] = [1 if (x == 'Male') else 0 for x in dataframe['gender']]
    dataframe['workclass'] = [0 if (x != 'Private') else 1 for x in dataframe['workclass']]
    dataframe['capital-gain'] = [1 if (x > 5000) else 0 for x in dataframe['capital-gain']]
    
    dataframe['income'] = [1 if (x == '>50K') else 0 for x in dataframe['income']]
    
    dataframe = dataframe[['race', 'marital-status', 'relationship', 'gender', 'workclass', 'capital-gain', 'income']]
            
    dataframe.to_csv(save_folder + save_name + '.csv')
    
    
def compas_data_clean(data_folder, data_name, save_folder, save_name):

    # load data file
    
    df = pd.read_csv(data_folder + data_name  + '.csv', header=0, na_values='?')
    
    print(df.shape)

    df = df[(df['race']=='African-American') | (df['race'] == "Caucasian")]

    df['race'] = [0 if v == 'African-American' else 1 for v in df['race']]
    df['score_text'] = [0 if v == 'Low' else 1 for v in df['score_text']]
    df['v_score_text'] = [0 if v == 'Low' else 1 for v in df['v_score_text']]
    df['priors_count'] = [0 if v < 5 else 1 for v in df['priors_count']]
    df['age_cat'] = [0 if v == 'Less than 25' else 1 for v in df['age_cat']]
    df['two_year_recid'] = [0 if v == 0 else 1 for v in df['two_year_recid']]
    
    df = df[['race', 'age_cat', 'score_text', 'v_score_text', 'priors_count', 'age_cat', 'two_year_recid']]

    df.to_csv(save_folder + save_name + '.csv')