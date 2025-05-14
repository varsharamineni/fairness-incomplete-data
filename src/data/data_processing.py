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