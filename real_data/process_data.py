from data.data_processing import *

german_data_clean(
    data_folder='real_data/raw/',
    data_name='german_data_credit',
    save_folder='real_data/processed/',
    save_name='german_clean'
)

adult_data_clean(
    data_folder='real_data/raw/',
    data_name='adult',
    save_folder='real_data/processed/',
    save_name='adult_clean'
)

compas_data_clean(
    data_folder='real_data/raw/',
    data_name='compas-scores-two-years_clean',
    save_folder='real_data/processed/',
    save_name='compas_clean'
)