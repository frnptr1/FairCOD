import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yaml
from typing import List, Optional, Union
import os



def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def ExecuteDataMapping(dataframe: pd.DataFrame, columns_added: str, dpc: int) -> pd.DataFrame:

    '''
    Handling the dataframe with original value, cleaned, normalized and mapped. This step comes right after the
    aggregation phase where, according to the aggregation and weighting strategy(ies) chosen, new columns have been 
    added to the original dataframe

    dataframe = dataframe to handle containing original data and new columns

    columns_added = columns that have been previously added to the dataframe    
    '''

    # Define the common folder path
    common_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mapping_config_file_path = os.path.join(common_folder_path, 'model', 'MappingModule_config.yaml')
    mapping_config = load_config(file_path=mapping_config_file_path)

    # Define the number of classes
    num_classes = len(mapping_config['mapping_class_to_Fib'].keys())
    # Define the landmarks of the intervals using linspace
    bins = np.linspace(0, 1, num_classes)

    # create the column of scaled aggregated value
    mmscaler = MinMaxScaler()

    scaled_aggregation_column_name = f'{columns_added}_{dpc}_scaled'
    FairCOD_class = f'{columns_added}_{dpc}_class'
    FairCOD_Fibonacci = f'{columns_added}_{dpc}_fairCOD'
    
    # minmax scaling of the column added
    minmax_scaled_results = mmscaler.fit_transform(dataframe[columns_added].to_numpy().reshape(-1, 1))
    dataframe[scaled_aggregation_column_name] = minmax_scaled_results

    # create the column class of FairCOD
    dataframe[FairCOD_class] = dataframe[scaled_aggregation_column_name].apply(lambda x: np.digitize(x, bins, right=True,)+1)

    # Create new column with Fibonacci FairCOD values
    dataframe[FairCOD_Fibonacci] = dataframe[FairCOD_class].map(mapping_config['mapping_class_to_Fib'])

    return dataframe