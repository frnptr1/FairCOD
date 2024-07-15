import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yaml
from typing import List, Optional, Union
import os
from ProjectionBuilder import ProjectionBuilder
from pathlib import Path


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def ExecuteDataMapping(dataframe: pd.DataFrame, columns_added: str, dpc: int, intervals_type:str, backlog:str) -> pd.DataFrame:

    '''
    Handling the dataframe with original value, cleaned, normalized and mapped. This step comes right after the
    aggregation phase where, according to the aggregation and weighting strategy(ies) chosen, new columns have been 
    added to the original dataframe

    dataframe = dataframe to handle containing original data and new columns
    columns_added = columns that have been previously added to the dataframe
    dpc = string reporting the last dpc
    intervals_type = string reporting what kind of intervals to use. 'linear' split the projection space linearly according
    to the number of classes. 'probability' split the projection space according to the probability distribution function of
    each class 
    '''

    # Define the common folder path
    # Get the project root directory
    project_root = Path(__file__).resolve().parent.parent.parent
    mapping_config_file_path = os.path.join(project_root, 'src', 'config', 'MappingModule_config.yaml')
    mapping_config = load_config(file_path=mapping_config_file_path)
    mapping_Fib2Class = mapping_config['mapping_Fib_to_class']

    # Define the number of classes -- total number of expected classes
    classes = mapping_config['mapping_class_to_Fib'].keys()
    num_classes = len(mapping_config['mapping_class_to_Fib'].keys())


    if intervals_type == 'linear': 
        # Define the landmarks of the intervals using linspace
        bins = np.linspace(0, 1, num_classes, endpoint=False)
        bins = np.append(bins, 1)

    elif intervals_type == 'probability':

        projection = ProjectionBuilder(dataframe['CoD'].map(mapping_Fib2Class), backlog=backlog)
        bins = projection.intervals_vector.to_numpy()
        

    # create the column of scaled aggregated value
    mmscaler = MinMaxScaler()

    scaled_aggregation_column_name = f'{columns_added}_scaled'
    FairCOD_class = f'{columns_added}_class'
    FairCOD_Fibonacci = f'{columns_added}_fairCOD'
    
    # minmax scaling of the column added
    minmax_scaled_results = mmscaler.fit_transform(dataframe[columns_added].to_numpy().reshape(-1, 1))
    dataframe[scaled_aggregation_column_name] = minmax_scaled_results

    # create the column class of FairCOD
    dataframe[FairCOD_class] = dataframe[scaled_aggregation_column_name].apply(lambda x: np.digitize(x, bins[:-1]))

    # Create new column with Fibonacci FairCOD values
    dataframe[FairCOD_Fibonacci] = dataframe[FairCOD_class].map(mapping_config['mapping_class_to_Fib'])

    # Finally, convert original CoD Fibonacci to CoD class
    dataframe['CoD_class'] = dataframe['CoD'].map(mapping_Fib2Class)

    return dataframe