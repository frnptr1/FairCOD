from AggregationModule import OWA, WA, WOWA
from DataPreparation import ExecuteDataPreparation
from MappingModule import ExecuteDataMapping
from ErrorMeasure import evaluate_WeightedMeanAbsoluteError
import WeightingModule
import pandas as pd
from tabulate import tabulate
import pprint
import os
import yaml
import sys
from datetime import datetime
import json

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_information(filename: str, information: str) -> None:
    """
    Save the given information to a text file.

    Args:
        filename (str): The name of the file to save the information in.
        information (str): The information to save.
    """
    with open(filename, 'w') as file:
        file.write(information)
    print(f"Information saved to {filename}")



def main():
    
    # I load weighting config
    common_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weighting_config_path = os.path.join(common_folder_path, 'model', 'Weighting_config.yaml')
    weighting_config = load_config(weighting_config_path)

    
    # I load information about data source and aggregation strategy and processing config
    filename_str = 'FairCOD-Dataset.xlsx' # to be parametrized lately
    aggregation = 'OWA' # to be parametrized lately
    save = False # to save results created as .csv files - to be parametrized lately
    # create ID session
    id_session = f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    file_path = os.path.join(common_folder_path, 'results', id_session) # if save == True, file path where csv are stored
    # os.makedirs(name=file_path, exist_ok=True)

    #################
    # PREPROCESSING #
    #################
    input_dfs, criteria_list = ExecuteDataPreparation(filename = filename_str)

    # preparing for processing
    res_dict = {}
    c_to_drop = ['DPC', 'Epic_Name', 'CoD', 'Backlog', 'Margin_to_deadline']


    ###########################
    # WEIGHTING & AGGREGATION #
    ###########################
    if aggregation == 'OWA':

        n = len(criteria_list)
        # retrieve weighting strategy according to aggregation chosen
        weighting_strategy = weighting_config[aggregation]['w']
        w = WeightingModule.CreateWeightsArray(weighting_strategy, n)

        for dpc, backlogs in input_dfs.items():

            res_dict[dpc] = {}

            for backlog_name, df in backlogs.items():

                aggregated_df, aggregated_column_name = OWA(dataframe= df, 
                                                            criteria_weights_df=w, 
                                                            dpc=dpc,
                                                            columns_to_drop=c_to_drop)
                
                res_dict[dpc][backlog_name] = aggregated_df


    elif aggregation == 'WOWA':

        n = len(criteria_list)
        # w vector -- apply strategy chosen
        weighting_strategy = weighting_config[aggregation]['w']
        w = WeightingModule.CreateWeightsArray(weighting_strategy, n)        
        # p vector
        p = weighting_config[aggregation]['p']

        for dpc, backlogs in input_dfs.items():

            res_dict[dpc] = {}

            for backlog_name, df in backlogs.items():

                aggregated_df, aggregated_column_name = WOWA(dataframe=df, 
                                                             w_weights_vector=w.to_numpy(),
                                                             p_weights_vector= p,
                                                             dpc=dpc,
                                                             columns_to_drop=c_to_drop)
                
                res_dict[dpc][backlog_name] = aggregated_df


    elif aggregation == 'WA':

        # retrieve weighting strategy according to aggregation chosen -- w vector
        w = weighting_config[aggregation]['w']

        for dpc, backlogs in input_dfs.items():

            res_dict[dpc] = {}

            for backlog_name, df in backlogs.items():

                aggregated_df, aggregated_column_name = WA(dataframe=df,
                                                           weights_dict=w,
                                                           dpc=dpc,
                                                           columns_to_drop=c_to_drop)
                
                res_dict[dpc][backlog_name] = aggregated_df


    ###########
    # MAPPING #
    ###########
    for dpc, backlogs in res_dict.items():

        for backlog_name, df in backlogs.items():

            aggregated_column_name = f"{aggregation}_{weighting_strategy}_{str(dpc)}"

            res_dict[dpc][backlog_name] = ExecuteDataMapping(dataframe= res_dict[dpc][backlog_name], 
                                                             columns_added=aggregated_column_name,
                                                             dpc=dpc,
                                                             intervals_type='probability',
                                                             backlog=backlog_name)
            
            predicted_class_column_name = aggregated_column_name + '_class'
            true_class_column_name = 'CoD_class'

            evaluate_WeightedMeanAbsoluteError(df = res_dict[dpc][backlog_name],
                                               column_name_true = true_class_column_name,
                                               column_name_pred = predicted_class_column_name,
                                               backlog = backlog_name,
                                               dpc = dpc)


            if save:
                os.makedirs(name=file_path, exist_ok=True)
                res_dict[dpc][backlog_name].to_csv(os.path.join(file_path, f'{dpc}_{backlog_name}_mapped.csv'))


    # Example usage
    info = f"Successfully completed session {id_session}\nSource of data {filename_str}\nAggregation {aggregation}"

    # append weighting strategy, if exists
    if weighting_strategy:
        info = "\n".join([info, f"Weighting strategy {weighting_strategy}"])

    # append the final vector w
    if isinstance(w,pd.DataFrame):
        info = "\n".join([info, f"W vector\n{w.to_json(indent=4)}"])    
    elif isinstance(w,dict):
        info = "\n".join([info, f"W vector\n{json.dumps(w, jsonindent=4)}"])

    # append also the vector p in case of WOWA
    if aggregation == "WOWA":
        additional_string = f"P vector {json.dumps(p, indent=4)}\n"
        info = "\n".join([info,additional_string])

    if save:
        filename = os.path.join(file_path, f"parameters_{id_session}.txt")
        save_information(filename, info)
    else:
        print(info)

if __name__ == "__main__":

    main()