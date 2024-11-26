# Import built-in libraries
from pathlib import Path
import sys
import pandas as pd
import os
import yaml
import sys
from datetime import datetime
import json


# Get the project root directory
# project_root = Path("./").resolve().parent
project_root = Path(__file__).resolve().parent.parent.parent


# Add the support_modules directory to Python's path
support_modules_path = os.path.join(project_root, 'src', 'support')
sys.path.append(support_modules_path)

# import customer moduels 
import shared
from AggregationModule import OWA, WA, WOWA
from DataPreparation import ExecuteDataPreparation
from MappingModule import ExecuteDataMapping
from ErrorMeasure import evaluate_WeightedMeanAbsoluteError
import WeightingModule
from DBConnectionModule import DBConnection


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
    
    # load config files
    weighting_config_path = os.path.join(project_root, 'src', 'config', 'Weighting_config.yaml')
    db_config_path = os.path.join(project_root, 'src','config', 'DBConnectionModule_config.yaml')
    weighting_config = load_config(weighting_config_path)
    db_config = load_config(db_config_path)
    
    # I load information about data source and aggregation strategy and processing config
    filename_str = 'FairCOD-Dataset-v2.xlsx' # to be parametrized lately
    aggregation = 'WOWA' # to be parametrized lately
    save = True # to save results created as .csv files - to be parametrized lately

    # create ID session
    time_session = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    file_path = os.path.join(project_root, 'results', time_session) # if save == True, file path where csv are stored

    # Create the full path for the database file
    db_path = os.path.join(project_root, 'results', "database-v3.db")

    db = DBConnection(db_position = db_path)

    # columns to drop
    c_to_drop = ['DPC', 'Epic_Name', 'CoD', 'Backlog', 'Margin_to_deadline']

    # make information available globally
    shared.time_session = time_session
    shared.project_root = project_root
    shared.support_modules_path = support_modules_path
    shared.weighting_config_path = weighting_config_path
    shared.db_config_path = db_config_path
    shared.aggregation = aggregation
    shared.db_path = db_path
    shared.columns_to_drop = c_to_drop
    shared.DBConnection = db


    #################
    # PREPROCESSING #
    #################
    input_dfs, criteria_list = ExecuteDataPreparation(filename = filename_str)

    # preparing for processing
    res_dict = {}


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
                                                             w_weights_vector=w,
                                                             p_weights_vector= p,
                                                             dpc=dpc,
                                                             columns_to_drop=None)
                
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


            num_of_epics_in_backlog = res_dict[dpc][backlog_name].shape[0]
            id_session = '_'.join([time_session, str(dpc), backlog_name])

            total_error, wmae, n_wmae = evaluate_WeightedMeanAbsoluteError(df = res_dict[dpc][backlog_name],
                                                                            column_name_true = true_class_column_name,
                                                                            column_name_pred = predicted_class_column_name,
                                                                            backlog = backlog_name,
                                                                            dpc = dpc,
                                                                            id_session=id_session,
                                                                            db=db)
            
            # Insert results in DB the command
            db.DB_InsertLine_results(id_param = id_session, 
                                     aggregation_param = aggregation, 
                                     weighting_param = weighting_strategy,
                                     num_crits_param= n,
                                     num_epics_param = num_of_epics_in_backlog,
                                     dpc_param=dpc, 
                                     backlog_param=backlog_name, 
                                     tot_error_param=total_error,
                                     wmae_param=wmae, 
                                     n_wmae_param=n_wmae)


            if save:
                os.makedirs(name=file_path, exist_ok=True)
                res_dict[dpc][backlog_name].to_csv(os.path.join(file_path, f'{dpc}_{backlog_name}_mapped.csv'))

    # Commit the changes
    db.connection.commit()

    # Close the connection
    db.connection.close()

    # Example usage
    info = f"Successfully completed session {time_session}\nSource of data {filename_str}\nAggregation {aggregation}"

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

    # now display content either in console or in txt file
    if save:
        filename = os.path.join(file_path, f"parameters_{time_session}.txt")
        save_information(filename, info)
    else:
        print(info)

if __name__ == "__main__":

    main()