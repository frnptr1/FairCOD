import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yaml
import os
from typing import Dict
from pathlib import Path

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def join_lists_keep_order(list1, list2):
    # Create a set of elements from list2 for faster lookup
    set2 = set(list2)

    # Initialize an empty result list
    result = []

    # Iterate over elements in list1 while maintaining the order
    for item in list1:
        if item in set2 and item not in result:
            # Append the element to the result list if it's present in list2 and not already in the result list
            result.append(item)

    return result


def function_mapping_critical_event_to_value(df, mapping):

  # Create new column with converted values
  df['Critical_Event_value'] = df['Critical_Event_type'].map(mapping)

  return df


def normalize_column(df, columns_to_norm):

    # check for null columns
    columns_all_zero = df.columns[(df == 0).all()].tolist()
    difference = [item for item in columns_to_norm if item not in columns_all_zero]

    df_to_normalize = df[difference].astype(float)

    # Calculate L2 norms for the selected columns
    norms = np.linalg.norm(df_to_normalize, axis=0)


    # Perform normalization
    df[difference] = df[difference].div(norms)

    return df


def drop_rows_by_value(df: pd.DataFrame, column:str='SOP_EOP_Relevant', value:str='Yes'):
    return df[df[column] != value]
 


def transform_Margin_to_deadline(df):
  """
  The column Margin to deadline contains the integer number of days between the
  deadline of the critical event type and the fixed today date (corresponding
  to the DPT date). If no critical event is specified, max value of 45077 should
  appear

  Args:
    df:

  Returns:

  """

  mmscaler = MinMaxScaler()
  resulting_array = 1 - mmscaler.fit_transform(np.array(df['Margin_to_deadline']).reshape(-1, 1))
  df['Margin_to_deadline_value'] = resulting_array

  return df


'''
# Main script
if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--data', type=str, required=True, help='The name of the data file to be read (without path).')
    args = parser.parse_args()
'''
def ExecuteDataPreparation(filename: str) -> Dict:
    

    # Get the project root directory
    project_root = Path(__file__).resolve().parent.parent.parent

    # Construct the data file path
    excel_file_path = os.path.join(project_root, 'data', filename)
    config_file_path = os.path.join(project_root, 'src', 'config', 'DataPreparation_config.yaml')

    # Read and process the excel file
    if os.path.exists(excel_file_path):
        
        # read the excel worksheet, if exists
        df = pd.read_excel(excel_file_path, sheet_name='Dataset', index_col='Epic_ID')
        df.convert_dtypes(convert_string=True)

    else:
        print(f"Error: The file '{excel_file_path}' does not exist.")


    # Read and process the config file
    if os.path.exists(config_file_path):
        # Load the config file
        config = load_config(config_file_path)
    else:
        print(f"Error: The file '{config_file_path}' does not exist.")


    clean_dataset_to_process = (df.pipe(function_mapping_critical_event_to_value, config['mapping_critical_event_to_value']).
                                pipe(pd.DataFrame.replace, to_replace=np.NAN, value=0).
                                pipe(drop_rows_by_value, column='SOP_EOP_Relevant', value='Yes').
                                pipe(pd.DataFrame.drop, config['columns_to_drop'], axis=1).
                                pipe(transform_Margin_to_deadline))

    
    # ordered fullset criteria contains, indeed, the ordered criteria according to results of study 1
    # the join with the columns list of the dataframe allows me to understand which subset of the ordered set
    # will be actually adopted in the analysis
    criteria_list = join_lists_keep_order(config['ordered_fullset_criteria'], clean_dataset_to_process.columns.to_list())

    # Create a dictionary to store DataFrames for each group
    dfs = {}
    # this list will help include past dpc epics in the same future backlog
    previous_dpc_included = []

    # Iterate through unique DPC dates
    for date in clean_dataset_to_process['DPC'].unique():
        
        date_dict = {}
        
        # once fixed the DPC date, iterate over the Backlogs (OLA, OVS, Websites)
        for backlog in clean_dataset_to_process['Backlog'].unique():
            
            # retrieve the dataframe to process
            filtered_df = clean_dataset_to_process[(clean_dataset_to_process['DPC'] == date) & (clean_dataset_to_process['Backlog'] == backlog)]

            if not filtered_df.empty:

                # check the condition whether standalone backlog or with previous epics
                if config['standalone']:

                    # in case in dpc xxx only new epics are included
                    date_dict[backlog] = filtered_df
                
                # if condition standalone is false
                else:

                    # add to the current dpc backlog also the backlogs presented in the previous dpc(s)
                    tmp = filtered_df.copy()
                    for previous_dpc in previous_dpc_included:
                        previous_dpc_date = clean_dataset_to_process[(clean_dataset_to_process['DPC'] == previous_dpc) & (clean_dataset_to_process['Backlog'] == backlog)]
                        tmp = pd.concat([tmp, previous_dpc_date], axis=0)

                    date_dict[backlog] = tmp

        # once the DPC date has been successfully introduced, store it in previous_dpc_date list
        previous_dpc_included.append(date)
        # store DPC dictionary
        dfs[date] = date_dict


    # now that the Dataframe has been splitted by DPC and backlong, normaliza columns
    for date in dfs.keys():

        for backlog in dfs[date]:

            dfs[date][backlog] = normalize_column(df=dfs[date][backlog], 
                                                  columns_to_norm=config['columns_to_normalize'])

            

    return dfs, criteria_list