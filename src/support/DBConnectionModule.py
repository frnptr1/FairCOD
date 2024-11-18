import sqlite3
from typing import Union
from pathlib import Path
import os
import yaml

class DBConnection():

    def __init__(self, db_position:str):
        
        # where the .db file is created and placed
        self.db_filepath = db_position
        self.connection = sqlite3.connect(db_position)
        self.cursor = self.connection.cursor()

        # Get the project root directory
        self.project_root = Path(__file__).resolve().parent.parent.parent
        # load configuration file where queries are stored
        self.db_config_path = os.path.join(self.project_root, 'src','config', 'DBConnectionModule_config.yaml')
        self.db_config = self.load_config()

        # init table result
        if 'sql_result_create_table' in self.db_config.keys():
            self.cursor.execute(self.db_config['sql_result_create_table'])

        # init table cod_estimation
        if 'sql_cod_estimation_create_table' in self.db_config.keys():
            self.cursor.execute(self.db_config['sql_cod_estimation_create_table'])


    def load_config(self):
        with open(self.db_config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config


    def DB_InsertLine_results(self, id_param, aggregation_param, weighting_param, num_crits_param, num_epics_param, dpc_param, backlog_param, tot_error_param, wmae_param, n_wmae_param):
        packed_params = (id_param, aggregation_param,  weighting_param, num_crits_param, num_epics_param, dpc_param, backlog_param, tot_error_param, wmae_param, n_wmae_param) 
        self.cursor.execute(self.db_config['sql_result_insert_into'], packed_params)

    def DB_InsertLine_cod_estimation(self, id_param, results_id, epic_name_param, epic_id_param, true_cod_param, pred_cod_param, class_difference_param, error_param):
        
        packed_params = (id_param, results_id, epic_name_param, epic_id_param, true_cod_param, pred_cod_param, class_difference_param, error_param) 
        self.cursor.execute(self.db_config['sql_cod_estimation_insert_into'], packed_params)
