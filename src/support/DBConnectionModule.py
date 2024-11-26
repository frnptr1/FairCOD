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

        # init table aggregation
        if 'sql_aggregation_create_table' in self.db_config.keys():
            self.cursor.execute(self.db_config['sql_aggregation_create_table'])

        # init table wowa
        if 'sql_wowa_create_table' in self.db_config.keys():
            self.cursor.execute(self.db_config['sql_wowa_create_table'])


    def load_config(self):
        with open(self.db_config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config


    # 
    def DB_InsertLine_results(self, id_param, aggregation_param, weighting_param, num_crits_param, num_epics_param, dpc_param, backlog_param, tot_error_param, wmae_param, n_wmae_param):
        
        '''
        exectue INSERT command to write record on results table

        Parameters
            id_param:
            aggregation_param:
            weighting_param:
            num_crits_param: 
            num_epics_param: 
            dpc_param: 
            backlog_param: 
            tot_error_param: 
            wmae_param: 
            n_wmae_param:

        Return:
            None
        
        '''
        
        
        packed_params = (id_param, aggregation_param,  weighting_param, num_crits_param, num_epics_param, dpc_param, backlog_param, tot_error_param, wmae_param, n_wmae_param) 
        self.cursor.execute(self.db_config['sql_result_insert_into'], packed_params)

    def DB_InsertLine_cod_estimation(self, id_param, results_id, epic_name_param, epic_id_param, true_cod_param, pred_cod_param, class_difference_param, error_param):
        
        packed_params = (id_param, results_id, epic_name_param, epic_id_param, true_cod_param, pred_cod_param, class_difference_param, error_param) 
        self.cursor.execute(self.db_config['sql_cod_estimation_insert_into'], packed_params)

    def DB_InsertLine_aggregation(self, 
                                  session_id_param: str, 
                                  project_id: str, 
                                  epic_name_param: str, 
                                  epic_id_param: str, 
                                  aggregated_value_param: float, 
                                  aggregation_param: str, 
                                  backlog_param: str, 
                                  true_cod_param: int) -> None:
        
        '''
        exectue INSERT command to write record on "aggregation" table

        Parameters
            session_id_param: str, 
            project_id: str, 
            epic_name_param: str, 
            epic_id_param: str, 
            aggregated_value_param: float,
            aggregation_param: str, 
            backlog_param: str, 
            true_cod_param: int)

        Return:
            None
        '''
        

        packed_params = (session_id_param,      # 20241120_112845_24.3_Webisites
                         project_id,            # 20241120_112845_24.3_Webisites_LPM-131
                         epic_name_param,       # The most beautiful epic
                         epic_id_param,         # LPM-131
                         aggregated_value_param,# 0.89
                         aggregation_param,     # WOWA
                         backlog_param,         # Websites
                         true_cod_param)        #40

        self.cursor.execute(self.db_config['sql_aggregation_insert_into'], packed_params)


    def DB_InsertLine_wowa(self, 
                           project_id_param: str, 
                           aggregated_value_param: float, 
                           position_param: int, 
                           criteria_param: str, 
                           criteria_value_param: float, 
                           p_weight_current_param: float, 
                           p_weight_previous_param: float, 
                           omega_star_current_param: float, 
                           omega_star_previous_param: float, 
                           wowa_weight_param: float, 
                           weighted_criteria_val_param: float) -> None:
        
        '''
        exectue INSERT command to write record on "wowa" table

        Parameters
            id_param:
            aggregation_param:
            weighting_param:
            num_crits_param: 
            num_epics_param: 
            dpc_param: 
            backlog_param: 
            tot_error_param: 
            wmae_param: 
            n_wmae_param:

        Return:
            None
        
        '''

        packed_params = (project_id_param, 
                         aggregated_value_param, 
                         position_param, 
                         criteria_param, 
                         criteria_value_param, 
                         p_weight_current_param, 
                         p_weight_previous_param, 
                         omega_star_current_param, 
                         omega_star_previous_param, 
                         wowa_weight_param, 
                         weighted_criteria_val_param)

        self.cursor.execute(self.db_config['sql_wowa_insert_into'], packed_params)
        
