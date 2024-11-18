import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


dataset_path = 'c:/Users/TCXBQOI/OneDrive - Volkswagen AG/FairCOD/data/FairCOD-Dataset.xlsx'

# read dataset
df = pd.read_excel(dataset_path, sheet_name='Dataset')

# casting values into proper dtype
df.convert_dtypes(convert_string=True)

# drop irrelevant columns for calculation
columns_to_drop = ["DPC", "Effort", "WSJF", "Status in D.P.O. for column DPC 23.3", "In DPC prio for 23.3?", "Comments", "Cost"]
# Columns to normalize
columns_to_normalize = ['EconomicImpact_Income', 'EconomicImpact_CostsSaving', 'EconomicImpact_LossPrevention', 'Margin_to_deadline']


df.drop(columns=columns_to_drop, inplace=True)

# Epic_ID is the index of the dataframe
df.set_index('Epic_ID', inplace=True)

# mapping categorical values of Critical Event type
mapping_critical_event_to_value = {
    'Vehicle SOP affected':1,
    'Digital Platform unavailability':1,
    'Vehicle regulation fulfillment':1,
    'Digital Platform regulation fulfillment':1,
    'Data regulation fulfillment':1,
    'Partner product roadmap accomplishment':0.8,
    'Digital Platform obsolescence':0.8
}


# Create new column with converted values
df['Critical_Event_value'] = df['Critical_Event_type'].map(mapping_critical_event_to_value)


# convert all Nan values into 0
df.replace(to_replace=np.NAN, value=0, inplace=True)


##########################################
##### L2 Normalize unbounded columns #####
##########################################

# Calculate L2 norms for the selected columns
norms = np.linalg.norm(df[columns_to_normalize], axis=0)

# Perform normalization
df[columns_to_normalize] = df[columns_to_normalize].div(norms)

##########################################
##### L2 Normalize unbounded columns #####
##########################################


def topsis(norm_dataset, weights, impacts):
    # Normalize the dataset
    # norm_dataset = dataset / np.sqrt(np.sum(dataset**2, axis=0))

    mapping_impact = { 1: 'positive', -1:'negative'}

    for i in range(norm_dataset.shape[1]):

      print(f'Criteria {norm_dataset.columns[i]} has {mapping_impact[impacts[i]]} on the model')

    # Weighted normalized decision matrix
    #weighted_norm_matrix = norm_dataset * weights * impacts
    weighted_norm_matrix = norm_dataset * impacts

    # Determine positive and negative ideal solutions
    positive_ideal = np.max(weighted_norm_matrix, axis=0)
    negative_ideal = np.min(weighted_norm_matrix, axis=0)

    print('Positive ideal \n', positive_ideal)
    print('Negative ideal \n', negative_ideal)

    # Calculate the Euclidean distances to positive and negative ideal solutions
    positive_distances = np.sqrt(np.sum((weighted_norm_matrix - positive_ideal)**2, axis=1))
    negative_distances = np.sqrt(np.sum((weighted_norm_matrix - negative_ideal)**2, axis=1))

    print('Positive distances \n', positive_distances)

    print('Negative distances \n', negative_distances)

    # Calculate the performance scores
    performance_scores = negative_distances / (positive_distances + negative_distances)

    # Calculate the relative closeness to the ideal solution
    relative_closeness = positive_distances / (positive_distances + negative_distances)

    norm_dataset['performance_scores'] = performance_scores
    norm_dataset['relative_closeness'] = relative_closeness
    norm_dataset['positive_distances'] = positive_distances
    norm_dataset['negative_distances'] = negative_distances

    print(performance_scores)

    # Rank the alternatives based on relative closeness
    #ranked_indices = np.argsort(relative_closeness)[::-1]  # Descending order

    #return relative_closeness, ranked_indices
    return positive_ideal, negative_ideal, norm_dataset



# Columns to discard
columns_to_not_process = ["Epic_Name", "CoD", "Backlog", "Critical_Event_type", "Strategy_Objective_Year", "Strategy_Objective", "Strategy_KeyResult",
                          "UserExperience_RateYourExperience", "UserExperience_FutureCustomerExperience_Initiative", "UserExperience_FutureCustomerExperience_Action",
                          "PlatformUsage_Platform", "PlatformUsage_KPI", "PlatformUsage_Impact", "Markets", "Prospects", "CustomerSatisfaction_L1Category",
                          "CustomerSatisfaction_L2SpecificPart", "CustomerSatisfaction_L2SpecificPart", "CustomerSatisfaction_L3Details", "CustomerSatisfaction_L4MoreInfo",
                          "Deadline"
                          ]


clean_dataset_to_process = df.drop(columns=columns_to_not_process)

# Example weights
weights = np.array(np.repeat(1/clean_dataset_to_process.shape[1],clean_dataset_to_process.shape[1]))

# Example impacts (1 for benefit, -1 for cost)
impacts = np.concatenate( (np.repeat(1, clean_dataset_to_process.shape[1]-1), np.full((1,), -1) ) )

# Run TOPSIS
#scores, rankings = topsis(df_processed, weights, impacts)
pos_ideal, neg_ideal, new_df = topsis(clean_dataset_to_process, weights, impacts)


mmscaler = MinMaxScaler()

# Min-Max scaling to performance scores to distribute them into 0-1 interval
new_df['scaled_performance'] = mmscaler.fit_transform(np.array(new_df['performance_scores']).reshape(-1, 1))


bins = np.array([0.0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 1])
new_df['class_fairCOD'] = new_df['scaled_performance'].apply(lambda x: np.digitize(x, bins))


# Mapping dictionaries
mapping_class_to_Fib = {
    1: 1,
    2: 2,
    3: 3,
    4: 5,
    5: 8,
    6: 13,
    7: 21,
    8: 40,
    9: 100
}


# Mapping dictionaries
mapping_Fib_to_class = {
    1: 1,
    2: 2,
    3: 3,
    5: 4,
    8: 5,
    13: 6,
    21: 7,
    40: 8,
    100: 9
}


# Create new column with converted values
new_df['fairCOD'] = new_df['class_fairCOD'].map(mapping_class_to_Fib)


# Transfer columns created 'performance scores', 'scaled_performance_scores', 'fairCOD', 'class_fairCOD' from new_df to the original df based on indices
df = pd.merge(df, new_df[['performance_scores', 'scaled_performance','fairCOD', 'class_fairCOD']], left_index=True, right_index=True, how='left')

# Create new column with converted values
df['class_COD'] = df['CoD'].map(mapping_Fib_to_class)

df['abs_difference'] = np.abs(df['class_fairCOD'] - df['class_COD'])
df['difference'] = df['class_fairCOD'] - df['class_COD']


print(df[['difference','abs_difference','Epic_Name', 'CoD', 'fairCOD']].sort_values(by='abs_difference',ascending=False).head(20))