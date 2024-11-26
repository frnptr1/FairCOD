import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict
import shared


def single_line_owa(item: pd.Series, weights_array: Union[np.ndarray, pd.Series, List]) -> np.ndarray:

    # convert weights_array to numpy array
    if isinstance(weights_array, np.ndarray):
        pass
    elif isinstance(weights_array, pd.Series):
        weights_array = weights_array.to_numpy()
    elif isinstance(weights_array, list):
        wights_array = np.array(wights_array)

    row_array = item.sort_values(ascending=False).to_numpy()

    res = np.sum(row_array * weights_array)

    return res


def OWA(dataframe: pd.DataFrame, criteria_weights_df: pd.DataFrame, dpc, columns_to_drop: Optional[List]) -> pd.DataFrame:


    # if there is the need to drop some support columns
    # if columns_to_drop:
    #    dataframe.drop(columns=columns_to_drop, axis=1, inplace=True)

    # loop over weights dataframe by columns
    for weighting_strategy in criteria_weights_df.columns:
        
        # setup name of new columns that will be created
        aggregation_column_name = f'OWA_{weighting_strategy}_{dpc}'

        # retrieve weights array
        weights_array = criteria_weights_df[weighting_strategy]

        # create the column of aggregated values
        dataframe[aggregation_column_name] = dataframe.drop(columns=columns_to_drop, axis=1).apply(single_line_owa, args=(weights_array,), axis=1)

    return dataframe, aggregation_column_name


def single_line_weighted_avg(item: pd.Series, weights: dict) -> np.ndarray:

    # convert weights_array to numpy array
    # if isinstance(weights_array, np.ndarray):
    #     pass
    # elif isinstance(weights_array, pd.Series):
    #     weights_array = weights_array.to_numpy()
    # elif isinstance(weights_array, list):
    #     wights_array = np.array(wights_array)

    row_values = np.array(item)

    val = 0
    for index, value in item.items():

        val += value*weights[index]

    return val


def WA(dataframe: pd.DataFrame, weights_dict: Dict, dpc, columns_to_drop: Optional[List]) -> pd.DataFrame:

    # setup name of new columns that will be created
    aggregation_column_name = f'WA_{dpc}'

    # create the column of aggregated values
    dataframe[aggregation_column_name] = dataframe.drop(columns=columns_to_drop, axis=1).apply(single_line_weighted_avg, args=(weights_dict,), axis=1)

    return dataframe, aggregation_column_name


def calculate_interpolation_params(x_points, y_points):
    """
    Calculate the slope (m) and intercept (b) for each interval
    defined by x_points and y_points.

    Parameters:
    x_points (array-like): x coordinates of the points
    y_points (array-like): y coordinates of the points

    Returns:
    list of tuples: Each tuple contains (m, b) for the interval
    """
    m_b_params = []
    for i in range(1, len(x_points)):
        x0, x1 = x_points[i-1], x_points[i]
        y0, y1 = y_points[i-1], y_points[i]
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m * x0
        m_b_params.append((m, b))
    return m_b_params


class PiecewiseLinearInterpolator:
    def __init__(self, x_points, y_points):
        self.x_points = np.array(x_points)
        self.y_points = np.array(y_points)
        self.params = calculate_interpolation_params(x_points, y_points)

    def interpolate(self, x_new):
        """
        Interpolate the y value for a new x point based on the
        piecewise linear interpolation.

        Parameters:
        x_new (float): The new x value to interpolate

        Returns:
        float: The interpolated y value
        """
        if x_new < self.x_points[0] or x_new > self.x_points[-1]:
            raise ValueError("x_new is out of the interpolation range.")

        # Find the interval x_new falls into
        for i in range(1, len(self.x_points)):
            if self.x_points[i-1] <= x_new <= self.x_points[i]:
                m, b = self.params[i-1]
                y_new = m * x_new + b
                return y_new
        raise ValueError("x_new did not fall into any interval. This should not happen.")


def single_line_wowa(item: pd.Series, interpolator: PiecewiseLinearInterpolator, p_vector: Dict, dpc:Optional[Union[float,int]]):

    # set everything ready to write aggregation results to DB
    epic_name = item.Epic_Name
    backlog = item.Backlog
    cod = item.CoD
    epic_id = item.name
    dpc = dpc
    session_id = f'{shared.time_session}_{dpc}_{backlog}'
    project_id = f'{session_id}_{epic_id}'
    aggregation = shared.aggregation
    cols_to_drop = shared.columns_to_drop

    # open the connection to the DB
    db = shared.DBConnection

    # once retrieve alla the info needed, clean the Series from unused cols
    item.drop(labels=cols_to_drop, inplace=True)
    
    # Build vector p for the item in input
    # vector p is tailored according to the line in input.
    # the values contained in it are always the same, but depending on the point of strengh of the
    # passed item, the values are ordered accordingly
    p_epic = []

    for col_name, val in item.sort_values(ascending=False).items():

        p_epic.append(p_vector[col_name])

    wowa_value = 0

    # print(f'### Evaluation epic {epic_id} ###')

    # Now that I the vector of p is ready, I can go on with the calculation of the weights w*
    for idx, (col_name, val) in enumerate(item.sort_values(ascending=False).items()):
        
        term_1 = np.sum(p_epic[0:idx+1])
        term_2 = np.sum(p_epic[0:idx])

        mapped_term_1 = interpolator.interpolate(term_1)
        mapped_term_2 = interpolator.interpolate(term_2)

        w_wowa = mapped_term_1-mapped_term_2

        wowa_term = val*w_wowa

        wowa_value += wowa_term

        # print(f'Position {idx:>3} \n -- Criteria {col_name:>50} -- term_1 {term_1:>5.2f} -- term_2 {term_2:>5.2f} -- mapped term_1 {mapped_term_1:>5.2f} -- mapped term_2 {mapped_term_2:>5.2f} -- w_{idx} {w_wowa:>5.2f} -- crit value {val:>5.2f} -- wowa_term {wowa_term:>5.2f}')

    print(project_id)

    db.DB_InsertLine_aggregation(session_id_param = session_id, 
                                 project_id = project_id, 
                                 epic_name_param = epic_name, 
                                 epic_id_param = epic_id, 
                                 aggregated_value_param = wowa_value, 
                                 aggregation_param = aggregation , 
                                 backlog_param = backlog, 
                                 true_cod_param = cod)

    db.connection.commit()

    print(f'### WOWA value for epic {epic_id}  is {wowa_value} ###')

    return wowa_value


def WOWA(dataframe: pd.DataFrame, w_weights_vector: Union[List, pd.DataFrame, np.ndarray], p_weights_vector: Dict, dpc, columns_to_drop: Optional[List]) -> pd.DataFrame:

    # Implementation of Weighted Ordered Weighted Average operator, as defined in 
    # Torra, Vicenç, and Yasuo Narukawa. Modeling decisions: information fusion and aggregation operators. Springer Science & Business Media, 2007.
    # Section 6.1.3

    # passed as pandas dataframe to take the column name
    weighting_strategy = w_weights_vector.columns[0]
    # setup name of new columns that will be created
    aggregation_column_name = f'WOWA_{weighting_strategy}_{dpc}'

    # then converted into numpy array as before
    w_weights_vector = w_weights_vector.to_numpy()

    w_acc = [np.sum(w_weights_vector[0:i+1]) for i in range(len(w_weights_vector))]
    w_acc.insert(0,0)


    # Example usage
    x_points = np.linspace(0,1,len(w_acc))
    y_points = w_acc

    # the trained interpolator is nothing but the interpolation function w* 
    trained_interpolator = PiecewiseLinearInterpolator(x_points, y_points)

    # create the column of aggregated values
    if columns_to_drop:
        dataframe[aggregation_column_name] = dataframe.drop(columns=columns_to_drop, axis=1).apply(single_line_wowa, args=(trained_interpolator,p_weights_vector, dpc), axis=1)
    else:
        dataframe[aggregation_column_name] = dataframe.apply(single_line_wowa, args=(trained_interpolator,p_weights_vector, dpc), axis=1)

    return dataframe, aggregation_column_name
