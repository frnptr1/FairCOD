import numpy as np
import pandas as pd
from AggregationModule import PiecewiseLinearInterpolator
from scipy.interpolate import interp1d


def create_EqualWeights(n):

  equalweights = np.repeat(1/n, n)

  return np.sort(equalweights)[::-1]




def create_RankSum(n):

  single_weight = lambda x: (2*(n+1-x)) / (n*(n+1))
  orders = np.arange(1,n+1)
  ranksum_weights = single_weight(orders)

  return np.sort(ranksum_weights)[::-1]



def create_FibonacciInverseWeights(n: int) ->np.ndarray :

  '''
  Create FibonacciInverse-based set of weights with dimension n

  Parameters:
    n: number of weights (and consequently length) to store in resulting array

  Returns:
    ndarray: array with weights sorted in descending order

  Scope:
    This function create a set of ordered weights according to the inverse Fibonacci sequence
    adopted in Digital Products development for scoring new epics.
    Fibonacci sequence adopted is {1,2,3,5,8,13,20,40,100}
    The considered sequence is instead {100,40,20,13,8,5,3,2,1}
    This because we want to give more weight to higher values when performing OWA/WOWA aggregation.
    The result is obtained interpolating a function to the set points {i/n, Sum(inverseFib_normalized_i)}
    Once interpolated, the function will receive in input linspace(0,1,n) so that will return the
    corresponding values
  '''
  inverseFib = [100,40,20,13,8,5,3,2,1]
  inverseFib_normalized = np.array(inverseFib) / sum(inverseFib)
  
  # accumulated weights for y-axis values
  y_points = [np.sum(inverseFib_normalized[0:i+1]) for i in range(len(inverseFib_normalized))]
  y_points.insert(0,0)
  # create the equally-spaced values for x-axis
  x_points = np.linspace(0,1,len(y_points))

  # train the interpolator
  #trained_interpolator =PiecewiseLinearInterpolator(x_points, y_points)
  trained_interpolator = interp1d(x_points, y_points, kind='quadratic')
  # where are placed the new x's?
  new_xs = np.linspace(0,1,n+1)

  #FibonacciInverseCumulativeWeights = list(map(lambda x: trained_interpolator.interpolate(x), new_xs))
  FibonacciInverseCumulativeWeights = list(map(lambda x: trained_interpolator(x), new_xs))
  
  
  # retrieve for the cumulative ones the single weights
  FibonacciInverseWeights = []
  for i in range(len(FibonacciInverseCumulativeWeights)-1):
    res = FibonacciInverseCumulativeWeights[i+1] - FibonacciInverseCumulativeWeights[i]
    FibonacciInverseWeights.append(np.round(res, 3))


  return np.sort(FibonacciInverseWeights)[::-1]






def create_RankExponent(n,p):

  orders = np.arange(1,n+1)

  generate_numerator = lambda x : np.power(n-x+1,p)

  numerator_array = generate_numerator(orders)

  denominator = np.sum(numerator_array)

  rankexponent_weights = numerator_array/denominator

  return np.sort(rankexponent_weights)[::-1]





def create_RankReciprocal(n):

  orders = np.arange(1,n+1)
  reciprocals = 1/orders
  denom = np.sum(reciprocals)

  rankreciprocal_weights = reciprocals/denom

  return np.sort(rankreciprocal_weights)[::-1]





def create_RankOrderCentroid(n):

  orders = np.arange(1,n+1)
  reciprocals = 1/orders
  constant = 1/n
  rankordercentroid_weights = np.array([])

  for i in range(n):

    rankordercentroid_weights = np.append(rankordercentroid_weights, constant * np.sum(reciprocals[i:]))

  return np.sort(rankordercentroid_weights)[::-1]



def CreateWeightsArray(weighting_strategy: str, n: int) -> pd.DataFrame:
    # Define a mapping of strategies to their corresponding functions
    strategy_functions = {
        "EqualWeights": create_EqualWeights,
        "RankSum": create_RankSum,
        "RankReciprocal": create_RankReciprocal,
        "RankOrderCentroid": create_RankOrderCentroid
    }
    
    if weighting_strategy in strategy_functions:
        # Call the corresponding function from the dictionary
        weights_vector = strategy_functions[weighting_strategy](n=n)
        return pd.DataFrame({f'{weighting_strategy}': weights_vector})
    
    elif "RankExponent" in weighting_strategy:
        if '_' in weighting_strategy:
            # Split the strategy and extract the exponent parameter
            strategy_base, p = weighting_strategy.split('_')
            weights_vector = create_RankExponent(n=n, p=int(p))
            return pd.DataFrame({f'{weighting_strategy}': weights_vector})
        else:
            print('RankExponent command not properly defined. Structure to follow is "RankExponent_<integer>"')
            return None
    
    else:
        print(f'Unknown weighting strategy: {weighting_strategy}')
        return None
