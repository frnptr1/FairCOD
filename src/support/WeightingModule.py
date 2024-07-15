import numpy as np
import pandas as pd



def create_EqualWeights(n):

  equalweights = np.repeat(1/n, n)

  return np.sort(equalweights)[::-1]




def create_RankSum(n):

  single_weight = lambda x: (2*(n+1-x)) / (n*(n+1))
  orders = np.arange(1,n+1)
  ranksum_weights = single_weight(orders)

  return np.sort(ranksum_weights)[::-1]




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
