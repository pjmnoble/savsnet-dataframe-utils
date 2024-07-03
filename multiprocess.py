import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

def parallelize_dataframe(df: pd.DataFrame, func, n_cores: int = -1) -> pd.DataFrame:
    """
    :param df: Dataframe to split, apply function in parallel and then reconstruct
    :param func: function to apply to split dataframe parts
    :param n_cores: cores to use (usually number of cores on machine -1)
    :return: reconstructed dataframe where function has been applied to all rows
    """
    if n_cores == -1:
        n_cores = cpu_count()
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

# helper function to unpack kwargs and call func
def apply_func_with_kwargs(split_df, func, kwargs):
    """
    Applies a function with additional keyword arguments to a DataFrame.

    Parameters:
    split_df (pd.DataFrame): A portion of the original DataFrame to process.
    func (callable): The function to apply to the DataFrame split.
    kwargs (dict): Additional keyword arguments to pass to the function.

    Returns:
    pd.DataFrame: The processed DataFrame split.
    """
    return func(split_df, **kwargs)

def parallelize_dataframe_multi_arg(df: pd.DataFrame, func, n_cores: int = -1, **kwargs):
    """
    Splits a DataFrame, applies a function with additional keyword arguments in parallel, 
    and then reconstructs the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to split, apply the function in parallel, and then reconstruct.
    func (callable): The function to apply to each split of the DataFrame.
    n_cores (int): The number of cores to use (defaults to number of cores on machine).
    kwargs (dict): Additional keyword arguments to pass to the function.

    Returns:
    pd.DataFrame: The reconstructed DataFrame where the function has been applied to all rows.
    """
    if n_cores == -1:
        n_cores = cpu_count()
    df_split = np.array_split(df, n_cores)

    with Pool(n_cores) as pool:
        results = pool.starmap(apply_func_with_kwargs, 
                               [(split_df, func, kwargs) for split_df in df_split])

    return pd.concat(results)
