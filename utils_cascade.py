#!/usr/bin/env python3
# -*- coding: utf-8 -*-


##########################################    Libraries    ##########################################


import numpy as np
import pandas as pd
import math as math
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import gower
import torch
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from kneebow.rotor import Rotor
from matplotlib.figure import Figure
from minisom import MiniSom
from scipy import optimize
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, OneHotEncoder
from statistics import mean 
from utils import *

# Configurando o nível de registro (pode ser DEBUG, INFO, WARNING, ERROR, ou CRITICAL)
logging.basicConfig(level=logging.INFO)


#####################################################################################################
#####################################################################################################


#### Execution Time (minutes and seconds) ####
def measure_execution_time_min_seg(start_time):
    """
    Calculates and prints the execution time from start_time to the current moment.

    Parameters:
    start_time (float): The measurement start time.

    Return:
    phase_name (str): Name of the phase for which the execution time is being measured.
    """
    end_time = time.time()
    elapsed_time = end_time - start_time 
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_min_seg = f"{minutes}m.{seconds}s"

    return time_min_seg




#### Execution Time (seconds) ####
def measure_execution_time_seg(start_time):
    """
    Calculates and prints the execution time from start_time to the current moment.

    Parameters:
    start_time (float): The measurement start time.

    Return:
    phase_name (str): Name of the phase for which the execution time is being measured.
    """
    end_time = time.time()
    time_seg = end_time - start_time 

    return time_seg




#### Missing Data Mechanism ####
def eraser(df, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values. 
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str, 
        Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str, 
        For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
        If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    
    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """
    X = df.to_numpy()                   
    to_torch = torch.is_tensor(X)       ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    
    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1-p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    else:
        mask = (torch.rand(X.shape) < p_miss).double()
    
    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    
    return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}




#### Split datasets ####
def split_dataset(df):
    """
    Split the dataset into complete dataset (df_filled) and incomplete dataset (df_null)
    
    Parameters
    ----------
    df : dataframe
    
    Returns
    ----------
    df_filled: complete dataset
    df_null  : imcomplete dataset
    """
    aux_df = df.copy()
    aux_df['count_null'] = aux_df.isnull().sum(axis=1)

    # Get the indexes of null rows
    index_null = aux_df.index[aux_df['count_null'] > 0].tolist()
    index_not_null = aux_df.index[aux_df['count_null'] == 0].tolist()

    # Split on df_filled ("base completa") and df_null ("base incompleta")
    df_filled = df.iloc[index_not_null]
    df_null = df.iloc[index_null] 

    return df_filled, df_null




#### Get binarized dataset ####
def get_binarized_df(df_null):
    """
    Binarizes the dataset with null values(df_null). 1 = null values ; 0 = non-null values
    
    Parameters
    ----------
    df_null : dataframe
    
    Returns
    ----------
    df_mask: binarized dataset
    """    
    df_mask = df_null.copy()
    # Replace null by 1 and not null by 0
    df_mask[~df_mask.isnull()] = 0  # not nan
    df_mask[df_mask.isnull()]  = 1  # nan
    
    return df_mask




#### Marking the labels ####
def labels_col_df(df_mask, labels):
    """
    Creates a num_clusters column
    
    Parameters
    ----------
    df_mask : dataframe
    
    Returns
    ----------
    df_mask: dataframe + labels column
    """    
    ## Fazendo a marcação dos clusters
    df_mask['num_cluster'] = labels
    
    return df_mask




#### Correlation ####
def correlation(df):
    """
    Pearson's correlation  
    
    Parameters
    ----------
    df : dataframe
        Dataframe that wants to do the correlation
    
    Returns
    ----------
    correl : correlation matrix
    """
    correl = df.corr()
    return correl




#### Data normalization ####
def normalize_data(df, apply_normalization=True, normalization_type="standard"):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values. 
    
    Parameters
    ----------
    df : dataframe
        Data to be scaled.

    apply_normalization : bool, default=True
        Indicates whether to apply normalization or not.
        
    normalization_type : str, default="standard"
        Indicates the type of scaler to be used. Options are "standard" (default), "minmax", "maxabs", or "hotencoder".
    
    Returns
    ----------
    If apply_normalization is True, returns a dictionary containing:
        scaled: dataframe with normalized data.
    If apply_normalization is False, returns the original dataframe.
    """
    
    if not apply_normalization:
        return df
    
    if normalization_type == "standard":
        scaler = StandardScaler()
    elif normalization_type == "minmax":
        scaler = MinMaxScaler()
    elif normalization_type == "maxabs":
        scaler = MaxAbsScaler()
    else:
        scaler = OneHotEncoder()
    
    scaled_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
    
    return df_scaled




#### Calculate the distance ####
def calc_distance(u, v, type):
  """
  Calculates the distance between two points. 
    
  Parameters
  ----------
  u : input array 
  v : input array
  type : str, 
    Indicates the type of scaler to be used. "euclidean" by default, "manhattan", "cosine" or "mahalanobis"
    
  Returns
  ----------
  A dictionnary containing:
  'dist':  distance between vectors u and v.
  """
        
  if type == "euclidean":
    #dist = 
    return distance.euclidean(u, v)
  elif type == "manhattan":
    return distance.cityblock(u, v)
  #elif type == "cosine":
  else:  
    return distance.cosine(u, v)
  #else:
  #  cm = np.cov(u, v, rowvar=False)         # covariance matrix
  #  np.set_printoptions(precision=4, suppress=True)
  #  icm = np.linalg.inv(cm)                 # inverse covariance matrix
  #  return distance.mahalanobis(u, v, icm)  




#### Calculate the distance ####
def gover_distance(df):
  """
  Calculates the Gover's distance
    
  Parameters
  ----------
  df : dataframe 
    
  Returns
  ----------
  'dist':  distance between vectors u and v.
  """
  distance_matrix = gower.gower_matrix(df)

  return  distance_matrix 




#### Calculate the metrics ####
def reg_metric(y_true, y_pred, type):
    """
    Calculates the difference between the observed and predicted values of a variable.
    
    Parameters
    ----------
    y_true : array
      Ground truth (correct) target values
    y-pred : array
      Estimated target values
    type : str, 
      Indicates the metric we want to calculate. "mse" by default, "rmse" or "r2"
    """

    if type == "mse":
        return mean_squared_error(y_true, y_pred, squared=True)
    elif type == "rmse":
        return mean_squared_error(y_true, y_pred, squared=False)
    else:
        return r2_score(y_true, y_pred)




#### Imputation error metrics ####
def imput_error(y_true, y_pred, type):
    """
    Calculates the error generated by the imputation
    
    Parameters
    ----------
    y_true : array
      Ground truth (correct) target values
    y-pred : array
      Estimated target values
    type : str, 
      Indicates the metric we want to calculate. "mae" by default, "min_abs_error", "mape", "min_abs_perc_error", "sim_error"
    """

    min_y = min(y_true)
    max_y = max(y_true)
    
    if type == "mape":
      return mean_absolute_percentage_error(y_true, y_pred)
    elif type == "min_abs_error":
      min_abs_error = min(abs(y_true - y_pred))
      return min_abs_error
    elif type == "min_abs_perc_error":
      min_abs_perc_error = min(abs(y_true - y_pred) / y_true)
      return min_abs_perc_error
    elif type == "sim_error":
      abs_sup = abs(y_true - y_pred)
      abs_inf = abs(max_y - min_y)
      sim_error = mean(abs_sup / abs_inf)
      return sim_error
    else:
      return mean_absolute_error(y_true, y_pred) 




#### Attribute correlation metrics #### 
def correlation_bias(df_original, df_imputed):
    """
    Calculates the correlation bias generated by the imputation
    
    Parameters
    ----------
    y_true : array
      Ground truth (correct) target values
    y-pred : array
      Estimated target values
    """

    num_col = len(df_original.columns)

    # Original dataset
    correl_real = df_original.corr()
    correl_real_sum = pd.DataFrame(correl_real.apply(lambda x: np.sum(x)), columns=['correl_sum'])
    correl_real_sum['OC'] = correl_real_sum['correl_sum'] / (num_col - 1)       

    # Imputed dataset
    correl_impt = df_imputed.corr()
    correl_impt_sum = pd.DataFrame(correl_impt.apply(lambda x: np.sum(x)), columns=['correl_sum'])
    correl_impt_sum['OC'] = correl_impt_sum['correl_sum'] / (num_col - 1)      

    acb = correl_real_sum['OC'] - correl_impt_sum['OC']        # Correlation Bias of the Attribute (ACB) 
    cb = sum(acb) 

    return cb 




#### Cluster ordering criteria #### 
def cluster_order_criterion(df_mask, order_cluster="noSort"): 
    """
    Determines the order of clusters according to the selected criteria
    
    Parameters
    ----------
    df_mask : dataframe
      Binarized dataframe of incomplete subset

    order_cluster : str
      Indicates the type of ordered to be used. "noSort" by default, "tupleLessMissing", "tupleMoreMissing", "fieldLessMissing", 
      "fieldMoreMissing", "fieldPerTupleLessMissing", "fieldPerTupleMoreMissing", "random"
    
    Returns
    ----------
    'ordered_list':  cluster order list
    """
    
    ncol = len(df_mask.columns) - 1
    df_mask['aux_cell_tupla'] = df_mask.iloc[:,0:ncol].sum(axis=1)     # number null cells per tuple

    # Creating an auxiliary dataframe
    df_ordem = df_mask.groupby(['num_cluster']).agg(qtd_tupla=('num_cluster','count'),
                                                         cel_nulas=('aux_cell_tupla','sum'),
                                                         cel_tupla=('aux_cell_tupla','mean')).reset_index()
    df_ordem['random'] = np.random.randint(0, len(df_ordem), size=len(df_ordem))                                                     

    if order_cluster == "tupleLessMissing":
        aux_ordem = df_ordem.sort_values(by=['qtd_tupla'], ascending=True)
        ordered_list = aux_ordem['num_cluster'].values.tolist()
        return ordered_list
    elif order_cluster == "tupleMoreMissing":
        aux_ordem = df_ordem.sort_values(by=['qtd_tupla'], ascending=False)
        ordered_list = aux_ordem['num_cluster'].values.tolist()
        return ordered_list
    elif order_cluster == "fieldLessMissing":
        aux_ordem = df_ordem.sort_values(by=['cel_nulas'], ascending=True)
        ordered_list = aux_ordem['num_cluster'].values.tolist()
        return ordered_list
    elif order_cluster == "fieldMoreMissing":
        aux_ordem = df_ordem.sort_values(by=['cel_nulas'], ascending=False)
        ordered_list = aux_ordem['num_cluster'].values.tolist()
        return ordered_list
    elif order_cluster == "fieldPerTupleLessMissing":
        aux_ordem = df_ordem.sort_values(by=['cel_tupla'], ascending=True)
        ordered_list = aux_ordem['num_cluster'].values.tolist()
        return ordered_list
    elif order_cluster == "fieldPerTupleMoreMissing":
        aux_ordem = df_ordem.sort_values(by=['cel_tupla'], ascending=False)
        ordered_list = aux_ordem['num_cluster'].values.tolist()
        return ordered_list
    elif order_cluster == "random":
        aux_ordem = df_ordem.sort_values(by=['random'], ascending=True)
        ordered_list = aux_ordem['num_cluster'].values.tolist()
        return ordered_list
    else:
        aux_ordem = df_ordem.sort_values(by=['num_cluster'], ascending=True)
        ordered_list = aux_ordem['num_cluster'].values.tolist()
        return ordered_list 




#### Attribute ordering criteria #### 
def attribute_order_criterion(df_mask, df_correl, order_column="noSort"): 
    """
    Determines the order of attributes according to the selected criteria
    
    Parameters
    ----------
    df_mask : dataframe
      Binarized dataframe of incomplete subset

    df_correl : dataframe
      Correlation matrix of complete subset

    order_column : str
      Indicates the type of ordered to be used. "noSort" by default, "lessCorrelation", "moreCorrelation", "lessMissing", "moreMissing"
    
    Returns
    ----------
    'ordered_col_list':  attributes order list
    """
    
    df_mask = df_mask.drop(columns=['num_cluster','aux_cell_tupla'])
    
    # Correlation
    correl_sum_abs = pd.DataFrame(df_correl.apply(lambda x: np.sum(np.abs(x))), columns=['correl_sum_abs'])   # sum of the absolute value of the columns
    
    # Creating an auxiliary dataframe
    mask_aux = df_mask.sum()         # amount of missing in the column
    mask_null_sum = pd.DataFrame(mask_aux, columns=['count_missing'])               
                                        

    if order_column == "lessCorrelation":
        aux_ordem = correl_sum_abs.sort_values(by=['correl_sum_abs'], ascending=True)
        ordered_col_list = aux_ordem.index.values.tolist()
        return ordered_col_list
    elif order_column == "moreCorrelation":
        aux_ordem = correl_sum_abs.sort_values(by=['correl_sum_abs'], ascending=False)
        ordered_col_list = aux_ordem.index.values.tolist()
        return ordered_col_list
    elif order_column == "lessMissing":
        aux_ordem = mask_null_sum.sort_values(by=['count_missing'], ascending=True)
        ordered_col_list = aux_ordem.index.values.tolist()
        return ordered_col_list
    elif order_column == "moreMissing":
        aux_ordem = mask_null_sum.sort_values(by=['count_missing'], ascending=False)
        ordered_col_list = aux_ordem.index.values.tolist()
        return ordered_col_list
    else:
        ordered_col_list = df_mask.columns.tolist()
        return ordered_col_list   




#### Attribute ordering criteria #### 
def get_eps(df, n_neighbors):
    """
    Generates the eps curve to determine the optimal value
    
    Parameters
    ----------
    n_neighbors: int, default=5
        The number of neighbors to use.       

    df: dataframe
    """
    # Calculate nn distance
    distances, indices = knn(df, n_neighbors=n_neighbors, metric='euclidean')
    kn_dist = distances[:,1:].reshape(-1)

    # Calculate eps distance
    eps_dist = np.sort(kn_dist)
    rotor = Rotor()
    curve_xy = np.concatenate([np.arange(eps_dist.shape[0]).reshape(-1,1), eps_dist.reshape(-1,1)],1)
    rotor.fit_rotate(curve_xy)
    rotor.plot_elbow()
    e_idx = rotor.get_elbow_index()

    return curve_xy[e_idx]




#### Get cluster dataframe #### 
def get_cluster_df(df_filled, df_null_, df_restorado_cl, attribute):
    """
    Auxiliary function that generates the dataset that will be used in clustering
    
    Parameters
    ----------
    df_filled: dataframe 
        Dataset with non-null

    df_null_: dataframe
        Dataset with nulls and within a specific cluster

    df_restorado_cl: dataframe
        Auxiliary dataset with nulls and within a specific cluster

    attribute: str
        Attribute to be imputed
    
    Returns
    ----------
    df_cluster  : dataframe that will be used to cluster
    df_attr_null: dataframe with 'current attribute' null for imputation
    """  
    ## Step 1: Slipt the dataframe by complete and incomplete subset (based on attribute)
    # Split df_null (base that will be used in the cluster) into 2 subsets
    df_attr = df_restorado_cl.dropna(subset=[attribute])           # dataframe with non-null 'current attribute'
    df_attr_clean = df_attr.dropna()                               # dataframe without any null attributes
    df_attr_null = df_null_[df_null_[attribute].isna()]            # dataframe with 'current attribute' null for imputation
    
    if df_attr_null.empty:
        return df_filled, df_attr_null                             # Skip if there are no null values in the current attribute for the current cluster

    # Creating the dataset that will be used to cluster (df_filled + df_attr_clean)
    df_cluster = pd.concat([df_filled, df_attr_clean])  #, ignore_index=True
    df_cluster = df_cluster.sort_index()

    return df_cluster, df_attr_null




#### Get averages dataframe #### 
# def get_averages_df(df_cluster, indices):   #averages_df, 
#     """
#     Auxiliary function that creates the dataset that has the averages of the nearest neighbors
  
#     Parameters
#     ----------
#     averages_df: dataframe 
#         Empty dataFrame to store the averages
#     df_cluster: dataframe
#         Dataset that will be used to cluster
#     indices: dataframe
#         NN's indices of the neighbors of each point. 
    
#     Returns
#     ----------
#     averages_df  : dataframe that will be used to cluster
#     """  
#     averages_df = pd.DataFrame(columns=df_cluster.columns)

#     for group_indices in indices:
#         # Select lines from df_cluster based on the current group of indices
#         selected_lines = df_cluster.iloc[group_indices]

#         # Calculate the average for each column
#         group_averages = selected_lines.mean()

#         # Append the group averages to the DataFrame with the corresponding index
#         group_averages = pd.DataFrame(group_averages).T
#         group_averages.index = [df_cluster.index[group_indices[0]]]

#         averages_df = pd.concat([averages_df, group_averages])

#     # Deduplicating the averages_df indexes
#     idx = np.unique(averages_df.index.values, return_index=True)[1]
#     averages_df = averages_df.iloc[idx]

#     return averages_df




#### Get averages dataframe #### 
def get_averages_df(df_cluster, indices):
    """
    Auxiliary function that creates the dataset that has the averages of the nearest neighbors
    
    Parameters
    ----------
    df_cluster: dataframe
        Dataset that will be used to cluster

    indices: dataframe
        NN's indices of the neighbors of each point. 
    
    Returns
    ----------
    averages_df  : dataframe that will be used to cluster
    """  
    averages_df = pd.DataFrame()

    for group_indices in indices:
        # Select lines from df_cluster based on the current group of indices
        selected_lines = df_cluster.iloc[group_indices]

        # Check if selected_lines is not empty
        if not selected_lines.empty:
            # Calculate the average for each column
            group_averages = selected_lines.mean()

            # Append the group averages to the DataFrame with the corresponding index
            group_averages = pd.DataFrame(group_averages).T
            group_averages.index = [df_cluster.index[group_indices[0]]]
           # Concatenate group_averages to averages_df
            averages_df = pd.concat([averages_df, group_averages])

    # Deduplicating the averages_df indexes
    idx = np.unique(averages_df.index.values, return_index=True)[1]
    averages_df = averages_df.iloc[idx]

    return averages_df




#### Get averages dataframe (Optimized) #### 
def get_averages_df_optimized(df_cluster, indices):
    averages_list = []

    for group_indices in indices:
        if len(group_indices) > 0:  # Certifique-se de que group_indices não está vazio
            selected_lines = df_cluster.iloc[group_indices]
            group_averages = selected_lines.mean().to_frame().T
            group_averages.index = [df_cluster.index[group_indices[0]]]
            averages_list.append(group_averages)

    if averages_list:  # Verifica se a lista não está vazia
        averages_df = pd.concat(averages_list)
        # Deduplicando os índices, caso seja necessário
        averages_df = averages_df.loc[~averages_df.index.duplicated(keep='first')]
    else:
        averages_df = pd.DataFrame()

    return averages_df




#### Get averages dataframe (Parallelizing) ####
def calculate_group_average(args):
    df_cluster, group_indices = args
    if group_indices:  # Certifique-se de que group_indices não está vazio
        selected_lines = df_cluster.iloc[group_indices]
        group_averages = selected_lines.mean().to_frame().T
        group_averages.index = [df_cluster.index[group_indices[0]]]
        return group_averages
    return None


def get_averages_df_parallel(df_cluster, indices):
    # Preparar os argumentos para cada tarefa paralela
    tasks = [(df_cluster, group_indices) for group_indices in indices]

    # Executar em paralelo usando ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(calculate_group_average, tasks))

    # Filtrar os resultados None e concatenar os DataFrames
    averages_df = pd.concat([result for result in results if result is not None])

    # Deduplicando os índices, caso seja necessário
    averages_df = averages_df.loc[~averages_df.index.duplicated(keep='first')]

    return averages_df





#### Restore dataset #### 
def restore_dataset(df_restorado, df_attr_null, averages_df, attribute):
    """
    Auxiliary function that restores the dataset by replacing nulls with the averages of the nearest neighbors
    
    Parameters
    ----------
    df_restorado: dataframe
        Dataset that will be imputed

    df_attr_null: dataframe
        Dataset with 'current attribute' null 

    averages_df: dataframe 
        Dataset with the averages of the nearest neighbors

    attribute: array
        Attribute list 
    
    Returns
    ----------
    df_restorado : dataset by replacing nulls with the averages of the nearest neighbors
    """  
    for i in range(df_attr_null.shape[0]):
        #Find the closest index in averages_df to the current row index in df_attr_null
        if i >= len(df_attr_null.index):
            continue  # Skip if i is out of bounds
        distances = np.abs(np.array(averages_df.index) - df_attr_null.index[i])

        if len(distances) == 0:
            continue  # Skip if there are no indices in distances

        closest_index = averages_df.index[np.argmin(distances)]

        #Replace null values in the current attribute with the corresponding average from averages_df
        df_restorado.loc[df_attr_null.index[i], attribute] = averages_df.at[closest_index, attribute]

    return df_restorado




#### SOM Clustering (model-based) #### 
def som(df, dim_x, dim_y, sigma, lr, max_iter):
    """
    Calculates SOM's clusters
    
    Parameters
    ----------
    df : dataframe

    dim_x : int
        x dimension of the SOM.

    dim_y : int
        y dimension of the SOM.

    sigma : float, optional (default=1.0)
        Spread of the neighborhood function, needs to be adequate to the dimensions of the map.
        (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)

    lr : float
        initial learning rate (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T)
        where T is #num_iteration/2)

    max_iter: int
        maximum iteration
    
    Returns
    ----------
    winner_coordinates: coordinates of the winning neuron
    labels            : label of the cluster to which it belongs
    """

    data = df.values
    #data_norm = normalize(df, type = "minmax")
    som_shape = (dim_x, dim_y)
    input_len = data.shape[1]

    # SOM
    som = MiniSom(dim_x, dim_y, input_len, sigma, lr)
    som.random_weights_init(data)                             # initialize the weights
    #som.random_weights_init(data_norm)                       # initialize the weights
    #starting_weights = som.get_weights().copy()              # saving the starting weights
    som.train_random(data, max_iter)                          # training the model with 100 iteration

    # Each neuron represents a cluster
    winner_coordinates = np.array([som.winner(x) for x in data]).T

    # Convert the bidimensional coordinates to a monodimensional index
    labels = np.ravel_multi_index(winner_coordinates, som_shape)  
    
    return winner_coordinates, labels    




#### K-prototype Clustering (partitional-based) #### 
def kprototypes(df, n_clusters, max_iter, categ_index):
    """
    Calculates Kprototypes' clusters
    
    Parameters
    ----------
    df: dataframe 
    
    n_clusters: int, default=8
        The number of clusters to form as well as the number of centroids to generate.

    max_iter: int, default=100
        Maximum number of iterations of the k-means algorithm for a single run.

    categ_index: array
        Index of categorical columns.
    
    Returns
    ----------
    labels : label of the cluster to which it belongs
    """
    kproto = KPrototypes(n_clusters=n_clusters, init='Huang', max_iter=max_iter, random_state=42) 
    kproto.fit_predict(df, categorical=categ_index)
    labels = kproto.labels_

    return labels   




#### K-modes Clustering (partitional-based) #### 
def kmodes(df, n_clusters):
    """
    Calculates Kmodes' clusters
    
    Parameters
    ----------
    df: dataframe 
    
    n_clusters: int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    
    Returns
    ----------
    labels : label of the cluster to which it belongs
    """
    kmodes = KModes(n_clusters=n_clusters, init='random') 
    labels = kmodes.fit_predict(df)

    return labels




#### KNN Clustering #### 
def knn(df, n_neighbors, metric):
    """
    Calculates Nearest Neighbors' clusters
    
    Parameters
    ----------
    df: dataframe
    
    n_neighbors: int, default=5
        The number of neighbors to use.       

    metric: str
        Indicates the metric we want to calculate. 'euclidean', 'minkowski', 'manhattan', 'cityblock', 'jaccard', 'haversine'
    
    Returns
    ----------
    distances : distances of to the neighbors of each point
    indices: indices of the neighbors of each point.
    """
    
    knn = NearestNeighbors(n_neighbors = n_neighbors, algorithm='ball_tree', metric= metric)
    knn.fit(df)
    
    distances, indices = knn.kneighbors(df)

    return distances, indices




#### Dbscan Clustering (density-based) #### 
def dbscan(df, eps, min_samples, metric):
    """
    Calculates Dbscan's clusters
    
    Parameters
    ----------
    eps: float, default=0.5
        The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not 
        a maximum bound on the distances of points within a cluster.

    min_samples: int, default=5
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes
        the point itself.    

    metric: str, or callable, default=’euclidean’
        The metric to use when calculating distance between instances in a feature array. If metric is a string or callable, 
        it must be one of the options allowed by sklearn.metrics.pairwise_distances for its metric parameter. If metric is 
        “precomputed”, X is assumed to be a distance matrix and must be square. X may be a sparse graph, in which case only
        “nonzero” elements may be considered neighbors for DBSCAN.

    df: 
        dataframe that should be clustered.
    
    Returns
    ----------
    labels : label of the cluster to which it belongs
    """

    if metric == "precomputed":
        gower_dist = gover_distance(df)        # calculate the Gower's distance
        dbs = DBSCAN(eps=eps, min_samples=min_samples, metric = 'precomputed')
        dbs.fit(gower_dist)
        labels = dbs.labels_      # -1 value represents noisy points could not assigned to any cluster
    else:
        dbs = DBSCAN(eps=eps, min_samples=min_samples, metric = metric)
        dbs.fit(df)
        labels = dbs.labels_      # -1 value represents noisy points could not assigned to any cluster

    return labels




#### Agglomerative Clustering (hierarquical-based) #### 
def agglomerative_cluster(df, n_clusters, affinity, linkage):
    """
    Calculates Agglomerative's clusters
    
    Parameters
    ----------
    n_clusters: int, default=2 

    affinity: str or callable, default=”euclidean”
        Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”. If 
        linkage is “ward”, only “euclidean” is accepted. If “precomputed”, a distance matrix is needed as input for the 
        fit method.

    linkage: {'ward', 'complete', 'average', 'single'}, default='ward'
        Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. 
        The algorithm will merge the pairs of cluster that minimize this criterion.
        'ward' minimizes the variance of the clusters being merged.
        'average' uses the average of the distances of each observation of the two sets.
        'complete' or 'maximum' linkage uses the maximum distances between all observations of the two sets.
        'single' uses the minimum of the distances between all observations of the two sets.  

    df: 
        dataframe that should be clustered.
    
    Returns
    ----------
    labels : label of the cluster to which it belongs
    """  

    if affinity == "precomputed":
        gower_dist = gover_distance(df)        # calculate the Gower's distance
        agg_cluster = AgglomerativeClustering(n_clusters = n_clusters, affinity='precomputed', linkage=linkage)
        labels = agg_cluster.fit_predict(gower_dist)
    else:
        agg_cluster = AgglomerativeClustering(n_clusters = n_clusters, affinity=affinity, linkage=linkage)
        labels = agg_cluster.fit_predict(df)

    return labels




#### Morphology Absence ####
def morphology_absence(df, cluster_algorithm="som", dim_x=None, dim_y=None, sigma=None, lr=None, max_iter=None, n_clusters=None,  
                       categ_index=None, eps=None, min_samples=None, metric=None, affinity=None, linkage=None):
    """
    Determines the order of attributes according to the selected criteria
    
    Parameters
    ----------
    df: dataframe 
      Binarized dataframe for clustering

    cluster_algorithm :  {'som', 'kprototypes', 'kmodes', 'dbscan', 'agglomerative_cluster'}, default='som'
      Cluster algorithms that accept multivariate data. This algorithm will help identify the pattern of nulls

    dim_x : int
        x dimension of the SOM.

    dim_y : int
        y dimension of the SOM.

    sigma : float, optional (default=1.0)
        Parameter to be used in the som algorithm
        Spread of the neighborhood function, needs to be adequate to the dimensions of the map.
        (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)

    lr : float
        Parameter to be used in the som algorithm
        initial learning rate (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T)
        where T is #num_iteration/2)

    max_iter: int, default=100
        Parameter to be used in the som or kprototypes algorithm
        Maximum number of iterations for a single run.
    
    n_clusters: int, default=8
        Parameter to be used in the agglomerative_cluster or kprototypes algorithm
        The number of clusters to form as well as the number of centroids to generate.

    categ_index: array
        Parameter to be used in the kprototypes algorithm
        Index of categorical columns.

    eps: float, default=0.5
        Parameter to be used in the dbscan algorithm
        The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not 
        a maximum bound on the distances of points within a cluster.

    min_samples: int, default=5
        Parameter to be used in the kprototypes algorithm
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes
        the point itself.    

    metric: str, or callable, default='euclidean'
        Parameter to be used in the dbscan algorithm
        The metric to use when calculating distance between instances in a feature array. If metric is a string or callable, 
        it must be one of the options allowed by sklearn.metrics.pairwise_distances for its metric parameter. If metric is 
        “precomputed”, X is assumed to be a distance matrix and must be square. X may be a sparse graph, in which case only
        “nonzero” elements may be considered neighbors for DBSCAN.

    affinity: str or callable, default='euclidean'
        Parameter to be used in the agglomerative_cluster algorithm
        Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”. If 
        linkage is “ward”, only “euclidean” is accepted. If “precomputed”, a distance matrix is needed as input for the 
        fit method.

    linkage: {'ward', 'complete', 'average', 'single'}, default='ward'
        Parameter to be used in the agglomerative_cluster algorithm
        Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. 
        The algorithm will merge the pairs of cluster that minimize this criterion.
        'ward' minimizes the variance of the clusters being merged.
        'average' uses the average of the distances of each observation of the two sets.
        'complete' or 'maximum' linkage uses the maximum distances between all observations of the two sets.
        'single' uses the minimum of the distances between all observations of the two sets.  
    
    Returns
    ----------
    labels : label of the cluster to which it belongs
    """
    
    if cluster_algorithm == "kmodes":
        labels = kmodes(df, n_clusters)   #, init='Huang'
        return labels
    elif cluster_algorithm == "kprototypes":
        labels = kprototypes(df, n_clusters, max_iter, categ_index)
        return labels
    elif cluster_algorithm == "dbscan":
        labels = dbscan(df, eps, min_samples, metric)
        return labels
    elif cluster_algorithm == "agglomerative_cluster":
        labels = agglomerative_cluster(df, n_clusters, affinity, linkage)
        return labels
    elif cluster_algorithm == "som":
        winner_coordinates, labels = som(df, dim_x, dim_y, sigma, lr, max_iter)
        return labels
    else:
        print("Error")
     




#### Univariate imputation (G1 group) #### 
def univariate_imputation(df_filled, df_null):
    """
    Perform univariate imputation
    
    Parameters
    ----------
    df_filled: dataframe 
        Complete dataset that will be used to calculate the average and median of the attributes

    df_null: dataframe
        Dataset that will have the null values replaced by average/median of df_filled
    
    Returns
    ----------
    df_g1_mean  : dataframe with average imputation
    df_g1_median: dataframe with median imputation
    """  
    # Full dataframe average
    attr_mean = df_filled.mean() 
    attr_mean = pd.DataFrame(attr_mean).T

    # Full dataframe median
    attr_median = df_filled.median() 
    attr_median = pd.DataFrame(attr_median).T

    df_g1_mean   = df_null.copy()
    df_g1_median = df_null.copy()

    for name, values in df_g1_mean.items():
        # Imputing by the average
        mean = attr_mean[name][0]
        df_g1_mean[name].fillna(mean, inplace = True)

        # Imputing by the median
        median = attr_median[name][0]
        df_g1_median[name].fillna(median, inplace = True)

    return df_g1_mean, df_g1_median




#### Cascade imputation (G2 group) #### 
def cascade_imputation(df_filled, df_null, df_mask, ordered_clu_list, ordered_col_list, knn_neighbors=3, knn_metric='euclidean'):
    """
    Perform cascade imputation=
    
    Parameters
    ----------
    df_filled: dataframe
        Dataframe with only non-null values

    df_null: dataframe
        Dataframe with null values

    df_mask: dataframe
        The binary df_null where 1 is null and 0 is non-null

    ordered_clu_list: array
        List of order of clusters to be used

    ordered_col_list: array
        List of order of attributes/columns to be used

    knn_neighbors: int, default=5
        The number of neighbors to use in KNN.       

    knn_metric: str
        Indicates the metric we want to calculate in KNN. 'euclidean', 'minkowski', 'manhattan', 'cityblock', 'jaccard', 'haversine'
    
    Returns
    ----------
    df_restorado : dataset with imputed data
    """
    
    time_seg1_sum = 0
    time_seg2_sum = 0
    time_seg3_sum = 0
    time_seg4_sum = 0

    #Initialize an empty DataFrame to store the restored data
    df_restorado = df_null.copy()

    for cluster in ordered_clu_list:
        #Select the num_cluster
        idx_cluster = df_mask.loc[df_mask['num_cluster'] == cluster].index
        df_null_ = df_null.filter(items=idx_cluster, axis=0)
        df_restorado_cl = df_restorado.filter(items=idx_cluster, axis=0)

        for attribute in ordered_col_list:            
            #Step 1: Create cluster dataframe by complete and incomplete subset (based on attribute)
            start_time1 = time.time()                                  # Inicializar a contagem do tempo etapa 1
            df_cluster, df_attr_null = get_cluster_df(df_filled, df_null_, df_restorado_cl, attribute)

            time_seg1 = measure_execution_time_seg(start_time1)    # Finalizar a contagem do tempo (etapa 1)
            time_seg1_sum = time_seg1_sum + time_seg1

            #Step 2: Split subset by cluster and find the average of each attribute per cluster (Running the cluster algorithm)
            start_time2 = time.time()                                  # Inicializar a contagem do tempo etapa 2
            aux_knn = df_cluster.drop(columns=[attribute])
            n_samples = len(aux_knn)
            #print("n_samples:", n_samples)

            # Reduz o número de vizinhos, se necessário
            if knn_neighbors > n_samples:
                knn_neighbors = n_samples
            #print("knn_neighbors:", knn_neighbors)
            
            distances_, indices = knn(aux_knn, n_neighbors=knn_neighbors, metric=knn_metric)
            
            time_seg2 = measure_execution_time_seg(start_time2)    # Finalizar a contagem do tempo (etapa 2)
            time_seg2_sum = time_seg2_sum + time_seg2

            #Initialize an empty DataFrame to store the averages
            #averages_df = pd.DataFrame(columns=df_cluster.columns)
            
            #Iterate over each group of indices
            start_time3 = time.time()                                  # Inicializar a contagem do tempo etapa 3
            averages_df = get_averages_df_optimized(df_cluster, indices)
            #for group_indices in indices:
                #averages_df = get_averages_df(df_cluster, indices)    
                #averages_df = get_averages_df_parallel(df_cluster, indices)

            time_seg3 = measure_execution_time_seg(start_time3)    # Finalizar a contagem do tempo (etapa 3)
            time_seg3_sum = time_seg3_sum + time_seg3

            #Use the restore_dataset function to fill null values in df_restorado
            start_time4 = time.time()                                  # Inicializar a contagem do tempo etapa 4
            df_restorado = restore_dataset(df_restorado, df_attr_null, averages_df, attribute)

            time_seg4 = measure_execution_time_seg(start_time4)    # Finalizar a contagem do tempo (etapa 3)
            time_seg4_sum = time_seg4_sum + time_seg4
    
    time_seg1_sum_ = f"{int(time_seg1_sum // 60)}m.{int(time_seg1_sum % 60)}s"
    time_seg2_sum_ = f"{int(time_seg2_sum // 60)}m.{int(time_seg2_sum % 60)}s"
    time_seg3_sum_ = f"{int(time_seg3_sum // 60)}m.{int(time_seg3_sum % 60)}s"
    time_seg4_sum_ = f"{int(time_seg4_sum // 60)}m.{int(time_seg4_sum % 60)}s"

    logging.info(f"Etapa 1 (df_cluster): {time_seg1_sum_}")
    logging.info(f"Etapa 2 (KNN): {time_seg2_sum_}")
    logging.info(f"Etapa 3 (calc médias cluster): {time_seg3_sum_}")
    logging.info(f"Etapa 4 (restorando dados): {time_seg4_sum_}")
    
    return df_restorado




#### Error Metrics #### 
def error_metrics(df_original, df_imputado):
    # Initialize an empty DataFrame to store the averages
    rmse_values_df = pd.DataFrame(columns=df_original.columns)
    sim_values_df = pd.DataFrame(columns=df_original.columns)

    for column in df_original.columns:
        y_true = df_original[column].values
        y_pred = df_imputado[column].values

        # Calculate RMSE for each column
        rmse = reg_metric(y_true, y_pred, type="rmse")
        rmse_values_df[column] = [rmse]

        # Calculate Similarity error for each column
        erro_imputacao = imput_error(y_true, y_pred, "sim_error")
        sim_values_df[column] = [erro_imputacao]
    
    # Calculates the average of error metrics generated by the imputation
    avg_rmse = rmse_values_df.mean(axis=1).values[0]            # Calculate RMSE
    avg_sim_error = sim_values_df.mean(axis=1).values[0]        # Calculate similarity error
    correl_bias = correlation_bias(df_original, df_imputado)    # Calculate correlation bias 

    return avg_rmse, avg_sim_error, correl_bias







#####################################################################################################
#####################################################################################################

#########
### FUNÇÕES QUE CARREGAM JÁ OS PARAMETROS 


# Função para realizar o ERASER com os parâmetros fornecidos
def eraser_with_params(df, params):
    ## Inicializar a contagem do tempo
    start_time = time.time()
    
    ## Extrair os parâmetros de cada simulação
    idx_simulacao = params['idx_simulacao']
    dados = params['dados']
    mecha = params['mec_ausencia']     # Mecanismo de ausência: MAR, MNAR, MCAR
    p_miss = params['pct_ausencia']    # % valores ausentes
    logging.info(f"Iniciando simulação {idx_simulacao}...")

    ## Generating null data
    aux_ausencia = eraser(df, p_miss=p_miss, mecha=mecha, p_obs=0.5)
    df_incompleto = pd.DataFrame(aux_ausencia['X_incomp'].numpy(), columns= df.columns)

    ## Finalizando a contagem do tempo
    time_mecha = measure_execution_time_min_seg(start_time)
    logging.info(f"Eraser do {mecha}_{p_miss} concluído. Tempo decorrido: {time_mecha}")
    
    return dados, mecha, p_miss, df_incompleto   # idx_simulacao, 




# Função para realizar a IMPUTAÇÃO UNIVARIADA com os parâmetros fornecidos
def univariate_imputation_with_params(df_incompleto):
    ## Inicializando a contagem do tempo
    logging.info(f"Inicio da imputação univariada.")
    start_time = time.time()     # Inicializando a contagem do tempo

    ## Imputação Univariada
    df_filled, df_null = split_dataset(df_incompleto)       # split dataset
        
    # Generating a univariate imputation for comparison with cascade imputation
    aux_mean, aux_median = univariate_imputation(df_filled, df_null)

    # Concatenate df_filled and df_cascade_imp 
    df_g1_mean   = pd.concat([df_filled, aux_mean]).sort_index()    
    df_g1_median = pd.concat([df_filled, aux_median]).sort_index()

    ## Finalizando a contagem do tempo
    time_g1 = measure_execution_time_min_seg(start_time)
    logging.info(f"Imputação univariada concluída! Tempo decorrido: {time_g1}")

    return df_g1_mean, df_g1_median, time_g1




# Função para realizar a IMPUTAÇÃO CASCATA com os parâmetros fornecidos
def cascade_imputation_with_params(df_incompleto, params):
    ## Extrair os parâmetros de cada simulação
    idx_simulacao = params['idx_simulacao']
    cluster_algorithm = params['cluster_algorithm']
    dim_x_som = params['dim_x_som'] 
    dim_y_som = params['dim_y_som'] 
    sigma_som = params['sigma_som'] 
    lr_som = params['lr_som']
    max_iter_som = params['max_iter_som'] 
    n_clusters_kmd = params['n_clusters_kmd'] 
    n_clusters_aggcl = params['n_clusters_aggcl'] 
    affinity_aggcl = params['affinity_aggcl']
    eps_dbscan = params['eps_dbscan']              
    min_samples_dbscan = params['min_samples_dbscan'] 
    metric_dbscan = params['metric_dbscan']
    order_cluster = params['order_cluster'] 
    order_column = params['order_column']
    n_clusters_knn = params['n_clusters_knn']  

    logging.info(f"----------------")
    logging.info(f"Inicio da imputação cascata da simulação {idx_simulacao}...")

    ## Inicializar a contagem do tempo
    start_time = time.time()

    ## Imputação em Cascata (step-by-step)
    df_filled, df_null = split_dataset(df_incompleto)       # split dataset  
    df_mask = get_binarized_df(df_null)                     # binarized dataset
    df_correl = correlation(df_filled)                      # correlation

    logging.info(f"Iniciando a morfologia da ausência usando {cluster_algorithm}")
    
    if cluster_algorithm == "kmodes": 
        labels = morphology_absence(df_mask, cluster_algorithm="kmodes", n_clusters=int(n_clusters_kmd))
    elif cluster_algorithm == "dbscan": 
        labels = morphology_absence(df_mask, cluster_algorithm="dbscan", eps=float(eps_dbscan), min_samples=int(min_samples_dbscan) , metric=metric_dbscan)  
    elif cluster_algorithm == "agglomerative_cluster" and affinity_aggcl == "precomputed": 
        labels = morphology_absence(df_mask, cluster_algorithm="agglomerative_cluster", n_clusters=int(n_clusters_aggcl), affinity="precomputed", linkage='complete')
    elif cluster_algorithm == "agglomerative_cluster" and affinity_aggcl != "precomputed":
        labels = morphology_absence(df_mask, cluster_algorithm="agglomerative_cluster", n_clusters=int(n_clusters_aggcl), affinity="euclidean", linkage='ward')    
    elif cluster_algorithm == "som":
        labels = morphology_absence(df_mask, cluster_algorithm="som", dim_x=int(dim_x_som), dim_y=int(dim_y_som), sigma=float(sigma_som), lr=float(lr_som), max_iter=int(max_iter_som)) 
    else:
        print('Error: Algoritmo não encontrado')

    ## Finalizar a contagem do tempo da Morfologia da Ausência
    time_morf_ausencia = measure_execution_time_min_seg(start_time)  
    logging.info(f"Morfologia da ausência concluída! Tempo decorrido: {time_morf_ausencia}")

    df_mask = labels_col_df(df_mask, labels)         # create label column 

    # Cluster ordering criterion
    ordered_clu_list = cluster_order_criterion(df_mask, order_cluster=order_cluster)

    # Attribute ordering criterion
    ordered_col_list = attribute_order_criterion(df_mask, df_correl, order_column=order_column)
            
    # Cascade imputation
    df_cascade_imp = cascade_imputation(df_filled, df_null, df_mask, ordered_clu_list, ordered_col_list, knn_neighbors=n_clusters_knn, knn_metric='euclidean')
        
    # Concatenate df_filled and df_cascade_imp
    df_g2_imput = pd.concat([df_filled, df_cascade_imp]).sort_index()


    ## Finalizar a contagem do tempo da Imputação Cascata
    time_g2 = measure_execution_time_min_seg(start_time) 
    logging.info(f"Imputação cascata concluída! Tempo decorrido: {time_g2}")

    
    return idx_simulacao, df_g2_imput, time_g2   # 