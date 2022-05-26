import mlrun
from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
import snowflake.connector as snow
import os
import numpy as np
from dask.distributed import Client
from dask.dataframe import from_delayed
from dask import delayed
from dask import dataframe as dd

import warnings
warnings.filterwarnings("ignore")
    
@delayed
def load(batch):
    try:
        print("BATCHING")
        df_ = batch.to_pandas()
        return df_
    except Exception as e:
        print(f"Failed on {batch} for {e}")
        pass

def load_delayed(dask_client, connection_info, query, out_dir, write_out=False, publish=False):        
    context = mlrun.get_or_create_ctx('dask-cluster')  
    sfAccount = context.get_secret('account')
    context.log_result('sfAccount', sfAccount)
    context.logger.info(f'sfAccount = {sfAccount}')
    # setup dask client from the MLRun dask cluster function
    if dask_client:
        client = mlrun.import_function(dask_client).client
        context.logger.info(f'Existing dask client === >>> {client}\n')
    else:
        client = Client()
        context.logger.info(f'\nNewly created dask client === >>> {client}\n')
        
    query = query

    conn = snow.connect(**connection_info)
    cur = conn.cursor()
    cur.execute(query)
    batches = cur.get_result_batches()
    print(f'batches len === {len(batches)}\n')
    
    dfs = []    
    for batch in batches:
        if batch.rowcount > 0:
            df = load(batch)
            dfs.append(df)        
    ddf = from_delayed(dfs)
    
    # materialize the query results set for some sample compute
    
    ddf_sum = ddf.sum().compute()
    ddf_mean = ddf.mean().compute()
    ddf_describe = ddf.describe().compute()
    ddf_grpby = ddf.groupby("C_CUSTKEY").count().compute()
    
    context.logger.info(f'sum === >>> {ddf_sum}\n')
    context.logger.info(f'mean === >>> {ddf_mean}\n')
    context.logger.info(f'ddf head === >>> {ddf.head()}\n')
    context.logger.info(f'ddf  === >>> {ddf}\n')

    context.log_result('number of rows', len(ddf.index))   
    
    context.log_dataset('dask_data_frame', ddf)
    context.log_dataset("my_df_describe", df=ddf_describe)
    context.log_dataset("my_df_grpby",    df=ddf_grpby)
    
    ddf.persist(name = 'customer')
    if publish and (not client.list_datasets()):    
        client.publish_dataset(customer=ddf)
        
    if write_out:
        dd.to_parquet(df=ddf, path=out_dir)
        context.log_result('parquet', out_dir)
