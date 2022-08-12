import os 
from tqdm import tqdm 
import pandas as pd 

def get_change_df(project, datadir = "../data/parsed"):
    import glob 
    datadir = os.path.join(datadir, project) if not datadir.endswith(project) else datadir
    datafiles = glob.glob(os.path.join(datadir, "*.json"))
    assert len(datafiles) > 0, datadir
    dfs = []
    for datafile in tqdm(datafiles):
        _df = pd.read_json(datafile).T
        if len(_df) > 0:
            dfs.append(_df)

    ret_df = pd.concat(dfs).rename(
        columns = {'authored_data':'authored_date'})
    return ret_df