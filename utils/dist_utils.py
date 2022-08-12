from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

def parse_date(change_hist_df):
    """
    """
    from datetime import datetime
    convert_to_ts = lambda str_chgd_date: datetime.strptime(
        str_chgd_date, '%Y %b %d %H:%M'
        ) if isinstance(str_chgd_date, str) else str_chgd_date#

    change_hist_df.authored_date = change_hist_df.authored_date.apply(convert_to_ts)
    return change_hist_df


def compute_change_vector(
    target_commit: str, 
    change_hist: pd.DataFrame,
    target_del_add_mths: List[Tuple[str,str]] = None, 
    n_min_mod:int = 1) -> Tuple:
    """
    currently, we won't think about the file renaming, etc. 
    """ 
    from utils.hunk_utils import get_changed_method 
    if not target_del_add_mths:
        target_del_add_mths, _, _ = get_changed_method(change_hist.loc[target_commit].changes)
        if len(target_del_add_mths) <= n_min_mod:
            return (None, None)

    chg_time_vector = {mod_mth:[] for mod_mth in target_del_add_mths}
    chg_commit_vector = {mod_mth:[] for mod_mth in target_del_add_mths}
    tcommit_time = change_hist.loc[target_commit].authored_date 

    # previous changes, INCLUDING the current one 
    upto_change_hist = change_hist.loc[change_hist.authored_date <= tcommit_time]
    for index_commit, row in list(upto_change_hist[['changes', 'authored_date']].iterrows()): 
        deleted_or_added_mths, _, _ = get_changed_method(row.changes)
        intersected_mths = set(target_del_add_mths).intersection(set(deleted_or_added_mths))
        for inter_mth in intersected_mths: 
            chg_time_vector[inter_mth].append(row.authored_date)
            chg_commit_vector[inter_mth].append(index_commit)
    return chg_time_vector, chg_commit_vector


def compute_dist(
    chgdat_of_a:np.ndarray, 
    chgdat_of_b:np.ndarray, 
    gran:str = 'sec', 
    _N_recent_time:int = None
    ) -> np.ndarray:
    """
    dist of a to b
    """
    N_recent_time = -1
    t_N_recent_time = N_recent_time if _N_recent_time is None else _N_recent_time

    if len(chgdat_of_a) == 0 or len(chgdat_of_b) == 0:
        return -1 # meaning nothing has been changeed recently
    else:
        # compute the closest dist for every past change
        closet_dist = np.min(
            np.abs(
                chgdat_of_a[:,None] - chgdat_of_b
                ),
            axis = 1
        ) 
        # take the average
        time_dist = np.mean(closet_dist).total_seconds() 
        if t_N_recent_time > 0 and time_dist > t_N_recent_time: # shouldn't occur
            assert False 

        if gran == 'sec':
            return time_dist 
        elif gran == 'min':
            return np.round(time_dist/60)
        elif gran == 'hour':
            return np.round(time_dist/(60*60))
        elif gran == 'day':
            return np.round(time_dist/(60*60*24))
        else:
            print ("Does not support this granularity ({})".format(gran))
            assert False 


def compute_dist_wsum(
    insp_time:np.datetime64, 
    chgdat_of_a:np.ndarray, 
    chgdat_of_b:np.ndarray, 
    gran:str = 'sec', 
    _N_recent_time:int = None
) -> np.ndarray:
    """
    dist of a to b
    """
    N_recent_time = -1
    t_N_recent_time = N_recent_time if _N_recent_time is None else _N_recent_time

    if len(chgdat_of_a) == 0 or len(chgdat_of_b) == 0:
        return -1 # meaning nothing has been changeed recently
    else:
        closet_dist = np.min(
            np.abs(
                chgdat_of_a[:,None] - chgdat_of_b),
            axis = 1
        ) 
        from datetime import timedelta
        closet_dist = np.vectorize(
            timedelta.total_seconds
        )(closet_dist)

        # checking for validity 
        t = np.mean(closet_dist)#.total_seconds() 
        if t_N_recent_time > 0 and t > t_N_recent_time:
            assert False 

        # time diff to the current inspection time
        dist_to_chg = insp_time - chgdat_of_a # timedelta
        dist_to_chg = np.array(
            list(map(lambda v: v.total_seconds(), dist_to_chg)))

        if gran == 'sec':
            closet_dist = closet_dist + 1
        elif gran == 'min':
            closet_dist = np.round(closet_dist/60) + 1
            dist_to_chg = np.round(dist_to_chg/60) 
        elif gran == 'hour':
            closet_dist = np.round(closet_dist/(60*60)) + 1
            dist_to_chg = np.round(dist_to_chg/(60*60)) 
        elif gran == 'day':
            closet_dist = np.round(closet_dist/(60*60*24)) + 1
            dist_to_chg = np.round(dist_to_chg/(60*60*24)) 
        else:
            print ("Does not support this granularity ({})".format(gran))
            assert False 

        # weighted sum
        closet_dist = 1/closet_dist 
        if np.min(dist_to_chg) == 0:
            dist_to_chg += 1
        closet_dist *= 1/dist_to_chg
        return np.sum(closet_dist)


def compute_authorship_vector(
    target_commit: str, 
    change_hist: pd.DataFrame,
    target_del_add_mths: List[Tuple[str,str]] = None, 
    n_min_mod:int = 1) -> Tuple:
    """
    target_commit: 
    """ 
    from utils.hunk_utils import get_changed_method 
    if not target_del_add_mths:
        target_del_add_mths, _, _ = get_changed_method(change_hist.loc[target_commit].changes)
        if len(target_del_add_mths) <= n_min_mod:
            return (None, None)

    tcommit_time = change_hist.loc[target_commit].authored_date 
    # previous changes, INCLUDING the current one 
    upto_change_hist = change_hist.loc[change_hist.authored_date <= tcommit_time]
    authorship_vector = {mod_mth:[] for mod_mth in target_del_add_mths}
    
    for index_commit, row in list(
        upto_change_hist[['changes', 'commit.author.name']].iterrows()
        ): 
        deleted_or_added_mths, _, _ = get_changed_method(row.changes)
        intersected_mths = set(target_del_add_mths).intersection(set(deleted_or_added_mths))
        author = row['commit.author.name']
        for inter_mth in intersected_mths: 
            authorship_vector[inter_mth].append(author)
    
    return authorship_vector


def compute_jaro_wrinkler_sim(
    mth_a:Tuple[str,str], 
    mth_b:Tuple[str,str]) -> float:
    """
    """
    from nltk.metrics.distance import jaro_winkler_similarity
    # class part
    def get_core_clss(class_k:str) -> str:
        ts = class_k.split("#")
        pkg_p = ts[0].split(".")[-1]
        cls_p = " ".join(ts[1:])
        return pkg_p + " " + cls_p 

    class_a = get_core_clss(mth_a[0])
    class_b = get_core_clss(mth_b[0])
    class_sim = jaro_winkler_similarity(class_a, class_b)
    
    # method part 
    def get_mthname(mth_k:str) -> str:
        mth = mth_k.split("(")[0]
        return mth 

    mthname_a = get_mthname(mth_a[1])
    mthname_b = get_mthname(mth_b[1])
    mth_sim = jaro_winkler_similarity(mthname_a, mthname_b)
    return class_sim, mth_sim 

# compute various types of distances
def compute_distances(
    insp_time:np.datetime64, 
    which_types:List[str],
    target_mths_info:Dict, 
    mth_pairs:List[Tuple[str,str]] = None
    ) -> Dict[str,float]:
    """
    which_types -> chgdat, author, lexical sim. 
    target_mths_info
        -> keys = (m1, m2), 
        -> value = Dict -> key,value 
            = ('chgdat', np.array),  
              ('authored', List<str> (authors list)),
    
    Return:
        key = (m1,m2)
        value = Dict -> key,value
            = ('chgdat', [np.float, np.float]),
              ('authorship', np.float32), 
              ('static', np.float32)
    """
    import numpy as np
    from itertools import combinations
    
    ret_dists = {}
    if mth_pairs is None:
        target_mths = list(target_mths_info.keys())
        if 'author' in target_mths: 
            target_mths.remove('author')
        mth_pairs = list(combinations(target_mths, 2))
        
    from tqdm import tqdm
    for a_mth, b_mth in tqdm(mth_pairs):
        a_info = target_mths_info[a_mth]
        b_info = target_mths_info[b_mth]
        
        ret_dists[(a_mth, b_mth)] = {}
        for which_type in which_types:
            if which_type == 'chgdat': # 
                a_dist_arr = np.array(a_info['chgdat'])
                if a_dist_arr is None:
                    assert False # have to decide what to do with this...
                a_dist_arr.sort()
                a_dist_arr = a_dist_arr[::-1] # from the latest to the oldest

                b_dist_arr = np.array(b_info['chgdat'])
                if b_dist_arr is None:
                    assert False # have to decide
                b_dist_arr.sort()
                b_dist_arr = b_dist_arr[::-1] # from the latest to the oldest

                # exclude the current changes only for the opponent
                if insp_time is None:
                    a_to_b_dist = compute_dist(
                        a_dist_arr[1:], b_dist_arr, gran = 'day')
                    b_to_a_dist = compute_dist(
                        b_dist_arr[1:], a_dist_arr, gran = 'day')
                else:
                    a_to_b_dist = compute_dist_wsum(
                        insp_time, a_dist_arr[1:], b_dist_arr, gran = 'day')
                    b_to_a_dist = compute_dist_wsum(
                        insp_time, b_dist_arr[1:], a_dist_arr, gran = 'day')

                ret_dists[(a_mth, b_mth)][
                    which_type] = [a_to_b_dist, b_to_a_dist]
            elif which_type == 'static':
                # for now, lexical similarity between two method identifiersg
                cls_jarow_sim, mth_jarow_sim = compute_jaro_wrinkler_sim(a_mth, b_mth)
                ret_dists[(a_mth, b_mth)][
                    which_type] = np.float32(cls_jarow_sim * mth_jarow_sim) # can be anything 
            elif which_type == 'authorship':
                # compare a list of authors that modified two methods & compute 
                a_authored = np.array(a_info['authored'])
                b_authored = np.array(b_info['authored'])
                if a_authored is None:
                    assert False 
                if b_authored is None:
                    assert False
                # need to change to authorship vector 
                author_pool_v = np.append(a_authored, b_authored)
                uniq_authors = np.unique(author_pool_v)
                a_authored_v = np.zeros(len(uniq_authors))
                b_authored_v = np.zeros(len(uniq_authors))
                for i,author in enumerate(uniq_authors):
                    a_authored_v[i] = np.sum(a_authored[1:] == author)
                    b_authored_v[i] = np.sum(b_authored[1:] == author) #...
                
                if np.sum(a_authored_v) > 0 and np.sum(b_authored_v) > 0:
                    denominator = np.linalg.norm(a_authored_v) * np.linalg.norm(b_authored_v)
                    numerator = np.dot(a_authored_v, b_authored_v)
                    sim = numerator/denominator
                elif np.sum(a_authored_v) == 0 and np.sum(b_authored_v) == 0:
                    # e.g., both are introduced together this time 
                    sim = 1. 
                else:
                    # if only one of them is 0, then there is no past author information to exploit
                    sim = 0. 
                ret_dists[(a_mth, b_mth)][which_type] = np.float32(sim)
                
    return ret_dists
