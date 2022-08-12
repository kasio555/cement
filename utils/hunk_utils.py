from typing import List, Dict, Tuple
import pandas as pd 
from bson.objectid import ObjectId

def parse_date(change_hist_df):
    """
    """
    from datetime import datetime
    convert_to_ts = lambda str_chgd_date: datetime.strptime(
        str_chgd_date, '%Y %b %d %H:%M'
        ) if isinstance(str_chgd_date, str) else str_chgd_date#

    change_hist_df.authored_date = change_hist_df.authored_date.apply(convert_to_ts)
    return change_hist_df


def get_hunk_ids(
    hunk_df:pd.DataFrame,
    project:str, commit:str) -> List:
    """
    retrieve a list of diff hunks in a given commit
    """
    rows = hunk_df.loc[(hunk_df.project == project) & (hunk_df.revision_hash == commit)]
    return rows.hunk_id.values


def get_target_rows(
    hunk_df:pd.DataFrame,
    project:str, commit:str) -> List:
    """
    retrieve a list of diff hunks in a given commit
    """
    rows = hunk_df.loc[(hunk_df.project == project) & (hunk_df.revision_hash == commit)]
    return rows


def get_changed_method(changes: Dict):
    """
    directly from orignal changes_df
    """
    some_deleted, some_added, skipped_mths = [], [], []
    for afile, old_new_chgs in changes.items():
        old_afile, old_new_chgs = old_new_chgs
        try:
            # for the old files 
            if isinstance(old_new_chgs['old'], dict): # due to LexerError
                for str_cls_mth_key, chgd_lnos in old_new_chgs['old'].items():
                    # chgd_ln
                    cls_mth_key = eval(str_cls_mth_key)
                    cls_key, mth_key = cls_mth_key 
                    if mth_key is None:
                        skipped_mths.append(cls_mth_key)
                        continue  
                    some_deleted.append(cls_mth_key)
        except KeyError: # sometimes, only old or new are in the keys
            pass 

        try:
            # for the new files 
            if isinstance(old_new_chgs['new'], dict): # due to LexerError
                for str_cls_mth_key, chgd_lnos in old_new_chgs['new'].items():
                    # chgd_ln
                    cls_mth_key = eval(str_cls_mth_key)
                    cls_key, mth_key = cls_mth_key 
                    if mth_key is None:
                        skipped_mths.append(cls_mth_key)
                        continue  
                    some_added.append(cls_mth_key)       
        except KeyError: # sometimes, only old or new are in the keys
            pass 

    deleted_or_added = list(set(some_deleted + some_added))
    return deleted_or_added, some_deleted, some_added


def get_belong_chgd_mth(
    changes: Dict, 
    afile:str, 
    lno:int, 
    is_old:bool) -> Tuple[str, str]:
    """
    directly from orignal changes_df
    """
    ret_mth = None
    # currntly, chanegs does not handle added file -> 
    # SO, THIS WILL BE SKIPPED (WILL LATER CHANGED TO HANDLE THIS-> in changes + track)
    try:
        old_new_chgs = changes[afile] 
    except KeyError:
        return None 

    k = 'old' if is_old else 'new'
    try:
        if isinstance(old_new_chgs[k], dict): # due to LexerError
            for str_cls_mth_key, chgd_lnos in old_new_chgs[k].items():
                #print (str_cls_mth_key, chgd_lnos)
                if lno in chgd_lnos: 
                    cls_mth_key = eval(str_cls_mth_key)
                    if cls_mth_key[1] is None:
                        break 
                    else:
                        ret_mth = cls_mth_key
    except KeyError: # sometimes, only old or new are in the keys
        pass 
    return ret_mth


def get_all_labels(
    hunk_df:pd.DataFrame, 
    project:str, 
    hunk_id:ObjectId) -> Dict:
    """
    """
    rows = hunk_df.loc[
        (hunk_df.project == project) & (
        hunk_df.hunk_id == hunk_id)]
    
    assert len(rows) == 1
    lno_labels = rows.label_dict_consensus.values[0]
    return lno_labels
    

def get_label(
    hunk_df:pd.DataFrame, 
    project:str, 
    hunk_id:ObjectId, 
    lno:int) -> str:
    """
    """
    lno_labels = get_all_labels(hunk_df, project, hunk_id)
    return lno_labels[lno]
    

def decide_label(
    labels:List[str], 
    simple:bool) -> str:
    """
    """
    is_bugfix = 'bugfix' in labels
    if simple:
        return 'bugfix' if is_bugfix else 'no_bugfix'
    else:
        # majority ...?
        if is_bugfix:
            return 'bugfix'
        else:
            if len(set(labels)) == 1:
                return labels[0]
            else:
                ## before compute the label with most votes, filter out those 
                ## related to documentation, whitespace, 
                # and failed to meet the consensus (i.e., no_bugfix)
                labels = [label for label in labels if label != 'whitespace']
                labels = [label for label in labels if label != 'test_doc_whitespace']
                # b/c no_bugfix is assigned to those failed to meet consensus
                labels = [label for label in labels if label != 'no_bugfix'] 
                #labels = [label for label in labels if label != 'None'] # 
                labels = [label for label in labels if label != 'documentation'] 
                # remaining: refactoring, test, unrelated 
                cnts = {}
                for label in labels:
                    try:
                        cnts[label] += 1
                    except KeyError:
                        cnts[label] = 1

                max_vote = -1; final_label = None
                for label,cnt in cnts.items():
                    if max_vote < cnt:
                        max_vote = cnt
                        final_label = label 
                return final_label 


def get_final_labels(
    new_mths_w_labels:Dict, 
    old_mths_w_labels:Dict, 
    simple:bool = True) -> Dict:
    """
    labels: 
        None, bugfix, documentation, no_bugfix, refactoring, 
        test, test_doc_whitespace, unrelated, whitespace 
    simple -> 
        True: only bugfix or no_bugfix (those that are not bugfix)
        False: use all labels 

    there shouldn't be None, since we assumed commits with statments 
    without consensus to be filtered out
    """
    combined_mth_labels = {}
    for mth_k, assigned_labels in new_mths_w_labels.items():
        combined_mth_labels[mth_k] = list(assigned_labels.values())

    for mth_k, assigned_labels in old_mths_w_labels.items():
        try:
            #print (mth_k, assigned_labels)
            combined_mth_labels[mth_k].extend(list(assigned_labels.values()))
        except KeyError:
            combined_mth_labels[mth_k] = list(assigned_labels.values())
    
    # decide the final labels
    for mth_k, assigned_labels in combined_mth_labels.items():
        combined_mth_labels[mth_k] = decide_label(assigned_labels, simple)
    return combined_mth_labels
