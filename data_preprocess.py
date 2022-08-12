"""
Compute 
"""
from typing import Dict, List
import utils.hunk_utils as hunk_utils
import utils.git_utils as git_utils
import utils.dist_utils as dist_utils
import utils.change_utils as change_utils
import pandas as pd 
from tqdm import tqdm 
import os 
import git
import pickle

N_MIN_MOD = 1 # the minium number of modified methods -1 

def add_chgdat_vector(
    target_mths_info:Dict, 
    commit:str, 
    change_hist: pd.DataFrame) -> Dict:
    """
    """
    chgd_time_vector, _ = dist_utils.compute_change_vector(
        commit, 
        change_hist, 
        list(target_mths_info.keys()), 
        n_min_mod = N_MIN_MOD)
    
    for k in target_mths_info.keys():
        if k == 'author':
            continue
        else:
            target_mths_info[k]['chgdat'] = None     

    for k in chgd_time_vector.keys():
        target_mths_info[k]['chgdat'] = chgd_time_vector[k]
    
    return target_mths_info


def add_authorship_vector(
    target_mths_info:Dict, 
    commit:str, 
    change_hist: pd.DataFrame) -> Dict:
    """
    """
    authorship_vector = dist_utils.compute_authorship_vector(
        commit, 
        change_hist,
        list(target_mths_info.keys()), 
        n_min_mod = N_MIN_MOD)

    for k in target_mths_info.keys():
        if k == 'author':
            continue
        else:
            target_mths_info[k]['authored'] = None   

    for k in authorship_vector.keys():
        target_mths_info[k]['authored'] = authorship_vector[k]

    return target_mths_info


def collect_changes(
    project:str,
    repo_path:str,
    changes_dir:str, 
    collect:bool = False, 
    target_cs_file:str = None, 
    only_by_diff:bool = False, #### -> need to think about this 
    ) -> pd.DataFrame:
    """
    only_by_diff -> this will process the diff chunks purely based on the 
    """
    if (
        collect or 
        (not os.path.exists(changes_dir)) or (
        not os.listdir(changes_dir))
    ):
        import changeCollector 
        os.makedirs(changes_dir, exist_ok=True)
        changeCollector.main(
            repo_path,
            None, 
            target_cs_file = target_cs_file, 
            postfix= '.java', 
            dest = changes_dir, 
            only_by_diff = only_by_diff)

    changes_df = change_utils.get_change_df(project, datadir = changes_dir)
    changes_df = hunk_utils.parse_date(changes_df)
    return changes_df


def get_all_mths(
    repo_path:str,
    commit:git.Commit,
    postfix:str = '.java'):
    """
    """
    from parser import parse, get_cls_mth_sigs
    import javalang
    
    srcinfos = {}
    commit_hash = commit.hexsha
    # parse java files exist in this repo 
    file_paths = git_utils.list_all_files(
        commit_hash, 
        repo_path, 
        postfix=postfix)
    #
    for file_path in tqdm(file_paths):
        out = git_utils.show_file(commit_hash, file_path, repo_path)
        try:
            pos_dict = parse(out)
            cls_mth_sigs = get_cls_mth_sigs(pos_dict)
        except (javalang.parser.JavaSyntaxError,
                javalang.tokenizer.LexerError,) as e:
            cls_mth_sigs = None
        srcinfos[file_path] = cls_mth_sigs

    return srcinfos


def gen_chgdv_vectors_pcommit(
    changes_df:pd.DataFrame, 
    commit:git.Commit, 
    repo_path:str, 
    postfix:str = '.java', 
    srcinfo_file:str = None, 
    ) -> Dict[str,Dict]:
    """
    changes_df -> return value of collect_changes
    ... currently implementing
    """
    if srcinfo_file is not None:
        import json
        if os.path.exists(srcinfo_file):
            with open(srcinfo_file) as f:
                srcinfos = json.load(f)
        else:
            srcinfos = get_all_mths(repo_path, commit, postfix = postfix)
            with open(srcinfo_file, 'w') as f:
                f.write(json.dumps(srcinfos))
    else:
        srcinfos = get_all_mths(repo_path, commit, postfix = postfix)

    commit_hash = commit.hexsha
    author = commit.author.name
    chgdv_pcommit = {}
    #chgdv_pcommit[commit_hash] = {
        #tuple(cls_mth_sig):{
            #'author':author
        #} 
        #for cls_mth_sigs in srcinfos.values() 
        #for cls_mth_sig in cls_mth_sigs}
    chgdv = {}; cnt = 0
    for cls_mth_sigs in srcinfos.values():
        if cls_mth_sigs is not None:
            for cls_mth_sig in cls_mth_sigs:
                chgdv[tuple(cls_mth_sig)] = {'author':author}
        else: # cls_mth_sigs is None -> due to javalang error
            #chgdv[tuple(cls_mth_sig)] = None
            cnt += 1; continue
    chgdv_pcommit[commit_hash] = chgdv    
    print (f"Out of {len(srcinfos)} files, failed to parse {cnt} files")
    
    add_chgdat_vector(chgdv_pcommit[commit_hash], commit_hash, changes_df)
    add_authorship_vector(chgdv_pcommit[commit_hash], commit_hash, changes_df)  
    return chgdv_pcommit
 

if __name__ == "__main__":
    import argparse 
    import time 

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type = str)
    parser.add_argument("-dst", "--dest", type = str, default = ".")
    parser.add_argument("-r", "--repo", type = str)
    parser.add_argument("-target", "--target_cs_file", type = str)
    parser.add_argument("-chgdir", "--changes_dir", type = str)
    parser.add_argument("-collect", "--collect_chgs", action = "store_true")
    parser.add_argument("-n", "--n_min_mod", type = int, default = None)
    args = parser.parse_args()

    compute_distance = False
    project = args.project
    repo_path = args.repo
    target_cs_file = args.target_cs_file

    os.makedirs(args.dest, exist_ok=True)
    # chgdv_file -> contains the past-change-vector for the methods modified
    # in a commit, per commit
    chgdv_file = os.path.join(args.dest, f"{project}.chgdv.pkl")
    if not os.path.exists(chgdv_file):
        if args.n_min_mod is not None:
            N_MIN_MOD = args.n_min_mod

        t1 = time.time()
        changes_df = collect_changes(
            project,
            repo_path,
            args.changes_dir,
            collect = args.collect_chgs, 
            target_cs_file = target_cs_file)
        t2 = time.time()
        print (f"Time for collecting past changes: {t2-t1}")
        
        # below compute the past-chg-info vectors for every modified methods
        chgdv_pcommit = {}
        size = len(changes_df)
        t1 = time.time()
        for idx, (commit_hash, row) in enumerate(tqdm(list(changes_df.iterrows()))):
            chgdv_pcommit[commit_hash] = {}
            author = row['commit.author.name']
            chgdv_pcommit[commit_hash]['author'] = author 
            target_del_add_mths, _, _ = hunk_utils.get_changed_method(row.changes)
            for mth in target_del_add_mths:
                chgdv_pcommit[commit_hash][mth] = {}
            add_chgdat_vector(chgdv_pcommit[commit_hash], commit_hash, changes_df)
            add_authorship_vector(chgdv_pcommit[commit_hash], commit_hash, changes_df)    
    
        print (
            f"Save the past change information vector for {project} in {chgdv_file}"
        )
        with open(chgdv_file, 'wb') as f:
            pickle.dump(chgdv_pcommit, f)
        t2 = time.time()
        print (f"Time for generating past-change-info vector: {t2-t1}")
    else:
        with open(chgdv_file, 'rb') as f:
            chgdv_pcommit = pickle.load(f)

    ## compute distance
    if compute_distance:
        types_of_dist = ['chgdat', 'authorship', 'static']
        destfile = os.path.join(args.dest, f"{project}.dist.pk")
        pairwise_dists = {}
        for commit, mod_mths_info in tqdm(chgdv_pcommit.items()):
            insp_time = changes_df.loc[commit].authored_date 
            pairwise_dists[commit] = dist_utils.compute_distances(
                insp_time, 
                types_of_dist, 
                mod_mths_info
            )

        with open(destfile, 'wb') as f:
            import pickle
            pickle.dump(pairwise_dists, f)



            

