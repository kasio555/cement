from typing import Dict, Tuple, List 
import numpy as np 

def select_top_n(
    dists:Dict, 
    top_n:int = 100
    ) -> Dict[Tuple[str,str],int]:
    """
    """
    if top_n > len(dists):
        return dists 
    else:
        sorted_dists = sorted(
            dists.items(), 
            key = lambda item:item[1], reverse = False
        )
        return {item[0]:item[1] for item in sorted_dists[:top_n]}


def cement(
    target:Tuple[str,str], 
    candidates:List[Tuple[str,str]],
    dists:Dict, 
    top_n:int = 100
) -> Dict[Tuple[str,str],int]:
    """
    from the computed distance (algorithm 1 + alpha), apply cement.
    dists -> expected to be the output of compute_distances

    Return (Dict):
        key = method id, value = rank
    """
    from tqdm import tqdm 
    from scipy.stats import rankdata

    # handle -1 case
    dist_vs_arr = np.array([v['chgdat'] for v in dists.values()])
    min_dist = np.min(dist_vs_arr)
    if min_dist < 0:
        max_dist = np.max(dist_vs_arr)
        indices_to_negd_1 = np.where(dist_vs_arr[:,0] < 0)[0]
        indices_to_negd_2 = np.where(dist_vs_arr[:,1] < 0)[0]
        dist_vs_arr[indices_to_negd_1,0] = max_dist
        dist_vs_arr[indices_to_negd_2,1] = max_dist
        for idx, k in enumerate(dists.keys()):
            dists[k]['chgdat'] = dist_vs_arr[idx]
    # handled
    
    # distance from target to candidates
    D_t_c = {}
    for cand in tqdm(candidates):
        try:
            d = dists[(target, cand)]['chgdat'][0] # target -> cand
        except KeyError:
            try: 
                d = dists[(cand, target)]['chgdat'][1]
            except KeyError:
                d = None 
        if d is not None:
            D_t_c[cand] = d

    # check ranks & set top_n
    _ranks = rankdata(list(D_t_c.values()), method = 'max')
    #_ranks_copied = [r for r in _ranks]
    _ranks.sort()
    try:
        top_n = np.where(_ranks <= top_n)[0][-1]
    except IndexError as e:
        #return {c:r for c,r in zip(_ranks_copied, list(D_t_c.keys()))}
        top_n = len(D_t_c)
    
    # sort & select top_n
    sorted_dists = sorted(
        D_t_c.items(), 
        key = lambda item:item[1], reverse = False
    )
    if top_n > len(sorted_dists):
        top_n = len(sorted_dists)
    D_t_c_n = dict(sorted_dists[:top_n])
    remain_Ds = dict(sorted_dists[top_n:])
    
    for cand in tqdm(D_t_c_n.keys()):
        try:
            d = dists[(cand, target)]['chgdat'][0] # cand -> target
        except KeyError:
            try: 
                d = dists[(target, cand)]['chgdat'][1]
            except KeyError:
                d = None 
        if d is not None:
            D_t_c_n[cand] *= d
        else:
            del D_t_c_n[cand]
    
    top_n_ranks = rankdata(
        [D_t_c_n[cand] for cand in D_t_c_n.keys()],
         method = 'max'
    )
    ranks = {c:r for c,r in zip(D_t_c_n.keys(), top_n_ranks)}

    remain_ranks = rankdata(
        [D_t_c[cand] for cand in remain_Ds.keys()],
         method = 'max'
    )
    ranks.update(
        {c:(r + top_n) for c,r in zip(remain_Ds.keys(), remain_ranks)}
    )
    return ranks 


if __name__ == "__main__":
    import os 
    import argparse
    import utils.dist_utils as dist_utils 
    import utils.git_utils as git_utils 
    from data_preprocess import collect_changes, gen_chgdv_vectors_pcommit

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repo", type = str)
    #parser.add_argument("-dst", "--dest", type = str)
    parser.add_argument("-chgdir", "--changes_dir", type = str)
    parser.add_argument("-collect", "--collect_chgs", action = "store_true")
    parser.add_argument("-p", "--project", type = str)
    parser.add_argument("-tcs", "--target_cs_file", type = str, default = None)
    args = parser.parse_args()

    changes_df = collect_changes(
        args.project,
        args.repo,
        args.changes_dir if args.changes_dir.endswith(
            args.project) else os.path.join(
                args.changes_dir, args.project),
        collect = args.collect_chgs, 
        target_cs_file = args.target_cs_file)
    print ("changes collected")
    import sys; sys.exit()
    
    commits = git_utils.get_target_commits(
        args.repo, 
        args.target_cs_file
    )
    print (f"Total {len(commits)} commits are retrived")
    # will look for only one
    # generate past-chg-vectors
    commit = commits[0]
    commit_hash = commit.hexsha
    chgdv_vector = gen_chgdv_vectors_pcommit(
        changes_df,
        commit,
        args.repo, 
        postfix = '.java', 
        srcinfo_file= f'{args.project}.{commit_hash[:8]}.srcinfos.json')
    print ("change vectors genereted")

    target = list(chgdv_vector[commit_hash].keys())[0]
    candidates = list(chgdv_vector[commit_hash].keys())[1:]
    mth_pairs = [(target, cand) for cand in candidates]
    print (f"mth pairs: {len(mth_pairs)}")

    types_of_dist = ['chgdat']
    insp_time = changes_df.loc[commit_hash].authored_date 
    dist_vector = dist_utils.compute_distances(
        None, #insp_time,
        types_of_dist,
        chgdv_vector[commit_hash], 
        mth_pairs = mth_pairs)
    print ("distance computed")

    with open('temp_wsum.pkl', 'wb') as f:
        import pickle
        pickle.dump(dist_vector, f)

    # example
    # + here, -1 dist values are handled here
    cand_ranks = cement(
        target,
        candidates,
        dist_vector, 
        top_n = 100
    )

    print ()
    print (f"For {target[0]}, {target[1]}")
    sorted_cand_ranks = sorted(
        cand_ranks.items(), 
        key = lambda item:item[1], reverse = False
    )
    #for idx, (cand, cand_r) in enumerate(cand_ranks.items()):
    for idx, (cand, rank) in enumerate(sorted_cand_ranks):
        if idx == 20:
            break
        print (f"\t{cand}: {rank}")
    