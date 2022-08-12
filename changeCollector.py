import parser 
import os, sys 
from typing import Dict, Tuple, List
from tqdm import tqdm
import utils.git_utils as git_utils
from utils.git_utils import Changes
import json


def get_changed_mth(
    pos_dict:Dict, 
    lno:int) -> Tuple[str,str]:
    """
    """
    clssig = None; methsig = None
    for cls in pos_dict['classes']:
        if cls['pos'][0] <= lno and cls['pos'][1] >= lno:
            clssig = cls['fullsig']
            for method in cls['methods']:
                if lno in method['lnos']:
                    methsig = method['fullsig']
                    break
            break
    return (clssig, methsig)


def get_change_dict(
    changed_line_nums: List[int],
    changes: Changes,
    pcid: str,
    file_path: str,
    is_old: bool,
    language:str = 'java'
    ) -> Dict:
    """
    Here, the parsing take place ...
    """
    from parser import parse

    change_dict = {}
    blob = changes.get_blob(pcid, file_path, is_old) # .... -> changes.diff ? 
    #pos_dict = get_posdict_from_blob(blob)
    out = git_utils.show_file(blob.hexsha, None, changes.repo_path)
    # parsing ... -> currently support only the 
    if language == 'java':
        pos_dict = parse(out)
    else:
        # currently only support java files
        assert False

    #return None
    for line_num in changed_line_nums:
        #change_clspath, change_methname = get_clsNmeth(pos_dict, line_num)
        change_clspath, change_methname  = get_changed_mth(pos_dict, line_num)
        if str((change_clspath, change_methname)) not in change_dict.keys():
            change_dict[str((change_clspath, change_methname))] = []
        change_dict[str((change_clspath, change_methname))].append(line_num)
    return change_dict


def main(
    repo_path:str, 
    commits:List[str] = None, 
    target_cs_file:str = None, 
    postfix:str = '.java',
    dest:str = "output", 
    only_by_diff:bool = False) -> Dict:
    """
    """
    import time, javalang 

    if commits is None:
        commits = git_utils.get_target_commits(
            repo_path, target_cs_file=target_cs_file)
    
    parsed_data = {}
    diffs = {}
    #cnt = 0; thr = 1 * len(commits)/50
    size = len(commits)
    for idx, commit in enumerate(tqdm(commits)):
        #cnt += 1
        #if cnt >= thr:
        #    break 
        group_idx = ((idx - 1) // 10 + 1) * 10
        if os.path.exists(os.path.join(dest, f"{str(group_idx)}.json")):
            print("Skip group", group_idx, f"because it already exists ({idx=})")
            continue
        #
        # flush currently parsed data
        if (idx - 1) % 10 == 0 and idx != 1:
            print(f"{idx}/{size}")
            with open(os.path.join(dest, f"{idx - 1}.json"), "w") as f:
                json.dump(parsed_data, f, indent=4)
                parsed_data = {}
                
        commit_hash = commit.hexsha
        diffs[commit_hash] = {}
        parsed_commit_data = {}
        parsed_commit_data["authored_data"] = time.strftime(
            "%Y %b %d %H:%M", time.gmtime(commit.authored_date)
        )
        parsed_commit_data["commit.message"] = commit.message
        parsed_commit_data["commit.author.name"] = commit.author.name

        try:
            changes = Changes(repo_path, commit, postfix)
        except git_utils.DiffExplodeException as e:
            print(e)
            continue
        
        # below assertion -> only one parent => might be disabled later 
        #assert len(changes.diffs) < 2
        if not changes.diffs or len(changes.diffs) >= 2: # empty diff
            continue

        pcid, diff_dict = changes.diffs.popitem()
        parsed_commit_data["pcid"] = pcid
        
        change_dict = {}
        # deleted 
        parsed_commit_data["deleted"] = list(diff_dict['deleted'].keys())
        parsed_commit_data["new"] = list(diff_dict['new'].keys())
        #continue 
        ######
        # here, let's think about how we deal with deleted and new 
        # for deleted 
        for old_file_path, (_, line_num_old, _) in diff_dict['deleted'].items():
            file_change_dict = {}
            if len(line_num_old):
                try:
                    file_change_dict["old"] = get_change_dict(
                        line_num_old, changes, pcid, old_file_path, True)
                except (
                    javalang.parser.JavaSyntaxError,
                    javalang.tokenizer.LexerError,) as e:
                    file_change_dict["old"] = (
                        e.__class__.__name__,
                        old_file_path,
                        line_num_old,)
            if len(file_change_dict):
                change_dict[old_file_path] = (f"del_{old_file_path}", file_change_dict)

        # for new 
        for file_path, (_, _, line_num_new) in diff_dict['new'].items():
            file_change_dict = {}
            if len(line_num_new):
                try:
                    file_change_dict["new"] = get_change_dict(
                           line_num_new, changes, pcid, file_path, False)
                except (
                    javalang.parser.JavaSyntaxError,
                    javalang.tokenizer.LexerError,) as e:
                    file_change_dict["new"] = (
                        e.__class__.__name__,
                        file_path,
                        line_num_new,)
            if len(file_change_dict):
                change_dict[file_path] = (f"new_{file_path}", file_change_dict)

        ######
        # have to think about file renaming, copied files, etc
        for file_path, (old_file_path, line_num_old, line_num_new) in diff_dict['modified'].items():
            file_change_dict = {}
            #if not file_path.endswith(postfix):
            #    continue
            # old lines
            if len(line_num_old):
                try:
                    file_change_dict["old"] = get_change_dict(
                        line_num_old, changes, pcid, old_file_path, True)
                except (
                    javalang.parser.JavaSyntaxError,
                    javalang.tokenizer.LexerError,) as e:
                    file_change_dict["old"] = (
                        e.__class__.__name__,
                        old_file_path,
                        line_num_old,)

            # new lines
            if len(line_num_new):
                try:
                    #print ("=", file_path, changes.commit)
                    file_change_dict["new"] = get_change_dict(
                           line_num_new, changes, pcid, file_path, False)
                except (
                    javalang.parser.JavaSyntaxError,
                    javalang.tokenizer.LexerError,) as e:
                    file_change_dict["new"] = (
                        e.__class__.__name__,
                        file_path,
                        line_num_new,)
            
            # save
            if len(file_change_dict):
                #change_dict[file_path] = file_change_dict
                change_dict[file_path] = (
                    old_file_path if old_file_path != file_path else None, 
                    file_change_dict
                )
        
        if not len(change_dict):
            continue
        parsed_commit_data["changes"] = change_dict
        parsed_data[commit.hexsha] = parsed_commit_data
    
    # saving
    if len(parsed_data):
        with open(os.path.join(dest, f"{idx}.json"), "w") as f:
            json.dump(parsed_data, f, indent=4)


if __name__ == "__main__":
    import argparse 
    import time 
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type = str)
    parser.add_argument("-dst", "--dest", type = str, default = ".")
    parser.add_argument("-r", "--repo", type = str)
    parser.add_argument("-target", "--target_cs_file", type = str)
    args = parser.parse_args()

    dest = os.path.join(
        args.dest, 
        args.project) if not args.dest.endswith(args.project) else args.dest
    os.makedirs(dest, exist_ok=True)
    
    t1 = time.time()
    main(
        args.repo, 
        None, 
        target_cs_file = args.target_cs_file, 
        postfix= '.java', 
        dest = dest)

    t2 = time.time()
    print (f"Time: {t2 - t1}")