from typing import List, Sequence, Tuple, Dict
import git 
import pandas as pd 
import os, sys
import subprocess
import difflib
import contextlib

def get_target_commits(
    repo_path:str, 
    target_cs_file:str = None) -> List:
    """
    """
    if target_cs_file is not None: 
        target_commits = pd.read_csv(target_cs_file)
        target_commits_d = {tc:True for tc in target_commits.commit.values}
    else:
        target_commits = None 

    repo_path = os.path.join(repo_path)
    repo = git.Repo(repo_path)
    all_commits = []
    for commit in repo.iter_commits():
        try:
            _ = target_commits_d[commit.hexsha]
            all_commits.append(commit)
        except KeyError:
            continue
        if len(all_commits) == len(target_commits):
            break 
        elif len(all_commits) == int(len(target_commits)/4):
            print ("inspected upto 25%")
        elif len(all_commits) == int(len(target_commits)/2):
            print ("inspected upto 50%")
        elif len(all_commits) == int(3*len(target_commits)/4):
            print ("inspected upto 75%")
    ## temporary
    #all_commits = []
    #for commit in repo.iter_commits():
        #if commit.hexsha == '7f9418073856d8e622104a0030573969b52b9501':
            #all_commits.append(commit)
            #break  
    ## temporary 
    print (f"Total {len(all_commits)} out of {len(target_commits)} commits are collected")
    return all_commits


def list_all_files(
    commit_hash:str, 
    repo_path:str, 
    postfix:str = '.java') -> List[str]:
    """
    """
    cmd = f"git ls-tree --name-only -r {commit_hash}"
    output = subprocess.check_output(
        cmd, shell=True, cwd=repo_path
        ).decode("utf-8", "backslashreplace")
    files = [file for file in output.split("\n") if file.endswith(postfix)]
    return files  


def show_file(
    commit_hash:str, 
    file_path:str, 
    repo_path:str) -> str:
    """
    here .. need to think about subproject case (...)
    """
    #try:
    cmd = f"git show {commit_hash}:{file_path}" \
        if file_path is not None else f"git show {commit_hash}"
    #print (cmd)
    output = subprocess.check_output(
        cmd, shell = True, cwd = repo_path
        ).decode('utf-8', 'backslashreplace')
    #except subprocess.CalledProcessError as e:
    #    pass
    return output 


# from gitparse 
def get_changed_lines(
    a_blob: git.Blob, b_blob: git.Blob, repo_path: str
    ) -> Tuple[List[int], List[int]]:
    """
    return (line_num_a, line_num_b)
    line_num_a: a blob에서 삭제된 line_num 들
    line_num_b: b blob에서 추가된 line_num 들
    # (취소) line_num_m: a 에서 b로 바뀐 line_num 들 (pair)
    현재 사용 기준, a_blob은 old blob, b_blob은 new blob
    """
    if a_blob is not None:
        a_blob_str = show_file(a_blob.hexsha, None, repo_path)
    else:
        a_blob_str = ""
    if b_blob is not None:
        b_blob_str = show_file(b_blob.hexsha, None, repo_path)
    else:
        b_blob_str = ""
    s = difflib.SequenceMatcher(
        None, a_blob_str.splitlines(), b_blob_str.splitlines()
    )
    # line_num_a: new, line_num_b: old
    line_num_a, line_num_b = [], []
    for tag, a1, a2, b1, b2 in s.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "delete":  # a에선 없고 b에선 있는 line
            line_num_a.extend(range(a1 + 1, a2 + 1))
        elif tag == "insert":  # a에선 있고, b에선 없는 line
            line_num_b.extend(range(b1 + 1, b2 + 1))
        elif tag == "replace":  # a에서 b로 바뀌는 line
            line_num_a.extend(range(a1 + 1, a2 + 1))
            line_num_b.extend(range(b1 + 1, b2 + 1))
    return line_num_a, line_num_b


## from gitparse
#def is_modified_file(diff: git.Diff) -> bool:
    #return (
        #not diff.deleted_file
        #and not diff.new_file
        #and not diff.renamed_file
        #and not diff.copied_file
    #)
 
# from gitparse
def get_parents(commit: git.Commit) -> Sequence[git.Commit]:
    return commit.parents

# from gitparse
def get_diff(new: git.Commit, old: git.Commit) -> git.DiffIndex:
    # for this, a_blob will contain the changes in the new file and 
    # b_blob will contain the changes in the old file 
    return new.diff(old)

# from gitparse
def get_blobs(diff: git.Diff) -> tuple:
    return diff.a_blob, diff.b_blob


# from gitparse
def get_changes(
    diff: git.Diff, repo_path: str
    ) -> Tuple[str, List[int], List[int]]:
    """
    return: (file, line_num_new, line_num_old)
    line_num_new: commit 이후 버전 기준 추가된 line_num 들
    line_num_old: commit 이전 버전 기준 삭제된 line_num 들
    # (취소) line_num_mod: commit 이전 버전에서 이후 버전으로 바뀐 line_num 들 (pair)
    """
    new_blob, old_blob = get_blobs(diff)
    assert new_blob is not None or old_blob is not None
    #if old_blob is None: # meaning new file 
        #new_file_path = new_blob.path
        ## here 
        #line_num_old, line_num_new = get_changed_lines(
            #None, new_blob, repo_path)
        #return new_file_path, None, None, None 
    #elif new_blob is None: # meaning deleted file
        #old_file_path = old_blob.path
        ## here 
        #line_num_old, line_num_new = get_changed_lines(
            #old_blob, None, repo_path)
        #return None, old_file_path, None, None, 
    #else:
    new_file_path = new_blob.path if new_blob is not None else None
    old_file_path = old_blob.path if old_blob is not None else None
    line_num_old, line_num_new = get_changed_lines(
        old_blob, new_blob, repo_path)
    return new_file_path, old_file_path, line_num_old, line_num_new 


# from gitparse
class DiffExplodeException(Exception):
    pass

# from gitparse
class BlobNotFoundException(Exception):
    pass

# from gitparse
class Changes:
    def __init__(self, repo_path: str, commit: git.Commit, postfix: str):
        self.repo_path = repo_path
        self.commit = commit
        self.author = commit.author.name
        self.cid = commit.hexsha
        self.postfix = postfix
        self.parents = {}
        
        for parent in get_parents(commit):
            pcid = parent.hexsha
            self.parents[pcid] = parent
        
        self.diffs = {}
        for pcid, parent in self.parents.items():
            #self.diffs[pcid] = {'deleted':[], 'new':[], 'modified':{}}
            self.diffs[pcid] = {'deleted':{}, 'new':{}, 'modified':{}}
            if len(list(get_diff(commit, parent))) > 1000:
                raise DiffExplodeException(
                    f"{pcid} has too many diffs ({len(list(get_diff(commit, parent)))})"
                )

            for diff in get_diff(commit, parent):
                if (
                    diff.a_path is not None and diff.a_path.endswith(self.postfix)) or (
                    diff.b_path is not None and diff.b_path.endswith(self.postfix)
                ):
                    new_file_path, old_file_path, line_num_old, line_num_new = get_changes(diff, repo_path)
                else:
                    continue 
                # set line_num_old or line_num_new is None, 
                # if and only it is either new or deleted file
                if new_file_path is None: # deleted file
                    #self.diffs[pcid]['deleted'].append(old_file_path)
                    self.diffs[pcid]['deleted'][old_file_path] = (
                        old_file_path, line_num_old, None
                    ) 
                elif old_file_path is None: # new file 
                    #self.diffs[pcid]['new'].append(new_file_path)
                    self.diffs[pcid]['new'][new_file_path] = (
                        None, None, line_num_new
                    ) 
                else:
                    # meaning a modified file 
                    self.diffs[pcid]['modified'][new_file_path] = (
                        old_file_path, line_num_old, line_num_new
                    ) 
                #self.diffs[pcid][file_path] = (line_num_old, line_num_new)

    def get_blob(self, pcid, file_path, is_old):
        """
        """
        parent = self.parents[pcid]
        for diff in get_diff(self.commit, parent):
            if is_old:
                if diff.b_blob and diff.b_blob.path == file_path:
                    return diff.b_blob
            else:
                if diff.a_blob and diff.a_blob.path == file_path:
                    return diff.a_blob

        raise BlobNotFoundException(
            f"{file_path}:{is_old=} not found in {pcid}")

