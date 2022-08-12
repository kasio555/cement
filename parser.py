import os
from typing import Dict, List, Tuple
import javalang
from tqdm import tqdm 
import utils.git_utils as git_utils


def get_lines(node: javalang.tree.Node) -> List[int]:
    """
    """ 
    import contextlib 
    lines = []
    for _, child in node.filter(javalang.tree.Node):
        with contextlib.suppress(TypeError):
            lines.append(int(child.position[0]))
    return sorted(list(set(lines)))


def get_full_cls(preClass_dict:Dict, clsname:str, inner:str) -> str:
    """
    """
    while inner:
        clsname = f"{inner}#{clsname}"
        outer_class = [
            c for c in preClass_dict if c["name"] == inner
            ][0]
        inner = outer_class["inner"]
    return clsname


def get_full_meth(method:Dict) -> str:
    """
    """
    methsig = method['name']
    if bool(method['paramtypes']):
        methsig = f"{methsig}({','.join(method['paramtypes'])})"
    return methsig  


def build_posdict(tree: javalang.tree.CompilationUnit) -> Dict:
    """
    modified Seongmin's gitparser.py build_posdict
    """
    import numpy as np 
    package_name = None if tree.package is None else tree.package.name
    ret = {"package": package_name, "classes": []}
    
    # class -> method 
    for path, node in tree.filter(javalang.tree.ClassDeclaration):
        classDeclNode: javalang.tree.ClassDeclaration = node
        class_name = classDeclNode.name
        cls_lnos = get_lines(classDeclNode)
        class_pos = (
            int(classDeclNode.position[0]),
            np.max(cls_lnos),
        )
        class_dict = {"name": class_name, "pos": class_pos, "lnos":cls_lnos, "methods": []}
        # get methods
        for method in classDeclNode.methods:
            mth_lnos = get_lines(method)
            method_dict = {
                "name": method.name,
                "pos": (int(method.position[0]), np.max(mth_lnos)),
                "lnos":mth_lnos, 
                "paramtypes": [p.type.name for p in method.parameters],
            }
            method_dict['fullsig'] = get_full_meth(method_dict)
            class_dict["methods"].append(method_dict)
        # check whether the current class has inner classes 
        # & set full classpath
        class_dict["inner"] = None
        for prev_class_dict in ret["classes"]:
            if (
                prev_class_dict["pos"][1] >= class_pos[1]
                and prev_class_dict["pos"][0] <= class_pos[0]
            ):
                class_dict["inner"] = prev_class_dict["name"] 
                break

        class_dict['fullsig'] = get_full_cls(
            ret['classes'],
            class_dict['name'], class_dict['inner'])
        class_dict['fullsig'] = f"{package_name}#{class_dict['fullsig']}"          
        ret["classes"].append(class_dict)

    return ret


def parse(file_content:str):
    """
    """
    import javalang 
    #try:
    tree: javalang.tree.CompilationUnit = javalang.parse.parse(file_content)   
    ret = build_posdict(tree)
    #except (javalang.parser.JavaSyntaxError,
    #    javalang.tokenizer.LexerError,) as e:
    #    ret = None 
    return ret 

def get_cls_mth_sigs(
    pos_dict:Dict) -> List[Tuple[str,str]]:
    cls_mth_sigs = []
    for cls in pos_dict['classes']:
        clssig = cls['fullsig']
        for method in cls['methods']:
            methsig = method['fullsig']
            cls_mth_sigs.append((clssig, methsig))
    return cls_mth_sigs

def main(
    repo:str, 
    commits:List[str] = None, 
    target_cs_file:str = None, 
    postfix:str = '.java',
    dest:str = "output") -> Dict:
    """
    """
    import pickle 
    if commits is None:
        commits = git_utils.get_target_commits(
            repo, target_cs_file=target_cs_file)
    
    srcinfo_pcommit = {}
    for idx, commit in enumerate(tqdm(commits)):
        commit_hash = commit.hexsha
        srcinfo_pcommit[commit_hash] = {}
        # parse java files exist in this repo 
        file_paths = git_utils.list_all_files(commit_hash, repo, postfix=postfix)
        for file_path in tqdm(file_paths):
            out = git_utils.show_file(commit_hash, file_path, repo)
            try:
                ret = parse(out)
                ret['file'] = file_path 
            except (javalang.parser.JavaSyntaxError,
                    javalang.tokenizer.LexerError,) as e:
                ret = None
            srcinfo_pcommit[commit_hash][file_path] = ret 
        
        if idx > 0 and idx % 10 == 0: 
            destfile = os.path.join(dest, f"{idx}.pkl")
            with open(destfile, 'wb') as f:
                pickle.dump(srcinfo_pcommit, f)
            srcinfo_pcommit = {} 


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
        target_cs_file = args.target_cs_file, 
        postfix= '.java', 
        dest = dest)
    t2 = time.time()
    print (f"Time: {t2 - t1}")