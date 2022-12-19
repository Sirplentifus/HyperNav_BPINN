from utility import set_directory
import shutil
import os

set_directory()
print("Deleted Folders:")

def print_path(full_path):
    index = full_path.find("pacs_bpinns")
    small_path = full_path[index:]
    print(small_path)

def remove_folder(name):
    if not os.path.isdir(name): return
    shutil.rmtree(name)
    print_path(name)


src_wd = os.getcwd()
for folder in os.listdir(src_wd):
    if folder[-3:] == ".py": continue
    folder_path = os.path.join(src_wd, folder)
    cache = os.path.join(folder_path, "__pycache__")
    remove_folder(cache)

outs_wd = os.path.join(src_wd, "..\outs")
for problem_folder in os.listdir(outs_wd):
    if problem_folder == ".gitkeep": continue
    problem_path = os.path.join(outs_wd, problem_folder)
    for case_folder in os.listdir(problem_path):
        if case_folder == ".gitkeep": continue
        case_path = os.path.join(problem_path, case_folder)
        trash = os.path.join(case_path, "trash")
        remove_folder(trash)

data_wd = os.path.join(src_wd, "..\data")
for problem_folder in os.listdir(data_wd):
    if problem_folder == ".gitkeep": continue
    problem_path = os.path.join(data_wd, problem_folder) 
    trash = os.path.join(problem_path, "trash")
    remove_folder(trash)

