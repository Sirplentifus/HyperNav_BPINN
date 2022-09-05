import os
import shutil
from datetime import datetime

def __create_keeper(path):
    """ Creation of the .gitkeep file """
    file_name = os.path.join(path,".gitkeep")
    open(file_name, 'w')

def __create_single_dir(base_path, last_path, keep = False, over = False):
    """ Creation of a single directory """
    folder_path = os.path.join(base_path, last_path)
    try: os.makedirs(folder_path)
    except: 
        if over:
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
    if keep: __create_keeper(folder_path) 
    return folder_path

def create_directories(par):
    """ Create all the directories we need to store the results """
    
    problem_folder  = __create_single_dir("../outs", par.problem, keep=True)
    algorith_folder = __create_single_dir(problem_folder, par.method, keep=True)
    case_time = f"{datetime.now().strftime('%Y.%m.%d-%H.%M.%S')}"
    case_name = par.case_name + "_" + case_time if par.utils["save_flag"] else "trash"
    if par.case_name == "default": case_name = "trash"
    case_folder = __create_single_dir(algorith_folder, case_name, over=True)
    
    path_plot   = __create_single_dir(case_folder, "plot")
    path_values = __create_single_dir(case_folder, "values")
    path_thetas = __create_single_dir(case_folder, "thetas")
    path_log    = __create_single_dir(case_folder, "log")
    __create_single_dir(path_values, "samples")

    return path_plot, path_values, path_thetas, path_log