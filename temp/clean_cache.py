import os
import shutil

src_wd = os.path.join(os.getcwd(), "src")
for folder in os.listdir(src_wd):
    if folder[-3:] == ".py": continue
    folder_path = os.path.join(src_wd, folder)
    cache = os.path.join(folder_path, "__pycache__")
    if os.path.isdir(cache):
        shutil.rmtree(cache)
        print(cache)
    if folder == "networks":
        folder_path = os.path.join(folder_path, "equations")
        cache = os.path.join(folder_path, "__pycache__")
        if os.path.isdir(cache):
            shutil.rmtree(cache)
            print(cache)

