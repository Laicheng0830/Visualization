import os


def search_file(root_dir,data_type):
    
    """
    遍历输入路径下的指定格式的文件，支持输入路径下有子目录
    
    Agrs：
        root_dir:需遍历的目录文件夹路径
        data_type:需遍历的文件的后缀，如.wav
    Outputs:
        是一个列表，列表每个元素是一个子列表，子列表第一个元素是文件所在的文件夹路径，第二个元素是文件的名称，带后缀
    """    
    
    file_path_list = []    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if(os.path.splitext(file)[-1].lower() == data_type):
                file_path_list.append([root,file])
    return file_path_list


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)