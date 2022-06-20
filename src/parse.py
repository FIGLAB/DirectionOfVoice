import os

def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

def get_all_subfolders_from_path(path,subject_folder):
    t_subfolders = []
    folder = path + subject_folder
    tr_folders = [ f.name for f in os.scandir(folder) if f.is_dir() ]
    for t in tr_folders:
        folder =  path + subject_folder + "/" + t + "/"
        subfolders =  fast_scandir(folder)
        if len(t_subfolders) == 0:
            t_subfolders = subfolders
        else:
            t_subfolders = t_subfolders + subfolders 
    return t_subfolders

def get_all_subfolders(path,subject_folder,tr_folders):
    t_subfolders = []
    for t in tr_folders:
        folder =  path + subject_folder + t
        subfolders =  fast_scandir(folder)
        if len(t_subfolders) == 0:
            t_subfolders = subfolders
        else:
            t_subfolders = t_subfolders + subfolders 
    return t_subfolders

def get_room_demographics(folder): 
    room = folder.split("/")[-2]
    rooms_s = room.split("_")
    session = int(rooms_s[-1][-1])
    geometry = "open"
    room_id = 0
    if rooms_s[2] == "nowall":
        geometry = "closed"
    if rooms_s[1] == "downstairs" and rooms_s[2] == "nowall":
        room_id = 0
    elif rooms_s[1] == "downstairs" and rooms_s[2] == "wall":
        room_id = 1
    elif rooms_s[1] == "upstairs" and rooms_s[2] == "nowall":
        room_id = 2
    elif rooms_s[1] == "upstairs" and rooms_s[2] == "wall":
        room_id = 3
    f_name = folder.split("/")[-1]
    position = f_name.split("_")[0]
    distance = f_name.split("_")[1]
    doa = f_name.split("_")[2]
    return room, session, geometry, room_id, position, distance, doa 

