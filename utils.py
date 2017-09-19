import os, collections
import cPickle as pickle


def create_local_model_path(common_path, model_name):
    return os.path.join(common_path, model_name)


def create_local_log_path(common_path, model_name):
    return os.path.join(common_path, model_name, "log")


def model_meta_file(model_path, file_prefix="models"):
    meta_files = [f for f in os.listdir(model_path) if f[-5:] == '.meta']
    final_model_files = [f for f in meta_files if file_prefix in f]
    if len(final_model_files) == 0:
        raise ValueError("failed to find any model meta files in {}".format(model_path))
    if len(final_model_files) > 1:
        print "warning, more than one model meta file is found in {}".format(model_path)
    return os.path.join(model_path, final_model_files[0])


def retrieve_reverse_token_dict(picke_file_path, key='reverse_token_dict'):
    with open(picke_file_path, 'rb') as raw_input:
        content = pickle.load(raw_input)
    return content[key]


def clear_folder(absolute_folder_path):
    if not os.path.exists(absolute_folder_path):
        os.makedirs(absolute_folder_path)
        return
    for file_name in os.listdir(absolute_folder_path):
        file_path = os.path.join(absolute_folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print 'failed to clear folder {}, with error {}'.foramt(absolute_folder_path, e)
