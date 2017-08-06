import os, sys, yaml, argparse, stat


def generate_tensorboard_script(logdir):
    file_name = "start_tensorboard.sh"
    with open(file_name, "w") as text_file:
        text_file.write("#!/bin/bash \n")
        text_file.write("tensorboard --logdir={}".format(logdir))
    st = os.stat(file_name)
    os.chmod(file_name, st.st_mode | stat.S_IEXEC)


def generate_multi_model_tensorboard_script(log_path_dict):
    '''function to create a srcript to start the tensoboard
    with logs from multiple models trained with different
    sets of hyper-parameters.

    Args:
        log_path_dict (dict): key-value pairs of model_name : model_log_path

    Returns:
        None
    '''
    file_name = "start_multi_model_tensorboard.sh"
    logdir = ",".join(["{}:{}".format(name, model_log_path) for name, model_log_path in log_path_dict.iteritems()])
    with open(file_name, "w") as text_file:
        text_file.write("#!/bin/bash \n")
        text_file.write("tensorboard --logdir={}".format(logdir))
    st = os.stat(file_name)
    os.chmod(file_name, st.st_mode | stat.S_IEXEC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', help='use all the yaml files', action='store_true')
    parser.add_argument('-l', '--list', nargs='+', help='list of yaml files')

    args = parser.parse_args()
    if args.all:
        mylist = os.listdir(os.getcwd())
        file_list = [file_name for file_name in mylist if ".yaml" in file_name]
    else:
        file_list = args.list
    model_dict = {}
    for file_name in file_list:
        print "file name: {}".format(file_name)
        with open(file_name, 'r') as yaml_file:
            single_model_dict = yaml.load(yaml_file)
        model_dict.update(single_model_dict)
    generate_multi_model_tensorboard_script(model_dict)

if __name__ == "__main__":
    main()
