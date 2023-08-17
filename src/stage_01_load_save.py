from src.utils.all_utils import read_yaml, create_directory

# argument parser
import argparse

import pandas as pd
import os



def get_data(config_path):
    config = read_yaml(config_path)
    
    # print(config)
    remote_data_path = config["data_source"]
    df = pd.read_csv(remote_data_path, sep = ";")
    
    # print(df.head())
    
    # save dataset in local directory
    # create path to directory: artifacts/raw_local_dir/data.csv
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    raw_local_dir = config["artifacts"]["raw_local_dir"]
    raw_local_file = config["artifacts"]["raw_local_file"]
    
    # join the path artifacts/raw_local_dir
    raw_local_dir_path = os.path.join(artifacts_dir, raw_local_dir)
    
    create_directory(dirs = [raw_local_dir_path])
    
    # join the path raw_local_dir/data.csv
    raw_local_file_path = os.path.join(raw_local_dir_path, raw_local_file)
    
    # save the data as csv file in the raw_local_dir directory
    df.to_csv(raw_local_file_path, sep = ",", index = False)


if __name__ == "__main__":
    # get args from cmd
    args = argparse.ArgumentParser()
    
    args.add_argument("--config", "-c", default = "config/config.yaml")
    
    parsed_args = args.parse_args()
    
    get_data(config_path = parsed_args.config)