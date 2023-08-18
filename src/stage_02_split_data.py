from src.utils.all_utils import read_yaml, create_directory, save_local_df

# argument parser
import argparse

import pandas as pd
import os
from sklearn.model_selection import train_test_split



def split_and_save(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    # save dataset in local directory
    # create path to directory: artifacts/raw_local_dir/data.csv
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    raw_local_dir = config["artifacts"]["raw_local_dir"]
    raw_local_file = config["artifacts"]["raw_local_file"]
    
    # join the path artifacts/raw_local_dir
    raw_local_file_path = os.path.join(artifacts_dir, raw_local_dir, raw_local_file)
    
    df = pd.read_csv(raw_local_file_path)

    # print(df.head())
    
    split_ratio = params["base"]["test_size"]
    random_state = params["base"]["random_state"]
    
    # split data into train and test
    train, test = train_test_split(df, test_size = split_ratio, random_state = random_state)
    
    split_data_dir = config["artifacts"]["split_data_dir"]
    
    # create directory: artifacts/split_data_dir
    create_directory([os.path.join(artifacts_dir, split_data_dir)])
    
    train_data_filename = config["artifacts"]["train"]
    test_data_filename = config["artifacts"]["test"]
    
    # set path of train.csv and test.csv
    train_data_path = os.path.join(artifacts_dir, split_data_dir, train_data_filename)
    test_data_path = os.path.join(artifacts_dir, split_data_dir, test_data_filename)
    
    # save train and test data to artifacts/split_data_dir
    for data, data_path in (train, train_data_path), (test, test_data_path):
        save_local_df(data, data_path)
        print(f"data saved at {data_path}")
    

if __name__ == "__main__":
    # get args from cmd
    args = argparse.ArgumentParser()
    
    args.add_argument("--config", "-c", default = "config/config.yaml")
    
    # read params.yaml file
    args.add_argument("--params", "-p", default = "params.yaml")
    
    
    parsed_args = args.parse_args()
    
    split_and_save(config_path = parsed_args.config, params_path = parsed_args.params)