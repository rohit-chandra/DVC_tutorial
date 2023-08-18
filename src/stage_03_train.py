from src.utils.all_utils import read_yaml, create_directory

# argument parser
import argparse

import pandas as pd
import os
from sklearn.linear_model import ElasticNet
import joblib

def train(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    # save dataset in local directory
    # create path to directory: artifacts/split_data_dir/train.csv
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    split_data_dir = config["artifacts"]["split_data_dir"]
    train_data_filename = config["artifacts"]["train"]
    
    # set path of train.csv
    train_data_path = os.path.join(artifacts_dir, split_data_dir, train_data_filename)
    
    train_df = pd.read_csv(train_data_path)
    
    train_y = train_df["quality"]
    train_x = train_df.drop("quality", axis = 1)
    
    # parameters of ElasticNet
    alpha = params["model_params"]["ElasticNet"]["alpha"]
    l1_ratio = params["model_params"]["ElasticNet"]["l1_ratio"]
    random_state = params["base"]["random_state"]
    
    lr = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, random_state = random_state)
    # train the model
    lr.fit(train_x, train_y)
    
    # save the model
    # define path to directory: artifacts/model_dir/model.joblib
    model_dir_name = config["artifacts"]["model_dir"]
    model_file_name = config["artifacts"]["model_file_name"]
    
    model_dir = os.path.join(artifacts_dir, model_dir_name)
    
    create_directory([model_dir])
    
    model_path = os.path.join(model_dir, model_file_name)
    
    joblib.dump(lr, model_path)

if __name__ == "__main__":
    # get args from cmd
    args = argparse.ArgumentParser()
    
    args.add_argument("--config", "-c", default = "config/config.yaml")
    
    # read params.yaml file
    args.add_argument("--params", "-p", default = "params.yaml")
    
    
    parsed_args = args.parse_args()
    
    train(config_path = parsed_args.config, params_path = parsed_args.params)