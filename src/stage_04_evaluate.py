from src.utils.all_utils import read_yaml, create_directory, save_reports

# argument parser
import argparse

import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np


def evaluate_metrics(actual_values, predicted_values):
    
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mae = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    
    return rmse, mae, r2 



def evaluate(config_path, params_path):
    
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    # save dataset in local directory
    # create path to directory: artifacts/split_data_dir/train.csv
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    split_data_dir = config["artifacts"]["split_data_dir"]
    test_data_filename = config["artifacts"]["test"]
    
    # set path of train.csv
    test_data_path = os.path.join(artifacts_dir, split_data_dir, test_data_filename)
    
    test_df = pd.read_csv(test_data_path)
    
    test_y = test_df["quality"]
    test_x = test_df.drop("quality", axis = 1)
    
    # define path to directory: artifacts/model_dir/model.joblib
    model_dir_name = config["artifacts"]["model_dir"]
    model_file_name = config["artifacts"]["model_file_name"]
    model_path = os.path.join(artifacts_dir, model_dir_name, model_file_name)
    
    # load the trained model
    lr = joblib.load(model_path)
    
    # make predictions
    predicted_values = lr.predict(test_x)
    rsme, mae, r2 = evaluate_metrics(test_y, predicted_values)
    
    #print(rsme, mar, r2)
    scores_dir = config["artifacts"]["reports_dir"]
    scores_filename = config["artifacts"]["scores"]
    scores_dir_path = os.path.join(artifacts_dir, scores_dir)
    
    create_directory([scores_dir_path])
    
    scores_filepath = os.path.join(scores_dir_path, scores_filename)
    
    # print(f"*******{scores_filepath}")
    save_reports(report = {"rmse": rsme, "mae": mae, "r2": r2}, report_path = scores_filepath)



if __name__ == "__main__":
    # get args from cmd
    args = argparse.ArgumentParser()
    
    args.add_argument("--config", "-c", default = "config/config.yaml")
    
    # read params.yaml file
    args.add_argument("--params", "-p", default = "params.yaml")
    
    
    parsed_args = args.parse_args()
    
    evaluate(config_path = parsed_args.config, params_path = parsed_args.params)