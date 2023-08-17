import yaml
import os


def read_yaml(path_to_yaml: str) -> dict:
    """
    Reads a yaml file
    :param path_to_yaml: [str] [path to yaml file]
    :return: [dict]
    """
    with open(path_to_yaml, mode='r') as yaml_file:
        content = yaml.safe_load(yaml_file)
        
    return content