from pathlib import Path
import logging
import json
import yaml
from ad_discordbot.modules.utils_shared import shared_path
import os

# Function to load .json, .yml or .yaml files
def load_file(file_path):
    try:
        file_suffix = Path(file_path).suffix.lower()

        if file_suffix in [".json", ".yml", ".yaml"]:
            with open(file_path, 'r', encoding='utf-8') as file:
                if file_suffix in [".json"]:
                    data = json.load(file)
                else:
                    data = yaml.safe_load(file)
            return data
        else:
            logging.error(f"Unsupported file format: {file_suffix}: {file_path}")
            return None
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while reading {file_path}: {str(e)}")
        return None
    
def merge_base(newsettings, basekey):
    def deep_update(original, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                deep_update(original[key], value)
            else:
                original[key] = value
    try:
        base_settings = load_file(shared_path.base_settings)
        keys = basekey.split(',')
        current_dict = base_settings
        for key in keys:
            if key in current_dict:
                current_dict = current_dict[key].copy()
            else:
                return None
        deep_update(current_dict, newsettings) # Recursively update the dictionary
        return current_dict
    except Exception as e:
        logging.error(f"Error loading '{shared_path.base_settings}' ({basekey}): {e}")
        return newsettings
    
def save_yaml_file(file_path, data):
    try:
        with open(file_path, 'w') as file:
            yaml.dump(data, file, encoding='utf-8', default_flow_style=False, width=float("inf"), sort_keys=False)
    except Exception as e:
        logging.error(f"An error occurred while saving {file_path}: {str(e)}")
        
        
def make_fp_unique(fp):
    c = 0 # already adds 1
    name, ext = fp.rsplit('.',1)
    
    if name.endswith(')'): # check if already a (1)
        name_, num = name.rsplit(' ',1)
        if num[1:-1].isdigit() and num[0] == '(':
            name = name_
    
    format_path = f'{name} ({{0}}).{ext}'
    while os.path.isfile(fp): # TODO could optimize with get dupe num
        c += 1
        fp = format_path.format(c)
    return fp