from modules.logs import import_track, log, get_logger; import_track(__file__, fp=True)
from pathlib import Path
import json
import yaml
from modules.utils_shared import shared_path
import os
log = get_logger(__name__)
logging = log

# Function to load .json, .yml or .yaml files
def load_file(file_path, default=None, missing_okay=False):
    try:
        file_suffix = Path(file_path).suffix.lower()

        if file_suffix in [".json", ".yml", ".yaml"]:
            with open(file_path, 'r', encoding='utf-8') as file:
                if file_suffix in [".json"]:
                    data = json.load(file)
                else:
                    data = yaml.safe_load(file)

            if data is None:
                return default
            return data

        else:
            log.error(f"Unsupported file format: {file_suffix}: {file_path}")
            return default

    except FileNotFoundError:
        if not missing_okay:
            log.error(f"File not found: {file_path}")
        return default

    except Exception as e:
        log.error(f"An error occurred while reading {file_path}: {str(e)}")
        return default

def merge_base(newsettings, basekey):
    def deep_update(original, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                deep_update(original[key], value)
            else:
                original[key] = value
    try:
        base_settings = load_file(shared_path.base_settings, {})
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
        log.error(f"Error loading '{shared_path.base_settings}' ({basekey}): {e}")
        return newsettings

def save_yaml_file(file_path, data):
    try:
        with open(file_path, 'w') as file:
            yaml.dump(data, file, encoding='utf-8', default_flow_style=False, width=float("inf"), sort_keys=False)
    except Exception as e:
        log.error(f"An error occurred while saving {file_path}: {str(e)}")


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