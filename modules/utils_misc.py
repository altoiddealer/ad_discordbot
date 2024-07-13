from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

import datetime
from datetime import timedelta
import math

# Adds missing keys/values
def fix_dict(set, req):
    for k, req_v in req.items():
        if k not in set:
            set[k] = req_v
        elif isinstance(req_v, dict):
            fix_dict(set[k], req_v)
    return set

# Updates matched keys, AND adds missing keys
def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    for k in d.keys() - u.keys():
        u[k] = d[k]
    return u

# Updates matched keys, AND adds missing keys, BUT sums together number values
def sum_update_dict(d, u):
    def get_decimal_places(value):
        # Function to get the number of decimal places in a float.
        if isinstance(value, float):
            return len(str(value).split('.')[1])
        else:
            return 0
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = sum_update_dict(d.get(k, {}), v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            current_value = d.get(k, 0)
            max_decimal_places = max(get_decimal_places(current_value), get_decimal_places(v))
            d[k] = round(current_value + v, max_decimal_places)
        else:
            d[k] = v
    for k in d.keys() - u.keys():
        u[k] = d[k]
    return u

# Updates matched keys, but DOES NOT add missing keys
def update_dict_matched_keys(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def random_value_from_range(value_range):
    if isinstance(value_range, (list, tuple)) and len(value_range) == 2:
        start, end = value_range
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            num_digits = max(len(str(start).split('.')[-1]), len(str(end).split('.')[-1]))
            value = random.uniform(start, end) if isinstance(start, float) or isinstance(end, float) else random.randint(start, end)
            value = round(value, num_digits)
            return value
    log.warning(f'Invalid value range "{value_range}". Defaulting to "0".')
    return 0

def convert_lists_to_tuples(dictionary:dict) -> dict:
    for key, value in dictionary.items():
        if isinstance(value, list) and len(value) == 2 and all(isinstance(item, (int, float)) for item in value) and not any(isinstance(item, bool) for item in value):
            dictionary[key] = tuple(value)
    return dictionary

def get_time(offset=0.0, time_format=None, date_format=None):
    try:
        new_time = ''
        new_date = ''
        current_time = datetime.now()
        if offset is not None and offset != 0.0:
            if isinstance(offset, int):
                current_time = datetime.now() + timedelta(days=offset)
            elif isinstance(offset, float):
                days = math.floor(offset)
                hours = (offset - days) * 24
                current_time = datetime.now() + timedelta(days=days, hours=hours)
        time_format = time_format if time_format is not None else '%H:%M:%S'
        date_format = date_format if date_format is not None else '%Y-%m-%d'
        new_time = current_time.strftime(time_format)
        new_date = current_time.strftime(date_format)
        return new_time, new_date
    except Exception as e:
        log.error(f"Error when getting date/time: {e}")
        return '', ''

# Converts seconds to other values
def format_time(seconds) -> str:
    if seconds < 60:
        return seconds, "secs"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}", "mins"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f}", "hrs"
    else:
        days = seconds / 86400
        return f"{days:.2f}", "days"

def format_time_difference(start_time, end_time) -> str:
    # Calculate difference in seconds and round to the nearest second
    difference_seconds = round(abs(end_time - start_time))
    
    # Calculate minutes, hours, and remaining seconds
    minutes, seconds = divmod(difference_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    # Format the result based on the time difference
    if difference_seconds < 60:
        return f"{difference_seconds} seconds"
    elif difference_seconds < 3600:
        return f"{minutes} minutes"
    elif seconds == 0:
        return f"{hours} hours"
    else:
        return f"{hours} hours and {minutes} minutes"

def get_normalized_weights(target:float, list_len:int, strength:float=1.0) -> list:
    # Generate normalized weights based on a triangular distribution centered around x
    target = max(0.0, min(1.0, target)) # ensure in range of 0.0 - 1.0
    target_index = target * (list_len - 1)
    # Create a simple triangular distribution for weights centered around target_index
    weights = [1.0 / (1.0 + abs(i - target_index) ** strength) for i in range(list_len)]
    # Normalize weights to sum up to 1.0
    total_weight = sum(weights)
    return [weight / total_weight for weight in weights]