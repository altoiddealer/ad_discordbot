from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

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

def get_normalized_weights(x:float, list_len:int) -> list:
    # Generate normalized weights based on a triangular distribution centered around x
    x = max(0.0, min(1.0, x)) # ensure in range of 0.0 - 1.0
    target_index = x * (list_len - 1)
    # Create a simple triangular distribution for weights centered around target_index
    weights = [1.0 / (1.0 + abs(i - target_index)) for i in range(list_len)]
    # Normalize weights to sum up to 1.0
    total_weight = sum(weights)
    return [weight / total_weight for weight in weights]