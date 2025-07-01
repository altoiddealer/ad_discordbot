from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

from datetime import datetime, timedelta
import math
import random
import copy
import base64
import filetype
import mimetypes
import re
from typing import Union, Optional, Any

def progress_bar(value, length=15):
    try:
        filled_length = int(length * value)
        bar = ':black_square_button:' * filled_length + ':black_large_square:' * (length - filled_length)
        return f'{bar}'
    except Exception:
        return 0
    
def consolidate_prompt_strings(prompt:str) -> str:
    ''' Removes duplicate prompt strings while preserving original order '''
    if not prompt:
        return ''
    negative_prompt_list = prompt.split(', ')
    unique_values_set = set()
    unique_values_list = []
    for value in negative_prompt_list:
        if value not in unique_values_set:
            unique_values_set.add(value)
            unique_values_list.append(value)
    return ', '.join(unique_values_list)

def check_probability(probability) -> bool:
    probability = max(0.0, min(1.0, probability))
    return random.random() < probability

def fix_dict(set, req, src: str | None = None, warned: bool = False, path=""):
    was_warned = warned
    ignored_keys = ['regenerate', '_continue', 'text', 'bot_in_character_menu', 'imgmodel_name', 'tags', 'override_settings']
    for k, req_v in req.items():
        current_path = f"{path}/{k}" if path else k  # Update the current path
        if k not in set:
            if k not in ignored_keys and not warned and src:  # Only log if warned is initially False
                log.warning(f'key "{current_path}" missing from "{src}".')
                log.info(f'Applying default value for "{current_path}": {repr(req_v)}.')
                was_warned = True
            set[k] = req_v
        elif isinstance(req_v, dict):
            set[k], child_warned = fix_dict(set[k], req_v, src, warned, current_path)
            was_warned = was_warned or child_warned  # Update was_warned if any child call was warned
    return set, was_warned

def extract_key(data: dict|list, config: Union[str, dict]) -> Any:
    if isinstance(config, dict):
        path = config.get("path")
        default = config.get("default", None)
    else:
        path = config
        default = None

    if not isinstance(path, str):
        raise ValueError("Path must be a string.")

    try:
        parts = re.findall(r'[^.\[\]]+|\[\d+\]', path)
        for part in parts:
            if re.fullmatch(r'\[\d+\]', part):  # list index
                idx = int(part[1:-1])
                if isinstance(data, list):
                    data = data[idx]
                else:
                    raise TypeError(f"Expected list for index access but got {type(data).__name__}")
            else:  # dict key
                if isinstance(data, dict):
                    data = data[part]
                else:
                    raise TypeError(f"Expected dict for key '{part}' but got {type(data).__name__}")
        return data
    except (KeyError, IndexError, TypeError) as e:
        if default is not None:
            return default
        raise ValueError(f"Failed to extract path '{path}': {e}")

def set_key(data: dict|list, path: str, value: Any) -> Any:
    if not isinstance(path, str):
        raise ValueError("Path must be a string.")

    parts = re.findall(r'[^.\[\]]+|\[\d+\]', path)
    current = data

    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1

        if re.fullmatch(r'\[\d+\]', part):  # list index
            idx = int(part[1:-1])
            if not isinstance(current, list):
                raise TypeError(f"Expected list at {'.'.join(parts[:i])}, got {type(current).__name__}")

            if is_last:
                while len(current) <= idx:
                    current.append(None)
                current[idx] = value
            else:
                while len(current) <= idx:
                    current.append({})
                if not isinstance(current[idx], (dict, list)):
                    current[idx] = {}
                current = current[idx]

        else:  # dict key
            if not isinstance(current, dict):
                raise TypeError(f"Expected dict at {'.'.join(parts[:i])}, got {type(current).__name__}")
            if is_last:
                current[part] = value
            else:
                if part not in current or not isinstance(current[part], (dict, list)):
                    current[part] = {}
                current = current[part]

    return data

def remove_keys(obj, keys_to_remove:list|set):
    if isinstance(obj, dict):
        return {key: remove_keys(value, keys_to_remove)
                for key, value in obj.items()
                if key not in keys_to_remove}
    elif isinstance(obj, list):
        return [remove_keys(item, keys_to_remove) for item in obj]
    else:
        return obj

# Safer version of update_dict
def deep_merge(base: dict, override: dict) -> dict:
    '''merge 2 dicts. "override" dict has priority'''
    result = copy.deepcopy(base)
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def update_dict(d:dict, u:dict, in_place=True, merge_unmatched=True, skip_none=False) -> dict:
    """
    Recursively updates dictionary `d` with dictionary `u`.

    Arguments:
        d (dict): The original dictionary to be updated.
        u (dict): The update dictionary.
        merge_unmatched (bool): If True, include all keys from both `d` and `u`.
                                If False, only update keys in `d` that are also in `u`.
        skip_none (bool): If True, ignore keys in `u` with value None.
        in_place (bool): If True, modifies `d` in-place. If False, returns a new updated dictionary.

    Returns:
        dict: The updated dictionary (either new or `d`, depending on `in_place`).
    """
    target = d if in_place else d.copy()

    keys = set(d) | set(u) if merge_unmatched else set(d) & set(u)

    for k in keys:
        d_val = d.get(k)
        u_val = u.get(k, d_val)

        if isinstance(d_val, dict) and isinstance(u_val, dict):
            target[k] = update_dict(
                d_val, u_val,
                merge_unmatched=merge_unmatched,
                skip_none=skip_none,
                in_place=in_place
            )
        elif skip_none and u.get(k) is None:
            target[k] = d_val
        else:
            target[k] = u_val

    return target

def sum_update_dict(d, u, updates_only=False, merge_unmatched=False, in_place=True):
    """
    Recursively updates dictionary `d` by summing numeric values from dictionary `u`.

    Arguments:
        d (dict): The target dictionary to be updated.
        u (dict): The update dictionary containing values to add or override.
        updates_only (bool): If True, return only keys from `u` that were updated and matched in `d`.
        merge_unmatched (bool): If True, include keys from `u` not found in `d`.
                                If False, ignore such keys.
        in_place (bool): If True, modifies `d` in-place. If False, does not mutate `d` or `u`.

    Returns:
        dict: A dictionary containing either the full merged result or just the updated keys.
    """
    def get_decimal_places(value):
        if isinstance(value, float):
            return len(str(value).split('.')[1])
        return 0

    # Make deep copies if not in-place
    d = copy.deepcopy(d) if not in_place else d
    u = copy.deepcopy(u) if not in_place else u

    result = {} if updates_only else d

    for k, v in u.items():
        if v is None:
            continue  # skip keys with None values
        key_in_d = k in d

        if not key_in_d and not merge_unmatched:
            continue  # skip keys not in d if merge_unmatched is False

        if isinstance(v, dict):
            updated_subdict = sum_update_dict(
                d.get(k, {}), v,
                updates_only=updates_only,
                merge_unmatched=merge_unmatched,
                in_place=False  # don't mutate subdicts during recursion
            )

            if updated_subdict or not updates_only:
                d[k] = d.get(k, {}) if isinstance(d.get(k), dict) else {}
                d[k].update(updated_subdict)

            if updates_only and (key_in_d or merge_unmatched):
                result[k] = updated_subdict

        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            current_value = d.get(k, 0)
            max_decimal_places = max(get_decimal_places(current_value), get_decimal_places(v))
            new_value = round(current_value + v, max_decimal_places)
            d[k] = new_value
            if updates_only and (key_in_d or merge_unmatched):
                result[k] = new_value
        else:
            d[k] = v
            if updates_only and (key_in_d or merge_unmatched):
                result[k] = v

    return result

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

def split_at_first_comma(data: str, return_before: bool = False) -> str:
    if "," in data:
        before, after = data.split(",", 1)
        return before if return_before else after
    return data

def is_base64(s: str) -> bool:
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False

def normalize_mime_type(fmt: str) -> str:
    """
    Normalize format strings, including converting MIME types like 'image/png' to 'png'.
    """
    if '/' in fmt:
        ext = mimetypes.guess_extension(fmt)
        if ext:
            return ext.lstrip('.') # remove dot from .jpg, .json, etc.
    return fmt.lower()

def guess_format_from_headers(headers: dict) -> str|None:
    """
    Attempts to infer the file format using HTTP headers.
    """
    content_type = headers.get("Content-Type") or headers.get("content-type")
    content_disposition = headers.get("Content-Disposition") or headers.get("content-disposition")

    # Try inferring from content-type
    if content_type:
        mime_type = content_type.split(";")[0].strip()
        ext = normalize_mime_type(mime_type)
        if ext:
            return ext

    # Try extracting from content-disposition filename
    if content_disposition:
        match = re.search(r'filename="?(?P<name>[^"]+)"?', content_disposition)
        if match:
            filename = match.group("name")
            ext_match = re.search(r"\.([a-zA-Z0-9]+)$", filename)
            if ext_match:
                return ext_match.group(1)

    return None  # fallback to guess_format_from_data

def guess_format_from_data(data, default=None) -> str:
    kind = filetype.guess(data)
    return kind.mime if kind else default

def detect_audio_format(data: bytes) -> str:
    if data.startswith(b'ID3') or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    elif data.startswith(b'RIFF') and b'WAVE' in data[8:16]:
        return "wav"
    else:
        return "unknown"

def image_bytes_to_data_uri(image_bytes: bytes, mime_type: str = "image/png") -> str:
    ### USAGE (IF EVER NEEDED)
        ## Attempt to detect image format
        #image = Image.open(io.BytesIO(raw_data))
        #mime_type = Image.MIME.get(image.format, "image/png")
        # Construct image data with URI
        #data_uri = image_bytes_to_data_uri(raw_data, mime_type)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

class ValueParser:
    """
    Utility to convert loosely formatted string inputs into structured Python values:
    - Handles primitives: int, float, bool, str
    - Parses list and dict syntax with nested support
    """

    def parse_value(self, value_str: str) -> Optional[Union[bool, int, float, str, list, dict]]:
        """Main entry point to parse any string value into a Python object."""
        try:
            original = value_str
            value_str = value_str.strip()

            if value_str.startswith('{') and value_str.endswith('}'):
                parsed = self._parse_dict(value_str)
            elif value_str.startswith('[') and value_str.endswith(']'):
                parsed = self._parse_list(value_str)
            else:
                parsed = self._parse_scalar(value_str)

            if not isinstance(parsed, str) or parsed != original:
                def _shorten(val, max_len=80):
                    val_str = str(val)
                    return val_str if len(val_str) <= max_len else val_str[:max_len] + "..."
                log.info(f"[ValueParser] Parsed '{_shorten(original)}' → {_shorten(parsed)!r}")

            return parsed
        except Exception as e:
            log.error(f"[ValueParser] Error parsing value: '{value_str}' — {e}")
            return None

    def _parse_scalar(self, value_str: str) -> Union[bool, int, float, str]:
        """Attempts to convert a string to bool, int, float, or falls back to str."""
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False

        # Try int
        try:
            return int(value_str)
        except ValueError:
            pass

        # Try float
        try:
            return float(value_str)
        except ValueError:
            pass

        # Unquote if needed
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]

        return value_str

    def _parse_list(self, value_str: str) -> list:
        """Parses a list from a string like '[1, 2, {a: 1}, "text"]'."""
        inner = value_str[1:-1].strip()

        # Handle nested lists or dicts
        items = self._split_top_level(inner, sep=',')

        return [self.parse_value(item.strip()) for item in items if item.strip()]

    def _parse_dict(self, value_str: str) -> dict:
        """Parses a dictionary from a string like '{a: 1, b: "text"}'."""
        inner = value_str[1:-1].strip()
        result = {}

        pairs = self._split_top_level(inner, sep=',')

        for pair in pairs:
            if ':' not in pair:
                log.warning(f"[ValueParser] Skipping invalid dict pair: '{pair}'")
                continue
            key_str, value_str = pair.split(':', 1)
            key = key_str.strip()
            if (key.startswith('"') and key.endswith('"')) or (key.startswith("'") and key.endswith("'")):
                key = key[1:-1]
            value = self.parse_value(value_str.strip())
            result[key] = value

        return result

    def _split_top_level(self, s: str, sep: str = ',') -> list[str]:
        """
        Splits a string by `sep` but ignores separators inside brackets/braces.
        Example: "1, [2, 3], {a: 4, b: 5}" → ['1', '[2, 3]', '{a: 4, b: 5}']
        """
        result = []
        depth = 0
        current = []
        bracket_pairs = {'{': '}', '[': ']'}
        opening = bracket_pairs.keys()
        closing = bracket_pairs.values()

        for char in s:
            if char in opening:
                depth += 1
            elif char in closing:
                depth -= 1
            if char == sep and depth == 0:
                result.append(''.join(current))
                current = []
            else:
                current.append(char)

        if current:
            result.append(''.join(current))

        return result

valueparser = ValueParser()
