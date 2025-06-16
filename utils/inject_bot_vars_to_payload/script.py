import sys
import os
import json
import yaml
import re

def load_input_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f), 'yaml'
        elif file_path.endswith('.json'):
            return json.load(f), 'json'
        else:
            raise ValueError("Unsupported file type.")

def save_output_file(data, file_path, file_type):
    base, ext = os.path.splitext(file_path)
    new_filename = f"{base}_updated{ext}"
    with open(new_filename, 'w', encoding='utf-8') as f:
        if file_type == 'json':
            json.dump(data, f, indent=2)
        elif file_type == 'yaml':
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return new_filename

def load_overrides():
    with open(os.path.join(os.path.dirname(__file__), '_default_overrides.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

def load_replacements():
    with open(os.path.join(os.path.dirname(__file__), '_known_bot_vars_edit_with_caution.yaml'), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def extract_key(data, config):
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
            else:
                if isinstance(data, dict):
                    data = data[part]
                else:
                    raise TypeError(f"Expected dict for key '{part}' but got {type(data).__name__}")
        return data
    except (KeyError, IndexError, TypeError) as e:
        if default is not None:
            return default
        raise ValueError(f"Failed to extract path '{path}': {e}")

def resolve_placeholders(config, context:dict):
    formatted_keys = []

    def _extract_from_context(path: str):
        try:
            if re.match(r'^[a-zA-Z_]\w*$', path):
                value = context.get(path)
            else:
                value = extract_key(context, path)
            if value is not None:
                formatted_keys.append(path.split('.')[0])
            return value
        except ValueError:
            return None

    def _resolve(config):
        if isinstance(config, str):
            stripped = config.strip()
            if re.fullmatch(r"\{[^\{\}]+\}", stripped):
                key_path = stripped[1:-1]
                value = _extract_from_context(key_path)
                return value if value is not None else config

        elif isinstance(config, dict):
            return {k: _resolve(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [_resolve(item) for item in config]

        return config

    updated = _resolve(config)

    if formatted_keys:
        formatted_keys = sorted(set(formatted_keys))

    return updated, formatted_keys

def inject_and_log(data, replacements, path=""):
    changes = []

    def recursive_inject(obj, path=""):
        if isinstance(obj, dict):
            for k in obj:
                current_path = f"{path}/{k}" if path else k
                if k in replacements:
                    new_val = replacements[k]
                    obj[k] = new_val
                    changes.append(f"{current_path}: {repr(new_val)}")
                else:
                    recursive_inject(obj[k], current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                recursive_inject(item, f"{path}[{i}]")

    recursive_inject(data, path)
    return data, changes

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: drag a JSON/YAML/YML file onto this script.")
        print("Optional second argument: 'resolve'")
        return

    input_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) == 3 else "inject"
    try:
        data, filetype = load_input_file(input_path)

        if mode == "resolve":
            changes = []
            comment = data.pop('_comment', None)
            if comment:
                changes.append("Removed '_comment' block")
            overrides = data.pop('__overrides__', None)
            if overrides:
                changes.append("Removed '__overrides__' block, and used it to resolve placeholders")
            else:
                overrides = load_overrides()
                changes.append("Resolved placeholders from '_default_overrides.json'")
            updated_data, log = resolve_placeholders(data, overrides)
            changes.insert(0, log)
        else:
            replacements = load_replacements()
            updated_data, changes = inject_and_log(data, replacements)
            if '__overrides__' not in updated_data:
                overrides = load_overrides()
                # Ensure __overrides__ is at the top
                from collections import OrderedDict
                updated_data = OrderedDict(list(overrides.items()) + list(updated_data.items()))
                changes.append("** Added '__overrides__' dict into file **")

        output_path = save_output_file(updated_data, input_path, filetype)

        log_path = os.path.splitext(input_path)[0] + "_update_log.txt"
        with open(log_path, 'w', encoding='utf-8') as log_file:
            if changes:
                log_file.write("The following keys were updated:\n")
                for line in changes:
                    log_file.write(f"- {line}\n")
            else:
                log_file.write("No keys were updated.\n")

        print(f"Updated file saved as: {output_path}")
        print(f"Log saved as: {log_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
