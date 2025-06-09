import sys
import os
import json
import yaml

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
    if len(sys.argv) != 2:
        print("Usage: drag a JSON/YAML/YML file onto this script.")
        return

    input_path = sys.argv[1]
    try:
        data, filetype = load_input_file(input_path)
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
