import json
import sys
import os

def process_json_file(input_file):
    # Extract the base name of the input file (without extension)
    base_filename = os.path.basename(input_file).replace('_multiple-history.json', '')
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for key, value in data.items():
        # Copy internal list to visible list
        value['visible'] = value.get('internal', []).copy()

        # Extract guild_name and channel_name for the filename
        guild_name = value.pop('guild_name', '')
        channel_name = value.pop('channel_name', '')

        # Construct the output file name
        output_file = f"{base_filename}_{guild_name}_{channel_name}.json"
        
        # Save only the modified dictionary content (without guild_name and channel_name)
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(value, out_f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_json_file>")
    else:
        input_file = sys.argv[1]
        if os.path.isfile(input_file):
            process_json_file(input_file)
        else:
            print(f"File not found: {input_file}")
