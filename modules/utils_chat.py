import yaml
import json
import os
from pathlib import Path
from PIL import Image, ImageOps
from modules.utils_shared import shared_path, load_file, is_tgwui_integrated
if is_tgwui_integrated:
    from modules.utils_tgwui import load_character

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

# credits for this code go to oobabooga
cache_folder = Path(os.path.join(shared_path.dir_user_characters, 'cache'))

# credits for this code go to oobabooga
def make_thumbnail(image):
    image = image.resize((350, round(image.size[1] / image.size[0] * 350)), Image.Resampling.LANCZOS)
    if image.size[1] > 470:
        image = ImageOps.fit(image, (350, 470), Image.LANCZOS)

    return image

# credits for this code go to oobabooga
def generate_pfp_cache(character):
    if not cache_folder.exists():
        cache_folder.mkdir()

    for path in [Path(f'{shared_path.dir_user_characters}/{character}.{extension}') for extension in ['png', 'jpg', 'jpeg']]:
        if path.exists():
            original_img = Image.open(path)
            original_img.save(Path(f'{cache_folder}/pfp_character.png'), format='PNG')

            thumb = make_thumbnail(original_img)
            thumb.save(Path(f'{cache_folder}/pfp_character_thumb.png'), format='PNG')

            return thumb

    return None

# credits for this code go to oobabooga
def build_pygmalion_style_context(data):
    context = ""
    if 'char_persona' in data and data['char_persona'] != '':
        context += f"{data['char_name']}'s Persona: {data['char_persona']}\n"

    if 'world_scenario' in data and data['world_scenario'] != '':
        context += f"Scenario: {data['world_scenario']}\n"

    if 'example_dialogue' in data and data['example_dialogue'] != '':
        context += f"{data['example_dialogue'].strip()}\n"

    context = f"{context.strip()}\n"
    return context

async def load_character_data(char_name, try_tgwui=False):
    char_data = {}
    char_dirs = [shared_path.dir_user_characters]
    if try_tgwui:
        char_dirs.append(os.path.join(shared_path.dir_tgwui, "characters"))
    for ext in ['.yaml', '.yml', '.json']:
        for char_dir in char_dirs:
            character_file = os.path.join(char_dir, f"{char_name}{ext}")
            if os.path.exists(character_file):
                loaded_data = load_file(character_file)
                if loaded_data is None:
                    continue

                char_data = dict(loaded_data)
                break  # Break the loop if data is successfully loaded

    if char_data is None:
        log.error(f"Failed to load data for: {char_name} (tried: .yaml/.yml/.json). Perhaps missing file?")

    return char_data

# credits for this code go to oobabooga
def load_bot_character(character, name1, name2, should_warn=True):
    context = greeting = ""
    greeting_field = 'greeting'
    picture = None

    filepath = None
    for extension in ["yml", "yaml", "json"]:
        filepath = Path(f'{shared_path.dir_user_characters}/{character}.{extension}')
        if filepath.exists():
            break

    if filepath is None or not filepath.exists():
        if should_warn:
            log.error(f"Could not find the character \"{character}\" inside 'user/characters/'. No character has been loaded.")
            raise ValueError
        return None, None, None, None, None

    file_contents = open(filepath, 'r', encoding='utf-8').read()
    data = json.loads(file_contents) if extension == "json" else yaml.safe_load(file_contents)

    for path in [Path(f"{cache_folder}/pfp_character.png"), Path(f"{cache_folder}/pfp_character_thumb.png")]:
        if path.exists():
            path.unlink()

    picture = generate_pfp_cache(character)

    # Finding the bot's name
    for k in ['name', 'bot', '<|bot|>', 'char_name']:
        if k in data and data[k] != '':
            name2 = data[k]
            break

    # Find the user name (if any)
    for k in ['your_name', 'user', '<|user|>']:
        if k in data and data[k] != '':
            name1 = data[k]
            break

    if 'context' in data:
        context = data['context'].strip()
    elif "char_persona" in data:
        context = build_pygmalion_style_context(data)
        greeting_field = 'char_greeting'

    greeting = data.get(greeting_field, greeting)
    return name1, name2, picture, greeting, context

def custom_load_character(char, n1, n2, try_tgwui=False):
    # Suppress ValueError if checking both locations
    should_warn = not try_tgwui
    name1, name2, picture, greeting, context = load_bot_character(char, n1, n2, should_warn)
    # Try native TGWUI function after local
    if not name2 and try_tgwui:
        try:
            name1, name2, picture, greeting, context = load_character(char, n1, n2)
        except ValueError:
            return None, None, None, None, None
        except Exception as e:
            log.error(f"Error loading character: {e}")
    return name1, name2, picture, greeting, context
