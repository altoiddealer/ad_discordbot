import yaml
import json
import os
from pathlib import Path
from PIL import Image, ImageOps
from modules.utils_shared import shared_path, load_file, is_tgwui_integrated
from modules.utils_tgwui import get_tgwui_functions

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

tgwui_base = Path(shared_path.dir_tgwui)
user_characters_dir = Path(shared_path.dir_user_characters)

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

    for extension in ['png', 'jpg', 'jpeg']:
        path = user_characters_dir / f"{character}.{extension}"
        if path.exists():
            original_img = Image.open(path)
            pfp_path = cache_folder / 'pfp_character.png'
            thumb_path = cache_folder / 'pfp_character_thumb.png'

            original_img.save(pfp_path, format='PNG')
            thumb = make_thumbnail(original_img)
            thumb.save(thumb_path, format='PNG')

            return str(thumb_path)

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
    char_dirs = [user_characters_dir]

    if try_tgwui:
        # Check for new nested location first
        new_tgwui_chars = tgwui_base / "user_data" / "characters"
        if new_tgwui_chars.exists():
            char_dirs.append(new_tgwui_chars)
        else:
            # Fallback to old path
            old_tgwui_chars = tgwui_base / "characters"
            char_dirs.append(old_tgwui_chars)

    # Search for character files with supported extensions
    for ext in ['.yaml', '.yml', '.json']:
        for char_dir in char_dirs:
            character_file = char_dir / f"{char_name}{ext}"
            if character_file.exists():
                loaded_data = load_file(character_file)
                if loaded_data is None:
                    continue

                char_data = dict(loaded_data)
                break  # Exit inner loop
        if char_data:
            break  # Exit outer loop once data is found

    if not char_data:
        log.error(
            f"Failed to load data for: {char_name} "
            f"(tried: .yaml/.yml/.json). Perhaps missing file?"
        )

    return char_data

# credits for this code go to oobabooga
def load_bot_character(character, name1, name2, should_warn=True):
    context = ""
    greeting = ""
    greeting_field = 'greeting'
    picture = None

    filepath = None
    file_ext = None

    # Locate the character file
    for ext in ["yml", "yaml", "json"]:
        candidate = user_characters_dir / f"{character}.{ext}"
        if candidate.exists():
            filepath = candidate
            file_ext = ext
            break

    # Handle missing file
    if filepath is None:
        if should_warn:
            log.error(
                f"Could not find the character \"{character}\" "
                f"inside 'user/characters/'. No character has been loaded."
            )
            raise ValueError(f"Character '{character}' not found.")
        return None, None, None, None, None

    # Load file contents
    with filepath.open('r', encoding='utf-8') as fh:
        if file_ext == "json":
            data = json.load(fh)
        else:
            data = yaml.safe_load(fh)

    # Remove cached profile pictures if they exist
    for path in [
        cache_folder / "pfp_character.png",
        cache_folder / "pfp_character_thumb.png"
    ]:
        path.unlink(missing_ok=True)

    # Generate new profile picture cache
    picture = generate_pfp_cache(character)

    # Finding the bot's name
    for k in ['name', 'bot', '<|bot|>', 'char_name']:
        if k in data and data[k]:
            name2 = data[k]
            break

    # Finding the user name (if any)
    for k in ['your_name', 'user', '<|user|>']:
        if k in data and data[k]:
            name1 = data[k]
            break

    # Extract context and greeting
    if 'context' in data and data['context']:
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
            load_character_func = get_tgwui_functions('load_character')
            name1, name2, picture, greeting, context = load_character_func(char, n1, n2)
        except ValueError:
            return None, None, None, None, None
        except Exception as e:
            log.error(f"Error loading character: {e}")
    return name1, name2, picture, greeting, context
