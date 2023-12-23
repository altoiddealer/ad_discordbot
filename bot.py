from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import random
import logging
import logging.handlers
import json
import re
import glob
import os
import warnings
import discord
from discord.ext import commands
from discord import app_commands
from discord import File
from discord.ext.commands.context import Context
from discord.ext.commands import clean_content
import typing
import torch
import io
import base64
import yaml
from PIL import Image, PngImagePlugin
import requests
import sqlite3
import pprint
import aiohttp
import math
import time
from itertools import product
from pydub import AudioSegment
import copy

session_history = {'internal': [], 'visible': []}
last_user_message = {'text': [], 'llm_prompt': []}
last_bot_message = {}

### Replace TOKEN with discord bot token, update A1111 address if necessary.
from ad_discordbot import config
TOKEN = config.discord['TOKEN'] 
A1111 = config.sd['A1111']

logging.basicConfig(format='%(levelname)s [%(asctime)s]: %(message)s (Line: %(lineno)d in %(funcName)s, %(filename)s )',
                    datefmt='%Y-%m-%d %H:%M:%S', 
                    level=logging.DEBUG)

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler = logging.handlers.RotatingFileHandler(
    filename='discord.log',
    encoding='utf-8',
    maxBytes=32 * 1024 * 1024,  # 32 MiB
    backupCount=5,  # Rotate through 5 files
)

# Intercept custom bot arguments
import sys
bot_arg_list = ["--limit-history", "--token"]
bot_argv = []
for arg in bot_arg_list:
    try:
        index = sys.argv.index(arg)
    except:
        index = None
    
    if index is not None:
        bot_argv.append(sys.argv.pop(index))
        bot_argv.append(sys.argv.pop(index))

import argparse
parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54))
parser.add_argument("--token", type=str, help="Discord bot token to use their API.")
parser.add_argument("--limit-history", type=int, help="When the history gets too large, performance issues can occur. Limit the history to improve performance.")
bot_args = parser.parse_args(bot_argv)

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="You have modified the pretrained model configuration to control generation")

import modules.extensions as extensions_module
from modules.extensions import apply_extensions
from modules.chat import chatbot_wrapper, load_character
from modules import shared
from modules import chat, utils
shared.args.chat = True
from modules.LoRA import add_lora_to_model
from modules.models import load_model
from threading import Lock, Thread
shared.generation_lock = Lock()

# Update the command-line arguments based on the interface values
def update_model_parameters(state, initial=False):
    elements = ui.list_model_elements()  # the names of the parameters
    gpu_memories = []

    for i, element in enumerate(elements):
        if element not in state:
            continue

        value = state[element]
        if element.startswith('gpu_memory'):
            gpu_memories.append(value)
            continue

        if initial and vars(shared.args)[element] != vars(shared.args_defaults)[element]:
            continue

        # Setting null defaults
        if element in ['wbits', 'groupsize', 'model_type'] and value == 'None':
            value = vars(shared.args_defaults)[element]
        elif element in ['cpu_memory'] and value == 0:
            value = vars(shared.args_defaults)[element]

        # Making some simple conversions
        if element in ['wbits', 'groupsize', 'pre_layer']:
            value = int(value)
        elif element == 'cpu_memory' and value is not None:
            value = f"{value}MiB"

        if element in ['pre_layer']:
            value = [value] if value > 0 else None

        setattr(shared.args, element, value)

    found_positive = False
    for i in gpu_memories:
        if i > 0:
            found_positive = True
            break

    if not (initial and vars(shared.args)['gpu_memory'] != vars(shared.args_defaults)['gpu_memory']):
        if found_positive:
            shared.args.gpu_memory = [f"{i}MiB" for i in gpu_memories]
        else:
            shared.args.gpu_memory = None

#Load Extensions   
shared.args.extensions = []
extensions_module.available_extensions = utils.get_available_extensions()

supported_tts_clients = ['alltalk_tts', 'coqui_tts', 'silero_tts', 'elevenlabs_tts']
tts_client = config.discord['tts_settings'].get('extension', '') # tts client
tts_api_key = None
if tts_client:
    if tts_client not in supported_tts_clients:
        print(f'tts client "{tts_client}" is not yet confirmed to be work. The "/speak" command will not be registered. List of supported tts_clients: {supported_tts_clients}')

    tts_api_key = config.discord['tts_settings'].get('api_key', None)
    if tts_client == 'alltalk_tts':
        tts_voice_key = 'voice'
        tts_lang_key = 'language'
    if tts_client == 'coqui_tts':
        tts_voice_key = 'voice'
        tts_lang_key = 'language'
    if tts_client == 'silero_tts':
        tts_voice_key = 'speaker'
        tts_lang_key = 'language'
    if tts_client == 'elevenlabs_tts':
        tts_voice_key = 'selected_voice'
        tts_lang_key = ''
    if tts_client not in shared.args.extensions:
        shared.args.extensions.append(tts_client)

# Embed templates
status_embed_json = {
    "title": "Status",
    "description": "You don't have a job queued.",
    "color": 39129,
    "timestamp": datetime.now().isoformat()
}
status_embed = discord.Embed().from_dict(status_embed_json)

info_embed_json = {
    "title": "Welcome to ad_discordbot!",
    "description": """
      **/helpmenu** - Display this message
      **/character** - Change character
      **/main** - Toggle if Bot always replies, per channel
      **/image** - prompt an image to be generated by A1111 (or try "draw <subject>")
      **/speak** - if TTS settings are enabled, the bot can speak your text
      **__Changing settings__** ('.../ad\_discordbot/dict\_.yaml' files)
      **/imgmodel** - Change A1111 model & payload settings
      **/imgtags** - Change preset trigger phrases with prompt adders
      **/llmcontext** - Change context
      **/llmstate** - Change state
      **/behavior** - Change context
      """,
    "url": "https://github.com/altoiddealer/ad_discordbot"
}
info_embed = discord.Embed().from_dict(info_embed_json)

# Load text-generation-webui
# Define functions
def get_available_models():
    return sorted([re.sub(".pth$", "", item.name) for item in list(Path(f"{shared.args.model_dir}/").glob("*")) if not item.name.endswith((".txt", "-np", ".pt", ".json", ".yaml"))], key=str.lower)

def get_available_extensions():
    return sorted(set(map(lambda x: x.parts[1], Path("extensions").glob("*/script.py"))), key=str.lower)

def get_model_specific_settings(model):
    settings = shared.model_config
    model_settings = {}

    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    return model_settings

def list_model_elements():
    elements = ["cpu_memory", "auto_devices", "disk", "cpu", "bf16", "load_in_8bit", "wbits", "groupsize", "model_type", "pre_layer"]
    for i in range(torch.cuda.device_count()):
        elements.append(f"gpu_memory_{i}")
    return elements

# Update the command-line arguments based on the interface values
def update_model_parameters(state, initial=False):
    elements = list_model_elements()  # the names of the parameters
    gpu_memories = []

    for i, element in enumerate(elements):
        if element not in state:
            continue

        value = state[element]
        if element.startswith("gpu_memory"):
            gpu_memories.append(value)
            continue

        if initial and vars(shared.args)[element] != vars(shared.args_defaults)[element]:
            continue

        # Setting null defaults
        if element in ["wbits", "groupsize", "model_type"] and value == "None":
            value = vars(shared.args_defaults)[element]
        elif element in ["cpu_memory"] and value == 0:
            value = vars(shared.args_defaults)[element]

        # Making some simple conversions
        if element in ["wbits", "groupsize", "pre_layer"]:
            value = int(value)
        elif element == "cpu_memory" and value is not None:
            value = f"{value}MiB"

        setattr(shared.args, element, value)

    found_positive = False
    for i in gpu_memories:
        if i > 0:
            found_positive = True
            break

    if not (initial and vars(shared.args)["gpu_memory"] != vars(shared.args_defaults)["gpu_memory"]):
        if found_positive:
            shared.args.gpu_memory = [f"{i}MiB" for i in gpu_memories]
        else:
            shared.args.gpu_memory = None

# Loading custom settings
settings_file = None
if shared.args.settings is not None and Path(shared.args.settings).exists():
    settings_file = Path(shared.args.settings)
elif Path("settings.json").exists():
    settings_file = Path("settings.json")
if settings_file is not None:
    print(f"Loading settings from {settings_file}...")
    new_settings = json.loads(open(settings_file, "r").read())
    for item in new_settings:
        shared.settings[item] = new_settings[item]

# Default extensions
extensions_module.available_extensions = get_available_extensions()
for extension in shared.settings["default_extensions"]:
    shared.args.extensions = shared.args.extensions or []
    if extension not in shared.args.extensions:
        shared.args.extensions.append(extension)

if shared.args.extensions and len(shared.args.extensions) > 0:
    extensions_module.load_extensions()
    
available_models = get_available_models()

# Model defined through --model
if shared.args.model is not None:
    shared.model_name = shared.args.model

# Only one model is available
elif len(available_models) == 1:
    shared.model_name = available_models[0]

# Select the model from a command-line menu
elif shared.model_name == "None" or shared.args.model_menu:
    if len(available_models) == 0:
        print("No models are available! Please download at least one.")
        sys.exit(0)
    else:
        print("The following models are available:\n")
        for i, model in enumerate(available_models):
            print(f"{i+1}. {model}")
        print(f"\nWhich one do you want to load? 1-{len(available_models)}\n")
        i = int(input()) - 1
        print()
    shared.model_name = available_models[i]

# If any model has been selected, load it
if shared.model_name != "None":

    model_settings = get_model_specific_settings(shared.model_name)
    shared.settings.update(model_settings)  # hijacking the interface defaults
    update_model_parameters(model_settings, initial=True)  # hijacking the command-line arguments

    # Load the model
    shared.model, shared.tokenizer = load_model(shared.model_name)
    if shared.args.lora:
        add_lora_to_model([shared.args.lora])

# Loading the bot
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True  # Enable reaction events
intents.guild_messages = True # Allows updating topic
client = commands.Bot(command_prefix=".", intents=intents)

# client_attributes = [attr for attr in dir(client) if not callable(getattr(client, attr)) and not attr.startswith("__")]
# print("client_attributes", client_attributes)

queues = []
blocking = False
busy_drawing = False
reply_count = 0

# Start a task processing loop
task_queue = asyncio.Queue()

async def process_tasks_in_background():
    while True:
        task = await task_queue.get()
        await task

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

# Updates matched keys, but DOES NOT add missing keys
def update_dict_matched_keys(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {str(e)}")
        return None

def save_yaml_file(file_path, data):
    try:
        with open(file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, width=float("inf"))
    except Exception as e:
        print(f"An error occurred while saving {file_path}: {str(e)}")

# Reset session_history
def reset_session_history():
    global session_history
    session_history = {'internal': [], 'visible': []}

def retain_last_user_message(text, llm_prompt):
    global last_user_message
    last_user_message['text'] = text
    last_user_message['llm_prompt'] = llm_prompt

async def delete_last_message(i):
    try:
        message_ids = last_bot_message.get(i.channel.id, [])
        if message_ids:
            for message_id in message_ids:
                message = await i.channel.fetch_message(message_id)
                if message:
                    await message.delete()
            print('Bot message(s) were deleted successfully.')
        else:
            print('No matching bot message(s) found to delete in this channel.')
    except Exception as e:
        print(e)

async def update_client_profile(change_username, change_avatar, char_name):
    try:
        if change_username and client.user.display_name != char_name:
            await client.user.edit(username=char_name)
        if change_avatar:
            folder = 'characters'
            picture_path = os.path.join(folder, f'{char_name}.png')
            if os.path.exists(picture_path):
                with open(picture_path, 'rb') as f:
                    picture = f.read()
                await client.user.edit(avatar=picture)
        update_last_change() # Store the current datetime in bot.db
    except Exception as e:
        print("Error change character username or avatar:", e)

async def character_loader(source):
    try:
        # Get data using textgen-webui native character loading function
        _, name, _, greeting, context = load_character(source, '', '')
        missing_keys = [key for key, value in {'name': name, 'greeting': greeting, 'context': context}.items() if not value]
        if any (missing_keys):
            print(f'Note that character "{source}" is missing the following info:"{missing_keys}".')
        textgen_data = {'name': name, 'greeting': greeting, 'context': context}
        # Check for extra bot data
        char_path = os.path.join("characters", f"{source}.yaml")
        char_data = load_yaml_file(char_path)
        char_data = dict(char_data)
        # Gather context specific keys from the character data
        extensions_value = {}
        vc_value = ''
        char_llmcontext = {}
        for key, value in char_data.items():
            if key in ['bot_description', 'bot_emoji', 'extensions', 'use_voice_channel']:
                char_llmcontext[key] = value
                if key == 'extensions':
                    extensions_value = value
                elif key == 'use_voice_channel':
                    vc_value = value
        await update_extensions(extensions_value)
        await voice_channel(vc_value)
        # Merge any extra data with the llmcontext data
        char_llmcontext.update(textgen_data)
        # Collect behavior data
        char_behavior = char_data.get('behavior', None)
        if char_behavior is None:
            behaviors = load_yaml_file('ad_discordbot/dict_behaviors.yaml')
            char_behavior = next((b for b in behaviors if b['behavior_name'] == 'Default'), client.settings['behavior'])
            print("No character specific Behavior settings. Using 'Default' ('ad_discordbot/dict_behaviors.yaml').")
        # Collect llmstate data
        char_llmstate = char_data.get('state', None)
        if char_llmstate is None:
            llmstates = load_yaml_file('ad_discordbot/dict_llmstates.yaml')
            char_llmstate = next((s for s in llmstates if s['llmstate_name'] == 'Default'), client.settings['llmstate']['state'])
            print("No character specific LLM State settings. Using 'Default' ('ad_discordbot/dict_llmstates.yaml').")
        # Commit the character data to client.settings
        client.settings['llmcontext'] = dict(char_llmcontext) # Replace the entire dictionary key
        update_dict(client.settings['behavior'], dict(char_behavior))
        update_dict(client.settings['llmstate']['state'], dict(char_llmstate))
        # Data for saving to activesettings.yaml (skipped in on_ready())
        return char_llmcontext, char_behavior, char_llmstate
    except Exception as e:
        print("Error getting character information during character change:", e)

def update_last_change():
    try:
        conn = sqlite3.connect('bot.db')
        c = conn.cursor()
        now = datetime.now()
        formatted_now = now.strftime('%Y-%m-%d %H:%M:%S')
        c.execute('''UPDATE last_change SET timestamp = ?''', (formatted_now,))
        conn.commit()
        conn.close()
        client.database = Database()
    except Exception as e:
        print(f"An error occurred while logging time of profile update to bot.db: {e}")

async def check_last_change(i, char_name):
    try:
        change_username = config.discord.get('change_username_with_character', '')
        change_avatar = config.discord.get('change_avatar_with_character', '')
        if i and client.user.display_name != char_name:
            # Check for cooldown before allowing profile change
            if change_username or change_avatar:
                last_change = client.database.last_change
                if last_change and datetime.now() < last_change + timedelta(minutes=10):
                    remaining_cooldown = last_change + timedelta(minutes=10) - datetime.now()
                    seconds = int(remaining_cooldown.total_seconds())
                    await i.channel.send(f'Please wait {seconds} before changing character again')
        return change_username, change_avatar
    except Exception as e:
        print(f"An error occurred while checking time of last discord profile update: {e}")

async def change_character(i, char_name):
    try:
        # Check last time username / avatar were changed
        change_username, change_avatar = await check_last_change(i, char_name)
        # Load the character
        char_llmcontext, char_behavior, char_llmstate = await character_loader(char_name)
        # Update discord username / avatar
        await update_client_profile(change_username, change_avatar, char_name)
        # Save the updated active_settings to activesettings.yaml
        active_settings = load_yaml_file('ad_discordbot/activesettings.yaml')
        active_settings['llmcontext'] = char_llmcontext
        active_settings['behavior'] = char_behavior
        active_settings['llmstate']['state'] = char_llmstate
        save_yaml_file('ad_discordbot/activesettings.yaml', active_settings)
        # Ensure all settings are synchronized
        update_client_settings() # Sync updated user settings to client
        # Clear chat history
        reset_session_history()
    except Exception as e:
        await i.channel.send(f"An error occurred while changing character: {e}")
        print(f"An error occurred while changing character: {e}")
    return

async def send_long_message(channel, message_text):
    """ Splits a longer message into parts while preserving sentence boundaries and code blocks """
    global last_bot_message
    bot_messages = []  # List to store message IDs to be deleted by cont or regen function
    activelang = ''
    # Helper function to ensure even pairs of code block markdown
    def ensure_even_code_blocks(chunk_text, code_block_inserted):
        nonlocal activelang  # Declare activelang as nonlocal to modify the global variable
        code_block_languages = ["asciidoc", "autohotkey", "bash", "coffeescript", "cpp", "cs", "css", "diff", "fix", "glsl", "ini", "json", "md", "ml", "prolog", "ps", "py", "tex", "xl", "xml", "yaml"]
        code_block_count = chunk_text.count("```")
        if code_block_inserted:
            # If a code block was inserted in the previous chunk, add a leading set of "```"
            chunk_text = f"```{activelang}\n" + chunk_text
            code_block_inserted = False  # Reset the code_block_inserted flag
        code_block_count = chunk_text.count("```")
        if code_block_count % 2 == 1:
            # Check last code block for syntax like "```yaml"
            last_code_block_index = chunk_text.rfind("```")
            last_code_block = chunk_text[last_code_block_index + len("```"):].strip()
            for lang in code_block_languages:
                if last_code_block.startswith(lang):
                    activelang = lang
                    break  # Stop checking if a match is found
            # If there is an odd number of code blocks, add a closing set of "```"
            chunk_text += "```"
            code_block_inserted = True
        return chunk_text, code_block_inserted
    if len(message_text) <= 1980:
        sent_message = await channel.send(message_text)
        bot_messages.append(sent_message.id)
    else:
        code_block_inserted = False  # Initialize code_block_inserted to False
        while message_text:
            # Find the last occurrence of either a line break or the end of a sentence
            last_line_break = message_text.rfind("\n", 0, 1980)
            last_sentence_end = message_text.rfind(". ", 0, 1980)
            # Determine the index to split the string
            if last_line_break >= 0 and last_sentence_end >= 0:
                # If both a line break and a sentence end were found, choose the one that occurred last
                chunk_length = max(last_line_break, last_sentence_end) + 1
            elif last_line_break >= 0:
                # If only a line break was found, use it as the split point
                chunk_length = last_line_break + 1
            elif last_sentence_end >= 0:
                # If only a sentence end was found, use it as the split point
                chunk_length = last_sentence_end + 2  # Include the period and space
            else:
                chunk_length = 1980 # If neither was found, split at the maximum limit of 2000 characters
            chunk_text = message_text[:chunk_length]
            chunk_text, code_block_inserted = ensure_even_code_blocks(chunk_text, code_block_inserted)
            sent_message = await channel.send(chunk_text)
            bot_messages.append(sent_message.id)
            message_text = message_text[chunk_length:]
            if len(message_text) <= 1980:
                # Send the remaining text as a single chunk if it's shorter than or equal to 2000 characters
                chunk_text, code_block_inserted = ensure_even_code_blocks(message_text, code_block_inserted)
                sent_message = await channel.send(chunk_text)
                bot_messages.append(sent_message.id)
                break
    # Store the list of message IDs in the global dictionary
    last_bot_message[channel.id] = bot_messages

## Function to automatically change image models
# Set the topic of the channel and announce imgmodel as configured
async def auto_announce_imgmodel(selected_imgmodel, selected_imgmodel_name):
    try:
        # Set the topic of the channel and announce imgmodel as configured
        if config.imgmodels['auto_change_models'].get('channel_announce', ''):
            channel = client.get_channel(config.imgmodels['auto_change_models']['channel_announce'])
            if config.imgmodels['update_topic'].get('enabled', False):
                await imgmodel_update_topic(channel, selected_imgmodel, selected_imgmodel_name)
            if config.imgmodels['announce_in_chat'].get('enabled', False):
                reply = await imgmodel_announce(selected_imgmodel, selected_imgmodel_name)
                if reply:
                    await channel.send(reply)
                else:
                    await channel.send(f"Updated imgmodel settings to: {selected_imgmodel_name}")
    except Exception as e:
        print("Error announcing automatically selected imgmodel:", e)

# Select imgmodel based on mode, while avoid repeating current imgmodel
async def auto_select_imgmodel(current_imgmodel_name, imgmodels, imgmodel_names, mode='random'):   
    try:         
        if mode == 'random':
            if current_imgmodel_name:
                matched_imgmodel = None
                for imgmodel, imgmodel_name in zip(imgmodels, imgmodel_names):
                    if imgmodel_name == current_imgmodel_name:
                        matched_imgmodel = imgmodel
                        break
                if len(imgmodels) >= 2 and matched_imgmodel is not None:
                    imgmodels.remove(matched_imgmodel)
            selected_imgmodel = random.choice(imgmodels)
        elif mode == 'cycle':
            if current_imgmodel_name in imgmodel_names:
                current_index = imgmodel_names.index(current_imgmodel_name)
                next_index = (current_index + 1) % len(imgmodel_names)  # Cycle to the beginning if at the end
                selected_imgmodel = imgmodels[next_index]
            else:
                selected_imgmodel = random.choice(imgmodels) # If no image model set yet, select randomly
                print("The previous imgmodel name was not matched in list of fetched imgmodels, so cannot 'cycle'. New imgmodel was instead picked at random.")
        return selected_imgmodel
    except Exception as e:
        print("Error automatically selecting image model:", e)

# Task to auto-select an imgmodel at user defined interval
async def auto_update_imgmodel_task(mode='random'):
    while True:
        frequency = config.imgmodels['auto_change_models'].get('frequency', 1.0)
        duration = frequency*3600 # 3600 = 1 hour
        await asyncio.sleep(duration)
        try:
            active_settings = load_yaml_file('ad_discordbot/activesettings.yaml')
            current_imgmodel_name = active_settings.get('imgmodel', {}).get('imgmodel_name', '')
            imgmodel_names = [imgmodel.get('sd_model_checkpoint', imgmodel.get('imgmodel_name', '')) for imgmodel in all_imgmodels]
            # Select an imgmodel automatically
            selected_imgmodel = await auto_select_imgmodel(current_imgmodel_name, all_imgmodels, imgmodel_names, mode)
            ## Update imgmodel
            selected_imgmodel_name = selected_imgmodel.get('imgmodel_name', '')
            # Process .yaml method
            if not config.imgmodels['get_imgmodels_via_api'].get('enabled', True):
                update_dict(active_settings.get('imgmodel', {}), selected_imgmodel)
                if selected_imgmodel.get('imgmodel_imgtags', '') is not None:
                    await change_imgtags(active_settings, selected_imgmodel['imgmodel_imgtags'])
            else: # Process A1111 API method
                await change_imgmodel_api(active_settings, selected_imgmodel, selected_imgmodel_name)
            save_yaml_file('ad_discordbot/activesettings.yaml', active_settings)
            update_client_settings() # Sync updated user settings to client
            # Load the imgmodel and VAE via A1111 API
            await task_queue.put(a1111_load_imgmodel(active_settings['imgmodel'].get('override_settings', {}))) # Process this in the background
            # Update size options for /image command
            await task_queue.put(update_size_options(active_settings.get('imgmodel').get('payload', {}).get('width', 512),active_settings.get('imgmodel').get('payload', {}).get('height', 512)))
            # Set the topic of the channel and announce imgmodel as configured
            await auto_announce_imgmodel(selected_imgmodel, selected_imgmodel_name)
            print(f"Updated imgmodel settings to: {selected_imgmodel_name}")
        except Exception as e:
            print(f"Error automatically updating image model: {e}")
        #await asyncio.sleep(duration)

imgmodel_update_task = None # Global variable allows process to be cancelled and restarted (reset sleep timer)

# Helper function to start auto-select imgmodel
async def start_auto_update_imgmodel_task():
    global imgmodel_update_task
    if imgmodel_update_task:
        imgmodel_update_task.cancel()
    if config.imgmodels['auto_change_models'].get('enabled', False):
        mode = config.imgmodels['auto_change_models'].get('mode', 'random')
        imgmodel_update_task = client.loop.create_task(auto_update_imgmodel_task(mode))

voice_client = None

async def voice_channel(vc_setting):
    global voice_client
    # Start voice client if configured, and not explicitly deactivated in character settings
    if voice_client is None and (vc_setting is None or vc_setting) and int(config.discord.get('tts_settings', {}).get('play_mode', 0)) != 1:
        try:
            if tts_client and tts_client in shared.args.extensions:
                if config.discord['tts_settings'].get('voice_channel', ''):
                    voice_channel = client.get_channel(config.discord['tts_settings']['voice_channel'])
                    voice_client = await voice_channel.connect()
                else:
                    print(f'Bot launched with {tts_client}, but no voice channel is specified in config.py')
            else:
                print(f'Character "use_voice_channel" = True, and "voice channel" is specified in config.py, but no "tts_client" is specified in config.py')
        except Exception as e:
            print(f"An error occurred while connecting to voice channel:", e)
    # Stop voice client if explicitly deactivated in character settings
    if voice_client and voice_client.is_connected():
        try:
            if vc_setting is False:
                print("New context has setting to disconnect from voice channel. Disconnecting...")
                await voice_client.disconnect()
                voice_client = None
        except Exception as e:
            print(f"An error occurred while disconnecting from voice channel:", e)

last_extension_params = {}

async def update_extensions(params):
    try:
        global last_extension_params
        if last_extension_params or params:
            if last_extension_params == params:
                return # Nothing needs updating
            last_extension_params = params # Update global dict
        # Add tts API key if one is provided in config.py
        if tts_api_key:
            if tts_client not in last_extension_params:
                last_extension_params[tts_client] = {'api_key': tts_api_key}
            else:
                last_extension_params[tts_client].update({'api_key': tts_api_key})
        # Update extension settings
        if last_extension_params:
            last_extensions = list(last_extension_params.keys())
            # Update shared.settings with last_extension_params
            for param in last_extensions:
                listed_param = last_extension_params[param]
                shared.settings.update({'{}-{}'.format(param, key): value for key, value in listed_param.items()})
        else:
            print(f'** No extension params for this character. Reloading extensions with initial values. **')            
        extensions_module.load_extensions()  # Load Extensions (again)
    except Exception as e:
        print(f"An error occurred while updating character extension settings:", e)

# If first time bot script is run
async def first_run():
    try:
        for guild in client.guilds: # Iterate over all guilds the bot is a member of
            text_channels = guild.text_channels
            if text_channels:
                default_channel = text_channels[0]  # Get the first text channel of the guild
                info_embed = discord.Embed().from_dict(info_embed_json)
                await default_channel.send(embed=info_embed)
                break  # Exit the loop after sending the message to the first guild
        print('Welcome to ad_discordbot! Use "/helpmenu" to see main commands. (https://github.com/altoiddealer/ad_discordbot) for more info.')
    except Exception as e:
        if str(e).startswith("403"):
            print("The bot tried to send a welcome message, but probably does not have access/permissions to your default channel (probably #General)")
        else:
            print(f"An error occurred while welcoming user to the bot:", e)
    finally:
        conn = sqlite3.connect('bot.db')
        c = conn.cursor()
        c.execute('''UPDATE first_run SET is_first_run = ?''', (0,)) # log "0" for false
        conn.commit()
        conn.close()
        client.database = Database()

# Function to overwrite default settings with activesettings
def update_client_settings():
    try:
        defaults = Settings() # Instance of the default settings
        defaults = defaults.settings_to_dict() # Convert instance to dict
        # Current user custom settings
        active_settings = load_yaml_file('ad_discordbot/activesettings.yaml')
        active_settings = dict(active_settings)
        # Add any missing required settings
        fixed_settings = fix_dict(active_settings, defaults)
        # Commit fixed settings to the discord client (always accessible)
        client.settings = fixed_settings
        # Update client behavior
        client.behavior = Behavior() # Instance of the default behavior
        behavior = active_settings['behavior']
        client.behavior.update_behavior_dict(behavior)
    except Exception as e:
        print("Error updating client settings:", e)

## On Ready
@client.event
async def on_ready():
    try:
        update_client_settings() # initialize with defaults, updated by current activesettings.yaml
        client.database = Database() # general bot settings
        # If first time running bot
        if client.database.first_run:
            await first_run()
        # This will be either the char name found in activesettings.yaml, or the default char name
        source = client.settings['llmcontext']['name']
        # If name doesn't match the bot's discord username, try to figure out best char data to initialize with
        if source != client.user.display_name:
            sources = [
                client.user.display_name, # Try current bot name
                client.settings['llmcontext']['name'], # Try last known name
                config.discord.get('char_name', '') # Try default name in config.py
            ]
            for source in sources:
                print(f'Trying to load character "{source}"...')
                try:
                    _, char_name, _, _, _ = load_character(source, '', '')
                    if char_name:
                        print(f'Initializing with character "{source}". Use "/character" for changing characters.')                            
                        break  # Character loaded successfully, exit the loop
                except Exception as e:
                    logging.error("Error loading character:", e)
        # Load character, but don't save it's settings to activesettings (Only user actions will result in modifications)
        await character_loader(source)
        # For processing tasks in the background
        client.loop.create_task(process_tasks_in_background())
        await task_queue.put(client.tree.sync()) # Process this in the background
        # task to change image models automatically
        await task_queue.put(start_auto_update_imgmodel_task()) # Process this in the background
        # For warning about image gen taking longer on first use
        client.fresh = True
        logging.info("Bot is ready")
    except Exception as e:
        print("Error with on_ready:", e)

async def a1111_online(i):
    try:
        r = requests.get(f'{A1111}/')
        status = r.raise_for_status()
        #logging.info(status)
        return True
    except Exception as exc:
        logging.warning(exc)
        info_embed.title = f"A1111 api is not running at {A1111}"
        info_embed.description = "Launch Automatic1111 with the `--api` commandline argument\nRead more [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)"
        await i.reply(embed=info_embed)        
        return False

## Starboard feature
# Fetch images already starboard'd
try:
    data = load_yaml_file('ad_discordbot/starboard_messages.yaml')
    if data is None: starboard_posted_messages = ""
    else: starboard_posted_messages = set(data)
except FileNotFoundError:
    starboard_posted_messages = ""

@client.event
async def on_raw_reaction_add(endorsed_img):
    if not config.discord.get('starboard', {}).get('enabled', False):
        return
    channel = await client.fetch_channel(endorsed_img.channel_id)
    message = await channel.fetch_message(endorsed_img.message_id)
    total_reaction_count = 0
    if config.discord['starboard'].get('emoji_specific', False):
        for emoji in config.discord['starboard'].get('react_emojis', []):
            reaction = discord.utils.get(message.reactions, emoji=emoji)
            if reaction:
                total_reaction_count += reaction.count
    else:
        for reaction in message.reactions:
            total_reaction_count += reaction.count
    if total_reaction_count >= config.discord['starboard'].get('min_reactions', 2):
        target_channel = client.get_channel(config.discord['starboard'].get('target_channel_id', ''))
        if target_channel and message.id not in starboard_posted_messages:
            # Create the message link
            message_link = f'[Original Message]({message.jump_url})'
            # Duplicate image and post message link to target channel
            if message.attachments:
                attachment_url = message.attachments[0].url
                await target_channel.send(message_link)
                await target_channel.send(attachment_url)
            elif message.embeds and message.embeds[0].image:
                image_url = message.embeds[0].image.url
                await target_channel.send(message_link)
                await target_channel.send(image_url)
            # Add the message ID to the set and update the file
            starboard_posted_messages.add(message.id)
            save_yaml_file('ad_discordbot/starboard_messages.yaml', list(starboard_posted_messages))

queued_tts = []  # Keep track of queued tasks

def after_playback(file, error):
    global queued_tts
    if error:
        print(f'Player error: {error}, output: {error.stderr.decode("utf-8")}')
    # Check save mode setting
    if int(config.discord['tts_settings'].get('save_mode', 0)) > 0:
        try:
            os.remove(file)
        except Exception as e:
            pass
    # Check if there are queued tasks
    if queued_tts:
        # Pop the first task from the queue and play it
        next_file = queued_tts.pop(0)
        source = discord.FFmpegPCMAudio(next_file)
        voice_client.play(source, after=lambda e: after_playback(next_file, e))

async def play_in_voice_channel(file):
    global voice_client, queued_tts
    if voice_client is None:
        print("**tts response detected, but bot is not connected to a voice channel.**")
        return
    # Queue the task if audio is already playing
    if voice_client.is_playing():
        queued_tts.append(file)
    else:
        # Otherwise, play immediately
        source = discord.FFmpegPCMAudio(file)
        voice_client.play(source, after=lambda e: after_playback(file, e))

async def upload_tts_file(i, tts_resp):
    filename = os.path.basename(tts_resp)
    directory = os.path.dirname(tts_resp)
    wav_size = os.path.getsize(tts_resp)
    if wav_size <= 8388608:  # Discord's maximum file size for audio (8 MB)
        with open(tts_resp, "rb") as file:
            tts_file = File(file, filename=filename)
        await i.channel.send(file=tts_file) # lossless .wav output
    else: # convert to mp3
        bit_rate = int(config.discord['tts_settings'].get('mp3_bit_rate', 128))
        mp3_filename = os.path.splitext(filename)[0] + '.mp3'
        mp3_path = os.path.join(directory, mp3_filename)
        audio = AudioSegment.from_wav(tts_resp)
        audio.export(mp3_path, format="mp3", bitrate=f"{bit_rate}k")
        mp3_size = os.path.getsize(mp3_path) # Check the size of the MP3 file
        if mp3_size <= 8388608:  # Discord's maximum file size for audio (8 MB)
            with open(mp3_path, "rb") as file:
                mp3_file = File(file, filename=mp3_filename)
            await i.channel.send(file=mp3_file)
        else:
            await i.channel.send("The audio file exceeds Discord limitation even after conversion.")
        # if save_mode > 0: os.remove(mp3_path) # currently broken

async def process_tts_resp(i, tts_resp):
    play_mode = int(config.discord.get('tts_settings', {}).get('play_mode', 0))
    # Upload to interaction channel
    if play_mode > 0:
        await upload_tts_file(i, tts_resp)
    # Play in voice channel
    if play_mode != 1:
        await play_in_voice_channel(tts_resp) # run task in background

async def fix_user_input(user_input):
    # Fix user_input by adding any missing required settings
    defaults = Settings() # Create an instance of the default settings
    defaults = defaults.settings_to_dict() # Convert instance to dict
    default_state = defaults['llmstate']['state']
    current_state = user_input['state']
    user_input['state'] = fix_dict(current_state, default_state)
    return user_input

async def chatbot_wrapper_wrapper(user_input, save_history):
    # await fix_user_input(user_input)
    loop = asyncio.get_event_loop()

    def process_responses():
        last_resp = None
        tts_resp = None
        name1_value = user_input['state']['name1']
        name2_value = user_input['state']['name2']
        if user_input['state']['custom_stopping_strings']:
            user_input['state']['custom_stopping_strings'] += f',"{name1_value}","{name2_value}"'
        else:
            user_input['state']['custom_stopping_strings'] = f'"{name1_value}","{name2_value}"'
        if user_input['state']['custom_stopping_strings']:
            user_input['state']['stopping_strings'] += f',"{name1_value}","{name2_value}"'
        else:
            user_input['state']['stopping_strings'] = f'"{name1_value}","{name2_value}"'
        for resp in chatbot_wrapper(text=user_input['text'], state=user_input['state'], regenerate=user_input['regenerate'], _continue=user_input['_continue'], loading_message=True, for_ui=False):
            i_resp = resp['internal']
            if len(i_resp) > 0:
                resp_clean = i_resp[len(i_resp) - 1][1]
                last_resp = resp_clean
            # look for tts response
            vis_resp = resp['visible']
            if len(vis_resp) > 0:
                last_vis_resp = vis_resp[-1][-1]
                if 'audio src=' in last_vis_resp:
                    match = re.search(r'audio src="file/(.*?\.(wav|mp3))"', last_vis_resp)
                    if match:
                        tts_resp = match.group(1)
        # Retain chat history
        if not client.behavior.ignore_history and save_history:
            global session_history
            session_history['internal'].append([user_input['text'], last_resp])
            session_history['visible'].append([user_input['text'], last_resp])

        return last_resp, tts_resp  # bot's reply

    # Offload the synchronous task to a separate thread using run_in_executor
    last_resp, tts_resp = await loop.run_in_executor(None, process_responses)

    return last_resp, tts_resp

async def llm_gen(i, queues, save_history):
    global blocking
    global reply_count
    previous_user_id = None
    while len(queues) > 0:
        if blocking:
            await asyncio.sleep(1)
            continue
        blocking = True
        reply_count += 1
        user_input = queues.pop(0)
        mention = list(user_input.keys())[0]
        user_input = user_input[mention]
        last_resp, tts_resp = await chatbot_wrapper_wrapper(user_input, save_history)
        if len(queues) >= 1:
            next_in_queue = queues[0]
            next_mention = list(next_in_queue.keys())[0]
            if mention != previous_user_id or mention != next_mention:
                last_resp = f"{mention} " + last_resp
        previous_user_id = mention  # Update the current user ID for the next iteration
        logging.info("reply sent: \"" + mention + ": {'text': '" + user_input["text"] + "', 'response': '" + last_resp + "'}\"")
        await send_long_message(i.channel, last_resp)
        if tts_resp: await task_queue.put(process_tts_resp(i, tts_resp)) # Process this in background
        blocking = False

## Dynamic Context feature
async def process_dynamic_context(i, user_input, text, llm_prompt, save_history):
    dynamic_context = config.dynamic_context
    matched_presets = []
    if dynamic_context.get('enabled', False):
        for preset in dynamic_context.get('presets', []):
            triggers = preset.get('triggers', [])
            on_prefix_only = preset.get('on_prefix_only', False)
            remove_trigger_phrase = preset.get('remove_trigger_phrase', False)
            load_history = preset.get('load_history', 0)
            save_history = preset.get('save_history', True)
            add_instruct = preset.get('add_instruct', {})
            if on_prefix_only:
                if any(text.lower().startswith(trigger) for trigger in triggers):
                    matched_presets.append((preset, load_history, save_history, remove_trigger_phrase, add_instruct))
            else:
                if any(trigger in text.lower() for trigger in triggers):
                    matched_presets.append((preset, load_history, save_history, remove_trigger_phrase, add_instruct))
        if matched_presets:
            print_content = f"Dynamic Context: "
            chosen_preset = matched_presets[0] # Only apply one preset. Priority is top-to-bottom in the config file.
            swap_char_name = chosen_preset[0].get('swap_character', '')
            load_history_value = chosen_preset[0]['load_history']
            save_history_value = chosen_preset[0]['save_history']
            # Swap Character handling:
            if swap_char_name:
                try:
                    character_path = os.path.join("characters", f"{swap_char_name}.yaml")
                    if character_path:
                        char_data = load_yaml_file(character_path)
                        char_data = dict(char_data)
                        name1 = i.author.display_name
                        name2 = ''
                        if char_data.get('state', {}):
                            user_input['state'] = char_data['state']
                            user_input['state']['name1'] = name1
                        if char_data['name']:
                            name2 = char_data['name']
                            user_input['state']['name2'] = name2
                            user_input['state']['character_menu'] = name2
                        if char_data.get('context', ''):
                            context = char_data['context']
                            context = await replace_character_names(context, name1, name2)
                            user_input['state']['context'] = context
                        await fix_user_input(user_input) # Add any missing required information
                except Exception as e:
                    print(f"An error occurred while loading the YAML file for swap_character:", e)
                print_content += f"Character: {swap_char_name}"
            else: print_content += f"Character: {user_input['state']['name2']}"
            # Trigger removal handling
            if chosen_preset[0]['remove_trigger_phrase']:
                for trigger in chosen_preset[0]['triggers']:
                    index = text.lower().find(trigger)
                    if index != -1:
                        end_index = index + len(trigger)
                        if text[end_index:end_index + 1] == ' ':
                            llm_prompt = text[:index] + text[end_index + 1:]
                        else:
                            llm_prompt = text[:index] + text[end_index:]
            # Instruction handling
            if chosen_preset[0]['add_instruct']:
                llm_prompt = chosen_preset[0]['add_instruct'].format(llm_prompt)
            print_content += f" | Prompt: {llm_prompt}"
            # History handling
            if load_history_value:
                if load_history_value < 0:
                    user_input['state']['history'] = {'internal': [], 'visible': []} # No history
                if load_history_value > 0:
                    # Calculate the number of items to retain (up to the length of session_history)
                    num_to_retain = min(load_history_value, len(session_history["internal"]))
                    user_input['state']['history']['internal'] = session_history['internal'][-num_to_retain:]
                    user_input['state']['history']['visible'] = session_history['visible'][-num_to_retain:]
                print_content += f" | History: {user_input['state']['history']['visible']}"
            else: print_content += f" | History: (default)"
            if save_history_value == False:
                save_history = False
            # Print results
            if dynamic_context['print_results']:
                print(print_content)
    return user_input, llm_prompt, save_history

def determine_date():
    current_time = ''
    if config.tell_bot_time.get('enabled', False):
        current_time = 0.0
        if current_time == 0.0:
            current_time = datetime.now()
        elif isinstance(current_time, int):
            current_time = datetime.now() + timedelta(days=current_time)
        elif isinstance(current_time, float):
            days = math.floor(current_time)
            hours = (current_time - days) * 24
            current_time = datetime.now() + timedelta(days=days, hours=hours)
        else:
            return None
        current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return current_time

def user_asks_for_image(i, text):
    # Check config for image trigger settings
    image_trigger_settings = config.imgprompt_settings.get('trigger_img_gen_by_phrase', {})
    if not image_trigger_settings.get('enabled', False):
        return False
    triggers = image_trigger_settings['triggers']
    if image_trigger_settings['on_prefix_only']:
        # Check if text begins with matched trigger
        if any(text.lower().startswith(trigger) for trigger in triggers):
            return True
    else:
        # Otherwise, check if text contains trigger anywhere
        if any(trigger in text.lower() for trigger in triggers):
            return True
    # Last method to trigger an image response
    if random.random() < client.behavior.reply_with_image:
        return True
    return False

async def create_image_prompt(user_input, llm_prompt, current_time, save_history):
    try:
        user_input['text'] = llm_prompt
        if 'llmstate_name' in user_input: del user_input['llmstate_name'] # Looks more legit without this
        if current_time and config.tell_bot_time.get('message', '') and config.tell_bot_time.get('mode', 0) >= 1:
            user_input['state']['context'] = config.tell_bot_time['message'].format(current_time) + user_input['state']['context']
        last_resp, tts_resp = await chatbot_wrapper_wrapper(user_input, save_history)
        if len(last_resp) > 2000: last_resp = last_resp[:2000]
        return last_resp, tts_resp
    except Exception as e:
        print(f"An error occurred while processing image prompt: {e}")

async def create_prompt_for_llm(i, user_input, llm_prompt, current_time, save_history):
    user_input['text'] = llm_prompt
    if 'llmstate_name' in user_input: del user_input['llmstate_name'] # Looks more legit without this
    if current_time and config.tell_bot_time.get('message', '') and config.tell_bot_time.get('mode', 0) != 1:
        user_input['state']['context'] = config.tell_bot_time['message'].format(current_time) + user_input['state']['context']
    num = check_num_in_queue(i)
    if num >=10:
        await i.channel.send(f'{i.author.mention} You have 10 items in queue, please allow your requests to finish before adding more to the queue.')
    else:
        queue(i, user_input)
        #pprint.pp(user_input)
        async with i.channel.typing():
            await llm_gen(i, queues, save_history)

async def replace_character_names(text, name1, name2):
    user = config.replace_char_names.get('replace_user', '')
    char = config.replace_char_names.get('replace_char', '')
    if user: text = text.replace(f'{user}', name1)
    if char: text = text.replace(f'{char}', name2)
    return text

async def initialize_user_input(i, text):
    user_input = copy.deepcopy(client.settings['llmstate'])
    user_input['text'] = text
    name1 = i.author.display_name
    name2 = client.settings['llmcontext']['name']
    context = client.settings['llmcontext']['context']
    context = await replace_character_names(context, name1, name2)
    user_input['state']['name1'] = name1
    user_input['state']['name2'] = name2
    user_input['state']['character_menu'] = name2
    user_input['state']['context'] = context
    # check for ignore history setting / start with default history settings
    if not client.behavior.ignore_history:
        user_input['state']['history'] = session_history
    return user_input

@client.event
async def on_message(i):
    ctx = Context(message=i,prefix=None,bot=client,view=None)
    text = await commands.clean_content().convert(ctx, i.content)
    if client.behavior.bot_should_reply(i, text): pass # Bot replies
    else: return # Bot does not reply to this message.
    global busy_drawing
    if busy_drawing:
        await i.channel.send("(busy generating an image, please try again after image generation is completed)")
        return
    if not client.database.main_channels and client.user.mentioned_in(i): await main(i) # if None, set channel as main
    # if @ mentioning bot, remove the @ mention from user prompt
    if text.startswith(f"@{client.user.display_name} "):
        text = text.replace(f"@{client.user.display_name} ", "", 1)
    # name may be a default name defined in config.py
    elif text.startswith(f"@{config.discord['char_name']} "):
        text = text.replace(f"@{config.discord['char_name']} ","", 1)
    llm_prompt = text # 'text' will be retained as user's raw text (without @ mention)
    # build user_input with defaults
    user_input = await initialize_user_input(i, text)
    # apply dynamic_context settings
    save_history = True
    user_input, llm_prompt, save_history = await process_dynamic_context(i, user_input, text, llm_prompt, save_history)
    # apply datetime to prompt
    current_time = determine_date()
    # save a global copy of text/llm_prompt for /regen cmd
    retain_last_user_message(text, llm_prompt)
    if user_asks_for_image(i, text):
        busy_drawing = True
        try:
            if await a1111_online(i):
                info_embed.title = "Prompting ..."
                info_embed.description = " "
                picture_frame = await i.reply(embed=info_embed)
                async with i.channel.typing():
                    image_prompt, tts_resp = await create_image_prompt(user_input, llm_prompt, current_time, save_history)
                    await picture_frame.delete()
                    await pic(i, text, image_prompt, tts_resp, neg_prompt=None, size=None, face_swap=None, controlnet=None)
                    await i.channel.send(image_prompt)
        except Exception as e:
            print(f"An error occurred in on_message: {e}")
        busy_drawing = False
    else:
        await create_prompt_for_llm(i, user_input, llm_prompt, current_time, save_history)
    return

#----BEGIN IMAGE PROCESSING----#

async def process_image_gen(payload, picture_frame, i):
    try:
        censor_mode = None
        do_censor = False
        if config.sd['nsfw_censoring'].get('enabled', False):
            censor_mode = config.sd['nsfw_censoring'].get('mode', 0) 
            triggers = config.sd['nsfw_censoring'].get('triggers', [])
            if any(trigger in payload['prompt'].lower() for trigger in triggers):
                do_censor = True
                if censor_mode == 1:
                    info_embed.title = "Image prompt was flagged as inappropriate."
                    await i.send("Image prompt was flagged as inappropriate.")
                    await picture_frame.delete()
                    return
        images = await a1111_txt2img(payload, picture_frame)
        if not images:
            info_embed.title = "No images generated"
            await i.send("No images were generated.")
            await picture_frame.edit(delete_after=5)
        else:
            client.fresh = False
            # Ensure the output directory exists
            output_dir = 'ad_discordbot/sd_outputs/'
            os.makedirs(output_dir, exist_ok=True)
            # If the censor mode is 0 (blur), prefix the image file with "SPOILER_"
            image_files = [discord.File(f'temp_img_{idx}.png', filename=f'SPOILER_temp_img_{idx}.png') if do_censor and censor_mode == 0 else discord.File(f'temp_img_{idx}.png') for idx in range(len(images))]
            await i.channel.send(files=image_files)
            # Save the image at index 0 with the date/time naming convention
            os.rename(f'temp_img_0.png', f'{output_dir}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_0.png')
            # Delete temporary image files except for the one at index 0
            for idx in range(1, len(images)):
                os.remove(f'temp_img_{idx}.png')
            await picture_frame.delete()
    except asyncio.TimeoutError:
        info_embed.title = "Timeout error"
        await i.send("Timeout error")
        await picture_frame.edit(delete_after=5)

def clean_payload(payload):
    # Remove duplicate negative prompts
    negative_prompt_list = payload.get('negative_prompt', '').split(', ')
    unique_values_set = set()
    unique_values_list = []
    for value in negative_prompt_list:
        if value not in unique_values_set:
            unique_values_set.add(value)
            unique_values_list.append(value)
    processed_negative_prompt = ', '.join(unique_values_list)
    payload['negative_prompt'] = processed_negative_prompt
    # Delete unwanted keys
    if not config.sd['extensions'].get('controlnet_enabled', False):
        del payload['alwayson_scripts']['controlnet'] # Delete all 'controlnet' keys if disabled by config
    if not config.sd['extensions'].get('reactor_enabled', False):
        del payload['alwayson_scripts']['reactor'] # Delete all 'reactor' keys if disabled by config
    # Workaround for denoising strength A1111 bug (will be fixed in next A1111 version)
    if not payload.get('enable_hr', False):
        payload['denoising_strength'] = None
    # Delete all empty keys
    keys_to_delete = []
    for key, value in payload.items():
        if value == "":
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del payload[key]
    return payload

def process_faces(payload, text):
    # Process Reactor (face swap)
    trigger_params = config.imgprompt_settings.get('trigger_faces_by_phrase', {})
    if trigger_params.get('enabled', False):
        matched_presets = []     
        for preset in trigger_params.get('presets', []):
            triggers = preset.get('triggers', '')
            preset_payload = {}  # Create a dictionary to hold faces for each preset
            if any(trigger in text.lower() for trigger in triggers):
                for key, value in preset.items():
                    if key == 'face_swap':
                        face_file_path = f'ad_discordbot/swap_faces/{value}'
                        if os.path.exists(face_file_path) and os.path.isfile(face_file_path):
                            if face_file_path.endswith((".txt", ".png", ".jpg")):
                                if face_file_path.endswith((".txt")):
                                    with open(face_file_path, "r") as txt_file:
                                        payload['alwayson_scripts']['reactor']['args'][0] = txt_file.read()
                                else:
                                    with open(face_file_path, "rb") as image_file:
                                        image_data = image_file.read()
                                        faceswapimg = base64.b64encode(image_data).decode('utf-8')
                                        payload['alwayson_scripts']['reactor']['args'][0] = faceswapimg
                                payload['alwayson_scripts']['reactor']['args'][1] = True
                            else:
                                print("Invalid file for face swap (must be .txt, .png, or .jpg).")
                        else:
                            print(f"File not found '{face_file_path}'.")
                matched_presets.append((preset, preset_payload))
        for preset, preset_payload in matched_presets:
            payload.update(preset_payload)  # Update the payload with each preset's settings
    return payload

def random_value_from_range(value_range):
    if isinstance(value_range, tuple) and len(value_range) == 2:
        start, end = value_range
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            num_digits = max(len(str(start).split('.')[-1]), len(str(end).split('.')[-1]))
            value = random.uniform(start, end) if isinstance(start, float) or isinstance(end, float) else random.randint(start, end)
            value = round(value, num_digits)
            return value
    print(f'Invalid range "{value_range}". Defaulting to "0".')
    return 0

def process_param_variances(payload, presets_list):
    processed_variances = []
    for presets in presets_list:
        processed_presets = {}
        for key, value in presets.items():
            if isinstance(value, dict):
                processed_presets[key] = process_param_variances([value])
            elif isinstance(value, (list, tuple)):
                if isinstance(value, tuple):
                    processed_presets[key] = random_value_from_range(value)
                elif isinstance(value, (int, float)):
                    processed_presets[key] = value
                else:
                    print(f'Invalid params "{key}", "{value}".')
        processed_variances.append(processed_presets)
    for processed_params in processed_variances:
        for key, value in processed_params.items():
            if key in payload:
                payload[key] += value
    return payload

def process_payload_mods(payload, text):
    # Process triggered mods
    trigger_params = config.imgprompt_settings.get('trigger_img_params_by_phrase', {})
    if trigger_params.get('enabled', False):
        matched_presets = []
        for preset in trigger_params.get('presets', {}):
            triggers = preset.get('triggers', '')
            preset_payload = {}  # Create a dictionary to hold payload settings for each preset
            if any(trigger in text.lower() for trigger in triggers):
                for key, value in preset.items():
                    if key != 'triggers': # All key-value pairs except "triggers"
                        preset_payload[key] = value
                matched_presets.append((preset, preset_payload))
        for preset, preset_payload in matched_presets:
            payload.update(preset_payload)  # Update the payload with each preset's settings
    return payload

def apply_lrctl(filtered_presets):
    scaling_settings = [v for k, v in config.sd.get('extensions', {}).get('lrctl', {}).items() if 'scaling' in k]
    scaling_settings = scaling_settings if scaling_settings else ['']
    matching_presets = [preset for preset in filtered_presets if re.findall(r'<lora:[^:]+:[^>]+>', preset.get('positive_prompt', ''))]
    if len(matching_presets) >= config.sd['extensions']['lrctl']['min_loras']:
        for i, preset in enumerate(matching_presets):
            positive_prompt = preset.get('positive_prompt', '')
            matches = re.findall(r'<lora:[^:]+:[^>]+>', positive_prompt)
            if matches:
                for match in matches:
                    lora_weight_match = re.search(r'(?<=:)\d+(\.\d+)?', match) # Extract lora weight
                    if lora_weight_match:
                        lora_weight = float(lora_weight_match.group())
                        # Selecting the appropriate scaling based on the index
                        scaling_key = f'lora_{i + 1}_scaling' if i+1 < len(scaling_settings) else 'additional_loras_scaling'
                        scaling_values = config.sd.get('extensions', {}).get('lrctl', {}).get(scaling_key, '')
                        if scaling_values:
                            scaling_factors = [round(float(factor.split('@')[0]) * lora_weight, 2) for factor in scaling_values.split(',')]
                            scaling_steps = [float(step.split('@')[1]) for step in scaling_values.split(',')]
                            # Construct/apply the calculated lora-weight string
                            new_lora_weight_str = f'{",".join(f"{factor}@{step}" for factor, step in zip(scaling_factors, scaling_steps))}'
                            updated_match = match.replace(str(lora_weight), new_lora_weight_str)
                            new_positive_prompt = positive_prompt.replace(match, updated_match)
                            preset['positive_prompt'] = new_positive_prompt
                            print(f"{match} was applied '{scaling_key}: {scaling_values}'. Result: {updated_match}")
    return filtered_presets

def expand_dictionary(presets):
    def expand_value(value):
        # Split the value on commas
        parts = value.split(',')
        expanded_values = []

        for part in parts:
            # Check if the part contains curly brackets
            if '{' in part and '}' in part:
                # Use regular expression to find all curly bracket groups
                matches = re.findall(r'\{([^}]+)\}', part)
                permutations = list(product(*[match.split('|') for match in matches]))

                # Replace each curly bracket group with permutations
                for perm in permutations:
                    expanded_part = part
                    for match in matches:
                        expanded_part = expanded_part.replace('{' + match + '}', perm[matches.index(match)], 1)
                    expanded_values.append(expanded_part)
            else:
                expanded_values.append(part)

        return ','.join(expanded_values)

    expanded_presets = []

    for preset in presets:
        expanded_preset = {}
        for key, value in preset.items():
            expanded_preset[key] = expand_value(value)
        expanded_presets.append(expanded_preset)

    return expanded_presets

def apply_presets(payload, presets, i, text):
    if presets:
        presets = expand_dictionary(presets)
        # Determine text to search
        search_mode = config.imgprompt_settings.get('trigger_search_mode', 'userllm')
        if search_mode == 'user':
            search_text = text
        elif search_mode == 'llm':
            search_text = payload["prompt"]
        elif search_mode == 'userllm':
            search_text = text + " " + payload["prompt"]
        else:
            print("Invalid search mode")
            return
        matched_presets = []
        for preset in presets:
            triggers = [t.strip() for t in preset['trigger'].split(',')]
            for trigger in triggers:
                trigger_regex = r"\b{}\b".format(re.escape(trigger.lower()))
                match = re.search(trigger_regex, search_text.lower())
                if match:
                    matched_presets.append((preset, match.start()))
                    break
        if matched_presets:
            # Sorts the matched presets based on their position index in search_text when matched
            matched_presets.sort(key=lambda x: x[1])
            matched_presets = [preset for preset, _ in matched_presets]
            # Collect 'trump' parameters for each matched preset
            trump_params = []
            for preset in matched_presets:
                if 'trumps' in preset:
                    trump_params.extend([param.strip() for param in preset['trumps'].split(',')])
            trump_params = list(set(trump_params)) # Remove duplicates from the trump_params list
            # Compare each matched preset's trigger to the list of trump parameters
            # Ignore the preset if its trigger exactly matches any trump parameter
            filtered_presets = [preset for preset in matched_presets if not any(trigger.lower() in trump_params for trigger in preset['trigger'].split(','))]
            # Extension which may apply scaling to triggered LORAs, if enabled in config.py
            if config.sd['extensions'].get('lrctl', {}).get('enabled', False):
                filtered_presets = apply_lrctl(filtered_presets)
            for preset in reversed(filtered_presets):
                payload['negative_prompt'] += preset.get('negative_prompt', '')
                if config.imgprompt_settings.get('insert_imgtags_in_prompt', True):
                    matched_text = None
                    triggers = [t.strip() for t in preset['trigger'].split(',')]
                    # Iterate through the triggers for the current preset
                    for trigger in triggers:
                        trigger_regex = r"\b{}\b".format(re.escape(trigger.lower()))
                        #pattern = re.escape(trigger.lower())
                        match = re.search(trigger_regex, payload['prompt'].lower())
                        if match:
                            matched_text = match.group(0)
                            break
                    if matched_text:
                        # Find the index of the first occurrence of matched_text
                        insert_index = payload['prompt'].lower().index(matched_text) + len(matched_text)
                        insert_text = preset.get('positive_prompt', '')
                        payload['prompt'] = payload['prompt'][:insert_index] + insert_text + payload['prompt'][insert_index:]
                        filtered_presets.remove(preset)
            # All remaining are appended to prompt
            for preset in filtered_presets:
                payload['prompt'] += preset.get('positive_prompt', '')

def apply_suffix2(payload, positive_prompt_suffix2, positive_prompt_suffix2_blacklist):
    if positive_prompt_suffix2:
        blacklist_phrases = positive_prompt_suffix2_blacklist.split(',')
        prompt_text = re.sub(r'<([^<>]+)>', r'\1', payload['prompt'])  # Remove angle brackets while preserving contents
        prompt_text = prompt_text.lower()
        if not any(phrase.lower() in prompt_text for phrase in blacklist_phrases):
            payload['prompt'] += positive_prompt_suffix2

async def pic(i, text, image_prompt, tts_resp=None, neg_prompt=None, size=None, face_swap=None, controlnet=None):
    global busy_drawing
    busy_drawing = True
    try:
        if await a1111_online(i):
            info_embed.title = "Processing"
            info_embed.description = " ... "  # await check_a1111_progress()
            if client.fresh:
                info_embed.description = "First request tends to take a long time, please be patient"
            picture_frame = await i.channel.send(embed=info_embed)
            info_embed.title = "Sending prompt to A1111 ..."
            # Initialize payload settings
            payload = {"prompt": image_prompt, "negative_prompt": '', "width": 512, "height": 512, "steps": 20}
            if neg_prompt: payload.update({"negative_prompt": neg_prompt})
            # Apply settings from imgmodel configuration
            imgmodel_payload = copy.deepcopy(client.settings['imgmodel'].get('payload', {}))
            payload.update(imgmodel_payload)
            payload['override_settings'] = copy.deepcopy(client.settings['imgmodel'].get('override_settings', {}))
            # Process payload triggers
            process_payload_mods(payload, text)
            # Process payload param variances
            if config.sd['param_variances'].get('enabled', False):
                payload = process_param_variances(payload, config.sd['param_variances'].get('presets', []))
            # Process face swap triggers
            process_faces(payload, text)
            if size: payload.update(size)
            # Process extensions
            if face_swap:
                payload['alwayson_scripts']['reactor']['args'][0] = face_swap # image in base64 format
                payload['alwayson_scripts']['reactor']['args'][1] = True # Enable
            if controlnet: payload['alwayson_scripts']['controlnet']['args'][0].update(controlnet)
            # Process ImgTagss
            positive_prompt_prefix = copy.copy(client.settings['imgtag'].get('positive_prompt_prefix', ''))
            positive_prompt_suffix = copy.copy(client.settings['imgtag'].get('positive_prompt_suffix', ''))
            positive_prompt_suffix2 = copy.copy(client.settings['imgtag'].get('positive_prompt_suffix2', ''))
            positive_prompt_suffix2_blacklist = copy.copy(client.settings['imgtag'].get('positive_prompt_suffix2_blacklist', ''))
            negative_prompt_prefix = copy.copy(client.settings['imgtag'].get('negative_prompt_prefix', ''))
            presets = copy.deepcopy(client.settings['imgtag'].get('presets', []))

            if positive_prompt_prefix:
                payload['prompt'] = f'{positive_prompt_prefix} {image_prompt}'
            if positive_prompt_suffix:
                payload['prompt'] += positive_prompt_suffix
            if negative_prompt_prefix:
                payload['negative_prompt'] += negative_prompt_prefix
            
            apply_presets(payload, presets, i, text)
            apply_suffix2(payload, positive_prompt_suffix2, positive_prompt_suffix2_blacklist)
            clean_payload(payload)
            await process_image_gen(payload, picture_frame, i)
            # Play tts response
            if tts_resp is not None:
                await task_queue.put(process_tts_resp(i, tts_resp)) # Process this in background
    except Exception as e:
        print(f"An error occurred: {e}")
    busy_drawing = False

## Code pertaining to /image command
# Function to update size options
async def update_size_options(new_width, new_height):
    global size_choices
    options = load_yaml_file('ad_discordbot/dict_cmdoptions.yaml')
    sizes = options.get('sizes', [])
    aspect_ratios = [size.get("ratio") for size in sizes.get('ratios', [])]
    average = average_width_height(new_width, new_height)
    size_choices.clear()  # Clear the existing list
    ratio_options = calculate_aspect_ratio_sizes(average, aspect_ratios)
    static_options = sizes.get('static_sizes', [])
    size_options = (ratio_options or []) + (static_options or [])
    size_choices.extend(
        app_commands.Choice(name=option['name'], value=option['name'])
        for option in size_options)
    await client.tree.sync()

def round_to_precision(val, prec):
    return round(val / prec) * prec

def res_to_model_fit(width, height, mp_target):
    mp = width * height
    scale = math.sqrt(mp_target / mp)
    new_wid = int(round_to_precision(width * scale, 64))
    new_hei = int(round_to_precision(height * scale, 64))
    return new_wid, new_hei

def calculate_aspect_ratio_sizes(avg, aspect_ratios):
    ratio_options = []
    mp_target = avg*avg
    doubleavg = avg*2
    for ratio in aspect_ratios:
        ratio_parts = tuple(map(int, ratio.replace(':', '/').split('/')))
        ratio_sum = ratio_parts[0]+ratio_parts[1]
        # Approximate the width and height based on the average and aspect ratio
        width = round((ratio_parts[0]/ratio_sum)*doubleavg)
        height = round((ratio_parts[1]/ratio_sum)*doubleavg)
        # Round to correct megapixel precision
        width, height = res_to_model_fit(width, height, mp_target)
        if width > height: aspect_type = "landscape"
        elif width < height: aspect_type = "portrait"
        else: aspect_type = "square"
        # Format the result
        size_name = f"{width} x {height} ({ratio} {aspect_type})"
        ratio_options.append({'name': size_name, 'width': width, 'height': height})
    return ratio_options

def average_width_height(width, height):
    avg = (width + height) // 2
    if (width + height) % 2 != 0: avg += 1
    return avg

options = load_yaml_file('ad_discordbot/dict_cmdoptions.yaml')
options = dict(options)
active_settings = load_yaml_file('ad_discordbot/activesettings.yaml')
active_settings = dict(active_settings)

sizes = options.get('sizes', {})
aspect_ratios = [size.get("ratio") for size in sizes.get('ratios', [])]

# Calculate the average and aspect ratio sizes
width = active_settings.get('imgmodel', {}).get('payload', {}).get('width', 512)
height = active_settings.get('imgmodel', {}).get('payload', {}).get('height', 512)
average = average_width_height(width, height)
ratio_options = calculate_aspect_ratio_sizes(average, aspect_ratios)
# Collect any defined static sizes
static_options = sizes.get('static_sizes', [])
# Merge dynamic and static sizes
size_options = (ratio_options or []) + (static_options or [])
style_options = options.get('styles', {})
controlnet_options = options.get('controlnet', {})

size_choices = [
    app_commands.Choice(name=option['name'], value=option['name'])
    for option in size_options]
style_choices = [
    app_commands.Choice(name=option['name'], value=option['name'])
    for option in style_options]
cnet_model_choices = [
    app_commands.Choice(name=option['name'], value=option['name'])
    for option in controlnet_options]

@client.hybrid_command(name="image", description='Generate an image using A1111')
@app_commands.choices(size=size_choices)
@app_commands.choices(style=style_choices)
@app_commands.choices(cnet_model=cnet_model_choices)
@app_commands.choices(cnet_map=[
    app_commands.Choice(name="Yes, map design is white on a black background", value="map"),
    app_commands.Choice(name="Yes, map design is black on a white background", value="invert_map"),
    app_commands.Choice(name="No, I attached a reference image", value="no_map"),
])

async def image(
    i: discord.Interaction,
    prompt: str,
    size: typing.Optional[app_commands.Choice[str]],
    style: typing.Optional[app_commands.Choice[str]],
    neg_prompt: typing.Optional[str],
    face_swap: typing.Optional[discord.Attachment],
    cnet_model: typing.Optional[app_commands.Choice[str]],
    cnet_input: typing.Optional[discord.Attachment],
    cnet_map: typing.Optional[app_commands.Choice[str]]
    ):

    global busy_drawing
    if busy_drawing:
        await i.channel.send("(busy generating an image, please try again after image generation is completed)")
        return
    busy_drawing = True
    try:
        text = prompt
        pos_style_prompt = prompt
        neg_style_prompt = ""
        size_dict = {}
        faceswapimg = ''
        controlnet_dict = {}

        message_content = f">>> **Prompt:** {pos_style_prompt}"

        if neg_prompt:
            neg_style_prompt = f"{neg_prompt}, {neg_style_prompt}"
            message_content += f" | **Negative Prompt:** {neg_prompt}"

        if style:
            selected_style_option = next((option for option in style_options if option['name'] == style.value), None)

            if selected_style_option:
                pos_style_prompt = selected_style_option.get('positive').format(prompt)
                neg_style_prompt = selected_style_option.get('negative')
            message_content += f" | **Style:** {style.value}"

        if size:
            selected_size_option = next((option for option in size_options if option['name'] == size.value), None)
            if selected_size_option:
                size_dict['width'] = selected_size_option.get('width')
                size_dict['height'] = selected_size_option.get('height')
            message_content += f" | **Size:** {size.value}"

        if config.sd['extensions']['reactor_enabled']:
            if face_swap:
                if face_swap.content_type and face_swap.content_type.startswith("image/"):
                    imgurl = face_swap.url
                    attached_img = await face_swap.read()
                    faceswapimg = base64.b64encode(attached_img).decode('utf-8')
                    message_content += f" | **Face Swap:** Image Provided"
                else:
                    await i.send("Please attach a valid image to use for Face Swap.",ephemeral=True)
                    return

        if config.sd['extensions']['controlnet_enabled']:
            if cnet_model:
                selected_cnet_option = next((option for option in controlnet_options if option['name'] == cnet_model.value), None)
                if selected_cnet_option:
                    controlnet_dict['model'] = selected_cnet_option.get('model')
                    controlnet_dict['module'] = selected_cnet_option.get('module')
                    controlnet_dict['guidance_end'] = selected_cnet_option.get('guidance_end')
                    controlnet_dict['weight'] = selected_cnet_option.get('weight')
                    controlnet_dict['enabled'] = True
                message_content += f" | **ControlNet:** Model: {cnet_model.value}"

            if cnet_input:
                if cnet_input.content_type and cnet_input.content_type.startswith("image/"):
                    imgurl = cnet_input.url
                    attached_img = await cnet_input.read()
                    cnetimage = base64.b64encode(attached_img).decode('utf-8')
                    controlnet_dict['input_image'] = cnetimage
                else:
                    await i.send("Invalid image. Please attach a valid image.",ephemeral=True)
                    return
                if cnet_map:
                    if cnet_map.value == "no_map":
                        controlnet_dict['module'] = selected_cnet_option.get('module')
                    if cnet_map.value == "map":
                        controlnet_dict['module'] = "none"
                    if cnet_map.value == "invert_map":
                        controlnet_dict['module'] = "invert (from white bg & black line)"
                    message_content += f", Module: {controlnet_dict['module']}"
                    message_content += f", Map Input: {cnet_map.value}"
                else:
                    message_content += f", Module: {controlnet_dict['module']}"

                if (cnet_model and not cnet_input) or (cnet_input and not cnet_model):
                    await i.send("ControlNet feature requires **both** selecting a model (cnet_model) and attaching an image (cnet_input).",ephemeral=True)
                    return

        await i.send(message_content)

        await pic(
            i,
            text,
            image_prompt=pos_style_prompt,
            neg_prompt=neg_style_prompt,
            size=size_dict if size_dict else None,
            face_swap=faceswapimg if face_swap else None,
            controlnet=controlnet_dict if controlnet_dict else None)
    except Exception as e:
        print(f"An error occurred: {e}")
    busy_drawing = False

#----END IMAGE PROCESSING----#

@client.hybrid_command(description="Reset the conversation")
async def reset(i):
    global reply_count
    user = i.author.display_name
    char_name = client.user.display_name
    reply_count = 0
    shared.stop_everything = True
    reset_session_history()  # Reset conversation
    await change_character(i, char_name)
    prompt = client.settings['llmcontext']['context']
    info_embed.title = f"{user} reset the conversation with {char_name}"
    info_embed.description = ""
    await i.reply(embed=info_embed)    
    logging.info("conversation reset: {'replies': " + str(reply_count) + ", 'user': '" + user + "', 'character': '" + char_name + "', 'prompt': '" + prompt + "'}")

@client.hybrid_command(description="Check the status of your reply queue position and wait time")
async def status(i):
    total_num_queued_jobs = len(queues)
    que_user_ids = [list(a.keys())[0] for a in queues]
    if i.author.mention in que_user_ids:
        user_position = que_user_ids.index(i.author.mention) + 1
        msg = f"{i.author.mention} Your job is currently {user_position} out of {total_num_queued_jobs} in the queue. Estimated time until response is ready: {user_position * 20/60} minutes."
    else:
        msg = f"{i.author.mention} doesn\'t have a job queued."
    status_embed.timestamp = datetime.now()
    status_embed.description = msg
    await i.send(embed=status_embed)

def get_active_setting(key):
    try:
        active_settings = load_yaml_file('ad_discordbot/activesettings.yaml')
        if key in active_settings:
            return active_settings[key]
        else:
            return None
    except Exception as e:
        print(f"Error loading ad_discordbot/activesettings.yaml ({key}):", e)
        return None

def generate_characters():
    try:
        cards = []
        # Iterate through files in the image folder
        for file in sorted(Path("characters").glob("*")):
            if file.suffix in [".json", ".yml", ".yaml"]:
                character = {}
                character['name'] = file.stem
                filepath = str(Path(file).absolute())
                with open(filepath, encoding='utf-8') as f:
                    data = json.load(f) if file.suffix == ".json" else yaml.safe_load(f)
                    character['bot_description'] = data.get("bot_description", None)
                    character['bot_emoji'] = data.get("bot_emoji", "💬")
                    cards.append(character)
        return cards
    except Exception as e:
        print(f"Error collecting character items for /character menu:", e)

class CharacterDropdown(discord.ui.Select):
    def __init__(self, i):
        options = [discord.SelectOption(label=character["name"], description=character["bot_description"], emoji=character["bot_emoji"]) for character in generate_characters()]
        super().__init__(placeholder='', min_values=1, max_values=1, options=options)
        self.i = i
    async def callback(self, interaction: discord.Interaction):
        character = self.values[0]
        await change_character(self.i, character)
        greeting = client.settings['llmcontext']['greeting']
        if greeting:
            name1 = 'You'
            name2 = character
            greeting = await replace_character_names(greeting, name1, name2)
        else:
            greeting = f'**{character}** has entered the chat"'
        await interaction.response.send_message(greeting)
        print(f'Loaded new character: "{character}".')
        return

@client.hybrid_command(description="Choose Character")
async def character(i):
    view = discord.ui.View()
    view.add_item(CharacterDropdown(i))
    await i.send('Choose Character:', view=view, ephemeral=True)
     
# Apply changes for Settings commands
async def update_active_settings(selected_item, active_settings_key):
    try:
        active_settings = load_yaml_file('ad_discordbot/activesettings.yaml')
        target_settings = active_settings.setdefault(active_settings_key, {})
        update_dict(target_settings, selected_item)
        save_yaml_file('ad_discordbot/activesettings.yaml', active_settings)
    except Exception as e:
        print(f"Error updating ad_discordbot/activesettings.yaml ({active_settings_key}):", e)

# Post settings to dedicated channel
async def post_active_settings():
    if config.discord['post_active_settings'].get('target_channel_id', ''):
        channel = await client.fetch_channel(config.discord['post_active_settings']['target_channel_id'])
        if channel:
            active_settings = load_yaml_file('ad_discordbot/activesettings.yaml')
            settings_content = yaml.dump(active_settings, default_flow_style=False)
            # Fetch and delete all existing messages in the channel
            async for message in channel.history(limit=None):
                await message.delete()
                await asyncio.sleep(0.5)  # minimum delay for discord limit
            # Send the entire settings content as a single message
            await send_long_message(channel, f"Current settings:\n```yaml\n{settings_content}\n```")
        else:
            print(f"Target channel with ID {target_channel_id} not found.")
    else:
        print("Channel ID must be specified in config.py")

class SettingsDropdown(discord.ui.Select):
    def __init__(self, data_file, label_key, active_settings_key):
        try:
            items = load_yaml_file(data_file)
            options = [
                discord.SelectOption(label=item[label_key], value=item[label_key])
                for item in items
            ]
            super().__init__(placeholder=f'Select a {label_key.replace("_", " ")}', options=options)
            self.data_file = data_file
            self.label_key = label_key
            self.active_settings_key = active_settings_key
        except FileNotFoundError:
            print(f"{data_file} not found.")
        except Exception as e:
            print(f"Error loading {data_file}:", e)
    async def callback(self, interaction: discord.Interaction):
        selected_item_name = self.values[0]
        try:
            items = load_yaml_file(self.data_file)
            selected_item = next(item for item in items if item[self.label_key] == selected_item_name)
            await update_active_settings(selected_item, self.active_settings_key)
            print(f"Updated {self.active_settings_key} settings to: {selected_item_name}")
            update_client_settings() # Sync updated user settings to client
            # If a new LLMContext is selected
            if self.active_settings_key == 'llmcontext':
                reset_session_history() # Reset conversation
            await interaction.response.send_message(f"Updated {self.active_settings_key} settings to: {selected_item_name}")
            if config.discord.get('post_active_settings', {}).get('enabled', False):
                await task_queue.put(post_active_settings())
        except Exception as e:
            print(f"Error updating ad_discordbot/activesettings.yaml ({self.active_settings_key}):", e)

@client.hybrid_command(description="Choose LLM context")
async def llmcontext(i):
    view = discord.ui.View()
    view.add_item(SettingsDropdown('ad_discordbot/dict_llmcontexts.yaml', 'llmcontext_name', 'llmcontext'))
    await i.send("Choose LLM context:", view=view, ephemeral=True)
@client.hybrid_command(description="Choose an llmstate")
async def llmstate(i):
    view = discord.ui.View()
    view.add_item(SettingsDropdown('ad_discordbot/dict_llmstates.yaml', 'llmstate_name', 'llmstate'))
    await i.send("Choose an llmstate:", view=view, ephemeral=True)
@client.hybrid_command(description="Choose a behavior")
async def behaviors(i):
    view = discord.ui.View()
    view.add_item(SettingsDropdown('ad_discordbot/dict_behaviors.yaml', 'behavior_name', 'behavior'))
    await i.send("Choose a behavior:", view=view, ephemeral=True)
@client.hybrid_command(description="Choose ImgTags")
async def imgtags(i):
    view = discord.ui.View()
    view.add_item(SettingsDropdown('ad_discordbot/dict_imgtags.yaml', 'imgtag_name', 'imgtag'))
    await i.send("Choose ImgTags:", view=view, ephemeral=True)

@client.hybrid_command(description="Toggle current channel as main channel for bot to auto-reply without needing to be called")
async def main(i):
    try:
        conn = sqlite3.connect('bot.db')
        c = conn.cursor()
        if i.channel.id in client.database.main_channels:
            client.database.main_channels.remove(i.channel.id) # If the channel is already in the main channels, remove it
            c.execute('''DELETE FROM main_channels WHERE channel_id = ?''', (i.channel.id,))
            action_message = f'Removed {i.channel.mention} from main channels. Use "/main" again if you want to add it back.'
        else:
            # If the channel is not in the main channels, add it
            client.database.main_channels.append(i.channel.id)
            c.execute('''INSERT OR REPLACE INTO main_channels (channel_id) VALUES (?)''', (i.channel.id,))
            action_message = f'Added {i.channel.mention} to main channels. Use "/main" again to remove it.'
        conn.commit()
        conn.close()
        client.database = Database()
        await i.reply(action_message)
    except Exception as e:
        print("Error toggling main channel setting:", e)

@client.hybrid_command(description="Display help menu")
async def helpmenu(i):
    info_embed = discord.Embed().from_dict(info_embed_json)
    await i.send(embed=info_embed)

@client.hybrid_command(description="Regenerate the bot's last reply")
async def regen(i):
    info_embed.title = f"Regenerating ... "
    info_embed.description = ""
    await i.reply(embed=info_embed)
    user_input = await initialize_user_input(i, text=last_user_message['llm_prompt'])
    user_input['regenerate'] = True
    last_resp, tts_resp = await chatbot_wrapper_wrapper(user_input, save_history=None)
    await send_long_message(i.channel, last_resp)
    if tts_resp: await task_queue.put(process_tts_resp(i, tts_resp)) # Process this in background

@client.hybrid_command(description="Continue the generation")
async def cont(i):
    info_embed.title = f"Continuing ... "
    info_embed.description = ""
    await i.reply(embed=info_embed)
    user_input = await initialize_user_input(i, text=last_user_message['llm_prompt'])
    user_input['_continue'] = True
    last_resp, tts_resp = await chatbot_wrapper_wrapper(user_input, save_history=None)
    await delete_last_message(i)
    await send_long_message(i.channel, last_resp)

@client.hybrid_command(description="Update dropdown menus without restarting bot script.")
async def sync(interaction: discord.Interaction):
    await task_queue.put(client.tree.sync()) # Process this in the background

## /imgmodel command
# Apply user defined filters to imgmodel list
async def filter_imgmodels(imgmodels):
    try:
        if config.imgmodels['filter'] or config.imgmodels['exclude']:
            filter_list = config.imgmodels['filter']
            exclude_list = config.imgmodels['exclude']
            imgmodels = [
                imgmodel for imgmodel in imgmodels
                if (
                    (not filter_list or any(re.search(re.escape(filter_text), imgmodel.get('imgmodel_name', '') + imgmodel.get('sd_model_checkpoint', ''), re.IGNORECASE) for filter_text in filter_list))
                    and (not exclude_list or not any(re.search(re.escape(exclude_text), imgmodel.get('imgmodel_name', '') + imgmodel.get('sd_model_checkpoint', ''), re.IGNORECASE) for exclude_text in exclude_list))
                )
            ]
        return imgmodels
    except Exception as e:
        print("Error filtering image model list:", e)

# Build list of imgmodels depending on user preference (user .yaml / A1111 API)
async def fetch_imgmodels():
    try:
        if not config.imgmodels['get_imgmodels_via_api']['enabled']:
            imgmodels_data = load_yaml_file('ad_discordbot/dict_imgmodels.yaml')
            imgmodels = copy.deepcopy(imgmodels_data)
        else:
            try:
                async with aiohttp.ClientSession() as session: # populate options from A1111 API
                    async with session.get(url=f'{A1111}/sdapi/v1/sd-models') as response:
                        if response.status == 200:
                            imgmodels = await response.json()
                            # Update 'title' keys in A1111 fetched list to be uniform with .yaml method
                            for imgmodel in imgmodels:
                                if 'title' in imgmodel:
                                    imgmodel['sd_model_checkpoint'] = imgmodel.pop('title')
                        else:
                            return ''
                            print(f"Error fetching image models from the API (response: '{response.status}')")
            except Exception as e:
                print("Error fetching image models via API:", e)
                return ''     
        if imgmodels:
            imgmodels = await filter_imgmodels(imgmodels)
            return imgmodels
    except Exception as e:
        print("Error fetching image models:", e)

# Update imgtags at same time as imgmodel, as configured
async def change_imgtags(active_settings, imgtag_name):
    try:
        imgtags_data = load_yaml_file('ad_discordbot/dict_imgtags.yaml')
        selected_imgmodel_imgtags = None
        for imgtags in imgtags_data:
            if imgtags.get('imgtag_name') == imgtag_name:
                selected_imgmodel_imgtags = imgtags
                break
        if selected_imgmodel_imgtags is not None:
            active_settings['imgtag'] = selected_imgmodel_imgtags
            active_settings['imgmodel']['imgmodel_imgtags'] = imgtag_name
    except Exception as e:
        print(f"Error updating imgtags for {imgtag_name} in ad_discordbot/activesettings.yaml:", e)

# Check filesize of selected imgmodel to assume resolution and imgtags 
async def guess_imgmodel_res(active_settings, selected_imgmodel_filename):
    try:
        file_size_bytes = os.path.getsize(selected_imgmodel_filename)
        file_size_gb = file_size_bytes / (1024 ** 3)  # 1 GB = 1024^3 bytes
        presets = config.imgmodels['get_imgmodels_via_api']['presets']
        matched_preset = None
        for preset in presets:
            if preset['max_filesize'] > file_size_gb:
                matched_preset = dict(preset)
                del matched_preset['max_filesize']
                break
        if matched_preset:
            if matched_preset.get('imgtag_name') is not None:
                # Update ImgTags
                await change_imgtags(active_settings, matched_preset.get('imgtag_name'))
                del matched_preset['imgtag_name']
            update_dict(active_settings.get('imgmodel').get('payload'), matched_preset)
            #active_settings['imgmodel']['payload']['width'] = matched_preset['width']
            #active_settings['imgmodel']['payload']['height'] = matched_preset['height']
    except Exception as e:
        print("Error guessing imgmodel resolution:", e)

# Change imgmodel using A1111 API method
async def change_imgmodel_api(active_settings, selected_imgmodel, selected_imgmodel_name):
    try:
        if config.imgmodels['get_imgmodels_via_api']['guess_model_res']:
            selected_imgmodel_filename = selected_imgmodel.get('filename')
            if selected_imgmodel_filename is not None:
                # Check filesize of selected imgmodel to assume resolution and imgtags 
                await guess_imgmodel_res(active_settings, selected_imgmodel_filename)
        active_settings['imgmodel']['override_settings']['sd_model_checkpoint'] = selected_imgmodel['sd_model_checkpoint']
        active_settings['imgmodel']['imgmodel_name'] = selected_imgmodel_name
        active_settings['imgmodel']['imgmodel_url'] = ''
    except Exception as e:
        print("Error updating image model:", e)

async def a1111_load_imgmodel(options):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url=f'{A1111}/sdapi/v1/options', json=options) as response:
                if response.status == 200:
                    await response.json()
                else:
                    print(f"Error loading image model in A1111 API (response: '{response.status}')")
    except Exception as e:
        print("Error loading image model in A1111:", e) 

# Announce imgmodel change as configured
async def imgmodel_announce(selected_imgmodel, selected_imgmodel_name):
    try:
        reply = ''
        reply_prefix = config.imgmodels['announce_in_chat']['reply_prefix']
        # Process .yaml method
        if not config.imgmodels['get_imgmodels_via_api']['enabled']:
            reply = f"{reply_prefix}{selected_imgmodel.get('imgmodel_name')}"
            if config.imgmodels['announce_in_chat']['include_url']:
                reply += " <" + selected_imgmodel.get('imgmodel_url', {}) + ">"
            if config.imgmodels['announce_in_chat']['include_params']:
                selected_imgmodel_override_settings_info = ", ".join(
                    f"{key}: {value}" for key, value in selected_imgmodel_override_settings.imgmodels())
                selected_imgmodel_payload_info = ", ".join(
                    f"{key}: {value}" for key, value in selected_imgmodel_payload.imgmodels())
                reply += f"\n```{selected_imgmodel_override_settings_info}, {selected_imgmodel_payload_info}```"
        else: # Process A1111 API method
            reply = f"{reply_prefix}{selected_imgmodel_name}"
        return reply
    except Exception as e:
        print("Error announcing imgmodel:", e)

# Update channel topic with new imgmodel info as configured
async def imgmodel_update_topic(channel, selected_imgmodel, selected_imgmodel_name):
    try:
        topic_prefix = config.imgmodels['update_topic']['topic_prefix']
        # Process .yaml method
        if not config.imgmodels['get_imgmodels_via_api']['enabled']:
            new_topic = f"{topic_prefix}{selected_imgmodel.get('imgmodel_name')}"
            if config.imgmodels['update_topic']['include_url']:
                new_topic += " " + selected_imgmodel.get('imgmodel_url', {})
        else: # Process A1111 API method
            new_topic = f"{topic_prefix}{selected_imgmodel_name}"
        await channel.edit(topic=new_topic)
    except Exception as e:
        print("Error updating channel topic:", e)

async def process_imgmodel_announce(i, selected_imgmodel, selected_imgmodel_name):
    try:
        # Set the topic of the channel and announce imgmodel as configured
        if config.imgmodels['update_topic']['enabled']:
            channel = i.channel
            await imgmodel_update_topic(channel, selected_imgmodel, selected_imgmodel_name)
        if config.imgmodels['announce_in_chat']['enabled']:
            reply = await imgmodel_announce(selected_imgmodel, selected_imgmodel_name)
            if reply:
                await i.send(reply)
            else:
                await i.send(f"Updated imgmodel settings to: {selected_imgmodel_name}")
    except Exception as e:
        print("Error announcing imgmodel:", e)

async def get_selected_imgmodel_data(selected_imgmodel_value):
    try:
        if not config.imgmodels['get_imgmodels_via_api']['enabled']:
            imgmodels = load_yaml_file('ad_discordbot/dict_imgmodels.yaml')
            selected_imgmodel = next(imgmodel for imgmodel in imgmodels if imgmodel['imgmodel_name'] == selected_imgmodel_value)
        else: # Collect imgmodel data for A1111 API method
            selected_imgmodel = {}
            for imgmodel in all_imgmodels:
                if imgmodel["imgmodel_name"] == selected_imgmodel_value:
                    selected_imgmodel = {
                        "sd_model_checkpoint": imgmodel["sd_model_checkpoint"],
                        "imgmodel_name": imgmodel.get("imgmodel_name"),
                        "filename": imgmodel.get("filename", None)
                    }
                    break
    except Exception as e:
        print("Error getting image model data:", e)
    return selected_imgmodel

async def process_imgmodel(i, selected_imgmodel_value):
    try:
        selected_imgmodel = await get_selected_imgmodel_data(selected_imgmodel_value)
        selected_imgmodel_name = selected_imgmodel.get('imgmodel_name')
        active_settings = load_yaml_file('ad_discordbot/activesettings.yaml')
        # Process .yaml method
        if not config.imgmodels['get_imgmodels_via_api']['enabled']:
            update_dict(active_settings.get('imgmodel', {}), selected_imgmodel)
            if selected_imgmodel.get('imgmodel_imgtags') is not None:
                await change_imgtags(active_settings, selected_imgmodel.get('imgmodel_imgtags'))
        else: # Process A1111 API method
            await change_imgmodel_api(active_settings, selected_imgmodel, selected_imgmodel_name)
        save_yaml_file('ad_discordbot/activesettings.yaml', active_settings)
        update_client_settings() # Sync updated user settings to client
        # Load the imgmodel and VAE via A1111 API
        await task_queue.put(a1111_load_imgmodel(active_settings['imgmodel']['override_settings'])) # Process this in the background
        # Update size options for /image command
        await task_queue.put(update_size_options(active_settings.get('imgmodel').get('payload').get('width'),active_settings.get('imgmodel').get('payload').get('height')))
        await process_imgmodel_announce(i, selected_imgmodel, selected_imgmodel_name)
        print(f"Updated imgmodel settings to: {selected_imgmodel_name}")
    except Exception as e:
        print("Error updating image model:", e)
    if config.discord['post_active_settings']['enabled']:
        await task_queue.put(post_active_settings())

all_imgmodels = []
all_imgmodels = asyncio.run(fetch_imgmodels())

if all_imgmodels:
    for imgmodel in all_imgmodels:
        if 'model_name' in imgmodel:
            imgmodel['imgmodel_name'] = imgmodel.pop('model_name')
    imgmodel_options = [app_commands.Choice(name=imgmodel["imgmodel_name"], value=imgmodel["imgmodel_name"]) for imgmodel in all_imgmodels[:25]]
    if len(all_imgmodels) > 25:
        imgmodel_options1 = [app_commands.Choice(name=imgmodel["imgmodel_name"], value=imgmodel["imgmodel_name"]) for imgmodel in all_imgmodels[25:50]]    
        if len(all_imgmodels) > 50:      
            imgmodel_options2 = [app_commands.Choice(name=imgmodel["imgmodel_name"], value=imgmodel["imgmodel_name"]) for imgmodel in all_imgmodels[50:75]]                       
            if len(all_imgmodels) > 75:
                imgmodel_options3 = [app_commands.Choice(name=imgmodel["imgmodel_name"], value=imgmodel["imgmodel_name"]) for imgmodel in all_imgmodels[75:100]]                       
                if len(all_imgmodels) > 100: print("'/imgmodel' command only allows up to 100 image models. Some models were omitted.")

    if len(all_imgmodels) <= 25:
        @client.hybrid_command(name="imgmodel", description='Choose an imgmodel')
        @app_commands.choices(imgmodels=imgmodel_options)
        async def imgmodel(i: discord.Interaction, imgmodels: typing.Optional[app_commands.Choice[str]]):
            selected_imgmodel = imgmodels.value if imgmodels is not None else ''
            await process_imgmodel(i, selected_imgmodel)

    elif 25 < len(all_imgmodels) <= 50:
        @client.hybrid_command(name="imgmodel", description='Choose an imgmodel (pick only one)')
        @app_commands.choices(models_1=imgmodel_options)
        @app_commands.choices(models_2=imgmodel_options1)
        async def imgmodel(i: discord.Interaction, models_1: typing.Optional[app_commands.Choice[str]], models_2: typing.Optional[app_commands.Choice[str]]):
            if models_1 and models_2:
                await i.send("More than one imgmodel was selected. Using the first selection.", ephemeral=True)
            selected_imgmodel = ((models_1 or models_2) and (models_1 or models_2).value) or ''
            await process_imgmodel(i, selected_imgmodel)

    elif 50 < len(all_imgmodels) <= 75:
        @client.hybrid_command(name="imgmodel", description='Choose an imgmodel (pick only one)')
        @app_commands.choices(models_1=imgmodel_options)
        @app_commands.choices(models_2=imgmodel_options1)
        @app_commands.choices(models_3=imgmodel_options2)
        async def imgmodel(i: discord.Interaction, models_1: typing.Optional[app_commands.Choice[str]], models_2: typing.Optional[app_commands.Choice[str]], models_3: typing.Optional[app_commands.Choice[str]]):
            if sum(1 for v in (models_1, models_2, models_3) if v) > 1:
                await i.send("More than one imgmodel was selected. Using the first selection.", ephemeral=True)
            selected_imgmodel = ((models_1 or models_2 or models_3) and (models_1 or models_2 or models_3).value) or ''
            await process_imgmodel(i, selected_imgmodel)

    elif 75 < len(all_imgmodels) <= 100:
        @client.hybrid_command(name="imgmodel", description='Choose an imgmodel (pick only one)')
        @app_commands.choices(models_1=imgmodel_options)
        @app_commands.choices(models_2=imgmodel_options1)
        @app_commands.choices(models_3=imgmodel_options2)
        @app_commands.choices(models_4=imgmodel_options3)
        async def imgmodel(i: discord.Interaction, models_1: typing.Optional[app_commands.Choice[str]], models_2: typing.Optional[app_commands.Choice[str]], models_3: typing.Optional[app_commands.Choice[str]], models_4: typing.Optional[app_commands.Choice[str]]):
            if sum(1 for v in (models_1, models_2, models_3, models_4) if v) > 1:
                await i.send("More than one imgmodel was selected. Using the first selection.", ephemeral=True)
            selected_imgmodel = ((models_1 or models_2 or models_3 or models_4) and (models_1 or models_2 or models_3 or models_4).value) or ''
            await process_imgmodel(i, selected_imgmodel)

## /Speak command
async def process_speak_silero_non_eng(i, lang):
    non_eng_speaker = None
    non_eng_model = None
    try:
        with open('extensions/silero_tts/languages.json', 'r') as file:
            languages_data = json.load(file)
        if lang in languages_data:
            default_voice = languages_data[lang].get('default_voice')
            if default_voice: non_eng_speaker = default_voice
            silero_model = languages_data[lang].get('model_id')
            if silero_model: non_eng_model = silero_model
            tts_args = {'silero_tts': {'language': lang, 'speaker': non_eng_speaker, 'model_id': non_eng_model}}
        if not (non_eng_speaker and non_eng_model):
            await i.send(f'Could not determine the correct voice and model ID for language "{lang}". Defaulting to English.', ephemeral=True)
            tts_args = {'silero_tts': {'language': 'English', 'speaker': 'en_1'}}
    except Exception as e:
        print(f"Error processing non-English voice for silero_tts: {e}")
        await i.send(f"Error processing non-English voice for silero_tts: {e}", ephemeral=True)
    return tts_args

async def process_speak_args(i, selected_voice=None, lang=None, user_voice=None):
    try:
        tts_args = {}
        if lang:
            if tts_client == 'elevenlabs_tts':
                if lang != 'English':
                    tts_args.setdefault(tts_client, {}).setdefault('model', 'eleven_multilingual_v1')
                    # Currently no language parameter for elevenlabs_tts
            else:
                tts_args.setdefault(tts_client, {}).setdefault(tts_lang_key, lang)
                tts_args[tts_client][tts_lang_key] = lang
        if selected_voice or user_voice:
            tts_args.setdefault(tts_client, {}).setdefault(tts_voice_key, 'temp_voice.wav' if user_voice else selected_voice)
        elif tts_client == 'silero_tts' and lang:
            if lang != 'English':
                tts_args = await process_speak_silero_non_eng(i, lang) # returns complete args for silero_tts
                if selected_voice: await i.send(f'Currently, non-English languages will use a default voice (not using "{selected_voice}")', ephemeral=True)
        elif tts_client in last_extension_params and tts_voice_key in last_extension_params[tts_client]:
            pass # Default to voice in last_extension_params
        elif f'{tts_client}-{tts_voice_key}' in shared.settings:
            pass # Default to voice in shared.settings
        else:
            await i.send("No voice was selected or provided, and a default voice was not found. Request will probably fail...", ephemeral=True)
        return tts_args
    except Exception as e:
        print(f"Error processing tts options: {e}")
        await i.send(f"Error processing tts options: {e}", ephemeral=True)

async def convert_and_resample_mp3(mp3_file, output_directory=None):
    try:
        audio = AudioSegment.from_mp3(mp3_file)
        if audio.channels == 2:
            audio = audio.set_channels(1)   # should be Mono
        audio = audio.set_frame_rate(22050) # ideal sample rate
        audio = audio.set_sample_width(2)   # 2 bytes for 16 bits
        output_directory = output_directory or os.path.dirname(mp3_file)
        wav_filename = os.path.splitext(os.path.basename(mp3_file))[0] + '.wav'
        wav_path = f"{output_directory}/{wav_filename}"
        audio.export(wav_path, format="wav")
        print(f'User provided file "{mp3_file}" was converted to .wav for "/speak" command')
        return wav_path
    except Exception as e:
        print(f"Error converting user's .mp3 to .wav: {e}")
        await i.send("An error occurred while processing the voice file.", ephemeral=True)
    finally:
        if mp3_file: os.remove(mp3_file)

async def process_user_voice(i, voice_input=None):
    try:
        if not (voice_input and getattr(voice_input, 'content_type', '').startswith("audio/")):
            return ''
        if tts_client != 'alltalk_tts' and tts_client != 'coqui_tts':
            await i.send("Sorry, current tts extension does not allow using a voice attachment (only works for 'alltalk_tts' and 'coqui_tts)", ephemeral=True)
            return ''
        voiceurl = voice_input.url
        voiceurl_without_params = voiceurl.split('?')[0]
        if not voiceurl_without_params.endswith((".wav", ".mp3")):
            await i.send("Invalid audio format. Please try again with a WAV or MP3 file.", ephemeral=True)
            return ''
        voice_data_ext = voiceurl_without_params[-4:]
        user_voice = f'extensions/{tts_client}/voices/temp_voice{voice_data_ext}'
        async with aiohttp.ClientSession() as session:
            async with session.get(voiceurl) as resp:
                if resp.status == 200:
                    voice_data = await resp.read()
                    with open(user_voice, 'wb') as f:
                        f.write(voice_data)
                else:
                    await i.send("Error downloading your audio file. Please try again.", ephemeral=True)
                    return ''
        if voice_data_ext == '.mp3':
            try:
                user_voice = await convert_and_resample_mp3(user_voice, output_directory=None)
            except:
                if user_voice: os.remove(user_voice)
        return user_voice
    except Exception as e:
        print(f"Error processing user provided voice file: {e}")
        await i.send("An error occurred while processing the voice file.", ephemeral=True)

async def process_speak(i, input_text, selected_voice=None, lang=None, voice_input=None):
    try:
        user_voice = await process_user_voice(i, voice_input)
        tts_args = await process_speak_args(i, selected_voice, lang, user_voice)
        info_embed.title = f"Generating speach from text ... "
        info_embed.description = ""
        message = await i.reply(embed=info_embed)
        user_input = await initialize_user_input(i, text=input_text)
        user_input['_continue'] = True
        user_input['state']['max_new_tokens'] = 1
        user_input['state']['min_length'] = 0
        user_input['state']['history'] = {'internal': [[input_text, input_text]], 'visible': [[input_text, input_text]]}
        await update_extensions(tts_args) # Update tts_client extension settings
        _, tts_resp = await chatbot_wrapper_wrapper(user_input, save_history=False)
        if tts_resp: await task_queue.put(process_tts_resp(i, tts_resp)) # Process this in background
        await update_extensions(client.settings['llmcontext'].get('extensions', {})) # Restore character specific extension settings
        if user_voice: os.remove(user_voice)
       # await i.send(f'**{i.author} requested text to speech:**\n{input_text}')
        await send_long_message(i.channel, (f'**{i.author} requested text to speech:**\n{input_text}'))
        await message.delete()
    except Exception as e:
        print(f"Error processing tts request: {e}")
        await i.send(f"Error processing tts request: {e}", ephemeral=True)

async def fetch_speak_options():
    try:
        ext = ''
        lang_list = []
        all_voicess = []
        if tts_client == 'coqui_tts' or tts_client == 'alltalk_tts':
            ext = '.wav'
            lang_list = ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French', 'German', 'Hungarian', 'Italian', 'Japanese', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Spanish', 'Turkish']
            tts_voices_dir = f'extensions/{tts_client}/voices'
            if os.path.exists(tts_voices_dir) and os.path.isdir(tts_voices_dir):
                all_voices = [voice_name[:-4] for voice_name in os.listdir(tts_voices_dir) if voice_name.endswith(".wav")]
        elif tts_client == 'silero_tts':
            lang_list = ['English', 'Spanish', 'French', 'German', 'Russian', 'Tatar', 'Ukranian', 'Uzbek', 'English (India)', 'Avar', 'Bashkir', 'Bulgarian', 'Chechen', 'Chuvash', 'Kalmyk', 'Karachay-Balkar', 'Kazakh', 'Khakas', 'Komi-Ziryan', 'Mari', 'Nogai', 'Ossetic', 'Tuvinian', 'Udmurt', 'Yakut']
            print('''There's too many Voice/language permutations to make them all selectable in "/speak" command. Loading a bunch of English options. Non-English languages will automatically play using respective default speaker.''')
            all_voices = [f"en_{i}" for i in range(1, 76)] # will just include English voices in select menus. Other languages will use defaults.
        elif tts_client == 'elevenlabs_tts':
            lang_list = ['English', 'German', 'Polish', 'Spanish', 'Italian', 'French', 'Portuegese', 'Hindi', 'Arabic']
            print('''Getting list of available voices for elevenlabs_tts for "/speak" command...''')
            from extensions.elevenlabs_tts.script import refresh_voices, update_api_key
            if tts_api_key:
                update_api_key(tts_api_key)
            all_voices = refresh_voices()
        all_voices.sort() # Sort alphabetically
        return ext, lang_list, all_voices
    except Exception as e:
        print(f'Error building options for "/speak" command: {e}')

if tts_client and tts_client in supported_tts_clients:
    ext, lang_list, all_voices = asyncio.run(fetch_speak_options())
    voice_options = [app_commands.Choice(name=voice_name.replace('_', ' ').title(), value=f'{voice_name}{ext}') for voice_name in all_voices[:25]]
    if len(all_voices) > 25:
        voice_options1 = [app_commands.Choice(name=voice_name.replace('_', ' ').title(), value=f'{voice_name}{ext}') for voice_name in all_voices[25:50]]    
        if len(all_voices) > 50:      
            voice_options2 = [app_commands.Choice(name=voice_name.replace('_', ' ').title(), value=f'{voice_name}{ext}') for voice_name in all_voices[50:75]]                       
            if len(all_voices) > 75: print("'/speak' command only allows up to 75 voices. Some voices were omitted.")
    if lang_list: lang_options = [app_commands.Choice(name=lang, value=lang) for lang in lang_list]
    else: lang_options = [app_commands.Choice(name='English', value='English')] # Default to English

    if len(all_voices) <= 25:
        @client.hybrid_command(name="speak", description='AI will speak your text using a selected voice')
        @app_commands.choices(voice=voice_options)
        @app_commands.choices(lang=lang_options)
        async def speak(i: discord.Interaction, input_text: str, voice: typing.Optional[app_commands.Choice[str]], lang: typing.Optional[app_commands.Choice[str]], voice_input: typing.Optional[discord.Attachment]):
            selected_voice = voice.value if voice is not None else ''
            voice_input = voice_input if voice_input is not None else ''
            lang = lang.value if lang is not None else ''
            await process_speak(i, input_text, selected_voice, lang, voice_input)

    elif 25 < len(all_voices) <= 50:
        @client.hybrid_command(name="speak", description='AI will speak your text using a selected voice (pick only one)')
        @app_commands.choices(voice_1=voice_options)
        @app_commands.choices(voice_2=voice_options1)
        @app_commands.choices(lang=lang_options)
        async def speak(i: discord.Interaction, input_text: str, voice_1: typing.Optional[app_commands.Choice[str]], voice_2: typing.Optional[app_commands.Choice[str]], lang: typing.Optional[app_commands.Choice[str]], voice_input: typing.Optional[discord.Attachment]):
            if voice_1 and voice_2:
                await i.send("A voice was picked from two separate menus. Using the first selection.", ephemeral=True)
            selected_voice = ((voice_1 or voice_2) and (voice_1 or voice_2).value) or ''
            voice_input = voice_input if voice_input is not None else ''
            lang = lang.value if lang is not None else ''
            await process_speak(i, input_text, selected_voice, lang, voice_input)

    elif 50 < len(all_voices) <= 75:
        @client.hybrid_command(name="speak", description='AI will speak your text using a selected voice (pick only one)')
        @app_commands.choices(voice_1=voice_options)
        @app_commands.choices(voice_2=voice_options1)
        @app_commands.choices(voice_3=voice_options2)
        @app_commands.choices(lang=lang_options)
        async def speak(i: discord.Interaction, input_text: str, voice_1: typing.Optional[app_commands.Choice[str]], voice_2: typing.Optional[app_commands.Choice[str]], voice_3: typing.Optional[app_commands.Choice[str]], lang: typing.Optional[app_commands.Choice[str]], voice_input: typing.Optional[discord.Attachment]):
            if sum(1 for v in (voice_1, voice_2, voice_3) if v) > 1:
                await i.send("A voice was picked from two separate menus. Using the first selection.", ephemeral=True)
            selected_voice = ((voice_1 or voice_2 or voice_3) and (voice_1 or voice_2 or voice_3).value) or ''
            voice_input = voice_input if voice_input is not None else ''
            lang = lang.value if lang is not None else ''
            await process_speak(i, input_text, selected_voice, lang, voice_input)

## Initialized default settings
# Sub-classes under a main class 'Settings'
class Behavior:
    def __init__(self):
        self.behavior_name = '' # label used for /behavior command
        self.chance_to_reply_to_other_bots = 0.5
        self.conversation_recency = 600
        self.go_wild_in_channel = True
        self.ignore_history = False
        self.ignore_parentheses = True
        self.only_speak_when_spoken_to = True
        self.reply_to_bots_when_addressed = 0.3
        self.reply_to_itself = 0.0
        self.reply_with_image = 0.0
        self.user_conversations = {}

    def update_behavior_dict(self, new_data):
        # Update specific attributes of the behavior instance
        for key, value in new_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def update_user_dict(self, user_id):
        # Update the last conversation time for a user
        self.user_conversations[user_id] = datetime.now()

    def in_active_conversation(self, user_id):
        # Check if a user is in an active conversation with the bot
        last_conversation_time = self.user_conversations.get(user_id)
        if last_conversation_time:
            time_since_last_conversation = datetime.now() - last_conversation_time
            return time_since_last_conversation.total_seconds() < self.conversation_recency
        return False

    def bot_should_reply(self, i, text):
        # Don't reply to @everyone or to itself
        if i.mention_everyone or i.author == client.user:
            return False
        # Whether to reply to other bots
        if i.author.bot and client.user.display_name.lower() in text.lower() and i.channel.id in client.database.main_channels:
            if 'bye' in text.lower(): # don't reply if another bot is saying goodbye
                return False
            return self.probability_to_reply(self.reply_to_bots_when_addressed)
        # Whether to reply when text is nested in parenthesis
        if self.ignore_parentheses and (i.content.startswith('(') and i.content.endswith(')')) or (i.content.startswith('<:') and i.content.endswith(':>')):
            return False
        # Whether to reply if only speak when spoken to
        if (self.only_speak_when_spoken_to and (client.user.mentioned_in(i) or any(word in i.content.lower() for word in client.user.display_name.lower().split()))) or (self.in_active_conversation(i.author.id) and i.channel.id in client.database.main_channels):
            return True
        reply = False
        # few more conditions
        if i.author.bot and i.channel.id in client.database.main_channels:
            reply = self.probability_to_reply(self.chance_to_reply_to_other_bots)
        if self.go_wild_in_channel and i.channel.id in client.database.main_channels:
            reply = True
        if reply:
            self.update_user_dict(i.author.id)
        return reply

class ImgModel:
    def __init__(self):
        self.imgmodel_imgtags = ''
        self.imgmodel_name = '' # label used for /imgmodel command
        self.imgmodel_url = ''
        self.override_settings = {}
        self.payload = {
            'alwayson_scripts': {}
        }

class ImgTag:
    def __init__(self):
        self.imgtag_name = '' # label used for /imgtag command
        self.negative_prompt_prefix = ''
        self.positive_prompt_prefix = ''
        self.positive_prompt_suffix = ''
        self.positive_prompt_suffix2 = ''
        self.positive_prompt_suffix2_blacklist = ''
        self.presets = []

class LLMContext:
    def __init__(self):
        self.bot_description = ''
        self.bot_emoji = ''
        self.context = 'The following is a conversation with an AI Large Language Model. The AI has been trained to answer questions, provide recommendations, and help with decision making. The AI follows user requests. The AI thinks outside the box.'
        self.extensions = {}
        self.greeting = '' # 'How can I help you today?'
        self.llmcontext_name = '' # label used for /llmcontext command
        self.name = 'AI'
        self.use_voice_channel = False

class LLMState:
    def __init__(self):
        self.llmstate_name = '' # label used for /llmstate command
        self.text = ''
        self.state = {
            # These are defaults for 'Midnight Enigma' preset
            'preset': '',
            'grammar_string': '',
            'add_bos_token': True,
            'auto_max_new_tokens': False,
            'ban_eos_token': False, 
            'character_menu': '',
            'chat_generation_attempts': 1,
            'chat_prompt_size': 2048,
            'custom_stopping_strings': '',
            'custom_token_bans': '',
            'do_sample': True,
            'early_stopping': False,
            'encoder_repetition_penalty': 1,
            'epsilon_cutoff': 0,
            'eta_cutoff': 0,
            'frequency_penalty': 0,
            'greeting': '',
            'guidance_scale': 1,
            'history': {'internal': [], 'visible': []},
            'length_penalty': 1,
            'max_new_tokens': 512,
            'max_tokens_second': 0,
            'min_length': 0,
            'min_p': 0.00,
            'mirostat_eta': 0.1,
            'mirostat_mode': 0,
            'mirostat_tau': 5,
            'mode': 'chat',
            'name1': '',
            'name1_instruct': '',
            'name2': '',
            'name2_instruct': '',
            'negative_prompt': '',
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'presence_penalty': 0,
            'repetition_penalty': 1.18,
            'repetition_penalty_range': 1024,
            'seed': -1.0,
            'skip_special_tokens': True,
            'stop_at_newline': False,
            'stopping_strings': '',
            'stream': True,
            'temperature': 0.98,
            'temperature_last': False,
            'tfs': 1,
            'top_a': 0,
            'top_k': 100,
            'top_p': 0.37,
            'truncation_length': 2048,
            'turn_template': '',
            'typical_p': 1,
            'chat_template_str': '''{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{- name1 + ': ' + message['content'] + '\\n'-}}\n        {%- else -%}\n            {{- name2 + ': ' + message['content'] + '\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}''',
            'instruction_template_str': '''{%- set found_item = false -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set found_item = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not found_item -%}\n    {{- '' + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + '\\n\\n' -}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- '' + message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{-'### Instruction:\\n' + message['content'] + '\\n\\n'-}}\n        {%- else -%}\n            {{-'### Response:\\n' + message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{-'### Response:\\n'-}}\n{%- endif -%}'''
            }
        self.regenerate = False
        self._continue = False

# Holds the default value of all sub-classes.
class Settings:
    def __init__(self):
        self.behavior = Behavior()
        self.imgmodel = ImgModel()
        self.imgtag = ImgTag()
        self.llmcontext = LLMContext()
        self.llmstate = LLMState()

    # Returns the value of Settings as a dictionary
    def settings_to_dict(self):
        return {
            'behavior': vars(self.behavior),
            'imgmodel': vars(self.imgmodel),
            'imgtag': vars(self.imgtag),
            'llmcontext': vars(self.llmcontext),
            'llmstate': vars(self.llmstate)
        }
    
    # Allows printing default values of Settings
    def __str__(self):
        attributes = ", ".join(f"{attr}={getattr(self, attr)}" for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__"))
        return f"{self.__class__.__name__}({attributes})"

class Database:
    def __init__(self):
        self.take_notes_about_users = None # not yet implemented
        self.learn_about_and_use_guild_emojis = None # not yet implemented
        self.read_chatlog = None # not yet implemented
        self.first_run = self.initialize_first_run()
        self.last_change = self.initialize_last_change()
        self.main_channels = self.initialize_main_channels()

    def initialize_last_change(self):
        conn = sqlite3.connect('bot.db')
        c = conn.cursor()
        c.execute('''SELECT name FROM sqlite_master WHERE type='table' AND name='last_change' ''')
        is_last_change_table_exists = c.fetchone()
        if not is_last_change_table_exists:
            c.execute('''CREATE TABLE IF NOT EXISTS last_change (timestamp TEXT)''')
            conn.commit()  # Commit the changes to persist them
            conn.close()
            return None
        c.execute('''SELECT timestamp FROM last_change''')
        timestamp = c.fetchone()
        conn.close()
        return timestamp[0] if timestamp else None

    def initialize_main_channels(self):
        conn = sqlite3.connect('bot.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS emojis (emoji TEXT UNIQUE, meaning TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS config (setting TEXT UNIQUE, value TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS main_channels (channel_id TEXT UNIQUE)''')
        c.execute('''SELECT channel_id FROM main_channels''')
        result = [int(row[0]) for row in c.fetchall()]
        conn.close()
        return result if result else []

    def initialize_first_run(self):
        conn = sqlite3.connect('bot.db')
        c = conn.cursor()
        c.execute('''SELECT name FROM sqlite_master WHERE type='table' AND name='first_run' ''')
        is_first_run_table_exists = c.fetchone()
        if not is_first_run_table_exists:
            c.execute('''CREATE TABLE IF NOT EXISTS first_run (is_first_run BOOLEAN)''')
            c.execute('''INSERT INTO first_run (is_first_run) VALUES (1)''')
            conn.commit()
            conn.close()
            return True
        c.execute('''SELECT COUNT(*) FROM first_run''')
        is_first_run_exists = c.fetchone()[0]
        conn.close()
        return is_first_run_exists == 0

def probability_to_reply(probability):
    # Determine if the bot should reply based on a probability
    return random.random() < probability

def queue(i, user_input):
    user_id = i.author.mention
    queues.append({user_id:user_input})
    logging.info(f'reply requested: "{user_id} asks {user_input["state"]["name2"]}: {user_input["text"]}"')

def check_num_in_queue(i):
    user = i.author.mention
    user_list_in_que = [list(i.keys())[0] for i in queues]
    return user_list_in_que.count(user)

async def a1111_txt2img(payload, picture_frame):
    async def save_images_and_return():
        async with aiohttp.ClientSession() as session:
            async with session.post(url=f'{A1111}/sdapi/v1/txt2img', json=payload) as response:
                r = await response.json()
                images = []
                for img_data in r['images']:
                    image = Image.open(io.BytesIO(base64.b64decode(img_data.split(",", 1)[0])))
                    png_payload = {"image": "data:image/png;base64," + img_data}
                    response2 = requests.post(url=f'{A1111}/sdapi/v1/png-info', json=png_payload)
                    pnginfo = PngImagePlugin.PngInfo()
                    pnginfo.add_text("parameters", response2.json().get("info"))
                    image.save(f'temp_img_{len(images)}.png', pnginfo=pnginfo)
                    images.append(image)
                return images

    async def track_progress():
        await check_a1111_progress(picture_frame)

    # Start both tasks concurrently
    images_task = asyncio.create_task(save_images_and_return())
    progress_task = asyncio.create_task(track_progress())

    # Wait for both tasks to complete
    await asyncio.gather(images_task, progress_task)

    images = await images_task  # Get the list of images after both tasks are done

    return images

def progress_bar(value, length=20):
    filled_length = int(length * value)
    bar = ':white_large_square:' * filled_length + ':white_square_button:' * (length - filled_length)
    return f'{bar}'

async def check_a1111_progress(picture_frame):
    async with aiohttp.ClientSession() as session:
        progress_data = {"progress":0}
        while progress_data['progress'] == 0:
            try:
                async with session.get(f'{A1111}/sdapi/v1/progress') as progress_response:
                    progress_data = await progress_response.json()
                    progress = progress_data['progress']
                    #print(f'Progress: {progress}%')
                    info_embed.title = 'Waiting for response from A1111 ...'
                    await picture_frame.edit(embed=info_embed)                    
                    await asyncio.sleep(1)
            except aiohttp.client_exceptions.ClientConnectionError:
                print('Connection closed, retrying in 1 seconds')
                await asyncio.sleep(1)
        while progress_data['state']['job_count'] > 0:
            try:
                async with session.get(f'{A1111}/sdapi/v1/progress') as progress_response:
                    progress_data = await progress_response.json()
                    #pprint.pp(progress_data)
                    progress = progress_data['progress'] * 100
                    if progress == 0 :
                        info_embed.title = f'Generating image: 100%'
                        info_embed.description = progress_bar(1)
                        await picture_frame.edit(embed=info_embed)
                        break
                    #print(f'Progress: {progress}%')
                    info_embed.title = f'Generating image: {progress:.0f}%'
                    info_embed.description = progress_bar(progress_data['progress'])
                    await picture_frame.edit(embed=info_embed)
                    await asyncio.sleep(1)
            except aiohttp.client_exceptions.ClientConnectionError:
                print('Connection closed, retrying in 1 seconds')
                await asyncio.sleep(1)

client.run(bot_args.token if bot_args.token else TOKEN, root_logger=True, log_handler=handler)