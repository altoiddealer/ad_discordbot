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
import cv2 # pip install OpenCV-Python
from itertools import product

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
extensions_module.available_extensions = utils.get_available_extensions()
if shared.args.extensions is not None and len(shared.args.extensions) > 0:
    extensions_module.load_extensions()

#Discord Bot
prompt = "This is a conversation with your Assistant. The Assistant is very helpful and is eager to chat with you and answer your questions."
your_name = "You"
llamas_name = "Assistant"

reply_embed_json = {
    "title": "Reply #X",
    "color": 39129,
    "timestamp": (datetime.now() - timedelta(hours=3)).isoformat(),
    "url": "https://github.com/xNul/chat-llama-discord-bot",
    "footer": {
        "text": "Contribute to ChatLLaMA on GitHub!",
    },
    "fields": [
        {
            "name": your_name,
            "value": ""
        },
        {
            "name": llamas_name,
            "value": ":arrows_counterclockwise:"
        }
    ]
}
reply_embed = discord.Embed().from_dict(reply_embed_json)

reset_embed_json = {
    "title": "Conversation has been reset",
    "description": "Replies: 0\nYour name: " + your_name + "\nLLaMA's name: " + llamas_name + "\nPrompt: " + prompt,
    "color": 39129,
    "timestamp": (datetime.now() - timedelta(hours=3)).isoformat(),
    "url": "https://github.com/xNul/chat-llama-discord-bot",
    "footer": {
        "text": "Contribute to ChatLLaMA on GitHub!"
    }
}

reset_embed = discord.Embed().from_dict(reset_embed_json)

status_embed_json = {
    "title": "Status",
    "description": "You don't have a job queued.",
    "color": 39129,
    "timestamp": (datetime.now() - timedelta(hours=3)).isoformat(),
    "url": "https://github.com/xNul/chat-llama-discord-bot",
    "footer": {
        "text": "Contribute to ChatLLaMA on GitHub!"
    }
}
status_embed = discord.Embed().from_dict(status_embed_json)

greeting_embed_json = {
    "title": "",
    "description": "",
    "thumbnail": ""
}
greeting_embed = discord.Embed().from_dict(greeting_embed_json)

info_embed_json = {
    "title": "How to use",
    "description": """
      **/character** - Change character
      **/main** - Set main channel for bot so it can reply without being called by name
      **/image** - prompt an image to be generated by A1111.
      **__Changing settings__**
      **/imgmodel** - Change A1111 model & payload settings as defined in /ad_discordbot/dict_imgmodels.yaml
      **/imgloras** - Change preset trigger phrases with prompt adders in /ad_discordbot/dict_imgloras.yaml
      **/llmcontext** - Change context as defined in /ad_discordbot/dict_llmcontexts.yaml
      **/llmstate** - Change state as defined in /ad_discordbot/dict_llmstates.yaml
      **/behavior** - Change context as defined in /ad_discordbot/dict_behaviors.yaml
      """
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

queues = []
blocking = False
reply_count = 0

# # Function to recursively update a dictionary
def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    # Add missing keys from the source dictionary to the target dictionary
    for k in d.keys() - u.keys():
        u[k] = d[k]
    return u

# def update_dict(d, u):
#     for k, v in u.items():
#         if isinstance(v, dict):
#             d[k] = update_dict(d.get(k, {}), v)
#         else:
#             d[k] = v
#     return d

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

async def change_profile(i, character):
    # Check for cooldown before allowing profile change
    change_name = config.discord['change_username_with_character']
    change_avatar = config.discord['change_avatar_with_character']

    if change_name or change_avatar:
        last_change = getattr(i.bot, "last_change", None)
        if last_change and datetime.now() < last_change + timedelta(minutes=10):
            remaining_cooldown = last_change + timedelta(minutes=10) - datetime.now()
            seconds = int(remaining_cooldown.total_seconds())
            await i.channel.send(f'Please wait {seconds} before changing character again')
            return
    try:
        # Load the new character's information
        new_char = load_character(character, '', '', instruct=False)
        name1, name2, picture, greeting, context, _ = new_char
        i.bot.llm_context = context

        if change_name and i.bot.user.display_name != name2:
            await i.bot.user.edit(username=name2)

        if change_avatar:
            folder = 'characters'
            picture_path = os.path.join(folder, f'{character}.png')
            if os.path.exists(picture_path):
                with open(picture_path, 'rb') as f:
                    picture = f.read()
                await i.bot.user.edit(avatar=picture)

        character_path = os.path.join("characters", f"{character}.yaml")
        char_data = load_yaml_file(character_path)
        active_settings = load_yaml_file('ad_discordbot/activesettings.yaml')

        # Update 'llmcontext' dictionary in the active settings directly from character file
        llmcontext_dict = {}
        for key in ['name', 'greeting', 'context', 'bot_description', 'bot_emoji']:
            if key in char_data:
                llmcontext_dict[key] = char_data[key]
        if llmcontext_dict:
            active_settings['llmcontext'].update(llmcontext_dict)
        # Update behavior in active settings
        if char_data['behavior']:
            update_dict(active_settings['behavior'], char_data['behavior'])
        # Update state in active settings
        if char_data['state']:
            update_dict(active_settings['llmstate']['state'], char_data['state'])

        # Save the updated active_settings to activesettings.yaml
        save_yaml_file('ad_discordbot/activesettings.yaml', active_settings)

        reset_session_history()

        # Send a greeting or a message if there's no greeting
        greeting = char_data.get('greeting')
        if greeting:
            await i.channel.send(greeting)
        else:
            await interaction.response.send_message(f"**{character} has entered the chat.**")

        # Update last_change timestamp
        i.bot.last_change = datetime.now()

    except (discord.HTTPException, Exception) as e:
        await i.channel.send(f"An error occurred: {e}")

    if i.bot.behavior.read_chatlog:
        pass

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

def chatbot_wrapper_wrapper(user_input, save_history):
    for resp in chatbot_wrapper(text=user_input['text'], state=user_input['state'], regenerate=user_input['regenerate'], _continue=user_input['_continue']):
        i_resp = resp['internal']
        if len(i_resp) > 0:
            resp_clean = i_resp[len(i_resp) - 1][1]
            last_resp = resp_clean
    # Retain chat history
    if not get_active_setting('behavior').get('ignore_history') and save_history:
        global session_history
        session_history['internal'].append([user_input['text'], last_resp])
        session_history['visible'].append([user_input['text'], last_resp])
    return last_resp # bot's reply text

async def llm_gen(i, queues, save_history):
    global blocking
    global reply_count

    while len(queues) > 0:
        blocking = True
        reply_count += 1
        user_input = queues.pop(0)
        mention = list(user_input.keys())[0]
        user_input = user_input[mention]
        last_resp = chatbot_wrapper_wrapper(user_input, save_history)
        logging.info("reply sent: \"" + mention + ": {'text': '" + user_input["text"] + "', 'response': '" + last_resp + "'}\"")
        await send_long_message(i.channel, last_resp)
        # if bot_args.limit_history is not None and len(user_input['state']['history']['visible']) > bot_args.limit_history:
        #     user_input['state']['history']['visible'].pop(0)
        #     user_input['state']['history']['internal'].pop(0)
    blocking = False

## Function to automatically change image models
# Build list of imgmodels depending on user preference (user .yaml / A1111 API)
async def auto_fetch_imgmodels():
    try:
        active_settings = load_yaml_file('ad_discordbot/activesettings.yaml')
        current_imgmodel_name = active_settings.get('imgmodel', {}).get('imgmodel_name', '')
        if not config.imgmodels['get_imgmodels_via_api']:
            items = load_yaml_file('ad_discordbot/dict_imgmodels.yaml')
        else:
            async with aiohttp.ClientSession() as session: # populate options from A1111 API
                async with session.get(url=f'{A1111}/sdapi/v1/sd-models') as response:
                    items = await response.json()
                    # Update 'title' keys to 'sd_model_checkpoint' to be uniform with .yaml method (the key for actual checkpoint file)
                    for item in items:
                        if 'title' in item:
                            item['sd_model_checkpoint'] = item.pop('title')
        return active_settings, current_imgmodel_name, items
    except Exception as e:
        print("Error fetching image models:", e)

# Apply user defined filters
async def auto_filter_imgmodels(items):
    try:                                
        if config.imgmodels['auto_change_models']['filter'] or config.imgmodels['exclude']:
            filter_list = config.imgmodels['auto_change_models']['filter']
            exclude_list = config.imgmodels['exclude']
            items = [
                item for item in items
                if (
                    (not filter_list or any(re.search(re.escape(filter_text), item.get('imgmodel_name', '') + item.get('sd_model_checkpoint', ''), re.IGNORECASE) for filter_text in filter_list))
                    and (not exclude_list or not any(re.search(re.escape(exclude_text), item.get('imgmodel_name', '') + item.get('sd_model_checkpoint', ''), re.IGNORECASE) for exclude_text in exclude_list))
                )
            ]
        items = items[:25]
        item_names = [item.get('imgmodel_name', item.get('sd_model_checkpoint', None)) for item in items]
        return items, item_names
    except Exception as e:
        print("Error filtering image model list:", e)

# Select imgmodel based on mode, while avoid repeating current imgmodel
async def auto_select_imgmodel(current_imgmodel_name, items, item_names, mode='random'):   
    try:         
        if mode == 'random':
            if current_imgmodel_name:
                matched_item = None
                for item, item_name in zip(items, item_names):
                    if item_name == current_imgmodel_name:
                        matched_item = item
                        break
                if len(items) >= 2 and matched_item is not None:
                    items.remove(matched_item)
            selected_item = random.choice(items)
        elif mode == 'cycle':
            if current_imgmodel_name:
                current_index = item_names.index(current_imgmodel_name)
                next_index = (current_index + 1) % len(item_names)  # Cycle to the beginning if at the end
                selected_item = items[next_index]
            else:
                selected_item = random.choice(items) # If no image model set yet, select randomly
        return selected_item
    except Exception as e:
        print("Error automatically selecting image model:", e)

# Update activesettings.yaml with selected imgmodel data
async def auto_change_imgmodel(selected_item, active_settings):
    try:
        if not config.imgmodels['get_imgmodels_via_api']:
            selected_item_name = selected_item['imgmodel_name']
            # update the entire imgmodel dict in activesettings.yaml
            await update_active_settings(selected_item, 'imgmodel')
        else:
            selected_item_name = selected_item['model_name']
            # update only the values accessible via A1111 API
            active_settings['imgmodel']['imgmodel_name'] = selected_item_name
            active_settings['imgmodel']['imgmodel_url'] = ''
            active_settings['imgmodel']['override_settings']['sd_model_checkpoint'] = selected_item['sd_model_checkpoint']
            save_yaml_file('ad_discordbot/activesettings.yaml', active_settings)
        # Update size options for /image command
        await update_size_options(active_settings.get('imgmodel').get('payload').get('width'),active_settings.get('imgmodel').get('payload').get('height'))
        print(f"Updated imgmodel settings to: {selected_item_name}")
        return selected_item_name
    except Exception as e:
        print("Error updating image model:", e)

# Task to auto-select an imgmodel at user defined interval
async def auto_update_imgmodel_task(mode='random'):
    while True:
        frequency = config.imgmodels['auto_change_models']['frequency']
        duration = frequency*3600 # 3600 = 1 hour
        await asyncio.sleep(duration)
        try:
            active_settings, current_imgmodel_name, items = await auto_fetch_imgmodels()
            items, item_names = await auto_filter_imgmodels(items)
            selected_item = await auto_select_imgmodel(current_imgmodel_name, items, item_names, mode)
            selected_item_name = await auto_change_imgmodel(selected_item, active_settings)
            # Process announcements
            if config.imgmodels['auto_change_models']['channel_announce']:
                channel = client.get_channel(config.imgmodels['auto_change_models']['channel_announce'])
                reply = await process_imgmodel_announce(channel, selected_item, selected_item_name)
                if reply:
                    await channel.send(reply)
                else:
                    await channel.send(f"Updated imgmodel settings to: {selected_item_name}")
        except Exception as e:
            print(f"Error automatically updating image model: {e}")

imgmodel_update_task = None # Global variable allows process to be cancelled and restarted (reset sleep timer)

# Helper function to start auto-select imgmodel
async def start_auto_update_imgmodel_task():
    global imgmodel_update_task
    if imgmodel_update_task:
        imgmodel_update_task.cancel()
    if config.imgmodels['auto_change_models']['enabled']:
        mode = config.imgmodels['auto_change_models']['mode']
        imgmodel_update_task = client.loop.create_task(auto_update_imgmodel_task(mode))

## On Ready
@client.event
async def on_ready():
    if not hasattr(client, 'context'):
        """Loads character profile based on Bot's display name"""
        sources = [
            client.user.display_name,
            get_active_setting('llmcontext').get('name'),
            config.discord['char_name']
        ]

        for source in sources:
            try:
                _, _, _, _, context, _ = load_character(source, '', '')
                if context:
                    client.llm_context = context
                    break  # Character loaded successfully, exit the loop
            except Exception as e:
                client.context = "no character loaded"
                logging.error("Error loading character:", e)
    client.fresh = True
    client.behavior = Behavior()
    try:
        client.behavior.__dict__.update(get_active_setting('behavior'))
    except Exception as e:
        logging.error("Error updating behavior:", e)
    logging.info("Bot is ready")
    await client.tree.sync()
    # task to change image models automatically
    await start_auto_update_imgmodel_task()

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

# 
@client.event
async def on_raw_reaction_add(endorsed_img):
    if not config.discord['starboard']['enabled']:
        return
    channel = await client.fetch_channel(endorsed_img.channel_id)
    message = await channel.fetch_message(endorsed_img.message_id)
    total_reaction_count = 0
    if config.discord['starboard']['emoji_specific']:
        for emoji in config.discord['starboard']['react_emojis']:
            reaction = discord.utils.get(message.reactions, emoji=emoji)
            if reaction:
                total_reaction_count += reaction.count
    else:
        for reaction in message.reactions:
            total_reaction_count += reaction.count
    if total_reaction_count >= config.discord['starboard']['min_reactions']:
        target_channel = client.get_channel(config.discord['starboard']['target_channel_id'])
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

## Dynamic Context feature
def process_dynamic_context(user_input, text, llm_prompt):
    dynamic_context = config.dynamic_context
    matched_presets = []
    if dynamic_context['enabled']:
        for preset in dynamic_context['presets']:
            triggers = preset['triggers']
            on_prefix_only = preset['on_prefix_only']
            remove_trigger_phrase = preset['remove_trigger_phrase']
            load_history = preset['load_history']
            save_history = preset['save_history']
            add_instruct = preset['add_instruct']
            if on_prefix_only:
                if any(text.lower().startswith(trigger) for trigger in triggers):
                    matched_presets.append((preset, load_history, save_history, remove_trigger_phrase, add_instruct))
            else:
                if any(trigger in text.lower() for trigger in triggers):
                    matched_presets.append((preset, load_history, save_history, remove_trigger_phrase, add_instruct))
        if matched_presets:
            print_content = f"Dynamic Context: "
            chosen_preset = matched_presets[0] # Only apply one preset. Priority is top-to-bottom in the config file.
            swap_character_name = chosen_preset[0]['swap_character']
            load_history_value = chosen_preset[0]['load_history']
            save_history_value = chosen_preset[0]['save_history']
            # Swap Character handling:
            if swap_character_name:
                try:
                    character_path = os.path.join("characters", f"{swap_character_name}.yaml")
                    if character_path:
                        char_data = load_yaml_file(character_path)
                        if char_data['name']: user_input['state']['name2'] = char_data['name']
                        if char_data['context']: user_input['state']['context'] = char_data['context']
                        if char_data['state']:
                            update_dict(user_input['state'], char_data['state'])
                except Exception as e:
                    print(f"An error occurred while loading the YAML file for swap_character:", e)
                print_content += f"Character: {swap_character_name}"
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
    if config.tell_bot_time['enabled']:
        if config.tell_bot_time['time_offset']: current_time = config.tell_bot_time['time_offset']
        else: current_time = 0.0
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
    image_trigger_settings = config.imgprompt_settings['trigger_img_gen_by_phrase']
    if not image_trigger_settings['enabled']:
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
    if random.random() < get_active_setting('behavior').get('reply_with_image'): #and client.behavior.bot_should_reply(i, text):
        return True
    return False

def create_image_prompt(user_input, llm_prompt, current_time, save_history):
    user_input['text'] = llm_prompt
    if 'llmstate_name' in user_input: del user_input['llmstate_name'] # Looks more legit without this
    if current_time and config.tell_bot_time['message'] and config.tell_bot_time['mode'] >= 1:
        user_input['state']['context'] = config.tell_bot_time['message'].format(current_time) + user_input['state']['context']
    last_resp = chatbot_wrapper_wrapper(user_input, save_history)
    if len(last_resp) > 2000: last_resp = last_resp[:2000]
    return last_resp

async def create_prompt_for_llm(i, user_input, llm_prompt, current_time, save_history):
    user_input['text'] = llm_prompt
    if 'llmstate_name' in user_input: del user_input['llmstate_name'] # Looks more legit without this
    if current_time and config.tell_bot_time['message'] and config.tell_bot_time['mode'] != 1:
        user_input['state']['context'] = config.tell_bot_time['message'].format(current_time) + user_input['state']['context']
    num = check_num_in_queue(i)
    if num >=10:
        await i.channel.send(f'{i.author.mention} You have 10 items in queue, please allow your requests to finish before adding more to the queue.')
    else:
        queue(i, user_input)
        #pprint.pp(user_input)
        async with i.channel.typing():
            await llm_gen(i, queues, save_history)

def initialize_user_input(i, data, text):
    user_input = get_active_setting('llmstate') # default state settings
    user_input['text'] = text
    user_input['state']['name1'] = i.author.display_name
    user_input['state']['name2'] = data.get('name')
    #user_input['state']['name2'] = client.user.display_name
    user_input['state']['context'] = data.get('context') # default character context
    #user_input['state']['context'] = client.llm_context
    # check for ignore history setting / start with default history settings
    if not get_active_setting('behavior').get('ignore_history'):
        user_input['state']['history'] = session_history
    # print(user_input)
    return user_input

@client.event
async def on_message(i):
    ctx = Context(message=i,prefix=None,bot=client,view=None)
    text = await commands.clean_content().convert(ctx, i.content)
    if client.behavior.bot_should_reply(i, text): pass # Bot replies
    else: return # Bot does not reply to this message.
    if client.behavior.main_channels == None and client.user.mentioned_in(i): main(i) # if None, set channel as main
    data = get_active_setting('llmcontext')
    # if @ mentioning bot, remove the @ mention from user prompt
    if text.startswith(f"@{client.user.display_name} "):
        text = text.replace(f"@{client.user.display_name} ", "", 1)
    elif text.startswith(f"@{data['name']} "):
        text = text.replace(f"@{data['name']} ", "", 1)
    elif text.startswith(f"@{config.discord['char_name']} "):
        text = text.replace(f"@{config.discord['char_name']} ","", 1)
    llm_prompt = text # 'text' will be retained as user's raw text (without @ mention)
    # build user_input with defaults
    user_input = initialize_user_input(i, data, text)
    # apply dynamic_context settings
    user_input, llm_prompt, save_history = process_dynamic_context(user_input, text, llm_prompt)
    # apply datetime to prompt
    current_time = determine_date
    # save a global copy of text/llm_prompt for /regen cmd
    retain_last_user_message(text, llm_prompt)
    if user_asks_for_image(i, text):
        if await a1111_online(i):
            info_embed.title = "Prompting ..."
            info_embed.description = " "
            picture_frame = await i.reply(embed=info_embed)
            async with i.channel.typing():
                image_prompt = create_image_prompt(user_input, llm_prompt, current_time, save_history)
                await picture_frame.delete()
                await pic(i, text, image_prompt, neg_prompt=None, size=None, face_swap=None, controlnet=None)
                await i.channel.send(image_prompt)
        return
    await create_prompt_for_llm(i, user_input, llm_prompt, current_time, save_history)
    return

@client.hybrid_command(description="Set current channel as main channel for bot to auto reply in without needing to be called")
async def main(i):
    if i.channel.id not in i.bot.behavior.main_channels:
        i.bot.behavior.main_channels.append(i.channel.id)
        conn = sqlite3.connect('bot.db')
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO main_channels (channel_id) VALUES (?)''', (i.channel.id,))
        conn.commit()
        conn.close()
        await i.reply(f'Bot main channel set to {i.channel.mention}')
    await i.reply(f'{i.channel.mention} set as main channel')

@client.hybrid_command(description="Display help menu")
async def helpmenu(i):
    info_embed = discord.Embed().from_dict(info_embed_json)
    await i.send(embed=info_embed)

@client.hybrid_command(description="Regenerate the bot's last reply")
async def regen(i):
    info_embed.title = f"Regenerating ... "
    info_embed.description = ""
    await i.reply(embed=info_embed)
    data = get_active_setting('llmcontext')
    user_input = initialize_user_input(i, data, text=last_user_message['llm_prompt'])
    user_input['regenerate'] = True
    last_resp = chatbot_wrapper_wrapper(user_input, save_history=None)
    await send_long_message(i.channel, last_resp)

@client.hybrid_command(description="Continue the generation")
async def cont(i):
    info_embed.title = f"Continuing ... "
    info_embed.description = ""
    await i.reply(embed=info_embed)
    data = get_active_setting('llmcontext')
    user_input = initialize_user_input(i, data, text=last_user_message['llm_prompt'])
    user_input['_continue'] = True
    last_resp = chatbot_wrapper_wrapper(user_input, save_history=None)
    await delete_last_message(i)
    await send_long_message(i.channel, last_resp)

#----BEGIN IMAGE PROCESSING----#

async def process_image_gen(payload, picture_frame, i):
    try:
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
            # Send all images to the channel
            image_files = [discord.File(f'temp_img_{idx}.png') for idx in range(len(images))]
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
    negative_prompt_list = payload['negative_prompt'].split(', ')
    unique_values_set = set()
    unique_values_list = []
    for value in negative_prompt_list:
        if value not in unique_values_set:
            unique_values_set.add(value)
            unique_values_list.append(value)
    processed_negative_prompt = ', '.join(unique_values_list)
    payload['negative_prompt'] = processed_negative_prompt
    # Delete unwanted keys
    if not config.sd['extensions']['controlnet_enabled']:
        del payload['alwayson_scripts']['controlnet'] # Delete all 'controlnet' keys if disabled by config
    if not config.sd['extensions']['reactor_enabled']:
        del payload['alwayson_scripts']['reactor'] # Delete all 'reactor' keys if disabled by config
    # Workaround for denoising strength A1111 bug (will be fixed in next A1111 version)
    if not payload.get('enable_hr'):
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
    trigger_params = config.imgprompt_settings['trigger_faces_by_phrase']
    if trigger_params['enabled']:
        matched_presets = []     
        for preset in trigger_params['presets']:
            triggers = preset['triggers']
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

def process_payload_mods(payload, text):
    # Process triggered mods
    trigger_params = config.imgprompt_settings['trigger_img_params_by_phrase']
    if trigger_params['enabled']:
        matched_presets = []     
        for preset in trigger_params['presets']:
            triggers = preset['triggers']
            preset_payload = {}  # Create a dictionary to hold payload settings for each preset
            if any(trigger in text.lower() for trigger in triggers):
                for key, value in preset.items():
                    if key != 'triggers': # All key-value pairs except "triggers"
                        preset_payload[key] = value
                matched_presets.append((preset, preset_payload))
        for preset, preset_payload in matched_presets:
            payload.update(preset_payload)  # Update the payload with each preset's settings
    return payload

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
        search_mode = config.imgprompt_settings['trigger_search_mode']
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
                if re.search(trigger_regex, search_text.lower()):
                    matched_presets.append(preset)
                    break
        if matched_presets:
            # Collect 'trump' parameters for each matched preset
            trump_params = []
            for preset in matched_presets:
                if 'trumps' in preset:
                    trump_params.extend([param.strip() for param in preset['trumps'].split(',')])
            trump_params = list(set(trump_params)) # Remove duplicates from the trump_params list
            # Compare each matched preset's trigger to the list of trump parameters
            # Ignore the preset if its trigger exactly matches any trump parameter
            filtered_presets = [preset for preset in matched_presets if not any(trigger.lower() in trump_params for trigger in preset['trigger'].split(','))]
            for preset in reversed(filtered_presets):
                payload['negative_prompt'] += preset['negative_prompt']
                matched_text = None
                triggers = [t.strip() for t in preset['trigger'].split(',')]
                # Iterate through the triggers for the current preset
                for trigger in triggers:
                    pattern = re.escape(trigger.lower())
                    #pattern = re.escape(trigger.lower())
                    match = re.search(pattern, payload['prompt'].lower())
                    if match:
                        matched_text = match.group(0)
                        break
                if matched_text:
                    # Find the index of the first occurrence of matched_text
                    insert_index = payload['prompt'].lower().index(matched_text) + len(matched_text)
                    insert_text = preset['positive_prompt']
                    payload['prompt'] = payload['prompt'][:insert_index] + insert_text + payload['prompt'][insert_index:]
                    filtered_presets.remove(preset)
            # All remaining are appended to prompt
            for preset in filtered_presets:
                payload['prompt'] += preset['positive_prompt']

def apply_suffix2(payload, positive_prompt_suffix2, positive_prompt_suffix2_blacklist):
    if positive_prompt_suffix2:
        blacklist_phrases = positive_prompt_suffix2_blacklist.split(',')
        prompt_text = re.sub(r'<([^<>]+)>', r'\1', payload['prompt'])  # Remove angle brackets while preserving contents
        prompt_text = prompt_text.lower()
        if not any(phrase.lower() in prompt_text for phrase in blacklist_phrases):
            payload['prompt'] += positive_prompt_suffix2

async def pic(i, text, image_prompt, neg_prompt=None, size=None, face_swap=None, controlnet=None):
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
        activepayload = get_active_setting('imgmodel').get('payload')
        payload.update(activepayload)
        activeoverride = get_active_setting('imgmodel').get('override_settings')
        payload['override_settings'] = activeoverride
        # Process payload triggers
        process_payload_mods(payload, text)
        # Process face swap triggers
        process_faces(payload, text)
        if size: payload.update(size)
        # Process extensions
        if face_swap:
            payload['alwayson_scripts']['reactor']['args'][0] = face_swap # image in base64 format
            payload['alwayson_scripts']['reactor']['args'][1] = True # Enable
        if controlnet: payload['alwayson_scripts']['controlnet']['args'][0].update(controlnet)
        # Process LORAs
        data = get_active_setting('imglora')
        positive_prompt_prefix = data.get("positive_prompt_prefix")
        positive_prompt_suffix = data.get("positive_prompt_suffix")
        positive_prompt_suffix2 = data.get("positive_prompt_suffix2")
        positive_prompt_suffix2_blacklist = data.get("positive_prompt_suffix2_blacklist")
        negative_prompt_prefix = data.get("negative_prompt_prefix")
        presets = data.get("presets")

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

@client.hybrid_command()
async def sync(interaction: discord.Interaction):
    await client.tree.sync()

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
active_settings = load_yaml_file('ad_discordbot/activesettings.yaml')

sizes = options.get('sizes', {})
aspect_ratios = [size.get("ratio") for size in sizes.get('ratios', [])]

# Calculate the average and aspect ratio sizes
width = active_settings.get('imgmodel').get('payload').get('width')
height = active_settings.get('imgmodel').get('payload').get('height')
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

#----END IMAGE PROCESSING----#

@client.hybrid_command(description="Reset the conversation with LLaMA")
async def reset(i):
    global reply_count
    your_name = i.author.display_name
    llamas_name = i.bot.user.display_name
    reply_count = 0
    shared.stop_everything = True
    reset_session_history  # Reset conversation
    await change_profile(i, llamas_name)
    prompt = i.bot.llm_context
    info_embed.title = f"Conversation with {llamas_name} reset"
    info_embed.description = ""
    await i.reply(embed=info_embed)    
    logging.info("conversation reset: {'replies': " + str(reply_count) + ", 'your_name': '" + your_name + "', 'llamas_name': '" + llamas_name + "', 'prompt': '" + prompt + "'}")

@client.hybrid_command(description="Check the status of your reply queue position and wait time")
async def status(i):
    total_num_queued_jobs = len(queues)
    que_user_ids = [list(a.keys())[0] for a in queues]
    if i.author.mention in que_user_ids:
        user_position = que_user_ids.index(i.author.mention) + 1
        msg = f"{i.author.mention} Your job is currently {user_position} out of {total_num_queued_jobs} in the queue. Estimated time until response is ready: {user_position * 20/60} minutes."
    else:
        msg = f"{i.author.mention} doesn\'t have a job queued."
    status_embed.timestamp = datetime.now() - timedelta(hours=3)
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

class CharacterDropdown(discord.ui.Select):
    def __init__(self, i):
        options = [discord.SelectOption(label=character["name"], description=character["bot_description"], emoji=character["bot_emoji"]) for character in generate_characters()]
        super().__init__(placeholder='', min_values=1, max_values=1, options=options)
        self.i = i

    async def callback(self, interaction: discord.Interaction):
        character = self.values[0]
        await change_profile(self.i, character)

@client.hybrid_command(description="Choose Character")
async def character(i):
    view = discord.ui.View()
    view.add_item(CharacterDropdown(i))
    await i.send('Choose Character:', view=view, ephemeral=True)
     
# Settings Commands
async def update_active_settings(selected_item, active_settings_key):
    try:
        active_settings = load_yaml_file('ad_discordbot/activesettings.yaml')
        update_dict(active_settings.get(active_settings_key, {}), selected_item)
        if active_settings_key == 'imgmodel':
             # Update size options for /image command
            await update_size_options(active_settings.get('imgmodel').get('payload').get('width'),active_settings.get('imgmodel').get('payload').get('height'))
            try: # Update ImgLORAs
                if selected_item['imgmodel_loras']:
                    imgloras_data = load_yaml_file('ad_discordbot/dict_imgloras.yaml')
                    selected_imgmodel_loras = None
                    for imgloras in imgloras_data:
                        if imgloras.get('imglora_name') == selected_item['imgmodel_loras']:
                            selected_imgmodel_loras = imgloras
                            break
                    if selected_imgmodel_loras is not None:
                        update_dict(active_settings.get('imglora', {}), selected_imgmodel_loras)
            except Exception as e:
                print("Error updating imgloras for selected imgmodel in ad_discordbot/activesettings.yaml:", e)
        save_yaml_file('ad_discordbot/activesettings.yaml', active_settings)
    except Exception as e:
        print(f"Error updating ad_discordbot/activesettings.yaml ({active_settings_key}):", e)

# Post settings to dedicated channel
async def post_active_settings():
    if config.discord['post_active_settings']['target_channel_id']:
        channel = await client.fetch_channel(config.discord['post_active_settings']['target_channel_id'])
        if channel:
            active_settings = load_yaml_file('ad_discordbot/activesettings.yaml')
            settings_content = yaml.dump(active_settings, default_flow_style=False)
            # Fetch and delete all existing messages in the channel
            async for message in channel.history(limit=None):
                await message.delete()
                await asyncio.sleep(0.3)  # minimum delay for discord limit
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
            # If a new LLMContext is selected
            if self.active_settings_key == 'llmcontext':
                reset_session_history # Reset conversation
            if self.active_settings_key == 'imgmodel':
                # task to change image models automatically
                await start_update_image_model_task()
                channel = interaction.channel
                reply = await process_imgmodel_announce(channel, selected_item, selected_item_name)
                if reply:
                    await interaction.response.send_message(reply)
                else:
                    await interaction.response.send_message(f"Updated {self.active_settings_key} settings to: {selected_item_name}")
            else:
                await interaction.response.send_message(f"Updated {self.active_settings_key} settings to: {selected_item_name}")
            if config.discord['post_active_settings']['enabled']:
                await post_active_settings()
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
@client.hybrid_command(description="Choose LORAs")
async def imgloras(i):
    view = discord.ui.View()
    view.add_item(SettingsDropdown('ad_discordbot/dict_imgloras.yaml', 'imglora_name', 'imglora'))
    await i.send("Choose LORAs:", view=view, ephemeral=True)

async def process_imgmodel_announce(channel, selected_item, selected_item_name):
    # Set the topic of the channel if enabled in config
    if config.imgmodels['update_topic']['enabled']:
        topic_prefix = config.imgmodels['update_topic']['topic_prefix']
        if not config.imgmodels['get_imgmodels_via_api']:
            new_topic = f"{topic_prefix}{selected_item['imgmodel_name']}"
            if config.imgmodels['update_topic']['include_url']:
                new_topic += " " + selected_item.get('imgmodel_url', {})
        else:
            new_topic = f"{topic_prefix}{selected_item_name}"
        await channel.edit(topic=new_topic)
    # Reply with image model name if enabled in config
    reply = ''
    if config.imgmodels['announce_in_chat']['enabled']:
        reply_prefix = config.imgmodels['announce_in_chat']['reply_prefix']
        if not config.imgmodels['get_imgmodels_via_api']:
            reply = f"{reply_prefix}{selected_item['imgmodel_name']}"
            if config.imgmodels['announce_in_chat']['include_url']:
                reply += " <" + selected_item.get('imgmodel_url', {}) + ">"
            if config.imgmodels['announce_in_chat']['include_params']:
                selected_imgmodel_override_settings_info = ", ".join(
                    f"{key}: {value}" for key, value in selected_imgmodel_override_settings.items())
                selected_imgmodel_payload_info = ", ".join(
                    f"{key}: {value}" for key, value in selected_imgmodel_payload.items())
                reply += f"\n```{selected_imgmodel_override_settings_info}, {selected_imgmodel_payload_info}```"
        else:
            reply = f"{reply_prefix}{selected_item_name}"
    return reply

class ImgModelDropdown(discord.ui.Select):
    def __init__(self, imgmodel_data):
        options = [
            discord.SelectOption(label=imgmodel["model_name"], value=imgmodel["title"])
            for imgmodel in imgmodel_data[:25]
        ]
        super().__init__(placeholder='Choose an Image Model', options=options)
        self.imgmodel_data = imgmodel_data
    async def callback(self, interaction: discord.Interaction):
        try:
            selected_item = self.values[0]
            selected_item_name = None
            for option in self.options:
                if option.value == selected_item:
                    selected_item_name = option.label
                    break
            active_settings = load_yaml_file('ad_discordbot/activesettings.yaml')
            active_settings['imgmodel']['override_settings']['sd_model_checkpoint'] = selected_item
            active_settings['imgmodel']['imgmodel_name'] = selected_item_name
            active_settings['imgmodel']['imgmodel_url'] = ''
            save_yaml_file('ad_discordbot/activesettings.yaml', active_settings)
            # Update size options for /image command
            await update_size_options(active_settings.get('imgmodel').get('payload').get('width'),active_settings.get('imgmodel').get('payload').get('height'))
            # Task to change image models automatically
            await start_update_image_model_task()
            # Set the topic of the channel if enabled in config
            channel = interaction.channel
            reply = await process_imgmodel_announce(channel, selected_item, selected_item_name)
            if reply:
                await interaction.response.send_message(reply)
            else:
                await interaction.response.send_message(f"Updated {self.active_settings_key} settings to: {selected_item_name}")
            print(f"Updated imgmodel settings to: {selected_item_name}")
        except Exception as e:
            print("Error updating image model:", e)
        if config.discord['post_active_settings']['enabled']:
            await post_active_settings()

@client.hybrid_command(description="Choose an imgmodel")
async def imgmodel(i):
    if not config.imgmodels['get_imgmodels_via_api']:
        # Uses ad_discordbot/dict_imgmodel.yaml
        view = discord.ui.View()
        view.add_item(SettingsDropdown('ad_discordbot/dict_imgmodels.yaml', 'imgmodel_name', 'imgmodel'))
        await i.send("Choose an imgmodel:", view=view, ephemeral=True)
    else:
        # Or, fetch imgmodel list from A1111 API
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url=f'{A1111}/sdapi/v1/sd-models') as response:
                    if response.status == 200:
                        imgmodel_data = await response.json()
                        if config.imgmodels['exclude']:
                            exclude_list = config.imgmodels['exclude']
                            imgmodel_data = [
                                imgmodel for imgmodel in imgmodel_data
                                if not exclude_list or not any(re.search(re.escape(exclude_text), imgmodel['title'], re.IGNORECASE) for exclude_text in exclude_list)
                            ]
                        total_models = len(imgmodel_data)
                        if total_models > 25:
                            omitted_models = total_models - 25
                            await i.send(f"Due to Discord limitations, only the first 25 models can be listed ({omitted_models} omitted).", ephemeral=True)
                            imgmodel_data = imgmodel_data[:25]
                        view = discord.ui.View()
                        view.add_item(ImgModelDropdown(imgmodel_data))
                        await i.send('Choose Image Model:', view=view)
                    else:
                        await i.send("Failed to fetch Image Models from the API.", ephemeral=True)
        except Exception as e:
            print(f"Error fetching Image Models: {e}")
            await i.send("An error occurred while fetching Image Models.", ephemeral=True)

class LLMUserInputs():
    # Initialize default state settings
    def __init__(self):
        self.settings = {
        "text": "",
        "state": {
            "history": {'internal': [], 'visible': []},
            "max_new_tokens": 400,
            "max_tokens_second": 0,
            "seed": -1.0,
            "temperature": 0.7,
            "top_p": 0.1,
            "top_k": 40,
            "tfs": 0,
            'top_a': 0,
            "typical_p": 1,
            "epsilon_cutoff": 0,
            "eta_cutoff": 0,
            "repetition_penalty": 1.18,
            "repetition_penalty_range": 0,
            "encoder_repetition_penalty": 1,
            "no_repeat_ngram_size": 0,
            "min_length": 50,
            "do_sample": True,
            "penalty_alpha": 0,
            "num_beams": 1,
            "length_penalty": 1,
            "early_stopping": False,
            "add_bos_token": True,
            "ban_eos_token": False, 
            "skip_special_tokens": True,
            "truncation_length": 2048,
            "custom_stopping_strings": f'"### Assistant","### Human","</END>","{client.user.display_name}"',
            "custom_token_bans": "",
            'auto_max_new_tokens': False,
            "name1": "",
            "name2": client.user.display_name,
            "name1_instruct": "",
            "name2_instruct": client.user.display_name,
            "greeting": "",
            "context": client.llm_context,
            "turn_template": "",
            "chat_prompt_size": 2048,
            "chat_generation_attempts": 1,
            "stop_at_newline": False,
            "mode": "chat",
            "stream": True,
            "mirostat_mode": 0,
            "mirostat_tau": 5.00,
            "mirostat_eta": 0.10,
            'guidance_scale': 1,
            'negative_prompt': ''
            },
        "regenerate": False,
        "_continue": False, 
        #"loading_message" : True
        }
        # Update state settings from active_settings
        state = get_active_setting('llmstate').get('state')
        self.settings['state'].update(state)

class Behavior:
    def __init__(self):
        # Settings for the bot's behavior. Intended to be accessed via a command in the future
        self.learn_about_and_use_guild_emojis = None
        self.take_notes_about_users = None
        self.read_chatlog = None
        # The above are not yet implemented
        self.ignore_history = False
        self.reply_with_image = 0.0
        self.reply_to_itself = 0.0
        self.chance_to_reply_to_other_bots = 0.5
        self.reply_to_bots_when_addressed = 0.3
        self.only_speak_when_spoken_to = True
        self.ignore_parentheses = True
        self.go_wild_in_channel = True
        self.user_conversations = {}
        self.conversation_recency = 600
        self.main_channels = self.initialize_main_channels()
    
    def initialize_main_channels(self):
        # Initialize main_channels from the database
        conn = sqlite3.connect('bot.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS emojis (emoji TEXT UNIQUE, meaning TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS config (setting TEXT UNIQUE, value TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS main_channels (channel_id TEXT UNIQUE)''')
        c.execute('''SELECT channel_id FROM main_channels''')
        result = [int(row[0]) for row in c.fetchall()]
        conn.close()
        return result if result else []

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
        if i.author.bot and client.user.display_name.lower() in text.lower() and i.channel.id in self.main_channels:
            if 'bye' in text.lower(): # don't reply if another bot is saying goodbye
                return False
            return self.probability_to_reply(self.reply_to_bots_when_addressed)
        # Whether to reply when text is nested in parenthesis
        if self.ignore_parentheses and (i.content.startswith('(') and i.content.endswith(')')) or (i.content.startswith('<:') and i.content.endswith(':>')):
            return False
        # Whether to reply if only speak when spoken to
        if (self.only_speak_when_spoken_to and (client.user.mentioned_in(i) or any(word in i.content.lower() for word in client.user.display_name.lower().split()))) or (self.in_active_conversation(i.author.id) and i.channel.id in self.main_channels):
            return True
        reply = False
        # few more conditions
        if i.author.bot and i.channel.id in self.main_channels:
            reply = self.probability_to_reply(self.chance_to_reply_to_other_bots)
        if self.go_wild_in_channel and i.channel.id in self.main_channels:
            reply = True
        if reply:
            self.update_user_dict(i.author.id)
        return reply

    def probability_to_reply(self, probability):
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
