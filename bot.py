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
from discord import app_commands, File
from discord.ext.commands.context import Context
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
from threading import Lock, Thread
from pydub import AudioSegment
import copy
from shutil import copyfile
import sys
import traceback

#################################################################
#################### DISCORD / BOT STARTUP ######################
#################################################################
# Start logger
logging.basicConfig(format='%(levelname)s [%(asctime)s]: %(message)s (Line: %(lineno)d in %(funcName)s, %(filename)s )',
                    datefmt='%Y-%m-%d %H:%M:%S', 
                    level=logging.INFO) # logging.DEBUG)

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler = logging.handlers.RotatingFileHandler(
    filename='discord.log',
    encoding='utf-8',
    maxBytes=32 * 1024 * 1024,  # 32 MiB
    backupCount=5,  # Rotate through 5 files
)

# Resolve legacy config method
class Config:
    def __init__(self):
        self.config = {}

    def legacy_required_values(self):
        from ad_discordbot import config
        required_configs = ['discord', 'sd', 'textgenwebui']
        for config_name in required_configs:
            try:
                config_value = getattr(config, config_name)
                self.config[config_name] = config_value
            except AttributeError:
                logging.warning(f"'config.py' is missing the '{config_name}' configuration.")
            else:
                if not config_value:
                    logging.warning(f"'config.py' has an empty value for '{config_name}' configuration.")
        try:
            self.config['dynamic_prompting'] = config.dynamic_prompting_enabled
        except:
            logging.warning("'config.py' is missing a new parameter 'dynamic_prompting_enabled'. Defaulting to 'True' (enabled) ")
            self.config['dynamic_prompting'] = True

    def init_config(self):
        try:
            config_path = os.path.join('ad_discordbot', 'config.yaml')
            with open(config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            logging.error("Main bot config file 'config.yaml' not found.")
            try:
                self.legacy_required_values()
                logging.info("Found legacy 'config.py'. Please migrate your settings from 'config.py' as it will soon be unsupported.")
            except FileNotFoundError:
                logging.error("Legacy config file 'config.py' not found.")
                sys.exit(2)
        except yaml.YAMLError as e:
            logging.error(f"Error loading 'config.yaml': {e}")
            sys.exit(2)

    def get_config_dict(self):
        return self.config

config = Config()
config.init_config()
config = config.get_config_dict()

# Discord bot token
TOKEN = config['discord'].get('TOKEN', None)

# Intercept custom bot arguments
def parse_bot_args():
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
    return bot_args
    
bot_args = parse_bot_args()

# Check bot token
bot_token = bot_args.token if bot_args.token else TOKEN
if not bot_token:
    logging.error("Discord bot token is required. Please refer to install instructions (https://github.com/altoiddealer/ad_discordbot).")
    sys.exit(2)

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="You have modified the pretrained model configuration to control generation")

# Set discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True  # Enable reaction events
intents.guild_messages = True # Allows updating topic
client = commands.Bot(command_prefix=".", intents=intents)

#################################################################
####################### DISCORD EMBEDS ##########################
#################################################################
def init_embeds(embed_color=0x1e1f22):
    embed_color = config['discord'].get('embed_settings', {}).get('color', 0x1e1f22)
    system_embed_info = None
    img_gen_embed_info = None
    img_send_embed_info = None
    change_embed_info = None
    flow_embed_info = None

    enabled_embeds = config['discord'].get('embed_settings', {}).get('show_embeds', {})

    if enabled_embeds.get('system', True):
        system_embed_info_json = {
            "title": "Welcome to ad_discordbot!",
            "description": """
            **/helpmenu** - Display this message
            **/character** - Change character
            **/main** - Toggle if Bot always replies, per channel
            **/image** - prompt an image to be generated (or try "draw <subject>")
            **/speak** - if TTS settings are enabled, the bot can speak your text
            **__Changing settings__** ('.../ad\_discordbot/dict\_.yaml' files)
            **/imgmodel** - Change Img model and any model-specific settings
            """,
            "url": "https://github.com/altoiddealer/ad_discordbot",
            "color": embed_color
        }
        system_embed_info = discord.Embed().from_dict(system_embed_info_json)

    if enabled_embeds.get('images', True):
        img_gen_embed_info = discord.Embed(title = "Processing image generation ...", description=" ", url='https://github.com/altoiddealer/ad_discordbot', color=embed_color)
        img_send_embed_info = discord.Embed(title= 'User requested an image ...', description=" ", url='https://github.com/altoiddealer/ad_discordbot', color=embed_color)

    if enabled_embeds.get('changes', True):
        change_embed_info = discord.Embed(title = "Changing model ...", description=" ", url='https://github.com/altoiddealer/ad_discordbot', color=embed_color)

    if enabled_embeds.get('flows', True):
        flow_embed_info = discord.Embed(title = 'Processing flow ... ', description=" ", url='https://github.com/altoiddealer/ad_discordbot/wiki/tags', color=embed_color)

    return system_embed_info, img_gen_embed_info, img_send_embed_info, change_embed_info, flow_embed_info

system_embed_info, img_gen_embed_info, img_send_embed_info, change_embed_info, flow_embed_info = init_embeds()


#################################################################
################### Stable Diffusion Startup ####################
#################################################################
sd_enabled = config['sd'].get('enabled', True)

if sd_enabled:
    SD_URL = config['sd'].get('SD_URL', None) # Get the URL from config.yaml
    if SD_URL is None:
        SD_URL = config['sd'].get('A1111', 'http://127.0.0.1:7860')

    async def sd_api(endpoint, method='get', json=None, retry=True):
        try:
            async with aiohttp.ClientSession() as session:
                request_method = getattr(session, method.lower())  
                async with request_method(url=f'{SD_URL}{endpoint}', json=json) as response:
                    if response.status == 200:
                        r = await response.json()
                        return r
                    else:
                        logging.error(f'{SD_URL}{endpoint} response: "{response.status}"')
                        if retry and response.status in [408, 500]:
                            logging.info("Retrying the request in 3 seconds...")
                            await asyncio.sleep(3)
                            return await sd_api(endpoint, method, json, retry=False)
                        
        except aiohttp.client.ClientConnectionError:
            logging.warning(f'Failed to connect to: "{SD_URL}{endpoint}", offline?')
        
        except Exception as e:
            if endpoint == '/sdapi/v1/server-restart' or endpoint == '/sdapi/v1/progress':
                return None
            else:
                logging.error(f'Error getting data from "{SD_URL}{endpoint}": {e}')
                traceback.print_exc()
                return e

    async def get_sd_sysinfo():
        try:
            r = await sd_api(endpoint='/sdapi/v1/cmd-flags', method='get', json=None, retry=False)
            if not r:
                raise Exception('Failed to connect to SD api, make sure to start it or disable the api in your config.yaml')
            
            ui_settings_file = r.get("ui_settings_file", "")
            if "webui-forge" in ui_settings_file:
                return 'SD WebUI Forge'
            elif "webui" in ui_settings_file:
                return 'A1111 SD WebUI'
            else:
                return 'SD WebUI'
        except Exception as e:
            logging.error(f"Error getting SD sysinfo API: {e}")
            return None

    SD_CLIENT = asyncio.run(get_sd_sysinfo()) # Stable Diffusion client name to use in messages, warnings, etc

    # Function to attempt restarting the SD WebUI Client in the event it gets stuck
    @client.hybrid_command(description=f"Immediately Restarts the {SD_CLIENT} server. Requires '--api-server-stop' SD WebUI launch flag.")
    async def restart_sd_client(ctx: discord.ext.commands.Context):
        try:
            system_embed = None
            await ctx.send(f"**`/restart_sd_client` __will not work__ unless {SD_CLIENT} was launched with flag: `--api-server-stop`**", delete_after=10)
            await sd_api(endpoint='/sdapi/v1/server-restart', method='post', json=None, retry=False)
            title = f"{ctx.author} used '/restart_sd_client'. Restarting {SD_CLIENT} ..."
            if system_embed_info:
                system_embed_info.title = title
                system_embed_info.description = f'Attempting to re-establish connection in 5 seconds (Attempt 1 of 10)'
                system_embed = await ctx.send(embed=system_embed_info)
            logging.info(title)
            response = None
            retry = 1
            while response is None and retry < 11:
                if system_embed_info:
                    system_embed_info.description = f'Attempting to re-establish connection in 5 seconds (Attempt {retry} of 10)'
                    if system_embed: system_embed = await system_embed.edit(embed=system_embed_info)
                await asyncio.sleep(5)
                response = await sd_api(endpoint='/sdapi/v1/progress', method='get', json=None, retry=False)             
                retry += 1
            if response:
                title = f"{SD_CLIENT} restarted successfully."
                if system_embed_info:
                    system_embed_info.title = title
                    system_embed_info.description = f"Connection re-established after {retry} out of 10 attempts."
                    if system_embed: system_embed = await system_embed.edit(embed=system_embed_info)
                logging.info(title)
            else:
                title = f"{SD_CLIENT} server unresponsive after Restarting."
                if system_embed_info:
                    system_embed_info.title = title
                    system_embed_info.description = f"Connection was not re-established after 10 attempts."
                    if system_embed: system_embed = await system_embed.edit(embed=system_embed_info)
                logging.error(title)
        except Exception as e:
            logging.error(f"Error resetting the {SD_CLIENT} server: {e}")

    if SD_CLIENT:
        logging.info(f"Initializing with SD WebUI enabled: '{SD_CLIENT}'")
    else:
        sd_enabled = False

#################################################################
##################### TEXTGENWEBUI STARTUP ######################
#################################################################
if not 'textgenwebui' in config:
    logging.warning("'config.py' is missing a new dictionary 'textgenwebui'. Enabling TGWUI by default.")
    textgenwebui_enabled = True
else:
    textgenwebui_enabled = config['textgenwebui'].get('enabled', True)

if textgenwebui_enabled:
    import modules.extensions as extensions_module
    from modules.chat import chatbot_wrapper, load_character, save_history, load_latest_history, find_all_histories
    from modules import shared
    from modules import chat, utils
    from modules.LoRA import add_lora_to_model
    from modules.models import load_model, unload_model
    from modules.models_settings import get_model_metadata, update_model_parameters, get_fallback_settings, infer_loader

## Majority of this code section is copypasta from modules/server.py

def init_textgenwebui_settings():
    # Loading custom settings
    settings_file = None
    # Check if a settings file is provided and exists
    if shared.args.settings is not None and Path(shared.args.settings).exists():
        settings_file = Path(shared.args.settings)
    # Check if settings file exists
    elif Path("settings.json").exists():
        settings_file = Path("settings.json")
    elif Path("settings.yaml").exists():
        settings_file = Path("settings.yaml")
    if settings_file is not None:
        logging.info(f"Loading settings from {settings_file}...")
        file_contents = open(settings_file, 'r', encoding='utf-8').read()
        new_settings = json.loads(file_contents) if settings_file.suffix == "json" else yaml.safe_load(file_contents)
        shared.settings.update(new_settings)

    # Fallback settings for models
    shared.model_config['.*'] = get_fallback_settings()
    shared.model_config.move_to_end('.*', last=False)  # Move to the beginning

# legacy version of load_extensions() which allows extension params to be updated during runtime
def load_extensions(extensions, available_extensions):
    extensions_module.state = {}
    for index, name in enumerate(shared.args.extensions):
        if name in available_extensions:
            if name != 'api':
                logging.info(f'Loading the extension "{name}"')
            try:
                try:
                    exec(f"import extensions.{name}.script")
                except ModuleNotFoundError:
                    logging.error(f"Could not import the requirements for '{name}'. Make sure to install the requirements for the extension.\n\nLinux / Mac:\n\npip install -r extensions/{name}/requirements.txt --upgrade\n\nWindows:\n\npip install -r extensions\\{name}\\requirements.txt --upgrade\n\nIf you used the one-click installer, paste the command above in the terminal window opened after launching the cmd script for your OS.")
                    raise
                extension = getattr(extensions, name).script
                extensions_module.apply_settings(extension, name)
                if hasattr(extension, "setup"):
                    logging.warning(f'Extension "{name}" is hasattr "setup". Skipping...')
                    continue
                extensions_module.state[name] = [True, index]
            except:
                logging.error(f'Failed to load the extension "{name}".')

tts_settings = {}
try:
    tts_settings = config.get('textgenwebui', {}).get('tts_settings', {})
except:
    tts_settings = config['discord'].get('tts_settings', {})

supported_tts_clients = ['alltalk_tts', 'coqui_tts', 'silero_tts', 'elevenlabs_tts']

def init_textgenwebui_extensions():
    # monkey patch load_extensions behavior from pre-commit b3fc2cd
    extensions_module.load_extensions = load_extensions

    shared.args.extensions = []
    extensions_module.available_extensions = utils.get_available_extensions()
    
    # If any TTS extension defined in config.py, set tts bot vars and add extension to shared.args.extensions
    tts_client = tts_settings.get('extension', '') # tts client
    tts_api_key = None
    tts_voice_key = None
    tts_lang_key = None
    if tts_client:
        if tts_client not in supported_tts_clients:
            logging.warning(f'tts client "{tts_client}" is not yet confirmed to be work. The "/speak" command will not be registered. List of supported tts_clients: {supported_tts_clients}')

        tts_api_key = tts_settings.get('api_key', None)
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

    # Activate the extensions
    for extension in shared.settings['default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)

    if shared.args.extensions and len(shared.args.extensions) > 0:
        extensions_module.load_extensions(extensions_module.extensions, extensions_module.available_extensions)

    return tts_client, tts_api_key, tts_voice_key, tts_lang_key

def init_textgenwebui_llmmodels():
    # Model defined through --model
    if shared.args.model is not None:
        shared.model_name = shared.args.model

    # Only one model is available
    elif len(all_llmmodels) == 1:
        shared.model_name = all_llmmodels[0]

    # Select the model from a command-line menu
    elif shared.args.model_menu:
        if len(all_llmmodels) == 0:
            logging.error("No LLM models are available! Please download at least one.")
            sys.exit(0)
        else:
            print('The following LLM models are available:\n')
            for index, model in enumerate(all_llmmodels):
                print(f'{index+1}. {model}')

            print(f'\nWhich one do you want to load? 1-{len(all_llmmodels)}\n')
            i = int(input()) - 1
            print()

        shared.model_name = all_llmmodels[i]

# Check user settings (models/config-user.yaml) to determine loader
def get_llm_model_loader(model):
    loader = None
    user_model_settings = {}
    settings = shared.user_config
    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                user_model_settings[k] = settings[pat][k]
    if 'loader' in user_model_settings:
        loader = user_model_settings['loader']
        return loader
    else:
        loader = infer_loader(model, user_model_settings)
    return loader

instruction_template_str = None

async def load_llm_model(loader=None):
    try:
        # If any model has been selected, load it
        if shared.model_name != 'None':
            p = Path(shared.model_name)
            if p.exists():
                model_name = p.parts[-1]
                shared.model_name = model_name
            else:
                model_name = shared.model_name

            model_settings = get_model_metadata(model_name)

            global instruction_template_str
            instruction_template_str = model_settings.get('instruction_template_str', '')

            update_model_parameters(model_settings, initial=True)  # hijack the command-line arguments
            # Load the model
            loop = asyncio.get_event_loop()
            shared.model, shared.tokenizer = await loop.run_in_executor(None, load_model, model_name, loader)
           # shared.model, shared.tokenizer = load_model(model_name, loader)
            if shared.args.lora:
                add_lora_to_model(shared.args.lora)
    except Exception as e:
        logging.error(f"An error occurred while loading LLM Model: {e}")

if textgenwebui_enabled:
    init_textgenwebui_settings()
    tts_client, tts_api_key, tts_voice_key, tts_lang_key = init_textgenwebui_extensions()
    # Get list of available models
    all_llmmodels = utils.get_available_models()
    init_textgenwebui_llmmodels()
    asyncio.run(load_llm_model())
    shared.generation_lock = Lock()

#################################################################
##################### BACKGROUND QUEUE TASK #####################
#################################################################
bg_task_queue = asyncio.Queue()

async def process_tasks_in_background():
    while True:
        task = await bg_task_queue.get()
        await task

#################################################################
######################## MISC FUNCTIONS #########################
#################################################################
# Function to load .json, .yml or .yaml files
def load_file(file_path):
    try:
        file_suffix = Path(file_path).suffix.lower()

        if file_suffix in [".json", ".yml", ".yaml"]:
            with open(file_path, 'r', encoding='utf-8') as file:
                if file_suffix in [".json"]:
                    data = json.load(file)
                else:
                    data = yaml.safe_load(file)
            return data
        else:
            logging.error(f"Unsupported file format: {file_suffix}")
            return None
    except FileNotFoundError:
        logging.error(f"File not found: {file_suffix}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while reading {file_path}: {str(e)}")
        return None

def merge_base(newsettings, basekey):
    def deep_update(original, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                deep_update(original[key], value)
            else:
                original[key] = value
    try:
        base_settings = load_file('ad_discordbot/dict_base_settings.yaml')
        keys = basekey.split(',')
        current_dict = base_settings
        for key in keys:
            if key in current_dict:
                current_dict = current_dict[key].copy()
            else:
                return None
        deep_update(current_dict, newsettings) # Recursively update the dictionary
        return current_dict
    except Exception as e:
        logging.error(f"Error loading ad_discordbot/dict_base_settings.yaml ({basekey}): {e}")
        return newsettings

# Send message response to user's interaction command
async def ireply(i, process):
    try:
        if task_semaphore.locked(): # If a queued item is currently being processed
            ireply = await i.reply(f'Your {process} request was added to the task queue', ephemeral=True, delete_after=5)
            # del_time = 5
        else:
            ireply = await i.reply(f'Processing your {process} request', ephemeral=True, delete_after=3)
        #     del_time = 1
    except Exception as e:
        logging.error(f"Error sending message response to user's interaction command: {e}")

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

def save_yaml_file(file_path, data):
    try:
        with open(file_path, 'w') as file:
            yaml.dump(data, file, encoding='utf-8', default_flow_style=False, width=float("inf"), sort_keys=False)
    except Exception as e:
        logging.error(f"An error occurred while saving {file_path}: {str(e)}")

async def send_long_message(channel, message_text):
    """ Splits a longer message into parts while preserving sentence boundaries and code blocks """
    activelang = ''

    # Helper function to ensure even pairs of code block markdown
    def ensure_even_code_blocks(chunk_text, code_block_inserted):
        nonlocal activelang  # Declare activelang as nonlocal to modify the global variable
        code_block_languages = ["asciidoc", "autohotkey", "bash", "coffeescript", "cpp", "cs", "css", "diff", "fix", "glsl", "ini", "json", "md", "ml", "prolog", "ps", "py", "tex", "xl", "xml", "yaml", "html"]
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
                if (last_code_block.lower()).startswith(lang):
                    activelang = lang
                    break  # Stop checking if a match is found
            # If there is an odd number of code blocks, add a closing set of "```"
            chunk_text += "```"
            code_block_inserted = True
        return chunk_text, code_block_inserted

    if len(message_text) <= 1980:
        sent_message = await channel.send(message_text)
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
            message_text = message_text[chunk_length:]
            if len(message_text) <= 1980:
                # Send the remaining text as a single chunk if it's shorter than or equal to 2000 characters
                chunk_text, code_block_inserted = ensure_even_code_blocks(message_text, code_block_inserted)
                sent_message = await channel.send(chunk_text)
                break

#################################################################
########################## BOT STARTUP ##########################
#################################################################
## Function to automatically change image models
# Select imgmodel based on mode, while avoid repeating current imgmodel
async def auto_select_imgmodel(current_imgmodel_name, imgmodel_names, mode='random'):   
    try:
        imgmodels = copy.deepcopy(all_imgmodels)
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
                logging.info("The previous imgmodel name was not matched in list of fetched imgmodels, so cannot 'cycle'. New imgmodel was instead picked at random.")
        return selected_imgmodel
    except Exception as e:
        logging.error(f"Error automatically selecting image model: {e}")

# Task to auto-select an imgmodel at user defined interval
async def auto_update_imgmodel_task(mode, duration):
    while True:
        await asyncio.sleep(duration)
        try:
            imgmodels_data = load_file('ad_discordbot/dict_imgmodels.yaml')
            auto_change_settings = imgmodels_data.get('settings', {}).get('auto_change_imgmodels', {})
            channel = auto_change_settings.get('channel_announce', None)
            if channel == 11111111111111111111: channel = None
            active_settings = load_file('ad_discordbot/activesettings.yaml')
            current_imgmodel_name = active_settings.get('imgmodel', {}).get('imgmodel_name', '')
            imgmodel_names = [imgmodel.get('imgmodel_name', '') for imgmodel in all_imgmodels]           
            # Select an imgmodel automatically
            selected_imgmodel = await auto_select_imgmodel(current_imgmodel_name, imgmodel_names, mode)
            if channel: channel = client.get_channel(channel)
            
            async with task_semaphore:
                # offload to ai_gen queue
                params = {'imgmodel': selected_imgmodel}
                await change_imgmodel_task('Automatically', channel, params)
                logging.info("Automatically updated imgmodel settings")
                
        except Exception as e:
            logging.error(f"Error automatically updating image model: {e}")
        #await asyncio.sleep(duration)

imgmodel_update_task = None # Global variable allows process to be cancelled and restarted (reset sleep timer)

if sd_enabled:
    # Register command for helper function to toggle auto-select imgmodel
    @client.hybrid_command(description='Toggles the automatic Img model changing task')
    async def toggle_auto_change_imgmodels(ctx):
        global imgmodel_update_task
        if imgmodel_update_task and not imgmodel_update_task.done():
            imgmodel_update_task.cancel()
            if ctx: await ctx.send("Auto-change Imgmodels task was cancelled.", ephemeral=True, delete_after=5)
            logging.info("Auto-change Imgmodels task was cancelled via '/toggle_auto_change_imgmodels_task'")
        else:
            await bg_task_queue.put(start_auto_change_imgmodels())
            if ctx: await ctx.send(f"Auto-change Img models task was started.", ephemeral=True, delete_after=5)

# helper function to begin auto-select imgmodel task
async def start_auto_change_imgmodels():
    try:
        global imgmodel_update_task
        imgmodels_data = load_file('ad_discordbot/dict_imgmodels.yaml')
        auto_change_settings = imgmodels_data.get('settings', {}).get('auto_change_imgmodels', {})
        mode = auto_change_settings.get('mode', 'random')
        frequency = auto_change_settings.get('frequency', 1.0)
        duration = frequency*3600 # 3600 = 1 hour
        imgmodel_update_task = client.loop.create_task(auto_update_imgmodel_task(mode, duration))
        logging.info(f"Auto-change Imgmodels task was started (Mode: '{mode}', Frequency: {frequency} hours).")
    except Exception as e:
        logging.error(f"Error starting auto-change Img models task: {e}")

# Try getting a valid character file source
def get_character():
    try:
        # This will be either the char name found in activesettings.yaml, or the default char name
        source = bot_settings.settings['llmcontext']['name']
        # If name doesn't match the bot's discord username, try to figure out best char data to initialize with
        if source != bot_settings.database.last_character:
            sources = [
                client.user.display_name, # Try current bot name
                bot_settings.settings['llmcontext']['name'] # Try last known name
            ]
            char_name = None
            for try_source in sources:
                logging.info(f'Trying to load character "{try_source}"...')
                try:
                    _, char_name, _, _, _ = load_character(try_source, '', '')
                    if char_name:
                        logging.info(f'Initializing with character "{try_source}". Use "/character" for changing characters.')       
                        source = try_source                     
                        break  # Character loaded successfully, exit the loop
                except Exception as e:
                    logging.error(f"Error loading character for chat mode: {e}")
            if not char_name:
                logging.error(f"Character not found in '/characters'. Tried files: {sources}")
                return None # return nothing because no character files exist anyway
        # Load character, but don't save it's settings to activesettings (Only user actions will result in modifications)
        return source
    except Exception as e:
        logging.error(f"Error trying to load character data: {e}")
        return None

# If first time bot script is run
async def first_run():
    try:
        for guild in client.guilds: # Iterate over all guilds the bot is a member of
            text_channels = guild.text_channels
            if text_channels and system_embed_info:
                default_channel = text_channels[0]  # Get the first text channel of the guild
                await default_channel.send(embed=system_embed_info)
                break  # Exit the loop after sending the message to the first guild
        logging.info('Welcome to ad_discordbot! Use "/helpmenu" to see main commands. (https://github.com/altoiddealer/ad_discordbot) for more info.')
    except Exception as e:
        if str(e).startswith("403"):
            logging.warning("The bot tried to send a welcome message, but probably does not have access/permissions to your default channel (probably #General)")
        else:
            logging.error(f"An error occurred while welcoming user to the bot: {e}")
    finally:
        with sqlite3.connect('bot.db') as conn:
            c = conn.cursor()
            c.execute('''UPDATE first_run SET is_first_run = ?''', (0,)) # log "0" for false
            conn.commit()
        bot_settings.database = Database()

# Unpack tag presets and add global tag keys
async def update_tags(tags):
    if not isinstance(tags, list):
        logging.warning(f'''One or more "tags" are improperly formatted. Please ensure each tag is formatted as a list item designated with a hyphen (-)''')
        return tags
    try:
        tags_data = load_file('ad_discordbot/dict_tags.yaml')
        global_tag_keys = tags_data.get('global_tag_keys', [])
        tag_presets = tags_data.get('tag_presets', [])
        updated_tags = []
        for tag in tags:
            if 'tag_preset_name' in tag:
                # Find matching tag preset in tag_presets
                for preset in tag_presets:
                    if 'tag_preset_name' in preset and preset['tag_preset_name'] == tag['tag_preset_name']:
                        # Merge corresponding tag presets
                        updated_tags.extend(preset.get('tags', []))
                        tag.pop('tag_preset_name', None)
                        break
            if tag:
                updated_tags.append(tag)
        # Add global tag keys to each tag item
        for tag in updated_tags:
            for key, value in global_tag_keys.items():
                if key not in tag:
                    tag[key] = value
        updated_tags = await expand_triggers(updated_tags) # expand any simplified trigger phrases
        return updated_tags
    except Exception as e:
        logging.error(f"Error loading tag presets: {e}")
        return tags

async def update_bot_base_tags():
    try:
        tags_data = load_file('ad_discordbot/dict_tags.yaml')
        base_tags_data = tags_data.get('base_tags', [])
        base_tags = copy.deepcopy(base_tags_data)
        settings_dict = bot_settings.get_settings_dict()
        settings_dict['tags'] = await update_tags(base_tags)
    except Exception as e:
        logging.error(f"Error updating client base tags: {e}")

# Function to overwrite default settings with activesettings
async def update_bot_settings():
    try:
        defaults = Settings() # Instance of the default settings
        defaults = defaults.settings_to_dict() # Convert instance to dict
        # Current user custom settings
        active_settings = load_file('ad_discordbot/activesettings.yaml')
        active_settings = dict(active_settings)
        # Add any missing required settings
        fixed_settings = fix_dict(active_settings, defaults)
        # Commit fixed settings to bot_settings (always accessible)
        bot_settings.settings = fixed_settings
        # Update client behavior
        bot_settings.behavior = Behavior() # Instance of the default behavior
        behavior = active_settings['behavior']
        bot_settings.behavior.update_behavior_dict(behavior)
        await update_bot_base_tags() # base tags from dict_tags.yaml
    except Exception as e:
        logging.error(f"Error updating client settings: {e}")

#################################################################
########################### ON READY ############################
#################################################################
@client.event
async def on_ready():
    try:
        await update_bot_base_tags()
        # If first time running bot
        if bot_settings.database.first_run:
            await first_run()
        if textgenwebui_enabled:
            source = get_character() # Try loading character data regardless of mode (chat/instruct)
            char_instruct = None
            if source:
                char_instruct, _, _, _ = await character_loader(source)
                bot_settings.database.update_last_setting(source, 'last_character')
            update_instruct = char_instruct or instruction_template_str or None # 'instruction_template_str' is global variable
            if update_instruct:
                settings_dict = bot_settings.get_settings_dict()
                settings_dict['llmstate']['state']['instruction_template_str'] = update_instruct
            # Initialize chat history
            autoload_history = config.get('textgenwebui', {}).get('chat_history', {}).get('autoload_history', False)
            if autoload_history:
                history_manager.load_history(bot_settings.settings['llmstate']['state'])
        # Create background task processing queue
        client.loop.create_task(process_tasks_in_background())
        # Start background task to sync the discord client tree
        await bg_task_queue.put(client.tree.sync())
        # Start background task to to change image models automatically
        if sd_enabled:
            imgmodels_data = load_file('ad_discordbot/dict_imgmodels.yaml')
            if imgmodels_data and imgmodels_data.get('settings', {}).get('auto_change_imgmodels', {}).get('enabled', False):
                await bg_task_queue.put(start_auto_change_imgmodels())
        logging.info("Bot is ready")
    except Exception as e:
        logging.error(f"Error with on_ready: {e}")

#################################################################
####################### DISCORD FEATURES ########################
#################################################################
# Starboard feature
starboard_posted_messages = set()
# Fetch images already starboard'd
data = load_file('ad_discordbot/starboard_messages.yaml')
if data:
    starboard_posted_messages = set(data)

@client.event
async def on_raw_reaction_add(endorsed_img):
    if not config['discord'].get('starboard', {}).get('enabled', False):
        return
    channel = await client.fetch_channel(endorsed_img.channel_id)
    message = await channel.fetch_message(endorsed_img.message_id)
    total_reaction_count = 0
    if config['discord']['starboard'].get('emoji_specific', False):
        for emoji in config['discord']['starboard'].get('react_emojis', []):
            reaction = discord.utils.get(message.reactions, emoji=emoji)
            if reaction:
                total_reaction_count += reaction.count
    else:
        for reaction in message.reactions:
            total_reaction_count += reaction.count
    if total_reaction_count >= config['discord']['starboard'].get('min_reactions', 2):
        
        target_channel_id = config['discord']['starboard'].get('target_channel_id', None)
        if target_channel_id == 11111111111111111111: 
            target_channel_id = None
        
        target_channel = client.get_channel(target_channel_id)
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

# Post settings to a dedicated channel
async def post_active_settings():
    target_channel_id = config['discord']['post_active_settings'].get('target_channel_id', None)
    if target_channel_id == 11111111111111111111: 
        target_channel_id = None
    
    if target_channel_id:
        target_channel = await client.fetch_channel(target_channel_id)
        if target_channel:
            active_settings = load_file('ad_discordbot/activesettings.yaml')
            settings_content = yaml.dump(active_settings, default_flow_style=False)
            
            async for message in target_channel.history(limit=None):
                await message.delete()
                await asyncio.sleep(0.5)  # minimum delay for discord limit
            # Send the entire settings content as a single message
            await send_long_message(target_channel, f"Current settings:\n```yaml\n{settings_content}\n```")
        else:
            logging.error(f"Target channel with ID {target_channel_id} not found.")
    else:
        logging.warning("Channel ID must be specified in config.py")

#################################################################
######################## TTS PROCESSING #########################
#################################################################
voice_client = None

async def voice_channel(vc_setting):
    global voice_client
    # Start voice client if configured, and not explicitly deactivated in character settings
    if voice_client is None and (vc_setting is None or vc_setting) and int(tts_settings.get('play_mode', 0)) != 1:
        try:
            if tts_client and tts_client in shared.args.extensions:
                if tts_settings.get('voice_channel', ''):
                    voice_channel = client.get_channel(tts_settings['voice_channel'])
                    voice_client = await voice_channel.connect()
                else:
                    logging.warning(f'Bot launched with {tts_client}, but no voice channel is specified in config.py')
            else:
                if not bot_settings.database.was_warned('char_tts'):
                    bot_settings.database.update_was_warned('char_tts', 1)
                    logging.warning(f'Character "use_voice_channel" = True, and "voice channel" is specified in config.py, but no "tts_client" is specified in config.py')
        except Exception as e:
            logging.error(f"An error occurred while connecting to voice channel: {e}")
    # Stop voice client if explicitly deactivated in character settings
    if voice_client and voice_client.is_connected():
        try:
            if vc_setting is False:
                logging.info("New context has setting to disconnect from voice channel. Disconnecting...")
                await voice_client.disconnect()
                voice_client = None
        except Exception as e:
            logging.error(f"An error occurred while disconnecting from voice channel: {e}")

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
            logging.warning(f'** No extension params for this character. Reloading extensions with initial values. **')            
        extensions_module.load_extensions(extensions_module.extensions, extensions_module.available_extensions)  # Load Extensions (again)
    except Exception as e:
        logging.error(f"An error occurred while updating character extension settings: {e}")

queued_tts = []  # Keep track of queued tasks

def after_playback(file, error):
    global queued_tts
    if error:
        logging.info(f'Message from audio player: {error}, output: {error.stderr.decode("utf-8")}')
    # Check save mode setting
    if int(tts_settings.get('save_mode', 0)) > 0:
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
        logging.warning("**tts response detected, but bot is not connected to a voice channel.**")
        return
    # Queue the task if audio is already playing
    if voice_client.is_playing():
        queued_tts.append(file)
    else:
        # Otherwise, play immediately
        source = discord.FFmpegPCMAudio(file)
        voice_client.play(source, after=lambda e: after_playback(file, e))

async def upload_tts_file(channel, tts_resp):
    filename = os.path.basename(tts_resp)
    directory = os.path.dirname(tts_resp)
    wav_size = os.path.getsize(tts_resp)
    if wav_size <= 8388608:  # Discord's maximum file size for audio (8 MB)
        with open(tts_resp, "rb") as file:
            tts_file = File(file, filename=filename)
        await channel.send(file=tts_file) # lossless .wav output
    else: # convert to mp3
        bit_rate = int(tts_settings.get('mp3_bit_rate', 128))
        mp3_filename = os.path.splitext(filename)[0] + '.mp3'
        mp3_path = os.path.join(directory, mp3_filename)
        audio = AudioSegment.from_wav(tts_resp)
        audio.export(mp3_path, format="mp3", bitrate=f"{bit_rate}k")
        mp3_size = os.path.getsize(mp3_path) # Check the size of the MP3 file
        if mp3_size <= 8388608:  # Discord's maximum file size for audio (8 MB)
            with open(mp3_path, "rb") as file:
                mp3_file = File(file, filename=mp3_filename)
            await channel.send(file=mp3_file)
        else:
            await channel.send("The audio file exceeds Discord limitation even after conversion.")
        # if save_mode > 0: os.remove(mp3_path) # currently broken

async def process_tts_resp(channel, tts_resp):
    play_mode = int(tts_settings.get('play_mode', 0))
    # Upload to interaction channel
    if play_mode > 0:
        await upload_tts_file(channel, tts_resp)
    # Play in voice channel
    if play_mode != 1:
        await bg_task_queue.put(play_in_voice_channel(tts_resp)) # run task in background

#################################################################
########################### ON MESSAGE ##########################
#################################################################
async def fix_llm_payload(llm_payload):
    # Fix llm_payload by adding any missing required settings
    defaults = Settings() # Create an instance of the default settings
    defaults = defaults.settings_to_dict() # Convert instance to dict
    default_state = defaults['llmstate']['state']
    current_state = llm_payload['state']
    llm_payload['state'] = fix_dict(current_state, default_state)
    return llm_payload

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
        logging.error(f"Error when getting date/time: {e}")
        return '', ''

async def swap_llm_character(char_name, user_name, llm_payload):
    try:
        char_data = await load_character_data(char_name)
        name1 = user_name
        name2 = ''
        if char_data.get('state', {}):
            llm_payload['state'] = char_data['state']
            llm_payload['state']['name1'] = name1
        if char_data.get('name', 'AI'):
            llm_payload['state']['name2'] = char_data['name']
            llm_payload['state']['character_menu'] = char_data['name']
        if char_data.get('context', ''):
            llm_payload['state']['context'] = char_data['context']
        llm_payload = await fix_llm_payload(llm_payload) # Add any missing required information
        return llm_payload
    except Exception as e:
        logging.error(f"An error occurred while loading the file for swap_character: {e}")
        return llm_payload

def format_prompt_with_recent_output(user, prompt):
    try:
        formatted_prompt = copy.copy(prompt)
        # Find all matches of {user_x} and {llm_x} in the prompt
        pattern = r'\{(user|llm|history)_([0-9]+)\}'
        matches = re.findall(pattern, prompt)
        # Iterate through the matches
        for match in matches:
            prefix, index = match
            index = int(index)
            if prefix in ['user', 'llm'] and 0 <= index <= 10:
                message_list = history_manager.recent_messages[prefix]
                if not message_list or index >= len(message_list):
                    continue
                matched_syntax = f"{prefix}_{index}"
                formatted_prompt = formatted_prompt.replace(f"{{{matched_syntax}}}", message_list[index])
            elif prefix == 'history' and 0 <= index <= 10:
                user_message = history_manager.recent_messages['user'][index] if index < len(history_manager.recent_messages['user']) else ''
                llm_message = history_manager.recent_messages['llm'][index] if index < len(history_manager.recent_messages['llm']) else ''
                formatted_history = f'"{user}:" {user_message}\n"{bot_settings.database.last_character}:" {llm_message}\n'
                matched_syntax = f"{prefix}_{index}"
                formatted_prompt = formatted_prompt.replace(f"{{{matched_syntax}}}", formatted_history)
        formatted_prompt = formatted_prompt.replace('{last_image}', '__temp/temp_img_0.png')
        return formatted_prompt
    except Exception as e:
        logging.error(f'An error occurred while formatting prompt with recent messages: {e}')
        return prompt

def process_tag_formatting(user, prompt, formatting):
    try:
        updated_prompt = copy.copy(prompt)
        format_prompt = formatting.get('format_prompt', None)
        time_offset = formatting.get('time_offset', None)
        time_format = formatting.get('time_format', None)
        date_format = formatting.get('date_format', None)
        # Tag handling for prompt formatting
        if format_prompt is not None:
            updated_prompt = format_prompt.replace('{prompt}', updated_prompt)
        # format prompt with any defined recent messages
        updated_prompt = format_prompt_with_recent_output(user, updated_prompt)
        # Format time if defined
        new_time, new_date = get_time(time_offset, time_format, date_format)
        updated_prompt = updated_prompt.replace('{time}', new_time)
        updated_prompt = updated_prompt.replace('{date}', new_date)
        if updated_prompt != prompt:
            logging.info(f'Prompt was formatted: {updated_prompt}')
        return updated_prompt
    except Exception as e:
        logging.error(f"Error formatting LLM prompt: {e}")
        return prompt

async def build_flow_queue(input_flow):   
    try: 
        flow = copy.copy(input_flow)
        total_flows = 0
        flow_base = {}
        for flow_dict in flow: # find and extract any 'flow_base' first
            if 'flow_base' in flow_dict:
                flow_base = flow_dict.get('flow_base')
                flow.remove(flow_dict)
                break
        for step in flow:
            if not step:
                continue
            flow_step = copy.copy(flow_base)
            flow_step.update(step)
            counter = 1
            flow_step_loops = flow_step.pop('flow_step_loops', 0)
            counter += (flow_step_loops - 1) if flow_step_loops else 0
            total_flows += counter
            while counter > 0:
                counter -= 1
                await flow_queue.put(flow_step)
        global flow_event
        flow_event.set() # flag that a flow is being processed. Check with 'if flow_event.is_set():'
    except Exception as e:
        logging.error(f"Error building Flow: {e}")

async def process_llm_payload_tags(user_name, channel, llm_payload, llm_prompt, mods):
    try:
        char_params = {}
        params = {}
        flow = mods.get('flow', None)
        save_to_history = mods.get('save_to_history', None)
        load_history = mods.get('load_histor', None)
        param_variances = mods.get('param_variances', {})
        state = mods.get('state', {})
        change_character = mods.get('change_character', None)
        swap_character = mods.get('swap_character', None)
        change_llmmodel = mods.get('change_llmmodel', None)
        swap_llmmodel = mods.get('swap_llmmodel', None)
        send_user_image = mods.get('send_user_image', None)
        # Process the tag matches
        if flow or save_to_history or load_history or param_variances or state or change_character or swap_character or change_llmmodel or swap_llmmodel or send_user_image:
            # Flow handling
            if flow is not None and not flow_event.is_set():
                await build_flow_queue(flow)
            # History handling
            if save_to_history is not None: llm_payload['save_to_history'] = save_to_history # Save this interaction to history (True/False)
            if load_history is not None:
                if load_history < 0:
                    llm_payload['state']['history'] = {'internal': [], 'visible': []} # No history
                    logging.info("[TAGS] History is being ignored")
                elif load_history > 0:
                    # Calculate the number of items to retain (up to the length of session_history)
                    num_to_retain = min(load_history, len(history_manager.session_history["internal"]))
                    llm_payload['state']['history']['internal'] = history_manager.session_history['internal'][-num_to_retain:]
                    llm_payload['state']['history']['visible'] = history_manager.session_history['visible'][-num_to_retain:]
                    logging.info(f'[TAGS] History is being limited to previous {load_history} exchanges')
            if param_variances:
                processed_params = process_param_variances(param_variances)
                logging.info(f'[TAGS] LLM Param Variances: {processed_params}')
                sum_update_dict(llm_payload['state'], processed_params) # Updates dictionary while adding floats + ints
            if state:
                update_dict(llm_payload['state'], state)
                logging.info(f'[TAGS] LLM State was modified')
            # Character handling
            char_params = change_character or swap_character or {} # 'character_change' will trump 'character_swap'
            if char_params:
                # Error handling
                if not any(char_params == char['name'] for char in all_characters):
                    logging.error(f'Character not found: {char_params}')
                else:
                    if char_params == change_character:
                        verb = 'Changing'
                        char_params = {'character': {'char_name': char_params, 'mode': 'change', 'verb': verb}}
                        await change_char_task(user_name, channel, 'Tags', char_params)
                    else:
                        verb = 'Swapping'
                        llm_payload = await swap_llm_character(swap_character, user_name, llm_payload)
                    logging.info(f'[TAGS] {verb} Character: {char_params}')
            # LLM model handling
            params = change_llmmodel or swap_llmmodel or {} # 'llmmodel_change' will trump 'llmmodel_swap'
            if params:
                if params == shared.model_name:
                    logging.info(f'[TAGS] LLM model was triggered to change, but it is the same as current ("{shared.model_name}").')
                    params = {} # return empty dict
                else:
                    mode = 'change' if params == change_llmmodel else 'swap'
                    verb = 'Changing' if mode == 'change' else 'Swapping'
                    # Error handling
                    if not any(params == model for model in all_llmmodels):
                        logging.error(f'LLM model not found: {params}')
                    else:
                        logging.info(f'[TAGS] {verb} LLM Model: {params}')
                        params = {'llmmodel': {'llmmodel_name': params, 'mode': mode, 'verb': verb}}
            # Send User Image handling
            if send_user_image:
                params['send_user_image'] = send_user_image
                logging.info(f"[TAGS] Sending user image: {send_user_image}")
        return llm_payload, llm_prompt, params
    except Exception as e:
        logging.error(f"Error processing LLM tags: {e}")
        return llm_payload, llm_prompt, {}

def collect_llm_tag_values(tags):
    llm_payload_mods = {
        'flow': None,
        'save_to_history': None,
        'load_history': None,
        'change_character': None,
        'swap_character': None,
        'change_llmmodel': None,
        'swap_llmmodel': None,
        'send_user_image': None,
        'param_variances': {},
        'state': {}
        }
    formatting = {
        'format_prompt': None,
        'time_offset': None,
        'time_format': None,
        'date_format': None
        }
    for tag in tags['matches']:
        # Values that will only apply from the first tag matches
        if 'flow' in tag and llm_payload_mods['flow'] is None:
            llm_payload_mods['flow'] = tag.pop('flow')
        if 'save_history' in tag and llm_payload_mods['save_to_history'] is None:
            llm_payload_mods['save_to_history'] = bool(tag.pop('save_history'))
        if 'load_history' in tag and llm_payload_mods['load_history'] is None:
            llm_payload_mods['load_history'] = int(tag.pop('load_history'))
        if 'change_character' in tag and llm_payload_mods['change_character'] is None:
            llm_payload_mods['change_character'] = str(tag.pop('change_character'))
        if 'swap_character' in tag and llm_payload_mods['swap_character'] is None:
            llm_payload_mods['swap_character'] = str(tag.pop('swap_character'))
        if 'change_llmmodel' in tag and llm_payload_mods['change_llmmodel'] is None:
            llm_payload_mods['change_llmmodel'] = str(tag.pop('change_llmmodel'))
        if 'swap_llmmodel' in tag and llm_payload_mods['swap_llmmodel'] is None:
            llm_payload_mods['swap_llmmodel'] = str(tag.pop('swap_llmmodel'))
        if 'send_user_image' in tag and llm_payload_mods['send_user_image'] is None:
            user_image_file = tag.pop('send_user_image')
            user_image_args = get_image_tag_args('User image', str(user_image_file), key=None, set_dir=None)
            user_image_args.pop('selected_folder')
            llm_payload_mods['send_user_image'] = user_image_args
        if 'format_prompt' in tag and formatting['format_prompt'] is None:
            formatting['format_prompt'] = str(tag.pop('format_prompt'))
        # Values that may apply repeatedly
        if 'time_offset' in tag:
            formatting['time_offset'] = float(tag.pop('time_offset'))
        if 'time_format' in tag:
            formatting['time_format'] = str(tag.pop('time_format'))
        if 'date_format' in tag:
            formatting['date_format'] = str(tag.pop('date_format'))
        if 'llm_param_variances' in tag:
            llm_param_variances = dict(tag.pop('llm_param_variances'))
            try:
                llm_payload_mods['param_variances'].update(llm_param_variances) # Allow multiple to accumulate.
            except:
                logging.warning("Error processing a matched 'llm_param_variances' tag; ensure it is a dictionary.")
        if 'state' in tag:
            state = dict(tag.pop('state'))
            try:
                llm_payload_mods['state'].update(state) # Allow multiple to accumulate.
            except:
                logging.warning("Error processing a matched 'state' tag; ensure it is a dictionary.")
    return llm_payload_mods, formatting

def process_tag_insertions(prompt, tags):
    try:
        # iterate over a copy of the matches, preserving the structure of the original matches list
        tuple_matches = copy.deepcopy(tags['matches'])
        tuple_matches = [item for item in tuple_matches if isinstance(item, tuple)]  # Filter out only tuples
        tuple_matches.sort(key=lambda x: -x[1])  # Sort the tuple matches in reverse order by their second element (start index)
        for item in tuple_matches:
            tag, start, end = item # unpack tuple
            phase = tag.get('phase', 'user')
            if phase == 'llm':
                insert_text = tag.pop('insert_text', None)
                insert_method = tag.pop('insert_text_method', 'after')  # Default to 'after'
                join = tag.pop('text_joining', ' ')
            else:
                insert_text = tag.get('positive_prompt', None)
                insert_method = tag.pop('positive_prompt_method', 'after')  # Default to 'after'
                join = tag.pop('img_text_joining', ' ')
            if insert_text is None:
                logging.error(f"Error processing matched tag {item}. Skipping this tag.")
            else:
                if insert_method == 'replace':
                    if insert_text == '':
                        prompt = prompt[:start] + prompt[end:].lstrip()
                    else:
                        prompt = prompt[:start] + insert_text + prompt[end:]
                elif insert_method == 'after':
                    prompt = prompt[:end] + join + insert_text + prompt[end:]
                elif insert_method == 'before':
                    prompt = prompt[:start] + insert_text + join + prompt[start:]
        # clean up the original matches list
        updated_matches = []
        for item in tags['matches']:
            if isinstance(item, tuple):
                tag, start, end = item
            else:
                tag = item
            phase = tag.get('phase', 'user')
            if phase == 'llm':
                tag.pop('insert_text', None)
                tag.pop('insert_text_method', None)
                tag.pop('text_joining', None)
            else:
                tag.pop('img_text_joining', None)
                tag.pop('positive_prompt_method', None)
            updated_matches.append(tag)
        tags['matches'] = updated_matches
        return prompt, tags
    except Exception as e:
        logging.error(f"Error processing LLM prompt tags: {e}")
        return prompt, tags

def process_tag_trumps(matches, trump_params=[]):
    try:
        # Collect all 'trump' parameters for all matched tags
        trump_params = set(trump_params)
        for tag in matches:
            if isinstance(tag, tuple):
                tag_dict = tag[0]  # get tag value if tuple
            else:
                tag_dict = tag
            if 'trumps' in tag_dict:
                trump_params.update([param.strip().lower() for param in tag_dict['trumps'].split(',')])
                del tag_dict['trumps']
        # Remove duplicates from the trump_params set
        trump_params = set(trump_params)
        # Iterate over all tags in 'matches' and remove 'trumped' tags
        untrumped_matches = []
        for tag in matches:
            if isinstance(tag, tuple):
                tag_dict = tag[0]  # get tag value if tuple
            else:
                tag_dict = tag
            if any(trigger.strip().lower() == trump.strip().lower() for trigger in tag_dict.get('trigger', '').split(',') for trump in trump_params):
                logging.info(f'''[TAGS] Tag with triggers "{tag_dict['trigger']}" was trumped by another tag.''')
            else:
                untrumped_matches.append(tag)
        return untrumped_matches, trump_params
    except Exception as e:
        logging.error(f"Error processing matched tags: {e}")
        return matches  # return original matches if error occurs

def match_tags(search_text, tags, phase='llm'):
    try:
        # Remove 'llm' tags if pre-LLM phase, to be added back to unmatched tags list at the end of function
        if phase == 'llm':
            llm_tags = tags['unmatched'].pop('llm', []) if 'user' in tags['unmatched'] else []
        updated_tags = copy.deepcopy(tags)
        matches = updated_tags['matches']
        unmatched = updated_tags['unmatched']
        for list_name, unmatched_list in tags['unmatched'].items():
            for tag in unmatched_list:
                if 'trigger' not in tag:
                    unmatched[list_name].remove(tag)
                    tag['phase'] = phase
                    matches.append(tag)
                    continue
                case_sensitive = tag.get('case_sensitive', False)
                triggers = [t.strip() for t in tag['trigger'].split(',')]
                for index, trigger in enumerate(triggers):
                    trigger_regex = r'\b{}\b'.format(re.escape(trigger))
                    if case_sensitive:
                        trigger_match = re.search(trigger_regex, search_text)
                    else:
                        trigger_match = re.search(trigger_regex, search_text, flags=re.IGNORECASE)
                    if trigger_match:
                        if not (tag.get('on_prefix_only', False) and trigger_match.start() != 0):
                            unmatched[list_name].remove(tag)
                            tag['phase'] = phase
                            tag['matched_trigger'] = trigger  # retain the matched trigger phrase
                            if (('insert_text' in tag and phase == 'llm') or ('positive_prompt' in tag and phase == 'img')):
                                matches.append((tag, trigger_match.start(), trigger_match.end()))  # Add as a tuple with start/end indexes if inserting text later
                            else:
                                if 'positive_prompt' in tag:
                                    tag['imgtag_matched_early'] = True
                                matches.append(tag)
                            break  # Exit the loop after a match is found
                    else:
                        if ('imgtag_matched_early' in tag) and (index == len(triggers) - 1): # Was previously matched in 'user' text, but not in 'llm' text.
                            tag['imgtag_uninserted'] = True
                            matches.append(tag)
        if matches:
            updated_tags['matches'], updated_tags['trump_params'] = process_tag_trumps(matches, tags['trump_params']) # trump tags
        # Add LLM sublist back to unmatched tags list if LLM phase
        if phase == 'llm':
            unmatched['llm'] = llm_tags
        if 'user' in unmatched:
            del unmatched['user'] # Remove after first phase. Controls the 'llm' tag processing at function start.
        return updated_tags
    except Exception as e:
        logging.error(f"Error matching tags: {e}")
        return tags

def sort_tags(all_tags):
    try:
        sorted_tags = {'matches': [], 'unmatched': {'user': [], 'llm': [], 'userllm': []}, 'trump_params': []}
        for tag in all_tags:
            if 'random' in tag:
                if not isinstance(tag['random'], (int, float)):
                    logging.error("Error: Value for 'random' in tags should be float value (ex: 0.8).")
                    continue # Skip this tag
                if not random.random() < tag['random']:
                    continue # Skip this tag
            search_mode = tag.get('search_mode', 'userllm')  # Default to 'userllm' if 'search_mode' is not present
            if search_mode in sorted_tags['unmatched']:
                sorted_tags['unmatched'][search_mode].append({k: v for k, v in tag.items() if k != 'search_mode'})
            else:
                logging.warning(f"Ignoring unknown search_mode: {search_mode}")
        return sorted_tags
    except Exception as e:
        logging.error(f"Error sorting tags: {e}")
        return all_tags

async def expand_triggers(all_tags):
    try:
        def expand_value(value):
            # Split the value on commas
            parts = value.split(',')
            expanded_values = []
            for part in parts:
                # Check if the part contains curly brackets
                if '{' in part and '}' in part:
                    # Use regular expression to find all curly bracket groups
                    group_matches = re.findall(r'\{([^}]+)\}', part)
                    permutations = list(product(*[group_match.split('|') for group_match in group_matches]))
                    # Replace each curly bracket group with permutations
                    for perm in permutations:
                        expanded_part = part
                        for part_match in group_matches:
                            expanded_part = expanded_part.replace('{' + part_match + '}', perm[group_matches.index(part_match)], 1)
                        expanded_values.append(expanded_part)
                else:
                    expanded_values.append(part)
            return ','.join(expanded_values)
        for tag in all_tags:
            if 'trigger' in tag:
                tag['trigger'] = expand_value(tag['trigger'])
        return all_tags
    except Exception as e:
        logging.error(f"Error expanding tags: {e}")
        return all_tags

# Function to convert string values to bool/int/float
def extract_value(value_str):
    try:
        value_str = value_str.strip()
        if value_str.lower() == 'true':
            return True
        elif value_str.lower() == 'false':
            return False
        elif '.' in value_str:
            try:
                return float(value_str)
            except ValueError:
                return value_str
        else:
            try:
                return int(value_str)
            except ValueError:
                return value_str
    except Exception as e:
        logging.error(f"Error converting string to bool/int/float: {e}")

def parse_tag_from_text_value(value_str):
    try:
        if value_str.startswith('{') and value_str.endswith('}'):
            inner_text = value_str[1:-1]  # Remove outer curly brackets
            key_value_pairs = inner_text.split(',')
            result_dict = {}
            for pair in key_value_pairs:
                key, value = parse_key_pair_from_text(pair)
                result_dict[key] = value
            return result_dict
        elif value_str.startswith('[') and value_str.endswith(']'):
            inner_text = value_str[1:-1]
            result_list = []
            # if list of lists
            if inner_text.startswith('[') and inner_text.endswith(']'):
                sublist_strings = re.findall(r'\[[^\[\]]*\]', inner_text)
                for sublist_string in sublist_strings:
                    sublist_string = sublist_string.strip()
                    sublist_values = parse_tag_from_text_value(sublist_string)
                    result_list.append(sublist_values)
            # if single list
            else:
                list_strings = inner_text.split(',')
                for list_str in list_strings:
                    list_str = list_str.strip()
                    list_value = parse_tag_from_text_value(list_str)
                    result_list.append(list_value)
            return result_list
        else:
            if (value_str.startswith("'") and value_str.endswith("'")):
                return value_str.strip("'")
            elif (value_str.startswith('"') and value_str.endswith('"')):
                return value_str.strip('"')
            else:
                return extract_value(value_str)
    except Exception as e:
        logging.error(f"Error parsing nested value: {e}")

def parse_key_pair_from_text(kv_pair):
    try:
        key_value = kv_pair.split(':')
        key = key_value[0].strip()
        value_str = ':'.join(key_value[1:]).strip()
        value = parse_tag_from_text_value(value_str)
        return key, value
    except Exception as e:
        logging.error(f"Error parsing nested value: {e}")

# Matches [[this:syntax]] and creates 'tags' from matches
# Can handle any structure including dictionaries, lists, even nested sublists.
def get_tags_from_text(text):
    try:
        tags_from_text = []
        pattern = r'\[\[([^\[\]]*?(?:\[\[.*?\]\][^\[\]]*?)*?)\]\]'
        matches = re.findall(pattern, text)
        detagged_text = re.sub(pattern, '', text)
        for match in matches:
            tag_dict = {}
            tag_pairs = match.split('|')
            for pair in tag_pairs:
                key, value = parse_key_pair_from_text(pair)
                tag_dict[key] = value
            tags_from_text.append(tag_dict)
        if tags_from_text:
            logging.info(f"[TAGS] Tags from text: '{tags_from_text}'")
        return detagged_text, tags_from_text
    except Exception as e:
        logging.error(f"Error getting tags from text: {e}")
        return text, []

async def get_tags(text):
    try:
        flow_step_tags = []
        if flow_queue.qsize() > 0:
            flow_step_tags = [await flow_queue.get()]
        base_tags = bot_settings.settings.get('tags', []) # base tags
        imgmodel_tags = bot_settings.settings['imgmodel'].get('tags', []) # imgmodel specific tags
        char_tags = bot_settings.settings['llmcontext'].get('tags', []) # character specific tags
        detagged_text, tags_from_text = get_tags_from_text(text)
        all_tags = tags_from_text + flow_step_tags + char_tags + imgmodel_tags + base_tags  # merge tags to one dictionary
        sorted_tags = sort_tags(all_tags) # sort tags into phases (user / llm / userllm)
        return detagged_text, sorted_tags
    except Exception as e:
        logging.error(f"Error getting tags: {e}")
        return text, []      

async def initialize_llm_payload(user, text):
    llm_payload = copy.deepcopy(bot_settings.settings['llmstate'])
    llm_payload['text'] = text
    name1 = user
    name2 = bot_settings.settings['llmcontext']['name']
    context = bot_settings.settings['llmcontext']['context']
    llm_payload['state']['name1'] = name1
    llm_payload['state']['name2'] = name2
    llm_payload['state']['name1_instruct'] = name1
    llm_payload['state']['name2_instruct'] = name2
    llm_payload['state']['character_menu'] = name2
    llm_payload['state']['context'] = context
    llm_payload['save_to_history'] = True
    llm_payload['state']['history'] = history_manager.session_history
    return llm_payload

def get_wildcard_value(matched_text, dir_path='ad_discordbot/wildcards'):
    selected_option = None
    braces_pat = r'{{([^{}]+?)}}(?=[^\w$:]|$$|$)'   # {{this syntax|separate items can be divided|another item}}
    search_phrase = matched_text[2:] if matched_text.startswith('##') else matched_text
    search_path = f"{search_phrase}.txt"
    # List files in the directory
    txt_files = glob.glob(os.path.join(dir_path, search_path))
    if txt_files:
        selected_file = random.choice(txt_files)
        with open(selected_file, 'r') as file:
            lines = file.readlines()
            filtered_lines = [line.strip() for line in lines if not line.startswith("#")]
            selected_option = random.choice(lines).strip()
    else:
        # If no matching .txt file is found, try to find a subdirectory
        subdirectories = glob.glob(os.path.join(dir_path, search_phrase))
        for subdir in subdirectories:
            if os.path.isdir(subdir):
                subdir_files = glob.glob(os.path.join(subdir, '*.txt'))
                if subdir_files:
                    selected_file = random.choice(subdir_files)
                    with open(selected_file, 'r') as file:
                        lines = file.readlines()
                        filtered_lines = [line.strip() for line in lines if not line.startswith("#")]
                        selected_option = random.choice(lines).strip()
    # Check if selected option has braces pattern
    if selected_option:
        braces_match = re.search(braces_pat, selected_option)
        if braces_match:
            braces_phrase = braces_match.group(1)
            selected_option = get_braces_value(braces_phrase)
        # Check if the selected line contains a nested value
        if selected_option.startswith('__') and selected_option.endswith('__'):
            # Extract nested directory path from the nested value
            nested_dir = selected_option[2:-2]  # Strip the first 2 and last 2 characters
            nested_dir_path = os.path.join(dir_path, nested_dir)  # Use os.path.join for correct path joining
            # Get the last component of the nested directory path
            search_phrase = os.path.split(nested_dir)[-1]
            # Remove the last component from the nested directory path
            nested_dir = os.path.join('ad_discordbot/wildcards', os.path.dirname(nested_dir))
            # Recursively check filenames in the nested directory
            selected_option = get_wildcard_value(search_phrase, nested_dir)
    return selected_option

def process_dynaprompt_options(options):
    weighted_options = []
    total_weight = 0
    for option in options:
        if '::' in option:
            weight, value = option.split('::')
            weight = float(weight)
        else:
            weight = 1.0
            value = option
        total_weight += weight
        weighted_options.append((weight, value))
    # Normalize weights
    normalized_options = [(round(weight / total_weight, 2), value) for weight, value in weighted_options]
    return normalized_options

def choose_dynaprompt_option(options, num_choices=1):
    chosen_values = random.choices(options, weights=[weight for weight, _ in options], k=num_choices)
    return [value for _, value in chosen_values]

def get_braces_value(matched_text):
    wildcard_pat = r'##[\w-]+(?=[^\w-]|$)'  # ##this-syntax represents a wildcard .txt file
    num_choices = 1
    separator = None
    if '$$' in matched_text:
        num_choices_str, options_text = matched_text.split('$$', 1)  # Split by the first occurrence of $$
        if '-' in num_choices_str:
            min_choices, max_choices = num_choices_str.split('-')
            min_choices = int(min_choices)
            max_choices = int(max_choices)
            num_choices = random.randint(min_choices, max_choices)
        else:
            num_choices = int(num_choices_str) if num_choices_str.isdigit() else 1  # Convert to integer if it's a valid number        
        separator_index = options_text.find('$$')
        if separator_index != -1:
            separator = options_text[:separator_index]
            options_text = options_text[separator_index + 2:]     
        options = options_text.split('|')  # Extract options after $$
    else:
        options = matched_text.split('|')
    # Process weighting options
    options = process_dynaprompt_options(options)
    # Choose option(s)
    chosen_options = choose_dynaprompt_option(options, num_choices)
    # Check for selected wildcards
    for index, option in enumerate(chosen_options):
        wildcard_match = re.search(wildcard_pat, option)
        if wildcard_match:
            wildcard_phrase = wildcard_match.group()
            wildcard_value = get_wildcard_value(matched_text=wildcard_phrase, dir_path='ad_discordbot/wildcards')
            if wildcard_value:
                chosen_options[index] = wildcard_value
    chosen_options = [option for option in chosen_options if option is not None]
    if separator:
        replaced_text = separator.join(chosen_options)
    else:
        replaced_text = ', '.join(chosen_options) if num_choices > 1 else chosen_options[0]
    return replaced_text

async def dynamic_prompting(user, text, i=None):
    if not config.get('dynamic_prompting_enabled', True):
        if not bot_settings.database.was_warned('dynaprompt'):
            bot_settings.database.update_was_warned('dynaprompt', 1)
            logging.warning("'config.yaml' is missing a new parameter 'dynamic_prompting_enabled'. Defaulting to 'True' (enabled) ")
    if not dynamic_prompting:
        return text
    # copy text for adding comments
    text_with_comments = text
    # define wildcards directions
    wildcard_dir = 'ad_discordbot/wildcards'
    os.makedirs(wildcard_dir, exist_ok=True)
    # define patterns
    braces_pat = r'{{([^{}]+?)}}(?=[^\w$:]|$$|$)'   # {{this syntax|separate items can be divided|another item}}
    wildcard_pat = r'##[\w-]+(?=[^\w-]|$)'          # ##this-syntax represents a wildcard .txt file
    # Process braces patterns
    braces_start_indexes = []
    braces_matches = re.finditer(braces_pat, text)
    braces_matches = sorted(braces_matches, key=lambda x: -x.start())  # Sort matches in reverse order by their start indices
    for match in braces_matches:
        braces_start_indexes.append(match.start())  # retain all start indexes for updating 'text_with_comments' for wildcard match phase
        matched_text = match.group(1)               # Extract the text inside the braces
        replaced_text = get_braces_value(matched_text)   
        # Replace matched text
        text = text.replace(match.group(0), replaced_text, 1)
        # Update comment
        highlighted_changes = '`' + replaced_text + '`'
        text_with_comments = text_with_comments.replace(match.group(0), highlighted_changes, 1)
    # Process wildcards not in braces
    wildcard_matches = re.finditer(wildcard_pat, text)
    wildcard_matches = sorted(wildcard_matches, key=lambda x: -x.start())  # Sort matches in reverse order by their start indices
    for match in wildcard_matches:
        matched_text = match.group()
        replaced_text = get_wildcard_value(matched_text=matched_text, dir_path=wildcard_dir)
        if replaced_text:
            start, end = match.start(), match.end()
            # Replace matched text
            text = text[:start] + replaced_text + text[end:]
            # Calculate offset based on the number of braces matches with lower start indexes
            offset = sum(1 for idx in braces_start_indexes if idx < start) * 2
            adjusted_start = start + offset
            adjusted_end = end + offset
            highlighted_changes = '`' + replaced_text + '`'
            text_with_comments = (text_with_comments[:adjusted_start] + highlighted_changes + text_with_comments[adjusted_end:])
    # send a message showing the selected options
    if i and (braces_matches or wildcard_matches):
        await i.reply(content=f"__Text with **[Dynamic Prompting](<https://github.com/altoiddealer/ad_discordbot/wiki/dynamic-prompting>)**__:\n>>> **{user}**: {text_with_comments}", mention_author=False, silent=True)
    return text

@client.event
async def on_message(i):
    try:
        text = i.clean_content # primarly converts @mentions to actual user names
        if textgenwebui_enabled and not bot_settings.behavior.bot_should_reply(i, text): return # Check that bot should reply or not
        bot_settings.database.update_last_time('last_user_msg') # Store the current datetime in bot.db
        # if @ mentioning bot, remove the @ mention from user prompt
        if text.startswith(f"@{bot_settings.database.last_character} "):
            text = text.replace(f"@{bot_settings.database.last_character} ", "", 1)
        # apply wildcards
        text = await dynamic_prompting(i.author, text, i)
        
        async with task_semaphore:
            async with i.channel.typing():
                logging.info(f'reply requested: {i.author} said: "{text}"')
                await on_message_task(i.author, i.channel, 'on_message', text, i)
                await run_flow_if_any(i.author, i.channel, 'on_message', text)

    except Exception as e:
        logging.error(f"An error occurred in on_message: {e}")

#################################################################
#################### QUEUED FROM ON MESSAGE #####################
#################################################################
async def on_message_task(user, channel, source, text, i):
    try:
        params = {}
        # collects all tags, sorted into sub-lists by phase (user / llm / userllm)
        text, tags = await get_tags(text)
        # match tags labeled for user / userllm.
        tags = match_tags(text, tags, phase='llm')
        # check if triggered to not respond with text
        should_gen_text = should_bot_do('should_gen_text', default=True, tags=tags)
        if should_gen_text:
            # build llm_payload with defaults
            llm_payload = await initialize_llm_payload(user.name, text)
            # make working copy of user's request (without @ mention)
            llm_prompt = copy.copy(text)
            # apply tags to prompt
            llm_prompt, tags = process_tag_insertions(llm_prompt, tags)
            # collect matched tag values
            llm_payload_mods, formatting = collect_llm_tag_values(tags)
            # apply tags relevant to LLM payload
            llm_payload, llm_prompt, params = await process_llm_payload_tags(user.name, channel, llm_payload, llm_prompt, llm_payload_mods)
            # apply formatting tags to LLM prompt
            llm_prompt = process_tag_formatting(user.name, llm_prompt, formatting)
            # offload to ai_gen queue
            llm_payload['text'] = llm_prompt
            await hybrid_llm_img_gen(user, channel, source, text, tags, llm_payload, params, i)
            return
        should_gen_image = should_bot_do('should_gen_image', default=False, tags=tags)
        if should_gen_image:
            if await sd_online(channel):
                await channel.send(f'Bot was triggered by Tags to not respond with text.\n**Processing image generation using your input as the prompt ...**', delete_after=5) # msg for if LLM model is unloaded
            llm_prompt = copy.copy(text)
            await img_gen_task(user.name, channel, source, llm_prompt, params, i, tags)
    except Exception as e:
        logging.error(f"An error occurred processing on_message request: {e}")

async def hybrid_llm_img_gen(user, channel, source, text, tags, llm_payload, params, i=None):
    try:
        change_embed = None
        img_gen_embed = None
        tts_resp = None
        img_note = ''
        # Check params to see if an LLM model change/swap was triggered by Tags
        llmmodel_params = params.get('llmmodel', {})
        send_user_image = params.get('send_user_image', None)
        mode = llmmodel_params.get('mode', 'change') # default to 'change' unless a tag was triggered with 'swap'
        if llmmodel_params:
            orig_llmmodel = copy.deepcopy(shared.model_name)                    # copy current LLM model name
            change_embed = await change_llmmodel_task(user, channel, params)    # Change LLM model
            if mode == 'swap' and change_embed: await change_embed.delete()                      # Delete embed before the second call
        # make a 'Prompting...' embed when generating text for an image response
        should_gen_image = should_bot_do('should_gen_image', default=False, tags=tags)
        if should_gen_image and textgenwebui_enabled:
            if await sd_online(channel):
                if shared.model_name == 'None':
                    await channel.send('**Processing image generation using message as the image prompt ...**', delete_after=5) # msg for if LLM model is unloaded
                else:
                    if img_gen_embed_info:
                        img_gen_embed_info.title = "Prompting ..."
                        img_gen_embed_info.description = " "
                        img_gen_embed = await channel.send(embed=img_gen_embed_info)
        # if no LLM model is loaded, notify that no text will be generated     
        if textgenwebui_enabled:
            if shared.model_name == 'None':
                if not bot_settings.database.was_warned('no_llmmodel'):
                    bot_settings.database.update_was_warned('no_llmmodel', 1)
                    await channel.send(f'(Cannot process text request: No LLM model is currently loaded. Use "/llmmodel" to load a model.)', delete_after=10)
                    logging.warning(f'Bot tried to generate text for {user}, but no LLM model was loaded')
            # generate text with textgen-webui
            last_resp, tts_resp = await llm_gen(llm_payload)
            # If no text was generated, treat user input at the response
            if last_resp is not None:
                logging.info("reply sent: \"" + user.name + ": {'text': '" + llm_payload["text"] + "', 'response': '" + last_resp + "'}\"")
            else:
                if should_gen_image and sd_enabled: last_resp = copy.copy(text)
                else: return
            # if LLM model swapping was triggered
            if mode == 'swap':
                params['llmmodel']['llmmodel_name'] = orig_llmmodel
                change_embed = await change_llmmodel_task(user, channel, params)    # Swap LLM Model back
                if change_embed: await change_embed.delete()                        # Delete embed again after the second call
        # process image generation (A1111 / Forge)
        if sd_enabled:
            tags = match_img_tags(last_resp, tags)
            if not should_gen_image:
                should_gen_image = should_bot_do('should_gen_image', default=False, tags=tags) # Check again post-LLM
            if should_gen_image:
                if img_gen_embed: await img_gen_embed.delete()
                await img_gen_task(user, channel, source, last_resp, params, i, tags)
        if tts_resp: await process_tts_resp(channel, tts_resp)
        mention_resp = update_mention(user.mention, last_resp) # @mention non-consecutive users
        should_send_text = should_bot_do('should_send_text', default=True, tags=tags)
        if should_send_text:
            await send_long_message(channel, mention_resp)
        if send_user_image:
            send_user_image = discord.File(send_user_image)
            await channel.send(file=send_user_image)
        return
    except Exception as e:
        logging.error(f'An error occurred while processing "{source}" request: {e}')
        if img_gen_embed_info:
            img_gen_embed_info.title = f'An error occurred while processing "{source}" request'
            img_gen_embed_info.description = e
            if img_gen_embed: await img_gen_embed.edit(embed=img_gen_embed_info)
            else: await channel.send(embed=img_gen_embed_info)
        if change_embed: await change_embed.delete()

#################################################################
##################### QUEUED LLM GENERATION #####################
#################################################################
# Add dynamic stopping strings
async def extra_stopping_strings(llm_payload):
    try:
        name1_value = llm_payload['state']['name1']
        name2_value = llm_payload['state']['name2']
        # Check and replace in custom_stopping_strings
        custom_stopping_strings = llm_payload['state']['custom_stopping_strings']
        if "name1" in custom_stopping_strings:
            custom_stopping_strings = custom_stopping_strings.replace("name1", name1_value)
        if "name2" in custom_stopping_strings:
            custom_stopping_strings = custom_stopping_strings.replace("name2", name2_value)
        llm_payload['state']['custom_stopping_strings'] = custom_stopping_strings
        # Check and replace in stopping_strings
        stopping_strings = llm_payload['state']['stopping_strings']
        if "name1" in stopping_strings:
            stopping_strings = stopping_strings.replace("name1", name1_value)
        if "name2" in stopping_strings:
            stopping_strings = stopping_strings.replace("name2", name2_value)
        llm_payload['state']['stopping_strings'] = stopping_strings
        return llm_payload
    except Exception as e:
        logging.error(f'An error occurred while updating stopping strings: {e}')
        return llm_payload
    
# Send LLM Payload - get response
async def llm_gen(llm_payload):
    try:
        if shared.model_name == 'None':
            return None, None
        llm_payload = await extra_stopping_strings(llm_payload)
        loop = asyncio.get_event_loop()

        # Subprocess prevents losing discord heartbeat
        def process_responses():
            last_resp = ''
            tts_resp = ''
            for resp in chatbot_wrapper(text=llm_payload['text'], state=llm_payload['state'], regenerate=llm_payload['regenerate'], _continue=llm_payload['_continue'], loading_message=True, for_ui=False):
                i_resp = resp.get('internal', [])
                if len(i_resp) > 0:
                    last_resp = i_resp[len(i_resp) - 1][1]
                # look for tts response
                vis_resp = resp.get('visible', [])
                if len(vis_resp) > 0:
                    last_vis_resp = vis_resp[-1][-1]
                    if 'audio src=' in last_vis_resp:
                        audio_format_match = re.search(r'audio src="file/(.*?\.(wav|mp3))"', last_vis_resp)
                        if audio_format_match:
                            tts_resp = audio_format_match.group(1)
            return last_resp, tts_resp  # bot's reply

        # Offload the synchronous task to a separate thread using run_in_executor
        last_resp, tts_resp = await loop.run_in_executor(None, process_responses)

        save_to_history = llm_payload.get('save_to_history', True)
        history_manager.manage_history(prompt=llm_payload['text'], reply=last_resp, save_to_history=save_to_history)

        return last_resp, tts_resp
    except Exception as e:
        logging.error(f'An error occurred in llm_gen(): {e}')
        if str(e).startswith('list index out of range'):
            logging.warning(f'Note (this may not be the cause of error): "regen" and "continue" commands only work if bot sent message during current session.')
        traceback.print_exc()
        return None, None

async def cont_regen_task(i, user, text, channel, source, message):
    try:
        cmd = ''
        system_embed = None
        llm_payload = await initialize_llm_payload(user, text)
        llm_payload['save_to_history'] = False
        if source == 'cont':
            cmd = 'Continuing'
            llm_payload['_continue'] = True
        else:
            cmd = 'Regenerating'
            llm_payload['regenerate'] = True
        if shared.model_name == 'None':
            await channel.send('(Cannot process text request: No LLM model is currently loaded. Use "/llmmodel" to load a model.)', delete_after=5)
            logging.warning(f'{user} used {cmd} but no LLM model was loaded')
            return
        if system_embed_info:
            system_embed_info.title = f'{cmd} ... '
            system_embed_info.description = f'{cmd} text for {user}'
            system_embed = await channel.send(embed=system_embed_info)
        last_resp, tts_resp = await llm_gen(llm_payload)
        if system_embed: await system_embed.delete()
        if last_resp is None:
            return
        logging.info("reply sent: \"" + user + ": {'text': '" + llm_payload["text"] + "', 'response': '" + last_resp + "'}\"")
        fetched_message = await channel.fetch_message(message)
        await fetched_message.delete()
        if tts_resp: await process_tts_resp(channel, tts_resp)
        if source == 'regen':
            await i.followup.send('__Regenerated text:__', silent=True)
        await send_long_message(channel, last_resp)
    except Exception as e:
        e_msg = f'An error occurred while processing "{cmd}"'
        logging.error(f'{e_msg}: {e}')
        if str(e).startswith('cannot unpack non-iterable NoneType object'):
            none_msg = f'Error: {cmd} only works on messages sent from the bot during current session.'
            logging.error(none_msg)
            await i.followup.send(none_msg, silent=True)
        else:
            await i.followup.send(e_msg, silent=True)
        if system_embed: await system_embed.delete()

async def speak_task(user, channel, text, params):
    try:
        system_embed = None
        if shared.model_name == 'None':
            await channel.send('Cannot process "/speak" request: No LLM model is currently loaded. Use "/llmmodel" to load a model.)', delete_after=5)
            logging.warning(f'Bot tried to generate tts for {user}, but no LLM model was loaded')
            return
        if system_embed_info:
            system_embed_info.title = f'{user} requested tts ... '
            system_embed_info.description = ''
            system_embed = await channel.send(embed=system_embed_info)
        llm_payload = await initialize_llm_payload(user.name, text)
        llm_payload['_continue'] = True
        llm_payload['state']['max_new_tokens'] = 1
        llm_payload['state']['history'] = {'internal': [[text, text]], 'visible': [[text, text]]}
        llm_payload['save_to_history'] = False
        tts_args = params.get('tts_args', {})
        await update_extensions(tts_args)
        _, tts_resp = await llm_gen(llm_payload)
        if system_embed: await system_embed.delete()
        if tts_resp is None:
            return
        await process_tts_resp(channel, tts_resp)
        # remove api key (don't want to share this to the world!)
        for sub_dict in tts_args.values():
            if 'api_key' in sub_dict:
                sub_dict.pop('api_key')
        if system_embed_info:      
            system_embed_info.title = f'{user.name} requested tts:'
            system_embed_info.description = f"**Params:** {tts_args}\n**Text:** {text}"
            system_embed = await channel.send(embed=system_embed_info)
        await update_extensions(bot_settings.settings['llmcontext'].get('extensions', {})) # Restore character specific extension settings
        if params.get('user_voice'): os.remove(params['user_voice'])
    except Exception as e:
        logging.error(f"An error occurred while generating tts for '/speak': {e}")
        if system_embed_info:
            system_embed_info.title = "An error occurred while generating tts for '/speak'"
            system_embed_info.description = e
            if system_embed: await system_embed.edit(embed=system_embed_info)

#################################################################
###################### QUEUED MODEL CHANGE ######################
#################################################################
# Process selected Img model
async def change_imgmodel_task(user, channel, params):
    try:
        change_embed = None
        await sd_online(channel) # Can't change Img model if not online!
        imgmodel_params = params.get('imgmodel')
        imgmodel_name = imgmodel_params.get('imgmodel_name', '')
        mode = imgmodel_params.get('mode', 'change')    # default to 'change
        verb = imgmodel_params.get('verb', 'Changing')  # default to 'Changing'
        imgmodel = await get_selected_imgmodel_data(imgmodel_name) # params will be checkpoint name
        imgmodel_name = imgmodel.get('imgmodel_name', '')
        # Was not 'None' and did not match any known model names/checkpoints
        if len(imgmodel) < 3:
            if channel and change_embed_info:
                change_embed_info.title = 'Failed to change Img model:'
                change_embed_info.description = f'Img model not found: {imgmodel_name}'
                change_embed = await channel.send(embed=change_embed_info)
            return False
        # if imgmodel_name != 'None': ### IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
        if channel and change_embed_info: # Auto-select imgmodel feature may not have a configured channel
            change_embed_info.title = f'{verb} Img model ... '
            change_embed_info.description = f'{verb} to {imgmodel_name}'
            change_embed = await channel.send(embed=change_embed_info)
        # Merge selected imgmodel/tag data with base settings
        imgmodel, imgmodel_name, imgmodel_tags = await merge_imgmodel_data(imgmodel)
        # Soft Img model update if swapping
        if mode == 'swap' or mode == 'swap_back':
            model_data = imgmodel.get('override_settings') or imgmodel['payload'].get('override_settings')
            _ = await sd_api(endpoint='/sdapi/v1/options', method='post', json=model_data, retry=True)
            if change_embed: await change_embed.delete()
            return True
        # Change Img model settings
        await update_imgmodel(channel, imgmodel, imgmodel_tags)
        # if imgmodel_name != 'None': ### IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
        if channel and change_embed_info:
            if change_embed: await change_embed.delete()
            change_embed_info.title = f"{user} changed Img model:"
            change_embed_info.description = f'**{imgmodel_name}**'
            change_embed = await channel.send(embed=change_embed_info)
        logging.info(f"Image model changed to: {imgmodel_name}")
        if config['discord']['post_active_settings']['enabled']:
            await bg_task_queue.put(post_active_settings())
    except Exception as e:
        logging.error(f"Error changing Img model: {e}")
        traceback.print_exc()
        if change_embed_info:
            change_embed_info.title = "An error occurred while changing Img model"
            change_embed_info.description = e
            if change_embed: await change_embed.edit(embed=change_embed_info)
            else:
                if channel: await channel.send(embed=change_embed_info)
        return False

# Process selected LLM model
async def change_llmmodel_task(user, channel, params):
    try:
        change_embed = None
        llmmodel_params = params.get('llmmodel', {})
        llmmodel_name = llmmodel_params.get('llmmodel_name')
        mode = llmmodel_params.get('mode', 'change')
        verb = llmmodel_params.get('verb', 'Changing')
        # Load the new model if it is different from the current one
        if shared.model_name != llmmodel_name:
            if change_embed_info:
                change_embed_info.title = f'{verb} LLM model ... '
                change_embed_info.description = f"{verb} to {llmmodel_name}"
                change_embed = await channel.send(embed=change_embed_info)
            if shared.model_name != 'None':
                unload_model()                  # If an LLM model is loaded, unload it
            try:
                shared.model_name = llmmodel_name   # set to new LLM model
                if shared.model_name != 'None':
                    bot_settings.database.update_was_warned('no_llmmodel', 0) # Reset warning message
                    loader = get_llm_model_loader(llmmodel_name)    # Try getting loader from user-config.yaml to prevent errors
                    await load_llm_model(loader)                    # Load an LLM model if specified
            except:
                if change_embed_info:
                    change_embed_info.title = "An error occurred while changing LLM Model. No LLM Model is loaded."
                    change_embed_info.description = e
                    if change_embed: await change_embed.delete()
                    await channel.send(embed=change_embed_info)
            if mode == 'swap':
                return change_embed             # return the embed so it can be deleted by the caller
            if change_embed_info:
                if llmmodel_name == 'None':
                    change_embed_info.title = f"{user} unloaded the LLM model"
                    change_embed_info.description = 'Use "/llmmodel" to load a new one'
                else:
                    change_embed_info.title = f"{user} changed LLM model:"
                    change_embed_info.description = f'**{llmmodel_name}**'
                if change_embed: await change_embed.delete()
                await channel.send(embed=change_embed_info)
            logging.info(f"LLM model changed to: {llmmodel_name}")
    except Exception as e:
        logging.error(f"An error occurred while changing LLM Model from '/llmmodel': {e}")
        traceback.print_exc()
        if change_embed_info:
            change_embed_info.title = "An error occurred while changing LLM model"
            change_embed_info.description = e
            if change_embed: await change_embed.delete()
            await channel.send(embed=change_embed_info)

#################################################################
#################### QUEUED CHARACTER CHANGE ####################
#################################################################
async def change_char_task(user, channel, source, params):
    try:
        change_embed = None
        char_params = params.get('character', {})
        char_name = char_params.get('char_name', {})
        verb = char_params.get('verb', 'Changing')
        mode = char_params.get('mode', 'change')
        if change_embed_info:
            change_embed_info.title = f'{verb} character ... '
            change_embed_info.description = f'{user} requested character {mode}: "{char_name}"'
            change_embed = await channel.send(embed=change_embed_info)
        # Change character
        await change_character(channel, source, char_name)
        greeting = bot_settings.settings['llmcontext']['greeting']
        if greeting:
            greeting = greeting.replace('{{user}}', 'user')
            greeting = greeting.replace('{{char}}', char_name)
        else:
            greeting = f'**{char_name}** has entered the chat"'
        if change_embed: await change_embed.delete()
        if change_embed_info:
            change_embed_info.title = f"{user} changed character:"
            change_embed_info.description = f'**{char_name}**'
            await channel.send(embed=change_embed_info)
        await send_long_message(channel, greeting)
        logging.info(f"Character changed to: {char_name}")
    except Exception as e:
        logging.error(f"An error occurred while changing character for /character: {e}")
        if change_embed_info:
            change_embed_info.title = "An error occurred while changing character"
            change_embed_info.description = e
            if change_embed: await change_embed.edit(embed=change_embed_info)

#################################################################
######################## MAIN TASK QUEUE ########################
#################################################################
def should_bot_do(key, default, tags={}):   # Used to check if should:
    try:                                    # - generate text
        if not sd_enabled:
            if key == 'should_gen_image' or key == 'should_send_image':
                return False
        if not textgenwebui_enabled:
            if key == 'should_gen_text' or key == 'should_send_text':
                return False
        matches = tags.get('matches', {})   # - generate image
        if matches:                         # - send text response
            for item in matches:            # - send image response
                if isinstance(item, tuple):
                    tag, start, end = item
                else:
                    tag = item
                if key in tag:
                    return bool(tag.get(key, default))
        return default
    except Exception as e:
        logging.error(f"An error occurred while checking if bot should do '{key}': {e}")
        return default

# For @ mentioning users who were not last replied to
previous_user_id = ''

def update_mention(user_id, last_resp=''):
    global previous_user_id
    mention_resp = copy.copy(last_resp)                    
    if user_id != previous_user_id:
        mention_resp = f"{user_id} {last_resp}"
    previous_user_id = user_id
    return mention_resp

flow_event = asyncio.Event()
flow_queue = asyncio.Queue()

task_semaphore = asyncio.Semaphore(1)

#################################################################
########################## QUEUED FLOW ##########################
#################################################################
async def format_next_flow(next_flow, user, text):
    flow_name = ''
    formatted_flow_tags = {}
    for key, value in next_flow.items():
        # see if any tag values have dynamic formatting (user prompt, LLM reply, etc)
        if isinstance(value, str):
            formatted_value = format_prompt_with_recent_output(user, value)       # output will be a string
            if formatted_value != value:                                        # if the value changed,
                formatted_value = parse_tag_from_text_value(formatted_value)    # convert new string to correct value type
            formatted_flow_tags[key] = formatted_value
        # get name for message embed
        if key == 'flow_step':
            flow_name = f": {value}"
        # format prompt before feeding it back into on_message_task()
        elif key == 'format_prompt':
            formatting = {'format_prompt': value}
            text = process_tag_formatting(user, text, formatting)
        # apply wildcards
        text = await dynamic_prompting(user, text, i=None)
    next_flow.update(formatted_flow_tags) # commit updates
    return flow_name, text

# function to get a copy of the next queue item while maintaining the original queue
async def peek_flow_queue(queue, user, text):
    temp_queue = asyncio.Queue()
    total_queue_size = queue.qsize()
    first_flow = None
    while queue.qsize() > 0:
        if queue.qsize() == total_queue_size:
            item = await queue.get()
            flow_name, formatted_text = await format_next_flow(item, user, text)
        else:
            item = await queue.get()
        await temp_queue.put(item)
    # Enqueue the items back into the original queue
    while temp_queue.qsize() > 0:
        item_to_put_back = await temp_queue.get()
        await queue.put(item_to_put_back)
    return flow_name, formatted_text

async def flow_task(user, channel, source, text):
    try:
        global flow_event
        flow_embed = None
        total_flow_steps = flow_queue.qsize()
        if flow_embed_info:
            flow_embed_info.title = f'Processing Flow for {user} with {total_flow_steps} steps'
            flow_embed_info.description = ''
            flow_embed = await channel.send(embed=flow_embed_info)
        while flow_queue.qsize() > 0:   # flow_queue items are removed in get_tags()
            flow_name, text = await peek_flow_queue(flow_queue, user, text)
            remaining_flow_steps = flow_queue.qsize()
            if flow_embed_info:
                flow_embed_info.description = flow_embed_info.description.replace("**Processing", ":white_check_mark: **")
                flow_embed_info.description += f'**Processing Step {total_flow_steps + 1 - remaining_flow_steps}/{total_flow_steps}**{flow_name}\n'
                if flow_embed: await flow_embed.edit(embed=flow_embed_info)
            await on_message_task(user, channel, source, text, i=None)
        if flow_embed_info:
            flow_embed_info.title = f"Flow completed for {user}"
            flow_embed_info.description = flow_embed_info.description.replace("**Processing", ":white_check_mark: **")
            if flow_embed: await flow_embed.edit(embed=flow_embed_info)
        flow_event.clear()              # flag that flow is no longer processing
        flow_queue.task_done()          # flow queue task is complete      
    except Exception as e:
        logging.error(f"An error occurred while processing a Flow: {e}")
        if flow_embed_info:
            flow_embed_info.title = "An error occurred while processing a Flow"
            flow_embed_info.description = e
            if flow_embed: await flow_embed.edit(embed=flow_embed_info)
            else: await channel.send(embed=flow_embed_info)
        flow_event.clear()
        flow_queue.task_done()
        
        
async def run_flow_if_any(user, channel, source, text):
    if flow_queue.qsize() > 0:
        # flows are activated in process_llm_payload_tags(), and is where the flow queue is populated
        await flow_task(user, channel, source, text)

#################################################################
#################### QUEUED IMAGE GENERATION ####################
#################################################################
async def sd_online(channel):
    try:
        r = requests.get(f'{SD_URL}/')
        status = r.raise_for_status()
        #logging.info(status)
        return True
    except Exception as exc:
        logging.warning(exc)
        if channel and system_embed_info:
            system_embed_info.title = f"{SD_CLIENT} api is not running at {SD_URL}"
            system_embed_info.description = f"Launch {SD_CLIENT} with `--api --listen` commandline arguments\nRead more [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)"
            await channel.send(embed=system_embed_info)        
        return False

async def sd_progress_warning(img_gen_embed):
    logging.error('Reached maximum retry limit')
    if img_gen_embed:
        img_gen_embed_info.title = f'Error getting progress response from {SD_CLIENT}.'
        img_gen_embed_info.description = 'Image generation will continue, but progress will not be tracked.'
        await img_gen_embed.edit(embed=img_gen_embed_info)

def progress_bar(value, length=15):
    try:
        filled_length = int(length * value)
        bar = ':black_square_button:' * filled_length + ':black_large_square:' * (length - filled_length)
        return f'{bar}'
    except Exception as e:
        return 0

async def fetch_progress(session):
    try:
        async with session.get(f'{SD_URL}/sdapi/v1/progress') as progress_response:
            return await progress_response.json()
    except aiohttp.ClientError as e:
        logging.warning(f'Failed to fetch progress: {e}')
        return None

async def check_sd_progress(channel, session):
    try:
        img_gen_embed = None
        img_gen_embed_info.title = f'Waiting for {SD_CLIENT} ...'
        img_gen_embed_info.description = ' '
        img_gen_embed = await channel.send(embed=img_gen_embed_info)
        await asyncio.sleep(1)
        retry_count = 0
        while retry_count < 5:
            progress_data = await fetch_progress(session)
            if progress_data and progress_data['progress'] != 0:
                break
            logging.warning(f'Waiting for progress response from {SD_CLIENT}, retrying in 1 second (attempt {retry_count + 1}/5)')
            await asyncio.sleep(1)
            retry_count += 1
        else:
            await sd_progress_warning(img_gen_embed)
            return
        retry_count = 0   
        while progress_data['state']['job_count'] > 0:
            progress_data = await fetch_progress(session)
            if progress_data:
                if retry_count < 5:
                    progress = progress_data['progress'] * 100
                    eta = progress_data['eta_relative']
                    if eta == 0:
                        img_gen_embed_info.title = f'Generating image: 100%'
                        img_gen_embed_info.description = f'{progress_bar(1)}'
                    else:
                        img_gen_embed_info.title = f'Generating image: {progress:.0f}%'
                        img_gen_embed_info.description = f"{progress_bar(progress_data['progress'])}"
                    if img_gen_embed: await img_gen_embed.edit(embed=img_gen_embed_info)
                    await asyncio.sleep(1)
                else:
                    logging.warning(f'Connection closed with {SD_CLIENT}, retrying in 1 second (attempt {retry_count + 1}/5)')
                    await asyncio.sleep(1)
                    retry_count += 1
            else:
                await sd_progress_warning(img_gen_embed)
                return
        if img_gen_embed: await img_gen_embed.delete()
    except Exception as e:
        logging.error(f'Error tracking {SD_CLIENT} image generation progress: {e}')

async def track_progress(channel):
    if img_gen_embed_info:
        async with aiohttp.ClientSession() as session:
            await check_sd_progress(channel, session)

async def layerdiffuse_hack(temp_dir, img_payload, images, pnginfo):
    try:
        ld_output = None
        for i, image in enumerate(images):
            if image.mode == 'RGBA':
                if i == 0:
                    return images
                ld_output = images.pop(i)
                break
        if ld_output is None:
            logging.warning("Failed to find layerdiffuse output image")
            return images
        # Workaround for layerdiffuse PNG infoReActor + layerdiffuse combination
        reactor = img_payload['alwayson_scripts'].get('reactor', {})
        if reactor and reactor['args'][1]:          # if ReActor was enabled:
            _, _, _, alpha = ld_output.split()      # Extract alpha channel from layerdiffuse output
            img0 = Image.open(f'{temp_dir}/temp_img_0.png') # Open first image (with ReActor output)
            img0 = img0.convert('RGBA')             # Convert it to RGBA 
            img0.putalpha(alpha)                    # apply alpha from layerdiffuse output
        else:                           # if ReActor was not enabled:
            img0 = ld_output            # Just replace first image with layerdiffuse output
        img0.save(f'{temp_dir}/temp_img_0.png', pnginfo=pnginfo) # Save the local image with correct pnginfo
        images[0] = img0 # Update images list
        return images
    except Exception as e:
        logging.error(f'Error processing layerdiffuse images: {e}')

async def apply_reactor_mask(temp_dir, images, pnginfo, reactor_mask):
    try:
        reactor_mask = Image.open(io.BytesIO(base64.b64decode(reactor_mask))).convert('L')
        orig_image = images[0]                                          # Open original image
        face_image = images.pop(1)                                      # Open image with faceswap applied
        face_image.putalpha(reactor_mask)                               # Apply reactor mask as alpha to faceswap image
        orig_image.paste(face_image, (0, 0), face_image)                # Paste the masked faceswap image onto the original
        orig_image.save(f'{temp_dir}/temp_img_0.png', pnginfo=pnginfo)  # Save the image with correct pnginfo
        images[0] = orig_image                                          # Replace first image in images list
        return images
    except Exception as e:
        logging.error(f'Error masking ReActor output images: {e}')

async def save_images_and_return(temp_dir, img_payload, endpoint):
    images = []
    pnginfo = None
    # save .json for debugging
    # with open("img_payload.json", "w") as file:
    #     json.dump(img_payload, file)
    try:
        r = await sd_api(endpoint=endpoint, method='post', json=img_payload, retry=True)
        if not isinstance(r, dict):
            return [], r
        for i, img_data in enumerate(r.get('images')):
            image = Image.open(io.BytesIO(base64.b64decode(img_data.split(",", 1)[0])))
            png_payload = {"image": "data:image/png;base64," + img_data}
            r2 = await sd_api(endpoint='/sdapi/v1/png-info', method='post', json=png_payload, retry=True) 
            if not isinstance(r2, dict):
                return [], r2
            png_info_data = r2.get("info")
            if i == 0:  # Only capture pnginfo from the first png_img_data
                pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text("parameters", png_info_data)
            image.save(f'{temp_dir}/temp_img_{i}.png', pnginfo=pnginfo) # save image to temp directory
            images.append(image) # collect a list of PIL images
    except Exception as e:
        logging.error(f'Error processing images: {e}')
        traceback.print_exc()
        return [], e
    return images, pnginfo

async def sd_img_gen(channel, temp_dir, img_payload, endpoint):
    try:
        reactor_args = img_payload.get('alwayson_scripts', {}).get('reactor', {}).get('args', [])
        last_item = reactor_args[-1] if reactor_args else None
        reactor_mask = reactor_args.pop() if isinstance(last_item, dict) else None
        #Start progress task and generation task concurrently
        images_task = asyncio.create_task(save_images_and_return(temp_dir, img_payload, endpoint))
        progress_task = asyncio.create_task(track_progress(channel))
        # Wait for both tasks to complete
        await asyncio.gather(images_task, progress_task)
        # Get the list of images and copy of pnginfo after both tasks are done
        images, pnginfo = await images_task
        if not images:
            if img_send_embed_info:
                img_send_embed_info.title = 'Error processing images.'
                img_send_embed_info.description = f'Error: "{str(pnginfo)}"\nIf {SD_CLIENT} remains unresponsive, consider using "/restart_sd_client" command.'
                await channel.send(embed=img_send_embed_info)
            return None
        # Apply ReActor mask
        reactor = img_payload.get('alwayson_scripts', {}).get('reactor', {})
        if len(images) > 1 and reactor and reactor_mask:
            images = await apply_reactor_mask(temp_dir, images, pnginfo, reactor_mask['mask'])
        # Workaround for layerdiffuse output
        layerdiffuse = img_payload.get('alwayson_scripts', {}).get('layerdiffuse', {})
        if len(images) > 1 and layerdiffuse and layerdiffuse['args'][0]:
            images = await layerdiffuse_hack(temp_dir, img_payload, images, pnginfo)
        return images
    except Exception as e:
        logging.error(f'Error processing images in {SD_CLIENT} API module: {e}')
        return []

async def process_image_gen(img_payload, censor_mode, channel, tags, endpoint, sd_output_dir='ad_discordbot/sd_outputs/'):
    try:
        # Ensure the necessary directories exist
        os.makedirs(sd_output_dir, exist_ok=True)
        temp_dir = 'ad_discordbot/user_images/__temp/'
        os.makedirs(temp_dir, exist_ok=True)
        # Generate images, save locally
        images = await sd_img_gen(channel, temp_dir, img_payload, endpoint)
        if not images:
            return
        # Send images to discord
        # If the censor mode is 1 (blur), prefix the image file with "SPOILER_"
        file_prefix = 'temp_img_'
        if censor_mode == 1:
            file_prefix = 'SPOILER_temp_img_'
        image_files = [discord.File(f'{temp_dir}/temp_img_{idx}.png', filename=f'{file_prefix}{idx}.png') for idx in range(len(images))]
        should_send_image = should_bot_do('should_send_image', default=True, tags=tags)
        if should_send_image:
            await channel.send(files=image_files)
        # Save the image at index 0 with the date/time naming convention
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        last_image = f'{sd_output_dir}/{timestamp}.png'
        os.rename(f'{temp_dir}/temp_img_0.png', last_image)
        copyfile(last_image, f'{temp_dir}/temp_img_0.png')
        # Delete temporary image files
        # for tempfile in os.listdir(temp_dir):
        #     os.remove(os.path.join(temp_dir, tempfile))
    except Exception as e:
        logging.error(f"An error occurred when processing image generation: {e}")

def clean_img_payload(img_payload):
    try:
        # Remove duplicate negative prompts
        negative_prompt_list = img_payload.get('negative_prompt', '').split(', ')
        unique_values_set = set()
        unique_values_list = []
        for value in negative_prompt_list:
            if value not in unique_values_set:
                unique_values_set.add(value)
                unique_values_list.append(value)
        processed_negative_prompt = ', '.join(unique_values_list)
        img_payload['negative_prompt'] = processed_negative_prompt
        # Delete unwanted extension keys
        if img_payload.get('alwayson_scripts', {}):
            # Warn Non-Forge:
            if SD_CLIENT != 'SD WebUI Forge':
                if img_payload['alwayson_scripts']['forge_couple']['args'].get('enabled', False):
                    logging.warning(f'forge_couple is not known to be compatible with "{SD_CLIENT}". Not applying forge_couple...')
                    bot_settings.database.update_was_warned('forgecouple', 1)
                if img_payload['alwayson_scripts']['layerdiffuse']['args'].get('enabled', False):
                    logging.warning(f'layerdiffuse is not known to be compatible with "{SD_CLIENT}". Not applying layerdiffuse...')
                    bot_settings.database.update_was_warned('layerdiffuse', 1)
            # Clean ControlNet
            if not config['sd']['extensions'].get('controlnet_enabled', False):
                del img_payload['alwayson_scripts']['controlnet'] # Delete all 'controlnet' keys if disabled by config
            # Clean ReActor
            if not config['sd']['extensions'].get('reactor_enabled', False):
                del img_payload['alwayson_scripts']['reactor'] # Delete all 'reactor' keys if disabled by config
            else:
                img_payload['alwayson_scripts']['reactor']['args'] = list(img_payload['alwayson_scripts']['reactor']['args'].values()) # convert dictionary to list
            # Clean Forge Couple
            if not config['sd']['extensions'].get('forgecouple_enabled', False) or img_payload.get('init_images', False):
                del img_payload['alwayson_scripts']['forge_couple'] # Delete all 'forge_couple' keys if disabled by config
            else:
                img_payload['alwayson_scripts']['forge_couple']['args'] = list(img_payload['alwayson_scripts']['forge_couple']['args'].values()) # convert dictionary to list
                img_payload['alwayson_scripts']['forge couple'] = img_payload['alwayson_scripts'].pop('forge_couple') # Add the required space between "forge" and "couple" ("forge couple")
            # Clean layerdiffuse
            if not config['sd']['extensions'].get('layerdiffuse_enabled', False):
                del img_payload['alwayson_scripts']['layerdiffuse'] # Delete all 'layerdiffuse' keys if disabled by config
            else:
                img_payload['alwayson_scripts']['layerdiffuse']['args'] = list(img_payload['alwayson_scripts']['layerdiffuse']['args'].values()) # convert dictionary to list
        # Workaround for denoising strength bug
        if not img_payload.get('enable_hr', False) and not img_payload.get('init_images', False):
            img_payload['denoising_strength'] = None
        # Delete all empty keys
        keys_to_delete = []
        for key, value in img_payload.items():
            if value == "":
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del img_payload[key]
        return img_payload
    except Exception as e:
        logging.error(f"An error occurred when cleaning img_payload: {e}")
        return img_payload

def apply_loractl(tags):
    try:
        if SD_CLIENT != 'A1111 SD WebUI':
            if not bot_settings.database.was_warned('loractl'):
                bot_settings.database.update_was_warned('loractl', 1)
                logging.warning(f'loractl is not known to be compatible with "{SD_CLIENT}". Not applying loractl...')
            return tags
        scaling_settings = [v for k, v in config['sd'].get('extensions', {}).get('lrctl', {}).items() if 'scaling' in k]
        scaling_settings = scaling_settings if scaling_settings else ['']
        # Flatten the matches dictionary values to get a list of all tags (including those within tuples)
        matched_tags = [tag if isinstance(tag, dict) else tag[0] for tag in tags['matches']]
        # Filter the matched tags to include only those with certain patterns in their text fields
        lora_tags = [tag for tag in matched_tags if any(re.findall(r'<lora:[^:]+:[^>]+>', text) for text in (tag.get('positive_prompt', ''), tag.get('positive_prompt_prefix', ''), tag.get('positive_prompt_suffix', '')))]
        if len(lora_tags) >= config['sd']['extensions']['lrctl']['min_loras']:
            for index, tag in enumerate(lora_tags):
                # Determine the key with a non-empty value among the specified keys
                used_key = next((key for key in ['positive_prompt', 'positive_prompt_prefix', 'positive_prompt_suffix'] if tag.get(key, '')), None)
                if used_key:  # If a key with a non-empty value is found
                    positive_prompt = tag[used_key]
                    lora_matches = re.findall(r'<lora:[^:]+:[^>]+>', positive_prompt)
                    if lora_matches:
                        for lora_match in lora_matches:
                            lora_weight_match = re.search(r'(?<=:)\d+(\.\d+)?', lora_match) # Extract lora weight
                            if lora_weight_match:
                                lora_weight = float(lora_weight_match.group())
                                # Selecting the appropriate scaling based on the index
                                scaling_key = f'lora_{index + 1}_scaling' if index+1 < len(scaling_settings) else 'additional_loras_scaling'
                                scaling_values = config['sd'].get('extensions', {}).get('lrctl', {}).get(scaling_key, '')
                                if scaling_values:
                                    scaling_factors = [round(float(factor.split('@')[0]) * lora_weight, 2) for factor in scaling_values.split(',')]
                                    scaling_steps = [float(step.split('@')[1]) for step in scaling_values.split(',')]
                                    # Construct/apply the calculated lora-weight string
                                    new_lora_weight_str = f'{",".join(f"{factor}@{step}" for factor, step in zip(scaling_factors, scaling_steps))}'
                                    updated_lora_match = lora_match.replace(str(lora_weight), new_lora_weight_str)
                                    new_positive_prompt = positive_prompt.replace(lora_match, updated_lora_match)                                   
                                    # Update the appropriate key in the tag dictionary
                                    tag[used_key] = new_positive_prompt
                                    logging.info(f'''[TAGS] loractl applied: "{lora_match}" > "{updated_lora_match}"''')
        return tags
    except Exception as e:
        logging.error(f"Error processing lrctl: {e}")
        return tags

def apply_imgcmd_params(img_payload, params):
    try:
        size = params.get('size', None) if params else None
        face_swap = params.get('face_swap', None) if params else None
        controlnet = params.get('controlnet', None) if params else None
        img2img = params.get('img2img', {})
        img2img_mask = img2img.get('mask', '')
        if img2img:
            img_payload['init_images'] = [img2img['image']]
            img_payload['denoising_strength'] = img2img['denoising_strength']
        if img2img_mask:
            img_payload['mask'] = img2img_mask
        if size: img_payload.update(size)
        if face_swap:
            img_payload['alwayson_scripts']['reactor']['args']['image'] = face_swap # image in base64 format
            img_payload['alwayson_scripts']['reactor']['args']['enabled'] = True # Enable
        if controlnet: img_payload['alwayson_scripts']['controlnet']['args'][0].update(controlnet)
        return img_payload
    except Exception as e:
        logging.error(f"Error initializing img payload: {e}")
        return img_payload

def process_img_prompt_tags(img_payload, tags):
    try:
        img_prompt, tags = process_tag_insertions(img_payload['prompt'], tags)
        updated_positive_prompt = copy.copy(img_prompt)
        updated_negative_prompt = copy.copy(img_payload['negative_prompt'])
        matches = tags['matches']
        for tag in matches:
            join = tag.get('img_text_joining', ' ')
            if 'imgtag_uninserted' in tag: # was flagged as a trigger match but not inserted
                logging.info(f'''[TAGS] "{tag['matched_trigger']}" not found in the image prompt. Appending rather than inserting.''')
                updated_positive_prompt = updated_positive_prompt + ", " + tag['positive_prompt']
            if 'positive_prompt_prefix' in tag:
                updated_positive_prompt = tag['positive_prompt_prefix'] + join + updated_positive_prompt
            if 'positive_prompt_suffix' in tag:
                updated_positive_prompt = updated_positive_prompt + join + tag['positive_prompt_suffix']
            if 'negative_prompt_prefix' in tag:
                join = join if updated_negative_prompt else ''
                updated_negative_prompt = tag['negative_prompt_prefix'] + join + updated_negative_prompt
            if 'negative_prompt' in tag:
                join = join if updated_negative_prompt else ''
                updated_negative_prompt = updated_negative_prompt + join + tag['negative_prompt']
            if 'negative_prompt_suffix' in tag:
                join = join if updated_negative_prompt else ''
                updated_negative_prompt = updated_negative_prompt + join + tag['negative_prompt_suffix']
        img_payload['prompt'] = updated_positive_prompt
        img_payload['negative_prompt'] = updated_negative_prompt
        return img_payload
    except Exception as e:
        logging.error(f"Error processing Img prompt tags: {e}")
        return img_payload

def random_value_from_range(value_range):
    if isinstance(value_range, (list, tuple)) and len(value_range) == 2:
        start, end = value_range
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            num_digits = max(len(str(start).split('.')[-1]), len(str(end).split('.')[-1]))
            value = random.uniform(start, end) if isinstance(start, float) or isinstance(end, float) else random.randint(start, end)
            value = round(value, num_digits)
            return value
    logging.warning(f'Invalid value range "{value_range}". Defaulting to "0".')
    return 0

def convert_lists_to_tuples(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, list) and len(value) == 2 and all(isinstance(item, (int, float)) for item in value) and not any(isinstance(item, bool) for item in value):
            dictionary[key] = tuple(value)
    return dictionary

def process_param_variances(param_variances):
    try:
        param_variances = convert_lists_to_tuples(param_variances) # Only converts lists containing ints and floats (not strings or bools) 
        processed_params = copy.deepcopy(param_variances)
        for key, value in param_variances.items():
            # unpack dictionaries assuming they contain variances
            if isinstance(value, dict):
                processed_params[key] = process_param_variances(value)
            elif isinstance(value, tuple):
                processed_params[key] = random_value_from_range(value)
            elif isinstance(value, bool):
                processed_params[key] = random.choice([True, False])
            elif isinstance(value, list):
                if all(isinstance(item, str) for item in value):
                    processed_params[key] = random.choice(value)
                elif all(isinstance(item, bool) for item in value):
                    processed_params[key] = random.choice(value)
                else:
                    logging.warning(f'Invalid params "{key}", "{value}" will not be applied.')
                    processed_params.pop(key)  # Remove invalid key
            else:
                logging.warning(f'Invalid params "{key}", "{value}" will not be applied.')
                processed_params.pop(key)  # Remove invalid key
        return processed_params
    except Exception as e:
        logging.error(f"Error processing param variances: {e}")
        return {}

def select_random_image_or_subdir(directory=None, root_dir=None, key=None):
    image_file_path = None
    contents = os.listdir(directory)    # List all files and directories in the given directory
    # Filter files to include only .png and .jpg extensions
    image_files = [f for f in contents if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.png', '.jpg'))]
    # If there are image files, choose one randomly
    if image_files:
        if key is not None:
            for filename in image_files:
                filename_without_extension = os.path.splitext(filename)[0]
                if filename_without_extension.lower() == key.lower():
                    image_file_path = os.path.join(directory, filename)
                    method = 'Random from folder'
                    return image_file_path, method
    # If no image files and root_dir is not None, try again one time using root_dir as the directory
    if root_dir is not None:
        image_file_path, method = select_random_image_or_subdir(directory=root_dir, root_dir=None, key=None)
        method = 'Random from folder'
        return image_file_path, method
    if image_files and not image_file_path:
        random_image = random.choice(image_files)
        image_file_path = os.path.join(directory, random_image)
        method = 'Random from folder'
        return image_file_path, method
    # If no image files, check for subdirectories
    subdirectories = [d for d in contents if os.path.isdir(os.path.join(directory, d))]
    # If there are subdirectories, select one randomly and recursively call select_random_image
    if subdirectories:
        random_subdir = random.choice(subdirectories)
        subdir_path = os.path.join(directory, random_subdir)
        return select_random_image_or_subdir(directory=subdir_path, root_dir=root_dir, key=key)
    # If neither image files nor subdirectories found, return None
    return None, None

def get_image_tag_args(extension, value, key=None, set_dir=None):
    args = {}
    image_file_path = ''
    method = ''
    try:
        home_path = os.path.join('ad_discordbot', 'user_images')
        full_path = os.path.join(home_path, value)
        # If value contains valid image extension
        if any(ext in value for ext in (".txt", ".png", ".jpg")): # extension included in value
            image_file_path = os.path.join(home_path, value)
        # ReActor specific
        elif ".safetensors" in value and extension == 'ReActor Enabled':
            args['image'] = ''
            args['source_type'] = 1
            args['face_model'] = value
            method = 'Face model'
        # If value was a directory to choose random image from
        elif os.path.isdir(full_path):
            cwd_path = os.getcwd()
            if set_dir:
                os_path = set_dir
                root_dir = full_path
            else:
                os_path = os.path.join(cwd_path, full_path)
                root_dir = None
            while True:
                image_file_path, method = select_random_image_or_subdir(directory=os_path, root_dir=root_dir, key=key)
                if image_file_path:
                    break  # Break the loop if an image is found and selected
                else:
                    if not os.listdir(os_path):
                        logging.warning(f'Valid file not found in a "{home_path}" or any subdirectories: "{value}"')
                        break  # Break the loop if no folders or images are found
        # If value does not specify an extension, but is also not a directory
        else:
            found = False
            for ext in (".txt", ".png", ".jpg"):
                temp_path = os.path.join(home_path, value + ext)
                if os.path.exists(temp_path):
                    image_file_path = temp_path
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"File '{value}' not found with supported extensions (.txt, .png, .jpg)")
        if image_file_path and os.path.isfile(image_file_path):
            if extension == "User image":
                return image_file_path # user image does not need to be converted to base64
            if image_file_path.endswith(".txt"):
                with open(image_file_path, "r") as txt_file:
                    base64_img = txt_file.read()
                    method = 'base64 from .txt'
            else:
                with open(image_file_path, "rb") as image_file:
                    image_data = image_file.read()
                    base64_img = base64.b64encode(image_data).decode('utf-8')
                    args['image'] = base64_img
                    if not method: # will already have value if random img picked from dir
                        method = 'Image file'
        if method:
            logging.info(f'[TAGS] {extension}: "{value}" ({method}).')
            if method == 'Random from folder':
                args['selected_folder'] = os.path.dirname(image_file_path)
        return args
    except Exception as e:
        logging.error(f"[TAGS] Error processing {extension} tag: {e}")
        return {}

async def process_img_payload_tags(img_payload, mods, params):
    try:
        imgmodel_params = None
        flow = mods.get('flow', None)
        img_censoring = mods.get('img_censoring', None)
        change_imgmodel = mods.get('change_imgmodel', None)
        swap_imgmodel = mods.get('swap_imgmodel', None)
        payload = mods.get('payload', None)
        param_variances = mods.get('param_variances', {})
        controlnet = mods.get('controlnet', [])
        forge_couple = mods.get('forge_couple', {})
        layerdiffuse = mods.get('layerdiffuse', {})
        reactor = mods.get('reactor', {})
        img2img = mods.get('img2img', {})
        img2img_mask = mods.get('img2img_mask', {})
        send_user_image = mods.get('send_user_image', None)
        endpoint = '/sdapi/v1/txt2img'
        # Process the tag matches
        if flow or img_censoring or change_imgmodel or swap_imgmodel or payload or param_variances or controlnet or forge_couple or layerdiffuse or reactor or img2img or img2img_mask or send_user_image:
            # Flow handling
            if flow is not None and not flow_event.is_set():
                await build_flow_queue(flow)
            # Img censoring handling
            if img_censoring and img_censoring > 0:
                img_payload['img_censoring'] = img_censoring
                logging.info(f"[TAGS] Censoring: {'Image Blurred' if img_censoring == 1 else 'Generation Blocked'}")
            # Imgmodel handling
            imgmodel_params = change_imgmodel or swap_imgmodel or None
            if imgmodel_params:
                    ## IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
                    ## if not change_imgmodel and swap_imgmodel and swap_imgmodel == 'None':
                        # _ = await sd_api(endpoint='/sdapi/v1/unload-checkpoint', method='post', json=None, retry=True)
                # 'change_imgmodel' will trump 'swap_imgmodel'
                current_imgmodel = bot_settings.settings['imgmodel'].get('override_settings', {}).get('sd_model_checkpoint') or bot_settings.settings['imgmodel']['payload'].get('override_settings', {}).get('sd_model_checkpoint') or ''
                if imgmodel_params == current_imgmodel:
                    logging.info(f'[TAGS] Img model was triggered to change, but it is the same as current ("{current_imgmodel}").')
                    imgmodel_params = None # return None
                else:
                    mode = 'change' if imgmodel_params == change_imgmodel else 'swap'
                    verb = 'Changing' if mode == 'change' else 'Swapping'
                    logging.info(f"[TAGS] {verb} Img model: '{imgmodel_params}'")
                    imgmodel_params = {'imgmodel': {'imgmodel_name': imgmodel_params, 'mode': mode, 'verb': verb, 'current_imgmodel': current_imgmodel}} # return dict
            # Payload handling
            if payload:
                if isinstance(payload, dict):
                    logging.info(f"[TAGS] Updated payload: '{payload}'")
                    update_dict(img_payload, payload)
                else:
                    logging.warning("A tag was matched with invalid 'payload'; must be a dictionary.")
            # Param variances handling
            if param_variances:
                processed_params = process_param_variances(param_variances)
                logging.info(f"[TAGS] Applied Param Variances: '{processed_params}'")
                sum_update_dict(img_payload, processed_params)
            # Controlnet handling
            if controlnet and config['sd']['extensions'].get('controlnet_enabled', False):       
                img_payload['alwayson_scripts']['controlnet']['args'] = controlnet
            # forge_couple handling
            if forge_couple and config['sd']['extensions'].get('forgecouple_enabled', False):
                img_payload['alwayson_scripts']['forge_couple']['args'].update(forge_couple)
                img_payload['alwayson_scripts']['forge_couple']['args']['enable'] = True
                logging.info(f"[TAGS] Enabled forge_couple: {forge_couple}")
            # layerdiffuse handling
            if layerdiffuse and config['sd']['extensions'].get('layerdiffuse_enabled', False):
                img_payload['alwayson_scripts']['layerdiffuse']['args'].update(layerdiffuse)
                img_payload['alwayson_scripts']['layerdiffuse']['args']['enabled'] = True
                logging.info(f"[TAGS] Enabled layerdiffuse: {layerdiffuse}")
            # ReActor face swap handling
            if reactor and config['sd']['extensions'].get('reactor_enabled', False):
                img_payload['alwayson_scripts']['reactor']['args'].update(reactor)
                if reactor.get('mask'):
                    img_payload['alwayson_scripts']['reactor']['args']['save_original'] = True
            # Img2Img handling
            if img2img:
                img_payload['init_images'] = [img2img]
                params['endpoint'] = '/sdapi/v1/img2img'
            # Inpaint Mask handling
            if img2img_mask:
                img_payload['mask'] = img2img_mask
            # Send User Image handling
            if send_user_image:
                logging.info(f"[TAGS] Sending user image: {send_user_image}")
        return img_payload, imgmodel_params, params
    except Exception as e:
        logging.error(f"Error processing Img tags: {e}")
        return img_payload, None

# The methods of this function allow multiple extensions with an identical "select image from random folder" value to share the first selected folder.
# The function will first try to find a specific image file based on the extension's key name (ex: 'canny.png' or 'img2img_mask.jpg')
def collect_img_extension_mods(mods):
    controlnet = mods.get('controlnet', [])
    reactor = mods.get('reactor', None)
    img2img = mods.get('img2img', None)
    img2img_mask = mods.get('img2img_mask', None)
    set_dir = None
    if img2img:
        try:
            img2img_args = get_image_tag_args('Img2Img', img2img, key='img2img', set_dir=set_dir)
            mods['img2img'] = img2img_args.get('image', '')
            if img2img_args:
                if set_dir is None:
                    set_dir = img2img_args.get('selected_folder', None)
                if img2img_mask:
                    img2img_mask_args = get_image_tag_args('Img2Img Mask', img2img_mask, key='img2img_mask', set_dir=set_dir)
                    mods['img2img_mask'] = img2img_mask_args.get('image', '')
                    if img2img_mask_args:
                        if set_dir is None:
                            set_dir = img2img_mask_args.get('selected_folder', None)
        except Exception as e:
            logging.error(f"Error collecting img2img tag values: {e}")
    if controlnet:
        try:
            for idx, controlnet_item in enumerate(controlnet):
                control_type = controlnet_item.pop('control_type', None) # remove control_type
                module = controlnet_item.get('module', None)
                prefix = control_type or module or None
                image = controlnet_item.get('image', None)
                mask_image = controlnet_item.get('mask', None) or controlnet_item.get('mask_image', None)
                # Update controlnet item with image information
                if image:
                    cnet_args = get_image_tag_args('ControlNet Image', image, key=prefix, set_dir=set_dir)
                    if not cnet_args:
                        controlnet[idx] = {}
                    else:
                        if set_dir is None:
                            set_dir = cnet_args.pop('selected_folder', None)
                        else:
                            cnet_args.pop('selected_folder')
                        controlnet[idx].update(cnet_args)
                        controlnet[idx]['enabled'] = True
                        # Update controlnet item with mask_image information
                        if mask_image:
                            key = f'{prefix}_mask' if prefix else None
                            cnet_mask_args = get_image_tag_args('ControlNet Mask', mask_image, key=key, set_dir=set_dir)
                            controlnet[idx]['mask_image'] = cnet_mask_args.get('image', None)
                            if cnet_mask_args:
                                if set_dir is None:
                                    set_dir = cnet_mask_args.get('selected_folder', None)
            mods['controlnet'] = controlnet
        except Exception as e:
            logging.error(f"Error collecting ControlNet tag values: {e}")
    if reactor:
        try:
            image = reactor.get('image', None)
            mask_image = reactor.get('mask', None)
            if image:
                reactor_args = get_image_tag_args('ReActor Enabled', image, key='reactor', set_dir=None)
                if reactor_args:
                    reactor_args.pop('selected_folder', None)
                    mods['reactor'].update(reactor_args)
                    mods['reactor']['enabled'] = True
                    if mask_image:
                        reactor_mask_args = get_image_tag_args('ReActor Mask', mask_image, key='reactor_mask', set_dir=set_dir)
                        mods['reactor']['mask'] = reactor_mask_args.get('image', '')
                        if reactor_mask_args and set_dir is None:
                            set_dir = reactor_mask_args.get('selected_folder', None)
        except Exception as e:
            logging.error(f"Error collecting ReActor tag values: {e}")
    return mods

def collect_img_tag_values(tags):
    sd_output_dir = 'ad_discordbot/sd_outputs/'
    img_payload_mods = {
        'flow': None,
        'img_censoring': None,
        'change_imgmodel': None,
        'swap_imgmodel': None,
        'payload': {},
        'param_variances': {},
        'controlnet': [],
        'forge_couple': {},
        'layerdiffuse': {},
        'reactor': {},
        'send_user_image': None,
        'img2img': {},
        'img2img_mask': {}
        }
    payload_order_hack = {}
    controlnet_args = {}
    forge_couple_args = {}
    layerdiffuse_args = {}
    reactor_args = {}
    try:
        for tag in tags['matches']:
            if isinstance(tag, tuple):
                tag = tag[0] # For tags with prompt insertion indexes
            for key, value in tag.items():
                if key == 'sd_output_dir':
                    sd_output_dir = str(value)
                elif key == 'flow' and img_payload_mods['flow'] is None:
                    img_payload_mods['flow'] = dict(value)
                elif key == 'img_censoring' and img_payload_mods['img_censoring'] is None:
                    img_payload_mods['img_censoring'] = int(value)
                elif key == 'change_imgmodel' and img_payload_mods['change_imgmodel'] is None:
                    img_payload_mods['change_imgmodel'] = str(value)
                elif key == 'swap_imgmodel' and img_payload_mods['swap_imgmodel'] is None:
                    img_payload_mods['swap_imgmodel'] = str(value)
                elif key == 'payload': # Allow multiple to accumulate
                    try:
                        if img_payload_mods['payload']:
                            payload_order_hack = dict(value)
                            update_dict(payload_order_hack, img_payload_mods['payload'])
                            img_payload_mods['payload'] = payload_order_hack                           
                        else:
                            img_payload_mods['payload'] = dict(value)
                    except:
                        logging.warning("Error processing a matched 'payload' tag; ensure it is a dictionary.")
                elif key == 'img_param_variances': # Allow multiple to accumulate
                    try:
                        update_dict(img_payload_mods['param_variances'], dict(value))
                    except:
                        logging.warning("Error processing a matched 'img_param_variances' tag; ensure it is a dictionary.")
                # get layerdiffuse tag params                    
                elif key == 'layerdiffuse':
                    img_payload_mods['layerdiffuse']['method'] = str(value)
                elif key.startswith('laydiff_'):
                    laydiff_key = key[len('laydiff_'):]
                    layerdiffuse_args[laydiff_key] = value
                # get any user image
                elif key == 'send_user_image':
                    user_image_args = get_image_tag_args('User image', str(value), key=None, set_dir=None)
                    user_image_args.pop('selected_folder')
                    img_payload_mods['send_user_image'] = user_image_args
                # get reactor tag params
                elif key == 'reactor':
                    img_payload_mods['reactor']['image'] = value
                elif key.startswith('reactor_'):
                    reactor_key = key[len('reactor_'):]
                    reactor_args[reactor_key] = value
                # get forge_couple tag params                    
                elif key == 'forge_couple':
                    if value.startswith('['):
                        img_payload_mods['forge_couple']['maps'] = list(value)
                    else: img_payload_mods['forge_couple']['direction'] = str(value)
                elif key.startswith('couple_'):
                    forge_couple_key = key[len('couple_'):]
                    if value.startswith('['):
                        forge_couple_args[forge_couple_key] = list(value)
                    else:
                        forge_couple_args[forge_couple_key] = str(value)
                # get controlnet tag params
                elif key.startswith('controlnet'):
                    index = int(key[len('controlnet'):]) if key != 'controlnet' else 0  # Determine the index (cnet unit) for main controlnet args
                    controlnet_args.setdefault(index, {}).update({'image': value, 'enabled': True})         # Update controlnet args at the specified index
                elif key.startswith('cnet'):
                    # Determine the index for controlnet_args sublist
                    if key.startswith('cnet_'):
                        index = int(key.split('_')[0][len('cnet'):]) if not key.startswith('cnet_') else 0  # Determine the index (cnet unit) for additional controlnet args
                    controlnet_args.setdefault(index, {}).update({key.split('_', 1)[-1]: value})   # Update controlnet args at the specified index
                # get any img2img
                elif key == 'img2img':
                    img_payload_mods['img2img'] = str(value) # get base64 in next function
                # get any inpaint mask
                elif key == 'img2img_mask':
                    img_payload_mods['img2img_mask'] = str(value) # get base64 in next function
        # Add the collected SD WebUI extension args to the img_payload_mods dict
        for index in sorted(set(controlnet_args.keys())):   # This flattens down any gaps between collected ControlNet units (ensures lowest index is 0, next is 1, and so on)
            cnet_basesettings = copy.copy(bot_settings.settings['imgmodel']['payload']['alwayson_scripts']['controlnet']['args'][0])  # Copy of required dict items
            cnet_unit_args = controlnet_args.get(index, {})
            cnet_unit = update_dict(cnet_basesettings, cnet_unit_args)
            img_payload_mods['controlnet'].append(cnet_unit)
        img_payload_mods['forge_couple'].update(forge_couple_args)
        img_payload_mods['layerdiffuse'].update(layerdiffuse_args)
        img_payload_mods['reactor'].update(reactor_args)

        img_payload_mods = collect_img_extension_mods(img_payload_mods)
        return sd_output_dir, img_payload_mods
    except Exception as e:
        logging.error(f"Error collecting Img tag values: {e}")
        return sd_output_dir, tags

def initialize_img_payload(img_prompt, neg_prompt):
    try:
        # Initialize img_payload settings
        img_payload = {"prompt": img_prompt, "negative_prompt": neg_prompt, "width": 512, "height": 512, "steps": 20, "resize_mode": 1}
        # Apply settings from imgmodel configuration
        imgmodel_img_payload = copy.deepcopy(bot_settings.settings['imgmodel'].get('payload', {}))
        img_payload.update(imgmodel_img_payload)
        img_payload['override_settings'] = copy.deepcopy(bot_settings.settings['imgmodel'].get('override_settings', {}))
        return img_payload
    except Exception as e:
        logging.error(f"Error initializing img payload: {e}")

def match_img_tags(img_prompt, tags):
    try:
        # Unmatch any previously matched tags which try to insert text into the img_prompt
        for tag in tags['matches'][:]:  # Iterate over a copy of the list
            if tag.get('imgtag_matched_early'): # extract text insertion key pairs from previously matched tags
                new_tag = {}
                tag_copy = copy.copy(tag)
                for key, value in tag_copy.items(): # Iterate over a copy of the tag
                    if (key in ["trigger", "matched_trigger", "imgtag_matched_early", "case_sensitive", "on_prefix_only", "search_mode", "img_text_joining", "phase"]
                        or key.startswith(('positive_prompt', 'negative_prompt'))):
                        new_tag[key] = value
                        if not key == 'phase':
                            del tag[key] # Remove the key from the original tag
                tags['unmatched']['userllm'].append(new_tag) # append to unmatched list
                # Remove tag items from original list that became an empty list
                if not tag:
                    tags['matches'].remove(tag)
        # match tags for 'img' phase.
        tags = match_tags(img_prompt, tags, phase='img')
        # Rematch any previously matched tags that failed to match text in img_prompt
        for tag in tags['unmatched']['userllm'][:]:  # Iterate over a copy of the list
            if tag.get('imgtag_matched_early') and tag.get('imgtag_uninserted'):
                tags['matches'].append(tag)
                tags['unmatched']['userllm'].remove(tag)
        return tags
    except Exception as e:
        logging.error(f"Error matching tags for img phase: {e}")
        return tags

async def img_gen_task(user, channel, source, img_prompt, params, i=None, tags={}):
    try:
        check_key = bot_settings.settings['imgmodel'].get('override_settings') or bot_settings.settings['imgmodel']['payload'].get('override_settings')
        if check_key.get('sd_model_checkpoint') == 'None': # Model currently unloaded
            await channel.send("**Cannot process image request:** No Img model is currently loaded")
            logging.warning(f'Bot tried to generate image for {user}, but no Img model was loaded')
        if not tags:
            img_prompt, tags = await get_tags(img_prompt)
            tags = match_img_tags(img_prompt, tags)
        # Initialize img_payload
        neg_prompt = params.get('neg_prompt', '')
        img_payload = initialize_img_payload(img_prompt, neg_prompt)
        # collect matched tag values
        sd_output_dir, img_payload_mods = collect_img_tag_values(tags)
        send_user_image = img_payload_mods.get('send_user_image', None)
        # Apply tags relevant to Img gen
        img_payload, imgmodel_params, params = await process_img_payload_tags(img_payload, img_payload_mods, params)
        # Check censoring
        censor_mode = None
        if img_payload.get('img_censoring', 0) > 0:
            censor_mode = img_payload['img_censoring']
            if censor_mode == 2:
                if img_send_embed_info:
                    img_send_embed_info.title = "Image prompt was flagged as inappropriate."
                    img_send_embed_info.description = ""
                    await channel.send(embed=img_send_embed_info)
                return censor_mode
        # Process loractl
        if config['sd']['extensions'].get('lrctl', {}).get('enabled', False):
            tags = apply_loractl(tags)
        # Apply tags relevant to Img prompts
        img_payload = process_img_prompt_tags(img_payload, tags)
        # Apply menu selections from /image command
        img_payload = apply_imgcmd_params(img_payload, params)
        # Clean anything up that gets messy
        img_payload = clean_img_payload(img_payload)
        # Change imgmodel if triggered by tags
        should_swap = False
        if imgmodel_params:
            img_payload['override_settings']['sd_model_checkpoint'] = imgmodel_params['imgmodel'].get('imgmodel_name')
            current_imgmodel = imgmodel_params['imgmodel'].get('current_imgmodel', '')
            should_swap = await change_imgmodel_task(user, channel, imgmodel_params)
        # Generate and send images
        endpoint = params.get('endpoint', '/sdapi/v1/txt2img')
        await process_image_gen(img_payload, censor_mode, channel, tags, endpoint, sd_output_dir)
        should_send_text = should_bot_do('should_send_text', default=True, tags=tags)
        should_gen_text = should_bot_do('should_gen_text', default=True, tags=tags)
        if (source == 'image' or (should_send_text and not should_gen_text)) and img_send_embed_info:
            img_send_embed_info.title = f"{user} requested an image:"
            img_send_embed_info.description = params.get('message', img_prompt)
            if i:
                if hasattr(i, 'followup'): await i.followup.reply(embed=img_send_embed_info)
                else: await i.reply(embed=img_send_embed_info)
            else: await channel.send(embed=img_send_embed_info)
        if send_user_image:
            send_user_image = discord.File(send_user_image)
            await channel.send(file=send_user_image)
        # If switching back to original Img model
        if should_swap:
            await change_imgmodel_task(user, channel, params={'imgmodel': {'imgmodel_name': current_imgmodel, 'mode': 'swap_back', 'verb': 'Swapping back to'}})
        return
    except Exception as e:
        logging.error(f"An error occurred in img_gen_task(): {e}")

#################################################################
######################## /IMAGE COMMAND #########################
#################################################################
if sd_enabled:

    # Updates size options for /image command
    def update_size_options(average):
        global size_choices
        options = load_file('ad_discordbot/dict_cmdoptions.yaml')
        sizes = options.get('sizes', [])
        aspect_ratios = [size.get("ratio") for size in sizes.get('ratios', [])]
        size_choices.clear()  # Clear the existing list
        ratio_options = calculate_aspect_ratio_sizes(average, aspect_ratios)
        static_options = sizes.get('static_sizes', [])
        size_options = (ratio_options or []) + (static_options or [])
        size_choices.extend(
            app_commands.Choice(name=option['name'], value=option['name'])
            for option in size_options)

    # Updates size ControlNet data for /image command
    async def update_cnet_options():
        global cnet_data
        if cnet_data:
            cnet_data = await get_cnet_data()

    # Update /image command options        
    async def update_image_cmd_menus(average):
        update_size_options(average)
        await update_cnet_options()
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

    async def get_imgcmd_choices(size_options, style_options):
        try:
            size_choices = [
                app_commands.Choice(name=option['name'], value=option['name'])
                for option in size_options]
            style_choices = [
                app_commands.Choice(name=option['name'], value=option['name'])
                for option in style_options]
            return size_choices, style_choices
        except Exception as e:
            logging.error(f"An error occurred while building choices for /image: {e}")  

    async def get_imgcmd_options():
        try:
            options = load_file('ad_discordbot/dict_cmdoptions.yaml')
            options = dict(options)
            active_settings = load_file('ad_discordbot/activesettings.yaml')
            active_settings = dict(active_settings)
            # Get sizes and aspect ratios from 'dict_cmdoptions.yaml'
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
            # Get style and controlnet options
            style_options = options.get('styles', {})
            return size_options, style_options
        except Exception as e:
            logging.error(f"An error occurred while building options for /image: {e}")

    async def get_cnet_data():
        filtered_cnet_data = {}
        if config['sd']['extensions'].get(f'controlnet_enabled', False):
            try:
                all_cnet_data = await sd_api(endpoint='/controlnet/control_types', method='get', json=None, retry=False)
                for key, value in all_cnet_data["control_types"].items():
                    if key == "All":
                        continue
                    if key in ["Reference", "Revision", "Shuffle"]:
                        value['name'] = key
                        filtered_cnet_data[key] = value
                    elif value["default_model"] != "None":
                        value['name'] = key
                        filtered_cnet_data[key] = value
            except:
                cnet_online = await ext_online('controlnet', '/controlnet/model_list')
                if cnet_online:
                    logging.warning("ControlNet is both enabled in config.py and detected. However, ad_discordbot relies on the '/controlnet/control_types' \
                        API endpoint which is missing. See here: (https://github.com/altoiddealer/ad_discordbot/wiki/troubleshooting).")
        return filtered_cnet_data

    async def ext_online(ext, endpoint):
        if config['sd']['extensions'].get(f'{ext}_enabled', False):
            try:
                online = await sd_api(endpoint=endpoint, method='get', json=None, retry=False)
                if online: return True
                else: return False
            except:
                logging.warning(f"{ext} is enabled in config.py, but was not responsive from {SD_CLIENT} API. Omitting option from '/image' command.")
        return False
    # Get size and style options for /image command
    size_options, style_options = asyncio.run(get_imgcmd_options())
    size_choices, style_choices = asyncio.run(get_imgcmd_choices(size_options, style_options))
    # Get filtered ControlNet data for /image command
    cnet_data = asyncio.run(get_cnet_data())
    # Test API reactor endpoint for /image command
    reactor_online = asyncio.run(ext_online('reactor', '/reactor/models'))

    if cnet_data and reactor_online:
        @client.hybrid_command(name="image", description=f'Generate an image using {SD_CLIENT}')
        @app_commands.describe(style='Applies a positive/negative prompt preset')
        @app_commands.describe(img2img='Diffuses from an input image instead of pure latent noise.')
        @app_commands.describe(img2img_mask='Masks the diffusion strength for the img2img input. Requires img2img.')
        @app_commands.describe(face_swap='For best results, attach a square (1:1) cropped image of a face, to swap into the output.')
        @app_commands.describe(controlnet='Guides image diffusion using an input image or map.')
        @app_commands.choices(size=size_choices)
        @app_commands.choices(style=style_choices)
        async def image(ctx: discord.ext.commands.Context, prompt: str, size: typing.Optional[app_commands.Choice[str]], style: typing.Optional[app_commands.Choice[str]], neg_prompt: typing.Optional[str], img2img: typing.Optional[discord.Attachment], img2img_mask: typing.Optional[discord.Attachment],
            face_swap: typing.Optional[discord.Attachment], controlnet: typing.Optional[discord.Attachment]):
            user_selections = {"prompt": prompt, "size": size.value if size else None, "style": style.value if style else None, "neg_prompt": neg_prompt, "img2img": img2img if img2img else None, "img2img_mask": img2img_mask if img2img_mask else None,
            "face_swap": face_swap if face_swap else None, "cnet": controlnet if controlnet else None}
            await process_image(ctx, user_selections)
    elif cnet_data and not reactor_online:
        @client.hybrid_command(name="image", description=f'Generate an image using {SD_CLIENT}')
        @app_commands.describe(style='Applies a positive/negative prompt preset')
        @app_commands.describe(img2img='Diffuses from an input image instead of pure latent noise.')
        @app_commands.describe(img2img_mask='Masks the diffusion strength for the img2img input. Requires img2img.')
        @app_commands.describe(controlnet='Guides image diffusion using an input image or map.')
        @app_commands.choices(size=size_choices)
        @app_commands.choices(style=style_choices)
        async def image(ctx: discord.ext.commands.Context, prompt: str, size: typing.Optional[app_commands.Choice[str]], style: typing.Optional[app_commands.Choice[str]], neg_prompt: typing.Optional[str], img2img: typing.Optional[discord.Attachment], img2img_mask: typing.Optional[discord.Attachment],
            controlnet: typing.Optional[discord.Attachment]):
            user_selections = {"prompt": prompt, "size": size.value if size else None, "style": style.value if style else None, "neg_prompt": neg_prompt, "img2img": img2img if img2img else None, "img2img_mask": img2img_mask if img2img_mask else None,
            "cnet": controlnet if controlnet else None}
            await process_image(ctx, user_selections)
    elif reactor_online and not cnet_data:
        @client.hybrid_command(name="image", description=f'Generate an image using {SD_CLIENT}')
        @app_commands.describe(style='Applies a positive/negative prompt preset')
        @app_commands.describe(img2img='Diffuses from an input image instead of pure latent noise.')
        @app_commands.describe(img2img_mask='Masks the diffusion strength for the img2img input. Requires img2img.')
        @app_commands.describe(face_swap='For best results, attach a square (1:1) cropped image of a face, to swap into the output.')
        @app_commands.choices(size=size_choices)
        @app_commands.choices(style=style_choices)
        async def image(ctx: discord.ext.commands.Context, prompt: str, size: typing.Optional[app_commands.Choice[str]], style: typing.Optional[app_commands.Choice[str]], neg_prompt: typing.Optional[str], img2img: typing.Optional[discord.Attachment], img2img_mask: typing.Optional[discord.Attachment], 
            face_swap: typing.Optional[discord.Attachment]):
            user_selections = {"prompt": prompt, "size": size.value if size else None, "style": style.value if style else None, "neg_prompt": neg_prompt, "img2img": img2img if img2img else None, "img2img_mask": img2img_mask if img2img_mask else None,
            "face_swap": face_swap if face_swap else None}
            await process_image(ctx, user_selections)
    else:
        @client.hybrid_command(name="image", description=f'Generate an image using {SD_CLIENT}')
        @app_commands.describe(style='Applies a positive/negative prompt preset')
        @app_commands.describe(img2img='Diffuses from an input image instead of pure latent noise.')
        @app_commands.describe(img2img_mask='Masks the diffusion strength for the img2img input. Requires img2img.')
        @app_commands.choices(size=size_choices)
        @app_commands.choices(style=style_choices)
        async def image(ctx: discord.ext.commands.Context, prompt: str,  size: typing.Optional[app_commands.Choice[str]], style: typing.Optional[app_commands.Choice[str]], neg_prompt: typing.Optional[str], img2img: typing.Optional[discord.Attachment], img2img_mask: typing.Optional[discord.Attachment]):
            user_selections = {"prompt": prompt, "size": size.value if size else None, "style": style.value if style else None, "neg_prompt": neg_prompt, "img2img": img2img if img2img else None, "img2img_mask": img2img_mask if img2img_mask else None}
            await process_image(ctx, user_selections)

    async def process_image(ctx, selections):
        # Do not process if SD WebUI is offline
        if not await sd_online(ctx.channel):
            await ctx.defer()
            return
        # User inputs from /image command
        prompt = selections.get('prompt', '')  
        size = selections.get('size', None)
        style = selections.get('style', None)
        neg_prompt = selections.get('neg_prompt', '')
        img2img = selections.get('img2img', None)
        img2img_mask = selections.get('img2img_mask', None)
        face_swap = selections.get('face_swap', None)
        cnet = selections.get('cnet', None)
        # Defaults
        endpoint = '/sdapi/v1/txt2img'
        neg_style_prompt = ""
        size_dict = {}
        faceswapimg = None
        img2img_dict = {}
        cnet_dict = {}
        try:
            prompt = await dynamic_prompting(ctx.author, prompt, i=None)
            message = f"**Prompt:** {prompt}"
            if size:
                selected_size = next((option for option in size_options if option['name'] == size), None)
                if selected_size:
                    size_dict['width'] = selected_size.get('width')
                    size_dict['height'] = selected_size.get('height')
                message += f" | **Size:** {size}"
            if style:
                selected_style_option = next((option for option in style_options if option['name'] == style), None)
                if selected_style_option:
                    prompt = selected_style_option.get('positive').format(prompt)
                    neg_style_prompt = selected_style_option.get('negative')
                message += f" | **Style:** {style}"
            if neg_prompt:
                neg_style_prompt = f"{neg_prompt}, {neg_style_prompt}"
                message += f" | **Negative Prompt:** {neg_prompt}"
            if img2img:
                async def process_image_img2img(img2img, img2img_dict, endpoint, message):
                    #Convert attached image to base64
                    attached_i2i_img = await img2img.read()
                    i2i_image = base64.b64encode(attached_i2i_img).decode('utf-8')
                    img2img_dict['image'] = i2i_image
                    # Ask user to select a Denoise Strength
                    denoise_options = []
                    for value in [round(0.05 * index, 2) for index in range(int(1 / 0.05) + 1)]:
                        denoise_options.append(discord.SelectOption(label=str(value), value=str(value), default=True if value == 0.40 else False))
                    denoise_options = denoise_options[:25]
                    denoise_select = discord.ui.Select(custom_id="denoise", options=denoise_options)
                    # Send Denoise Strength select menu in a view
                    submit_button = discord.ui.Button(style=discord.ButtonStyle.primary, label="Submit")
                    view = discord.ui.View()
                    view.add_item(denoise_select)
                    view.add_item(submit_button)
                    select_message = await ctx.send("Select denoise strength for img2img:", view=view, ephemeral=True)
                    interaction = await client.wait_for("interaction", check=lambda interaction: interaction.message.id == select_message.id)
                    denoising_strength = interaction.data.get("values", ["0.40"])[0]
                    img2img_dict['denoising_strength'] = float(denoising_strength)
                    await interaction.response.defer() # defer response for this interaction
                    await select_message.delete()
                    endpoint = '/sdapi/v1/img2img' # Change API endpoint to img2img
                    message += f" | **Img2Img**, denoise strength: {denoising_strength}"
                    return img2img_dict, endpoint, message
                try:
                    img2img_dict, endpoint, message = await process_image_img2img(img2img, img2img_dict, endpoint, message)
                except Exception as e:
                    logging.error(f"An error occurred while configuring Img2Img for /image command: {e}")
            if img2img_mask:
                if img2img:
                    attached_img2img_mask_img = await img2img_mask.read()
                    img2img_mask_img = base64.b64encode(attached_img2img_mask_img).decode('utf-8')
                    img2img_dict['mask'] = img2img_mask_img
                    message += f" | **Inpainting:** Image Provided"
                else:
                    await ctx.send("Inpainting requires im2img. Not applying img2img_mask mask...", ephemeral=True)
            if face_swap:
                attached_face_img = await face_swap.read()
                faceswapimg = base64.b64encode(attached_face_img).decode('utf-8')
                message += f" | **Face Swap:** Image Provided"
            if cnet:
                async def process_image_controlnet(cnet, cnet_dict, message):
                    try:
                        # Convert attached image to base64
                        attached_cnet_img = await cnet.read()
                        cnetimage = base64.b64encode(attached_cnet_img).decode('utf-8')
                        cnet_dict['image'] = cnetimage
                    except:
                        logging.error(f"Error decoding ControlNet input image for '/image' command: {e}")
                    try:
                        # Ask user to select a Control Type
                        cnet_control_type_options = [discord.SelectOption(label=key, value=key) for key in cnet_data]
                        control_type_select = discord.ui.Select(options=cnet_control_type_options, placeholder="Select ControlNet Control Type", custom_id="cnet_control_type_select")
                        # Send Control Type select menu in a view
                        view = discord.ui.View()
                        view.add_item(control_type_select)
                        select_message = await ctx.send("### Select ControlNet Control Type:", view=view, ephemeral=True)
                        interaction = await client.wait_for("interaction", check=lambda interaction: interaction.message.id == select_message.id)
                        selected_control_type = interaction.data.get("values")[0]
                        selected_control_type = cnet_data[selected_control_type]
                        await interaction.response.defer() # defer response for this interaction
                        await select_message.delete()
                    except Exception as e:
                        logging.error(f"An error occurred while setting ControlNet Control Type in '/image' command: {e}")
                    # View containing Selects for ControlNet Module, Model, Start and End
                    class CnetControlView(discord.ui.View):
                        def __init__(self, cnet_data, selected_control_type):
                            super().__init__()
                            self.cnet_dict = {'module': selected_control_type["default_option"], 'model': selected_control_type["default_model"], 'guidance_start': 0.00, 'guidance_end': 1.00}
                        # Dropdown Menu for Module
                        module_options = [discord.SelectOption(label=module_option, value=module_option, default=True if module_option == selected_control_type["default_option"] else False,
                            description='Default' if module_option == selected_control_type["default_option"] else None) for module_option in selected_control_type["module_list"]]
                        @discord.ui.select(options=module_options, placeholder="Select ControlNet Module", custom_id="cnet_module_select")
                        async def module_select(self, select, interaction):
                            self.cnet_dict['module'] = select.data['values'][0]
                            await select.response.defer()
                        # Dropdown Menu for Model
                        model_options = [discord.SelectOption( label=model_option, value=model_option, default=True if model_option == selected_control_type["default_model"] else False,
                            description='Default' if model_option == selected_control_type["default_model"] else '') for model_option in selected_control_type["model_list"]]
                        @discord.ui.select(options=model_options, placeholder="Select ControlNet Model", custom_id="cnet_model_select", disabled=selected_control_type.get("default_model") == 'None')
                        async def model_select(self, select, interaction):
                            self.cnet_dict['model'] = select.data['values'][0]
                            await select.response.defer()
                        # Dropdown Menu for Start
                        start_options = []
                        for value in [round(0.05 * index, 2) for index in range(int(1 / 0.05) + 1)]:
                            start_options.append(discord.SelectOption(label=str(value), value=str(value), default=True if value == 0.00 else False))
                        @discord.ui.select(options=start_options, placeholder="Select Start Guidance (0.0 - 1.0)", custom_id="cnet_start_select")
                        async def start_select(self, select, interaction):
                            self.cnet_dict['guidance_start'] = float(select.data['values'][0])
                            await select.response.defer()
                        # Dropdown Menu for End
                        end_options = []
                        for value in [round(0.05 * index, 2) for index in range(int(1 / 0.05) + 1)]:
                            end_options.append(discord.SelectOption(label=str(value), value=str(value), default=True if value == 1.00 else False))
                        @discord.ui.select(options=end_options, placeholder="Select End Guidance (0.0 - 1.0)", custom_id="cnet_end_select")
                        async def end_select(self, select, interaction):
                            self.cnet_dict['guidance_end'] = float(select.data['values'][0])
                            await select.response.defer()
                        # Submit button
                        @discord.ui.button(label='Submit', style=discord.ButtonStyle.primary, custom_id="cnet_submit")
                        async def submit_button(self, button, interaction):
                            await button.response.defer()
                            self.stop()
                    # Function to build Select Options based on the selected ControlNet Module
                    def make_cnet_options(selected_module):
                        # Defaults
                        options_a = [discord.SelectOption(label='Not Applicable', value='64')]
                        options_b = [discord.SelectOption(label='Not Applicable', value='64')]
                        label_a = 'Not Applicable'
                        label_b = 'Not Applicable'
                        if (selected_module in ["canny", "mlsd", "normal_midas", "scribble_xdog", "softedge_teed"]
                            or selected_module.startswith(('blur', 'depth_leres', 'recolor_', 'reference', 'CLIP-G', 'tile_colorfix'))):
                            try:
                                # Initialize Specific Options
                                options_a = []
                                options_b = []
                                # Defaults
                                round_a = 2
                                range_a = 1
                                default_a = 10
                                round_b = 2
                                range_b = 256
                                default_b = 0
                                if selected_module.startswith('blur'):
                                    label_a = 'Sigma'
                                    range_a = 64
                                    default_a = 3
                                elif selected_module == 'canny':
                                    label_a = 'Low Threshold'
                                    round_a = 0
                                    range_a = 256
                                    default_a = 7
                                    label_b = 'High Threshold'
                                    round_b = 0
                                    default_b = 16
                                elif selected_module.startswith('depth_leres'):
                                    label_a = 'Remove Near %'
                                    round_a = 1
                                    range_a = 100
                                    default_a = 0
                                    label_b = 'Remove Background %'
                                    round_b = 1
                                    range_b = 100
                                elif selected_module == 'mlsd':
                                    label_a = 'MLSD Value Threshold'
                                    range_a = 2
                                    default_a = 0
                                    label_b = 'MLSD Distance Threshold'
                                    range_b = 20
                                elif selected_module == 'normal_midas':
                                    label_a = 'Normal Background Threshold'
                                    default_a = 8
                                elif selected_module.startswith('recolor'):
                                    label_a = 'Gamma Correction'
                                    round_a = 3
                                    range_a = 2
                                elif selected_module.startswith('reference'):
                                    label_a = 'Style Fidelity'
                                elif selected_module.startswith('CLIP-G'): # AKA 'Revision'
                                    label_a = 'Noise Augmentation'
                                    default_a = 0
                                elif selected_module == 'scribble_xdog':
                                    label_a = 'XDoG Threshold'
                                    range_a = 64
                                elif selected_module == 'softedge_teed':
                                    label_a = 'Safe Steps'
                                    default_a = 8
                                    range_a = 10
                                    round_a = 0
                                elif selected_module.startswith('tile_colorfix'):
                                    label_a = 'Variation'
                                    round_a = 0
                                    range_a = 32
                                    default_a = 5
                                    if selected_module == 'tile_colorfix+sharp':
                                        label_b = 'Sharpness'
                                        round_b = 0
                                        range_b = 2
                                        default_b = 10
                                for index, value in enumerate([round(index * (range_a / 20), round_a) for index in range(20 + 1)]):
                                    value = float(value) if round_a else int(value)
                                    options_a.append(discord.SelectOption(label=str(value), value=str(value), default=index == default_a))
                                for index, value in enumerate([round(index * (range_b / 20), round_b) for index in range(20 + 1)]):
                                    value = float(value) if round_b else int(value)
                                    options_b.append(discord.SelectOption(label=str(value), value=str(value), default=index == default_b))
                            except:
                                logging.error(f"Error building ControlNet options for '/image' command: {e}")
                                return [discord.SelectOption(label='Not Applicable', value='64')], 'Not Applicable', [discord.SelectOption(label='Not Applicable', value='64')], 'Not Applicable'
                        return options_a, label_a, options_b, label_b
                    try:
                        cnet_control_view = CnetControlView(cnet_data, selected_control_type)
                        view_message = await ctx.send('### Select ControlNet Options\n • **Module**\n • **Model**\n • **Start** (0.0 - 1.0)\n • **End** (0.0 - 1.0)\n(if unsure, just Submit with Defaults)', 
                            view=cnet_control_view, ephemeral=True)
                        await cnet_control_view.wait()
                        cnet_dict.update(cnet_control_view.cnet_dict)
                        selected_module = cnet_dict['module']   # For next step
                        await view_message.delete()
                        options_a, label_a, options_b, label_b = make_cnet_options(selected_module)
                    except Exception as e:
                        logging.error(f"An error occurred while configuring initial ControlNet options from '/image' command: {e}")
                    # View containing Selects for ControlNet Weight and Additional Options
                    class CnetOptionsView(discord.ui.View):
                        def __init__(self, options_a, label_a, options_b, label_b):
                            super().__init__()
                            self.cnet_dict = {'weight': 1.00}
                        # Dropdown Menu for Weight
                        weight_options = []
                        for value in [round(0.05 * index, 2) for index in range(int(1 / 0.05) + 1)]:
                            weight_options.append(discord.SelectOption(label=str(value), value=str(value), default=True if value == 1.00 else False))
                        @discord.ui.select(options=weight_options, placeholder="Select ControlNet Weight", custom_id="cnet_weight_select")
                        async def weight_select(self, select, interaction):
                            self.cnet_dict['weight'] = float(select.data['values'][0])
                            await select.response.defer()
                        # Dropdown Menu for Options A
                        @discord.ui.select(options=options_a, placeholder=label_a, custom_id="cnet_options_a_select", disabled=label_a == 'Not Applicable')
                        async def thresh_a_select(self, select, interaction):
                            self.cnet_dict['threshold_a'] = float(select.data['values'][0]) if '.' in select.data['values'][0] else int(select.data['values'][0])
                            await select.response.defer()
                        # Dropdown Menu for Options B
                        @discord.ui.select(options=options_b, placeholder=label_b, custom_id="cnet_options_b_select", disabled=label_b == 'Not Applicable')
                        async def options_b_select(self, select, interaction):
                            self.cnet_dict['threshold_b'] = float(select.data['values'][0]) if '.' in select.data['values'][0] else int(select.data['values'][0])
                            await select.response.defer()
                        # Submit button
                        @discord.ui.button(label='Submit', style=discord.ButtonStyle.primary, custom_id="cnet_submit")
                        async def submit_button(self, button, interaction):
                            await button.response.defer()
                            self.stop()
                    try:
                        view = CnetOptionsView(options_a, label_a, options_b, label_b)
                        message_a = f'\n • **{label_a}**' if label_a != 'Not Applicable' else ''
                        message_b = f'\n • **{label_b}**' if label_b != 'Not Applicable' else ''
                        view_message = await ctx.send(f'### Select ControlNet Options\n • **Weight** (0.0 - 1.0){message_a}{message_b}\n(if unsure, just Submit with Defaults)', view=view, ephemeral=True) 
                        await view.wait()
                        cnet_dict.update(view.cnet_dict)
                        await view_message.delete()
                    except Exception as e:
                        logging.error(f"An error occurred while configuring secondary ControlNet options from /image command: {e}")
                    cnet_dict.update({'enabled': True, 'save_detected_map': True})
                    message += f" | **ControlNet:** (Module: {cnet_dict['module']}, Model: {cnet_dict['model']})"
                    return cnet_dict, message
                try:
                    cnet_dict, message = await process_image_controlnet(cnet, cnet_dict, message)
                except Exception as e:
                    logging.error(f"An error occurred while configuring ControlNet for /image command: {e}")
            params = {'neg_prompt': neg_style_prompt, 'size': size_dict, 'img2img': img2img_dict, 'face_swap': faceswapimg, 'controlnet': cnet_dict, 'endpoint': endpoint, 'message': message}
            await ireply(ctx, 'image') # send a response msg to the user
            
            async with task_semaphore:
                async with ctx.channel.typing():
                    # offload to ai_gen queue
                    logging.info(f'{ctx.author} used "/image": "{prompt}"')
                    await img_gen_task(ctx.author, ctx.channel, 'image', prompt, params, i=None, tags={})
                    await run_flow_if_any(ctx.author, ctx.channel, 'image', prompt)

        except Exception as e:
            logging.error(f"An error occurred in image(): {e}")
            traceback.print_exc()

#################################################################
######################### MISC COMMANDS #########################
#################################################################
if system_embed_info:
    @client.hybrid_command(description="Display help menu")
    async def helpmenu(ctx):
        await ctx.send(embed=system_embed_info)

@client.hybrid_command(description="Toggle current channel as main channel for bot to auto-reply without needing to be called")
async def main(i):
    try:
        with sqlite3.connect('bot.db') as conn:
            c = conn.cursor()
            if i.channel.id in bot_settings.database.main_channels:
                bot_settings.database.main_channels.remove(i.channel.id) # If the channel is already in the main channels, remove it
                c.execute('''DELETE FROM main_channels WHERE channel_id = ?''', (i.channel.id,))
                action_message = f'Removed {i.channel.mention} from main channels. Use "/main" again if you want to add it back.'
            else:
                # If the channel is not in the main channels, add it
                bot_settings.database.main_channels.append(i.channel.id)
                c.execute('''INSERT OR REPLACE INTO main_channels (channel_id) VALUES (?)''', (i.channel.id,))
                action_message = f'Added {i.channel.mention} to main channels. Use "/main" again to remove it.'
            conn.commit()
        bot_settings.database = Database()
        await i.reply(action_message)
    except Exception as e:
        logging.error(f"Error toggling main channel setting: {e}")

@client.hybrid_command(description="Update dropdown menus without restarting bot script.")
async def sync(ctx: discord.ext.commands.Context):
    try:
        await ctx.reply('Syncing client tree. Note: Menus may not update instantly.', ephemeral=True, delete_after=10)
        logging.info(f"{ctx.author} used '/sync' to sync the client.tree (refresh commands).")
        await bg_task_queue.put(client.tree.sync()) # Process this in the background
    except Exception as e:
        logging.error(f"Error syncing client.tree with '/sync': {e}")

#################################################################
######################### LLM COMMANDS ##########################
#################################################################
if textgenwebui_enabled:
    # /reset command - Resets current character
    @client.hybrid_command(description="Reset the conversation with current character")
    async def reset_character(ctx: discord.ext.commands.Context):
        try:
            shared.stop_everything = True
            await ireply(ctx, 'character reset') # send a response msg to the user
            
            async with task_semaphore:
                # offload to ai_gen queue
                logging.info(f'{ctx.author} used "/reset": "{bot_settings.database.last_character}"')
                params = {'character': {'char_name': bot_settings.database.last_character, 'verb': 'Resetting', 'mode': 'reset'}}
                await change_char_task(ctx.author, ctx.channel, 'reset', params)
                
        except Exception as e:
            logging.error(f"Error with /reset: {e}")

    # /reset command - Resets current character
    @client.hybrid_command(description="Saves the current conversation to a new file in text-generation-webui/logs/")
    async def save_conversation(ctx: discord.ext.commands.Context):
        try:
            await ctx.reply('Saved current conversation history', ephemeral=True)
            history_manager.save_history()
        except Exception as e:
            logging.error(f"Error with /reset: {e}")

    # Context menu command to Regenerate last reply
    @client.tree.context_menu(name="regenerate")
    async def regen_llm_gen(i: discord.Interaction, message: discord.Message):
        text = message.content
        await i.response.defer(thinking=False)
        
        async with task_semaphore:
            async with i.channel.typing():
                # offload to ai_gen queue
                logging.info(f'{i.user.display_name} used "Regenerate"')
                await cont_regen_task(i, i.user.display_name, text, i.channel, 'regen', message.id)
                await run_flow_if_any(i.user.display_name, i.channel, 'regen', text)

    # Context menu command to Continue last reply
    @client.tree.context_menu(name="continue")
    async def continue_llm_gen(i: discord.Interaction, message: discord.Message):
        text = message.content
        await i.response.defer(thinking=False)
        
        async with task_semaphore:
            async with i.channel.typing():
                # offload to ai_gen queue
                logging.info(f'{i.user.display_name} used "Continue"')
                await cont_regen_task(i, i.user.display_name, text, i.channel, 'cont', message.id)
                await run_flow_if_any(i.user.display_name, i.channel, 'cont', text)

async def load_character_data(char_name):
    char_data = None
    for ext in ['.yaml', '.yml', '.json']:
        character_file = os.path.join("characters", f"{char_name}{ext}")
        if os.path.exists(character_file):
            char_data = load_file(character_file)
            if char_data is None:
                continue
            
            char_data = dict(char_data)
            break  # Break the loop if data is successfully loaded
            
    if char_data is None:
        logging.error(f"Failed to load data for: {char_name}, perhaps missing file?")
        
    return char_data

# Collect character information
async def character_loader(source):
    try:
        # Get data using textgen-webui native character loading function
        _, name, _, greeting, context = load_character(source, '', '')
        missing_keys = [key for key, value in {'name': name, 'greeting': greeting, 'context': context}.items() if not value]
        if any (missing_keys):
            logging.warning(f'Note that character "{source}" is missing the following info:"{missing_keys}".')
        textgen_data = {'name': name, 'greeting': greeting, 'context': context}
        # Check for extra bot data
        char_data = await load_character_data(source)
        char_instruct = char_data.get('instruction_template_str', None)
        # Merge with basesettings
        char_data = merge_base(char_data, 'llmcontext')
        # Reset warning for character specific TTS
        bot_settings.database.update_was_warned('char_tts', 0)
        # Gather context specific keys from the character data
        char_llmcontext = {}
        for key, value in char_data.items():
            if key == 'extensions':
                await update_extensions(value)
                char_llmcontext['extensions'] = value
            elif key == 'use_voice_channel':
                await voice_channel(value)
                char_llmcontext['use_voice_channel'] = value
            elif key == 'tags':
                value = await update_tags(value) # Unpack any tag presets
                char_llmcontext['tags'] = value
        # Merge llmcontext data and extra data
        char_llmcontext.update(textgen_data)
        # Collect behavior data
        char_behavior = char_data.get('behavior', {})
        char_behavior = merge_base(char_behavior, 'behavior')
        # Collect llmstate data
        char_llmstate = char_data.get('state', {})
        char_llmstate = merge_base(char_llmstate, 'llmstate,state')
        char_llmstate['character_menu'] = source
        # Commit the character data to bot_settings.settings
        settings_dict = bot_settings.get_settings_dict()
        settings_dict['llmcontext'] = dict(char_llmcontext) # Replace the entire dictionary key
        update_dict(settings_dict['behavior'], dict(char_behavior))
        update_dict(settings_dict['llmstate']['state'], dict(char_llmstate))
        # Update stored database value for character
        bot_settings.database.update_last_setting(source, 'last_character')
        # Print mode in cmd
        logging.info(f"Initializing in {bot_settings.settings['llmstate']['state']['mode']} mode")
        # Data for saving to activesettings.yaml (skipped in on_ready())
        return char_instruct, char_llmcontext, char_behavior, char_llmstate
    except Exception as e:
        logging.error(f"Error loading character. Check spelling and file structure. Use bot cmd '/character' to try again. {e}")
        return # TODO

# Task to manage discord profile updates
delayed_profile_update_task = None

async def delayed_profile_update(username, avatar, remaining_cooldown):
    try:
        await asyncio.sleep(remaining_cooldown)
        if username:
            for guild in client.guilds:
                client_member = guild.get_member(client.user.id)
                await client_member.edit(nick=username)
        if avatar:
            await client.user.edit(avatar=avatar)
        logging.info(f"Updated discord client profile (Username: {username}; Avatar: {'Updated' if avatar else 'Unchanged'}).\n Profile can be updated again in 10 minutes.")
        bot_settings.database.update_last_time('last_change')  # Store the current datetime in bot.db
    except Exception as e:
        logging.error(f"Error while changing character username or avatar: {e}")

async def update_client_profile(channel, char_name):
    try:
        global delayed_profile_update_task
        # Cancel delayed profile update task if one is already pending
        if delayed_profile_update_task and not delayed_profile_update_task.done():
            delayed_profile_update_task.cancel()
        # Do not update profile if name is same and no update task is scheduled
        elif all(guild.get_member(client.user.id).display_name == char_name for guild in client.guilds):
            return
        avatar = None
        folder = 'characters'
        picture_path = os.path.join(folder, f'{char_name}.png')
        if os.path.exists(picture_path):
            with open(picture_path, 'rb') as f:
                avatar = f.read()
        # Check for cooldown before allowing profile change
        last_change = bot_settings.database.last_change
        last_change = datetime.strptime(last_change, '%Y-%m-%d %H:%M:%S')
        last_cooldown = last_change + timedelta(minutes=10)
        if datetime.now() >= last_cooldown:
            # Apply changes immediately if outside 10 minute cooldown
            delayed_profile_update_task = asyncio.create_task(delayed_profile_update(char_name, avatar, 0))
        else:
            remaining_cooldown = last_cooldown - datetime.now()
            seconds = int(remaining_cooldown.total_seconds())
            await channel.send(f'**Due to Discord limitations, character name/avatar will update in {seconds} seconds.**', delete_after=10)
            logging.info(f"Due to Discord limitations, character name/avatar will update in {remaining_cooldown} seconds.")
            delayed_profile_update_task = asyncio.create_task(delayed_profile_update(char_name, avatar, seconds))
    except Exception as e:
        logging.error(f"An error occurred while updating Discord profile: {e}")

# Apply character changes
async def change_character(channel, source, char_name):
    try:
        # Load the character
        char_instruct, char_llmcontext, char_behavior, char_llmstate = await character_loader(char_name)
        update_instruct = char_instruct or instruction_template_str or None # 'instruction_template_str' is global variable
        if update_instruct:
            settings_dict = bot_settings.get_settings_dict()
            settings_dict['llmstate']['state']['instruction_template_str'] = update_instruct
        # Update stored database value for character
        bot_settings.database.update_last_setting(char_name, 'last_character')
        # Update discord username / avatar
        await update_client_profile(channel, char_name)
        # Save the updated active_settings to activesettings.yaml
        active_settings = load_file('ad_discordbot/activesettings.yaml')
        active_settings['llmcontext'] = char_llmcontext
        active_settings['behavior'] = char_behavior
        active_settings['llmstate']['state'] = char_llmstate
        save_yaml_file('ad_discordbot/activesettings.yaml', active_settings)
        # Ensure all settings are synchronized
        await update_bot_settings() # Sync updated user settings
        # Set history
        autoload_history = config.get('textgenwebui', {}).get('chat_history', {}).get('autoload_history', False)
        if autoload_history and source != 'reset':
            history_manager.load_history(bot_settings.settings['llmstate']['state'])
        else:
            history_manager.reset_session_history()
    except Exception as e:
        await channel.send(f"An error occurred while changing character: {e}")
        logging.error(f"An error occurred while changing character: {e}")
    return

async def process_character(ctx, selected_character_value):
    try:
        if not selected_character_value:
            await ctx.reply('**No character was selected**.', ephemeral=True, delete_after=5)
            return
        char_name = Path(selected_character_value).stem
        await ireply(ctx, 'character change') # send a response msg to the user

        async with task_semaphore:
            # offload to ai_gen queue
            logging.info(f'{ctx.author} used "/character": "{char_name}"')
            params = {'character': {'char_name': char_name, 'verb': 'Changing', 'mode': 'change'}}
            await change_char_task(ctx.author, ctx.channel, 'character', params)
            
    except Exception as e:
        logging.error(f"Error processing selected character from /character command: {e}")

def get_all_characters():
    all_characters = []
    filtered_characters = []
    try:
        for file in sorted(Path("characters").glob("*")):
            if file.suffix in [".json", ".yml", ".yaml"]:
                character = {}
                character['name'] = file.stem
                all_characters.append(character)

                char_data = load_file(file)
                if char_data is None:
                    continue
                
                char_data = dict(char_data)
                if char_data.get('bot_in_character_menu', True):
                    filtered_characters.append(character)

    except Exception as e:
        logging.error(f"An error occurred while getting all characters: {e}")
    return all_characters, filtered_characters

if textgenwebui_enabled:
    all_characters, filtered_characters = get_all_characters()
    if filtered_characters:
        character_options = [app_commands.Choice(name=character["name"][:100], value=character["name"]) for character in filtered_characters[:25]]
        character_options_label = f'{character_options[0].name[0]}-{character_options[-1].name[0]}'.lower()
        if len(filtered_characters) > 25:
            character_options1 = [app_commands.Choice(name=character["name"][:100], value=character["name"]) for character in filtered_characters[25:50]]
            character_options1_label = f'{character_options1[0].name[0]}-{character_options1[-1].name[0]}'.lower()
            if character_options1_label == character_options_label:
                character_options1_label = f'{character_options1_label}_1'
            if len(filtered_characters) > 50:
                character_options2 = [app_commands.Choice(name=character["name"][:100], value=character["name"]) for character in filtered_characters[50:75]]
                character_options2_label = f'{character_options2[0].name[0]}-{character_options2[-1].name[0]}'.lower()
                if character_options2_label == character_options_label or character_options2_label == character_options1_label:
                    character_options2_label = f'{character_options2_label}_2'
                if len(filtered_characters) > 75:
                    character_options3 = [app_commands.Choice(name=character["name"][:100], value=character["name"]) for character in filtered_characters[75:100]]
                    character_options3_label = f'{character_options3[0].name[0]}-{character_options3[-1].name[0]}'.lower()
                    if character_options3_label == character_options_label or character_options3_label == character_options1_label or character_options3_label == character_options2_label:
                        character_options3_label = f'{character_options2_label}_3'
                    if len(filtered_characters) > 100:
                        filtered_characters = filtered_characters[:100]
                        logging.warning("'/character' command only allows up to 100 characters. Some characters were omitted.")

        if len(filtered_characters) <= 25:
            @client.hybrid_command(name="character", description='Choose an character')
            @app_commands.rename(characters=f'characters_{character_options_label}')
            @app_commands.describe(characters=f'characters {character_options_label.upper()}')
            @app_commands.choices(characters=character_options)
            async def character(ctx: discord.ext.commands.Context, characters: typing.Optional[app_commands.Choice[str]]):
                selected_character = characters.value if characters is not None else ''
                await process_character(ctx, selected_character)

        elif 25 < len(filtered_characters) <= 50:
            @client.hybrid_command(name="character", description='Choose an character (pick only one)')
            @app_commands.rename(characters_1=f'characters_{character_options_label}')
            @app_commands.describe(characters_1=f'characters {character_options_label.upper()}')
            @app_commands.choices(characters_1=character_options)
            @app_commands.rename(characters_2=f'characters_{character_options1_label}')
            @app_commands.describe(characters_2=f'characters {character_options1_label.upper()}')
            @app_commands.choices(characters_2=character_options1)
            async def character(ctx: discord.ext.commands.Context, characters_1: typing.Optional[app_commands.Choice[str]], characters_2: typing.Optional[app_commands.Choice[str]]):
                if characters_1 and characters_2:
                    await ctx.send("More than one character was selected. Using the first selection.", ephemeral=True)
                selected_character = ((characters_1 or characters_2) and (characters_1 or characters_2).value) or ''
                await process_character(ctx, selected_character)

        elif 50 < len(filtered_characters) <= 75:
            @client.hybrid_command(name="character", description='Choose an character (pick only one)')
            @app_commands.rename(characters_1=f'characters_{character_options_label}')
            @app_commands.describe(characters_1=f'characters {character_options_label.upper()}')
            @app_commands.choices(characters_1=character_options)
            @app_commands.rename(characters_2=f'characters_{character_options1_label}')
            @app_commands.describe(characters_2=f'characters {character_options1_label.upper()}')
            @app_commands.choices(characters_2=character_options1)
            @app_commands.rename(characters_3=f'characters_{character_options2_label}')
            @app_commands.describe(characters_3=f'characters {character_options2_label.upper()}')
            @app_commands.choices(characters_3=character_options2)
            async def character(ctx: discord.ext.commands.Context, characters_1: typing.Optional[app_commands.Choice[str]], characters_2: typing.Optional[app_commands.Choice[str]], characters_3: typing.Optional[app_commands.Choice[str]]):
                if sum(1 for v in (characters_1, characters_2, characters_3) if v) > 1:
                    await ctx.send("More than one character was selected. Using the first selection.", ephemeral=True)
                selected_character = ((characters_1 or characters_2 or characters_3) and (characters_1 or characters_2 or characters_3).value) or ''
                await process_character(ctx, selected_character)

        elif 75 < len(filtered_characters) <= 100:
            @client.hybrid_command(name="character", description='Choose an character (pick only one)')
            @app_commands.rename(characters_1=f'characters_{character_options_label}')
            @app_commands.describe(characters_1=f'characters {character_options_label.upper()}')
            @app_commands.choices(characters_1=character_options)
            @app_commands.rename(characters_2=f'characters_{character_options1_label}')
            @app_commands.describe(characters_2=f'characters {character_options1_label.upper()}')
            @app_commands.choices(characters_2=character_options1)
            @app_commands.rename(characters_3=f'characters_{character_options2_label}')
            @app_commands.describe(characters_3=f'characters {character_options2_label.upper()}')
            @app_commands.choices(characters_3=character_options2)
            @app_commands.rename(characters_4=f'characters_{character_options3_label}')
            @app_commands.describe(characters_4=f'characters {character_options3_label.upper()}')
            @app_commands.choices(characters_4=character_options3)
            async def character(ctx: discord.ext.commands.Context, characters_1: typing.Optional[app_commands.Choice[str]], characters_2: typing.Optional[app_commands.Choice[str]], characters_3: typing.Optional[app_commands.Choice[str]], characters_4: typing.Optional[app_commands.Choice[str]]):
                if sum(1 for v in (characters_1, characters_2, characters_3, characters_4) if v) > 1:
                    await ctx.send("More than one character was selected. Using the first selection.", ephemeral=True)
                selected_character = ((characters_1 or characters_2 or characters_3 or characters_4) and (characters_1 or characters_2 or characters_3 or characters_4).value) or ''
                await process_character(ctx, selected_character)

#################################################################
####################### /IMGMODEL COMMAND #######################
#################################################################
# Apply user defined filters to imgmodel list
async def filter_imgmodels(imgmodels):
    try:
        imgmodels_data = load_file('ad_discordbot/dict_imgmodels.yaml')
        filter_list = imgmodels_data.get('settings', {}).get('filter', None)
        exclude_list = imgmodels_data.get('settings', {}).get('exclude', None)
        if filter_list or exclude_list:
            imgmodels = [
                imgmodel for imgmodel in imgmodels
                if (
                    (not filter_list or any(re.search(re.escape(filter_text), imgmodel.get('imgmodel_name', '') + imgmodel.get('sd_model_checkpoint', ''), re.IGNORECASE) for filter_text in filter_list))
                    and (not exclude_list or not any(re.search(re.escape(exclude_text), imgmodel.get('imgmodel_name', '') + imgmodel.get('sd_model_checkpoint', ''), re.IGNORECASE) for exclude_text in exclude_list))
                )
            ]
        return imgmodels
    except Exception as e:
        logging.error(f"Error filtering image model list: {e}")

# Build list of imgmodels depending on user preference (user .yaml / API)
async def fetch_imgmodels():
    try:
        try:
            imgmodels = await sd_api(endpoint='/sdapi/v1/sd-models', method='get', json=None, retry=False)
            # Update 'title' keys in fetched list for uniformity
            for imgmodel in imgmodels:
                if 'title' in imgmodel:
                    imgmodel['sd_model_checkpoint'] = imgmodel.pop('title')
        except Exception as e:
            logging.error(f"Error fetching image models from {SD_CLIENT} API: {e}")
            if str(e).startswith('Cannot connect to host'):
                logging.warning('"/imgmodels" command will initialize as an empty list. Stable Diffusion must be running before launching ad_discordbot.')
            return ''
        if imgmodels:
            imgmodels = await filter_imgmodels(imgmodels)
            return imgmodels
    except Exception as e:
        logging.error(f"Error fetching image models: {e}")

async def update_imgmodel(channel, selected_imgmodel, selected_imgmodel_tags):
    try:
        active_settings = load_file('ad_discordbot/activesettings.yaml')
        current_w, current_h = active_settings['imgmodel'].get('payload', {}).get('width', 512), active_settings['imgmodel'].get('payload', {}).get('height', 512)
        new_w, new_h = selected_imgmodel.get('payload', {}).get('width', 512), selected_imgmodel.get('payload', {}).get('height', 512)
        active_settings['imgmodel'] = selected_imgmodel
        active_settings['imgmodel']['tags'] = selected_imgmodel_tags
        save_yaml_file('ad_discordbot/activesettings.yaml', active_settings)
        await update_bot_settings() # Sync updated user settings
        ### IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
        # if selected_imgmodel['imgmodel_name'] == 'None':
        # _ = await sd_api(endpoint='/sdapi/v1/unload-checkpoint', method='post', json=None)
        #     change_embed.title = 'Unloaded Img model'
        #     change_embed.description = ''
        #     await channel.send(embed=change_embed)
        #     return
        # Load the imgmodel and VAE via API
        model_data = active_settings['imgmodel'].get('override_settings') or active_settings['imgmodel']['payload'].get('override_settings')
        _ = await sd_api(endpoint='/sdapi/v1/options', method='post', json=model_data, retry=True)
        # Update size options for /image command
        current_avg = average_width_height(current_w, current_h)    # get current average width/height
        new_avg = average_width_height(new_w, new_h)                # get new average width/height
        if current_avg != new_avg:                                  # Update size options in menus if they are different
            await bg_task_queue.put(update_image_cmd_menus(new_avg))
    except Exception as e:
        logging.error(f"Error updating settings with the selected imgmodel data: {e}")

# Check filesize/filters with selected imgmodel to assume resolution / tags
async def guess_model_data(selected_imgmodel, presets):
    try:
        filename = selected_imgmodel.get('filename', None)
        if not filename:
            return ''
        # Check filesize of selected imgmodel to assume resolution and tags 
        file_size_bytes = os.path.getsize(filename)
        file_size_gb = file_size_bytes / (1024 ** 3)  # 1 GB = 1024^3 bytes
        match_counts = []
        for preset in presets:
            # no guessing needed for exact match
            exact_match = preset.pop('exact_match', '')
            if exact_match and selected_imgmodel.get('imgmodel_name') == exact_match:
                logging.info(f'Applying exact match imgmodel preset for "{exact_match}".')
                return preset
            # score presets by how close they match the selected imgmodel
            filter_list = preset.pop('filter', [])
            exclude_list = preset.pop('exclude', [])
            match_count = 0
            if filter_list:
                if all(re.search(re.escape(filter_text), filename, re.IGNORECASE) for filter_text in filter_list):
                    match_count += 1
                else:
                    match_count -= 1
            if exclude_list:
                if not any(re.search(re.escape(exclude_text), filename, re.IGNORECASE) for exclude_text in exclude_list):
                    match_count += 1
                else:
                    match_count -= 1
            if 'max_filesize' in preset and preset['max_filesize'] > file_size_gb:
                match_count += 1
                del preset['max_filesize']
            match_counts.append((preset, match_count))
        match_counts.sort(key=lambda x: x[1], reverse=True)  # Sort presets based on match counts
        matched_preset = match_counts[0][0] if match_counts else ''
        return matched_preset
    except Exception as e:
        logging.error(f"Error guessing selected imgmodel data: {e}")

async def merge_imgmodel_data(selected_imgmodel):
    try:
        selected_imgmodel_name = selected_imgmodel.get('imgmodel_name')
        ### IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
        # if selected_imgmodel_name == 'None': # Unloading model
        #     selected_imgmodel_tags = []
        #     return selected_imgmodel, selected_imgmodel_name, selected_imgmodel_tags
        # Get tags if defined
        selected_imgmodel_tags = None
        imgmodel_settings = {'payload': {}, 'override_settings': {}}
        imgmodels_data = load_file('ad_discordbot/dict_imgmodels.yaml')
        if imgmodels_data.get('settings', {}).get('auto_change_imgmodels', {}).get('guess_model_params', True):
            imgmodel_presets = copy.deepcopy(imgmodels_data.get('presets', []))
            matched_preset = await guess_model_data(selected_imgmodel, imgmodel_presets)
            if matched_preset:
                selected_imgmodel_tags = matched_preset.pop('tags', None)
                imgmodel_settings['payload'] = matched_preset.get('payload', {})
        imgmodel_settings['override_settings']['sd_model_checkpoint'] = selected_imgmodel['sd_model_checkpoint']
        imgmodel_settings['imgmodel_name'] = selected_imgmodel_name
        # Replace input dictionary
        selected_imgmodel = imgmodel_settings
        # Merge the selected imgmodel data with base imgmodel data
        selected_imgmodel = merge_base(selected_imgmodel, 'imgmodel')
        # Unpack any tag presets
        selected_imgmodel_tags = await update_tags(selected_imgmodel_tags)
        return selected_imgmodel, selected_imgmodel_name, selected_imgmodel_tags
    except Exception as e:
        logging.error(f"Error merging selected imgmodel data with base imgmodel data: {e}")

async def get_selected_imgmodel_data(selected_imgmodel_value):
    try:
        selected_imgmodel = {}
        ### IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
        # Unloading the current Img model
        # if selected_imgmodel_value == 'None':
        #     selected_imgmodel = {'override_settings': {'sd_model_checkpoint': 'None'}, 'imgmodel_name': 'None'}
        #     return selected_imgmodel
        # if selected_imgmodel_value == 'Exit':
        #      selected_imgmodel = {'imgmodel_name': 'None were selected'}
        #     return selected_imgmodel
        all_imgmodel_data = copy.deepcopy(all_imgmodels)
        for imgmodel in all_imgmodel_data:
            # check that the value matches a valid checkpoint
            if imgmodel.get('imgmodel_name') == selected_imgmodel_value:
                selected_imgmodel = {
                    "sd_model_checkpoint": imgmodel["sd_model_checkpoint"],
                    "imgmodel_name": imgmodel.get("imgmodel_name"),
                    "filename": imgmodel.get("filename", None)
                }
                break
        if not selected_imgmodel:
            logging.error(f'Img model not found: {selected_imgmodel_value}')
        return selected_imgmodel 
    except Exception as e:
        logging.error(f"Error getting selected imgmodel data: {e}")
        return {}

async def process_imgmodel(ctx, selected_imgmodel_value):
    try:
        if not selected_imgmodel_value:
            await ctx.reply('**No Img model was selected**.', ephemeral=True, delete_after=5)
            return
        await ireply(ctx, 'Img model change') # send a response msg to the user
        
        async with task_semaphore:
            # offload to ai_gen queue
            logging.info(f'{ctx.author} used "/imgmodel": "{selected_imgmodel_value}"')
            params = {'imgmodel': {'imgmodel_name': selected_imgmodel_value}}
            await change_imgmodel_task(ctx.author, ctx.channel, params)
            
    except Exception as e:
        logging.error(f"Error processing selected imgmodel from /imgmodel command: {e}")

if sd_enabled:
    all_imgmodels = []
    all_imgmodels = asyncio.run(fetch_imgmodels())

    if all_imgmodels:
        for imgmodel in all_imgmodels:
            if 'model_name' in imgmodel:
                imgmodel['imgmodel_name'] = imgmodel.pop('model_name')
        
        ### IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
        # unload_options = [app_commands.Choice(name="Unload Model", value="None"),
        # app_commands.Choice(name="Do Not Unload Model", value="Exit")]

        imgmodel_options = [app_commands.Choice(name=imgmodel["imgmodel_name"][:100], value=imgmodel["imgmodel_name"]) for imgmodel in all_imgmodels[:25]]
        imgmodel_options_label = f'{imgmodel_options[0].name[0]}-{imgmodel_options[-1].name[0]}'.lower()
        if len(all_imgmodels) > 25:
            imgmodel_options1 = [app_commands.Choice(name=imgmodel["imgmodel_name"][:100], value=imgmodel["imgmodel_name"]) for imgmodel in all_imgmodels[25:50]]
            imgmodel_options1_label = f'{imgmodel_options1[0].name[0]}-{imgmodel_options1[-1].name[0]}'.lower()
            if imgmodel_options1_label == imgmodel_options_label:
                imgmodel_options1_label = f'{imgmodel_options1_label}_1'
            if len(all_imgmodels) > 50:
                imgmodel_options2 = [app_commands.Choice(name=imgmodel["imgmodel_name"][:100], value=imgmodel["imgmodel_name"]) for imgmodel in all_imgmodels[50:75]]
                imgmodel_options2_label = f'{imgmodel_options2[0].name[0]}-{imgmodel_options2[-1].name[0]}'.lower()
                if imgmodel_options2_label == imgmodel_options_label or imgmodel_options2_label == imgmodel_options1_label:
                    imgmodel_options2_label = f'{imgmodel_options2_label}_2'
                if len(all_imgmodels) > 75:
                    imgmodel_options3 = [app_commands.Choice(name=imgmodel["imgmodel_name"][:100], value=imgmodel["imgmodel_name"]) for imgmodel in all_imgmodels[75:100]]
                    imgmodel_options3_label = f'{imgmodel_options3[0].name[0]}-{imgmodel_options3[-1].name[0]}'.lower()
                    if imgmodel_options3_label == imgmodel_options_label or imgmodel_options3_label == imgmodel_options1_label or imgmodel_options3_label == imgmodel_options2_label:
                        imgmodel_options3_label = f'{imgmodel_options2_label}_3'
                    if len(all_imgmodels) > 100:
                        all_imgmodels = all_imgmodels[:100]
                        logging.warning("'/imgmodel' command only allows up to 100 image models. Some models were omitted.")

        if len(all_imgmodels) <= 25:
            @client.hybrid_command(name="imgmodel", description='Choose an imgmodel')
            @app_commands.rename(imgmodels=f'imgmodels_{imgmodel_options_label}')
            @app_commands.describe(imgmodels=f'Imgmodels {imgmodel_options_label.upper()}')
            @app_commands.choices(imgmodels=imgmodel_options)
            async def imgmodel(ctx: discord.ext.commands.Context, imgmodels: typing.Optional[app_commands.Choice[str]]):
        # @app_commands.choices(unload=unload_options) ### IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
        # async def imgmodel(ctx: discord.ext.commands.Context, imgmodels: typing.Optional[app_commands.Choice[str]], unload: typing.Optional[app_commands.Choice[str]]):
        #     if imgmodels and unload:
        #         await ctx.send("More than one option was selected. Using the first selection.", ephemeral=True)
        #     selected_imgmodel = ((imgmodels or unload) and (imgmodels or unload).value) or ''
                selected_imgmodel = imgmodels.value if imgmodels is not None else ''
                await process_imgmodel(ctx, selected_imgmodel)

        elif 25 < len(all_imgmodels) <= 50:
            @client.hybrid_command(name="imgmodel", description='Choose an imgmodel (pick only one)')
            @app_commands.rename(models_1=f'imgmodels_{imgmodel_options_label}')
            @app_commands.describe(models_1=f'Imgmodels {imgmodel_options_label.upper()}')
            @app_commands.choices(models_1=imgmodel_options)
            @app_commands.rename(models_2=f'imgmodels_{imgmodel_options1_label}')
            @app_commands.describe(models_2=f'Imgmodels {imgmodel_options1_label.upper()}')
            @app_commands.choices(models_2=imgmodel_options1)
            async def imgmodel(ctx: discord.ext.commands.Context, models_1: typing.Optional[app_commands.Choice[str]], models_2: typing.Optional[app_commands.Choice[str]]):
        # @app_commands.choices(unload=unload_options) ### IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
        # async def imgmodel(ctx: discord.ext.commands.Context, models_1: typing.Optional[app_commands.Choice[str]], models_2: typing.Optional[app_commands.Choice[str]], unload: typing.Optional[app_commands.Choice[str]]):
        #     if sum(1 for v in (models_1, models_2, unload) if v) > 1:
                if models_1 and models_2:
                    await ctx.send("More than one option was selected. Using the first selection.", ephemeral=True)
        #     selected_imgmodel = ((models_1 or models_2 or unload) and (models_1 or models_2 or unload).value) or ''
                selected_imgmodel = ((models_1 or models_2) and (models_1 or models_2).value) or ''
                await process_imgmodel(ctx, selected_imgmodel)

        elif 50 < len(all_imgmodels) <= 75:
            @client.hybrid_command(name="imgmodel", description='Choose an imgmodel (pick only one)')
            @app_commands.rename(models_1=f'imgmodels_{imgmodel_options_label}')
            @app_commands.describe(models_1=f'Imgmodels {imgmodel_options_label.upper()}')
            @app_commands.choices(models_1=imgmodel_options)
            @app_commands.rename(models_2=f'imgmodels_{imgmodel_options1_label}')
            @app_commands.describe(models_2=f'Imgmodels {imgmodel_options1_label.upper()}')
            @app_commands.choices(models_2=imgmodel_options1)
            @app_commands.rename(models_3=f'imgmodels_{imgmodel_options2_label}')
            @app_commands.describe(models_3=f'Imgmodels {imgmodel_options2_label.upper()}')
            @app_commands.choices(models_3=imgmodel_options2)
            async def imgmodel(ctx: discord.ext.commands.Context, models_1: typing.Optional[app_commands.Choice[str]], models_2: typing.Optional[app_commands.Choice[str]], models_3: typing.Optional[app_commands.Choice[str]]):
        # @app_commands.choices(unload=unload_options) ### IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
        # async def imgmodel(ctx: discord.ext.commands.Context, models_1: typing.Optional[app_commands.Choice[str]], models_2: typing.Optional[app_commands.Choice[str]], models_3: typing.Optional[app_commands.Choice[str]], unload: typing.Optional[app_commands.Choice[str]]):
        #     if sum(1 for v in (models_1, models_2, models_3, unload) if v) > 1:
                if sum(1 for v in (models_1, models_2, models_3) if v) > 1:
                    await ctx.send("More than one option was selected. Using the first selection.", ephemeral=True)
        #     selected_imgmodel = ((models_1 or models_2 or models_3 or unload) and (models_1 or models_2 or models_3 or unload).value) or ''
                selected_imgmodel = ((models_1 or models_2 or models_3) and (models_1 or models_2 or models_3).value) or ''
                await process_imgmodel(ctx, selected_imgmodel)

        elif 75 < len(all_imgmodels) <= 100:
            @client.hybrid_command(name="imgmodel", description='Choose an imgmodel (pick only one)')
            @app_commands.rename(models_1=f'imgmodels_{imgmodel_options_label}')
            @app_commands.describe(models_1=f'Imgmodels {imgmodel_options_label.upper()}')
            @app_commands.choices(models_1=imgmodel_options)
            @app_commands.rename(models_2=f'imgmodels_{imgmodel_options1_label}')
            @app_commands.describe(models_2=f'Imgmodels {imgmodel_options1_label.upper()}')
            @app_commands.choices(models_2=imgmodel_options1)
            @app_commands.rename(models_3=f'imgmodels_{imgmodel_options2_label}')
            @app_commands.describe(models_3=f'Imgmodels {imgmodel_options2_label.upper()}')
            @app_commands.choices(models_3=imgmodel_options2)
            @app_commands.rename(models_4=f'imgmodels_{imgmodel_options3_label}')
            @app_commands.describe(models_4=f'Imgmodels {imgmodel_options3_label.upper()}')
            @app_commands.choices(models_4=imgmodel_options3)
            async def imgmodel(ctx: discord.ext.commands.Context, models_1: typing.Optional[app_commands.Choice[str]], models_2: typing.Optional[app_commands.Choice[str]], models_3: typing.Optional[app_commands.Choice[str]], models_4: typing.Optional[app_commands.Choice[str]]):
        # @app_commands.choices(unload=unload_options) ### IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
        # async def imgmodel(ctx: discord.ext.commands.Context, models_1: typing.Optional[app_commands.Choice[str]], models_2: typing.Optional[app_commands.Choice[str]], models_3: typing.Optional[app_commands.Choice[str]], models_4: typing.Optional[app_commands.Choice[str]], unload: typing.Optional[app_commands.Choice[str]]):
        #     if sum(1 for v in (models_1, models_2, models_3, models_4, unload) if v) > 1:
                if sum(1 for v in (models_1, models_2, models_3, models_4) if v) > 1:
                    await ctx.send("More than one option was selected. Using the first selection.", ephemeral=True)
        #     selected_imgmodel = ((models_1 or models_2 or models_3 or models_4 or unload) and (models_1 or models_2 or models_3 or models_4 or unload).value) or ''
                selected_imgmodel = ((models_1 or models_2 or models_3 or models_4) and (models_1 or models_2 or models_3 or models_4).value) or ''
                await process_imgmodel(ctx, selected_imgmodel)

#################################################################
####################### /LLMMODEL COMMAND #######################
#################################################################
# Process selected LLM model
async def process_llmmodel(ctx, selected_llmmodel):
    try:
        if not selected_llmmodel:
            await ctx.reply('**No LLM model was selected**.', ephemeral=True, delete_after=5)
            return
        await ireply(ctx, 'LLM model change') # send a response msg to the user
        
        async with task_semaphore:
            # offload to ai_gen queue
            logging.info(f'{ctx.author} used "/llmmodel": "{selected_llmmodel}"')
            params = {'llmmodel': {'llmmodel_name': selected_llmmodel, 'verb': 'Changing', 'mode': 'change'}}
            await change_llmmodel_task(ctx.author, ctx.channel, params)
            
    except Exception as e:
        logging.error(f"Error processing /llmmodel command: {e}")

if textgenwebui_enabled and all_llmmodels:
    llmmodel_options = [app_commands.Choice(name=llmmodel[:100], value=llmmodel) for llmmodel in all_llmmodels[:25]]
    llmmodel_options_label = f'{llmmodel_options[1].name[0]}-{llmmodel_options[-1].name[0]}'.lower() # Using second "Name" since first name is "None"
    if len(all_llmmodels) > 25:
        llmmodel_options1 = [app_commands.Choice(name=llmmodel[:100], value=llmmodel) for llmmodel in all_llmmodels[25:50]]
        llmmodel_options1_label = f'{llmmodel_options1[0].name[0]}-{llmmodel_options1[-1].name[0]}'.lower()
        if llmmodel_options1_label == llmmodel_options_label:
            llmmodel_options1_label = f'{llmmodel_options1_label}_1'
        if len(all_llmmodels) > 50:
            llmmodel_options2 = [app_commands.Choice(name=llmmodel[:100], value=llmmodel) for llmmodel in all_llmmodels[50:75]]
            llmmodel_options2_label = f'{llmmodel_options2[0].name[0]}-{llmmodel_options2[-1].name[0]}'.lower()
            if llmmodel_options2_label == llmmodel_options_label or llmmodel_options2_label == llmmodel_options1_label:
                llmmodel_options2_label = f'{llmmodel_options2_label}_2'
            if len(all_llmmodels) > 75:
                llmmodel_options3 = [app_commands.Choice(name=llmmodel[:100], value=llmmodel) for llmmodel in all_llmmodels[75:100]]
                llmmodel_options3_label = f'{llmmodel_options3[0].name[0]}-{llmmodel_options3[-1].name[0]}'.lower()
                if llmmodel_options3_label == llmmodel_options_label or llmmodel_options3_label == llmmodel_options1_label or llmmodel_options3_label == llmmodel_options2_label:
                    llmmodel_options3_label = f'{llmmodel_options3_label}_3'
                if len(all_llmmodels) > 100:
                    all_llmmodels = all_llmmodels[:100]
                    logging.warning("'/llmmodel' command only allows up to 100 LLM models. Some models were omitted.")

    if len(all_llmmodels) <= 25:
        @client.hybrid_command(name="llmmodel", description='Choose an LLM model')
        @app_commands.rename(llmmodels=f'llm-models_{llmmodel_options_label}')
        @app_commands.describe(llmmodels=f'LLM models {llmmodel_options_label.upper()}')
        @app_commands.choices(llmmodels=llmmodel_options)
        async def llmmodel(ctx: discord.ext.commands.Context, llmmodels: typing.Optional[app_commands.Choice[str]]):
            selected_llmmodel = llmmodels.value if llmmodels is not None else ''
            await process_llmmodel(ctx, selected_llmmodel)

    elif 25 < len(all_llmmodels) <= 50:
        @client.hybrid_command(name="llmmodel", description='Choose an LLM model (pick only one)')
        @app_commands.rename(models_1=f'llm-models_{llmmodel_options_label}')
        @app_commands.describe(models_1=f'LLM models {llmmodel_options_label.upper()}')
        @app_commands.choices(models_1=llmmodel_options)
        @app_commands.rename(models_2=f'llm-models_{llmmodel_options1_label}')
        @app_commands.describe(models_2=f'LLM models {llmmodel_options1_label.upper()}')
        @app_commands.choices(models_2=llmmodel_options1)
        async def llmmodel(ctx: discord.ext.commands.Context, models_1: typing.Optional[app_commands.Choice[str]], models_2: typing.Optional[app_commands.Choice[str]]):
            if models_1 and models_2:
                await ctx.send("More than one LLM model was selected. Using the first selection.", ephemeral=True)
            selected_llmmodel = ((models_1 or models_2) and (models_1 or models_2).value) or ''
            await process_llmmodel(ctx, selected_llmmodel)

    elif 50 < len(all_llmmodels) <= 75:
        @client.hybrid_command(name="llmmodel", description='Choose an LLM model (pick only one)')
        @app_commands.rename(models_1=f'llm-models_{llmmodel_options_label}')
        @app_commands.describe(models_1=f'LLM models {llmmodel_options_label.upper()}')
        @app_commands.choices(models_1=llmmodel_options)
        @app_commands.rename(models_2=f'llm-models_{llmmodel_options1_label}')
        @app_commands.describe(models_2=f'LLM models {llmmodel_options1_label.upper()}')
        @app_commands.choices(models_2=llmmodel_options1)
        @app_commands.rename(models_3=f'llm-models_{llmmodel_options2_label}')
        @app_commands.describe(models_3=f'LLM models {llmmodel_options2_label.upper()}')
        @app_commands.choices(models_3=llmmodel_options2)
        async def llmmodel(ctx: discord.ext.commands.Context, models_1: typing.Optional[app_commands.Choice[str]], models_2: typing.Optional[app_commands.Choice[str]], models_3: typing.Optional[app_commands.Choice[str]]):
            if sum(1 for v in (models_1, models_2, models_3) if v) > 1:
                await ctx.send("More than one LLM model was selected. Using the first selection.", ephemeral=True)
            selected_llmmodel = ((models_1 or models_2 or models_3) and (models_1 or models_2 or models_3).value) or ''
            await process_llmmodel(ctx, selected_llmmodel)

    elif 75 < len(all_llmmodels) <= 100:
        @client.hybrid_command(name="llmmodel", description='Choose an LLM model (pick only one)')
        @app_commands.rename(models_1=f'llm-models_{llmmodel_options_label}')
        @app_commands.describe(models_1=f'LLM models {llmmodel_options_label.upper()}')
        @app_commands.choices(models_1=llmmodel_options)
        @app_commands.rename(models_2=f'llm-models_{llmmodel_options1_label}')
        @app_commands.describe(models_2=f'LLM models {llmmodel_options1_label.upper()}')
        @app_commands.choices(models_2=llmmodel_options1)
        @app_commands.rename(models_3=f'llm-models_{llmmodel_options2_label}')
        @app_commands.describe(models_3=f'LLM models {llmmodel_options2_label.upper()}')
        @app_commands.choices(models_3=llmmodel_options2)
        @app_commands.rename(models_4=f'llm-models_{llmmodel_options3_label}')
        @app_commands.describe(models_4=f'LLM models {llmmodel_options3_label.upper()}')
        @app_commands.choices(models_4=llmmodel_options3)
        async def llmmodel(ctx: discord.ext.commands.Context, models_1: typing.Optional[app_commands.Choice[str]], models_2: typing.Optional[app_commands.Choice[str]], models_3: typing.Optional[app_commands.Choice[str]], models_4: typing.Optional[app_commands.Choice[str]]):
            if sum(1 for v in (models_1, models_2, models_3, models_4) if v) > 1:
                await ctx.send("More than one LLM model was selected. Using the first selection.", ephemeral=True)
            selected_llmmodel = ((models_1 or models_2 or models_3 or models_4) and (models_1 or models_2 or models_3 or models_4).value) or ''
            await process_llmmodel(ctx, selected_llmmodel)

#################################################################
####################### /SPEAK COMMAND #######################
#################################################################
async def process_speak_silero_non_eng(ctx, lang):
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
            await ctx.send(f'Could not determine the correct voice and model ID for language "{lang}". Defaulting to English.', ephemeral=True)
            tts_args = {'silero_tts': {'language': 'English', 'speaker': 'en_1'}}
    except Exception as e:
        logging.error(f"Error processing non-English voice for silero_tts: {e}")
        await ctx.send(f"Error processing non-English voice for silero_tts: {e}", ephemeral=True)
    return tts_args

async def process_speak_args(ctx, selected_voice=None, lang=None, user_voice=None):
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
                tts_args = await process_speak_silero_non_eng(ctx, lang) # returns complete args for silero_tts
                if selected_voice: await ctx.send(f'Currently, non-English languages will use a default voice (not using "{selected_voice}")', ephemeral=True)
        elif tts_client in last_extension_params and tts_voice_key in last_extension_params[tts_client]:
            pass # Default to voice in last_extension_params
        elif f'{tts_client}-{tts_voice_key}' in shared.settings:
            pass # Default to voice in shared.settings
        else:
            await ctx.send("No voice was selected or provided, and a default voice was not found. Request will probably fail...", ephemeral=True)
        return tts_args
    except Exception as e:
        logging.error(f"Error processing tts options: {e}")
        await ctx.send(f"Error processing tts options: {e}", ephemeral=True)

async def convert_and_resample_mp3(ctx, mp3_file, output_directory=None):
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
        logging.info(f'User provided file "{mp3_file}" was converted to .wav for "/speak" command')
        return wav_path
    except Exception as e:
        logging.error(f"Error converting user's .mp3 to .wav: {e}")
        await ctx.send("An error occurred while processing the voice file.", ephemeral=True)
    finally:
        if mp3_file: os.remove(mp3_file)

async def process_user_voice(ctx, voice_input=None):
    try:
        if not (voice_input and getattr(voice_input, 'content_type', '').startswith("audio/")):
            return ''
        if tts_client != 'alltalk_tts' and tts_client != 'coqui_tts':
            await ctx.send("Sorry, current tts extension does not allow using a voice attachment (only works for 'alltalk_tts' and 'coqui_tts)", ephemeral=True)
            return ''
        voiceurl = voice_input.url
        voiceurl_without_params = voiceurl.split('?')[0]
        if not voiceurl_without_params.endswith((".wav", ".mp3")):
            await ctx.send("Invalid audio format. Please try again with a WAV or MP3 file.", ephemeral=True)
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
                    await ctx.send("Error downloading your audio file. Please try again.", ephemeral=True)
                    return ''
        if voice_data_ext == '.mp3':
            try:
                user_voice = await convert_and_resample_mp3(ctx, user_voice, output_directory=None)
            except:
                if user_voice: os.remove(user_voice)
        return user_voice
    except Exception as e:
        logging.error(f"Error processing user provided voice file: {e}")
        await ctx.send("An error occurred while processing the voice file.", ephemeral=True)

async def process_speak(ctx, input_text, selected_voice=None, lang=None, voice_input=None):
    try:
        if not (selected_voice or voice_input):
            await ctx.reply('**No voice was selected**.', ephemeral=True, delete_after=5)
            return
        user_voice = await process_user_voice(ctx, voice_input)
        tts_args = await process_speak_args(ctx, selected_voice, lang, user_voice)
        await ireply(ctx, 'tts') # send a response msg to the user
        
        async with task_semaphore:
            async with ctx.channel.typing():
                # offload to ai_gen queue
                logging.info(f'{ctx.author} used "/speak": "{input_text}"')
                params = {'tts_args': tts_args, 'user_voice': user_voice}
                await speak_task(ctx.author, ctx.channel, input_text, params)
                await run_flow_if_any(ctx.author, ctx.channel, 'speak', input_text)
            
    except Exception as e:
        logging.error(f"Error processing tts request: {e}")
        await ctx.send(f"Error processing tts request: {e}", ephemeral=True)

async def fetch_speak_options():
    try:
        lang_list = []
        all_voicess = []
        if tts_client == 'coqui_tts' or tts_client == 'alltalk_tts':
            lang_list = ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French', 'German', 'Hungarian', 'Italian', 'Japanese', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Spanish', 'Turkish']
            if tts_client == 'coqui_tts':
                from extensions.coqui_tts.script import get_available_voices
            elif tts_client == 'alltalk_tts':
                from extensions.alltalk_tts.script import get_available_voices
            all_voices = get_available_voices()
        elif tts_client == 'silero_tts':
            lang_list = ['English', 'Spanish', 'French', 'German', 'Russian', 'Tatar', 'Ukranian', 'Uzbek', 'English (India)', 'Avar', 'Bashkir', 'Bulgarian', 'Chechen', 'Chuvash', 'Kalmyk', 'Karachay-Balkar', 'Kazakh', 'Khakas', 'Komi-Ziryan', 'Mari', 'Nogai', 'Ossetic', 'Tuvinian', 'Udmurt', 'Yakut']
            logging.warning('''There's too many Voice/language permutations to make them all selectable in "/speak" command. Loading a bunch of English options. Non-English languages will automatically play using respective default speaker.''')
            all_voices = [f"en_{index}" for index in range(1, 76)] # will just include English voices in select menus. Other languages will use defaults.
        elif tts_client == 'elevenlabs_tts':
            lang_list = ['English', 'German', 'Polish', 'Spanish', 'Italian', 'French', 'Portuegese', 'Hindi', 'Arabic']
            logging.info('''Getting list of available voices for elevenlabs_tts for "/speak" command...''')
            from extensions.elevenlabs_tts.script import refresh_voices, update_api_key
            if tts_api_key:
                update_api_key(tts_api_key)
            all_voices = refresh_voices()
        all_voices.sort() # Sort alphabetically
        return lang_list, all_voices
    except Exception as e:
        logging.error(f"Error building options for '/speak' command: {e}")

if textgenwebui_enabled and tts_client and tts_client in supported_tts_clients:
    lang_list, all_voices = asyncio.run(fetch_speak_options())
    voice_options = [app_commands.Choice(name=voice_name.replace('_', ' ').title(), value=f'{voice_name}') for voice_name in all_voices[:25]]
    voice_options_label = f'{voice_options[0].name[0]}-{voice_options[-1].name[0]}'.lower()
    if len(all_voices) > 25:
        voice_options1 = [app_commands.Choice(name=voice_name.replace('_', ' ').title(), value=f'{voice_name}') for voice_name in all_voices[25:50]]
        voice_options1_label = f'{voice_options1[0].name[0]}-{voice_options1[-1].name[0]}'.lower()
        if voice_options1_label == voice_options_label:
            voice_options1_label = f'{voice_options1_label}_1'
        if len(all_voices) > 50:
            voice_options2 = [app_commands.Choice(name=voice_name.replace('_', ' ').title(), value=f'{voice_name}') for voice_name in all_voices[50:75]]
            voice_options2_label = f'{voice_options2[0].name[0]}-{voice_options2[-1].name[0]}'.lower()
            if voice_options2_label == voice_options_label or voice_options2_label == voice_options1_label:
                voice_options2_label = f'{voice_options2_label}_2'
            if len(all_voices) > 75:
                all_voices = all_voices[:75]
                logging.warning("'/speak' command only allows up to 75 voices. Some voices were omitted.")
    if lang_list: lang_options = [app_commands.Choice(name=lang, value=lang) for lang in lang_list]
    else: lang_options = [app_commands.Choice(name='English', value='English')] # Default to English

    if len(all_voices) <= 25:
        @client.hybrid_command(name="speak", description='AI will speak your text using a selected voice')
        @app_commands.rename(voice=f'voices_{voice_options_label}')
        @app_commands.describe(voice=f'Voices {voice_options_label.upper()}')
        @app_commands.choices(voice=voice_options)
        @app_commands.choices(lang=lang_options)
        async def speak(ctx: discord.ext.commands.Context, input_text: str, voice: typing.Optional[app_commands.Choice[str]], lang: typing.Optional[app_commands.Choice[str]], voice_input: typing.Optional[discord.Attachment]):
            selected_voice = voice.value if voice is not None else ''
            voice_input = voice_input if voice_input is not None else ''
            lang = lang.value if lang is not None else ''
            await process_speak(ctx, input_text, selected_voice, lang, voice_input)

    elif 25 < len(all_voices) <= 50:
        @client.hybrid_command(name="speak", description='AI will speak your text using a selected voice (pick only one)')
        @app_commands.rename(voice_1=f'voices_{voice_options_label}')
        @app_commands.describe(voice_1=f'Voices {voice_options_label.upper()}')
        @app_commands.choices(voice_1=voice_options)
        @app_commands.rename(voice_2=f'voices_{voice_options1_label}')
        @app_commands.describe(voice_2=f'Voices {voice_options1_label.upper()}')
        @app_commands.choices(voice_2=voice_options1)
        @app_commands.choices(lang=lang_options)
        async def speak(ctx: discord.ext.commands.Context, input_text: str, voice_1: typing.Optional[app_commands.Choice[str]], voice_2: typing.Optional[app_commands.Choice[str]], lang: typing.Optional[app_commands.Choice[str]], voice_input: typing.Optional[discord.Attachment]):
            if voice_1 and voice_2:
                await ctx.send("A voice was picked from two separate menus. Using the first selection.", ephemeral=True)
            selected_voice = ((voice_1 or voice_2) and (voice_1 or voice_2).value) or ''
            voice_input = voice_input if voice_input is not None else ''
            lang = lang.value if lang is not None else ''
            await process_speak(ctx, input_text, selected_voice, lang, voice_input)

    elif 50 < len(all_voices) <= 75:
        @client.hybrid_command(name="speak", description='AI will speak your text using a selected voice (pick only one)')
        @app_commands.rename(voice_1=f'voices_{voice_options_label}')
        @app_commands.describe(voice_1=f'Voices {voice_options_label.upper()}')
        @app_commands.choices(voice_1=voice_options)
        @app_commands.rename(voice_2=f'voices_{voice_options1_label}')
        @app_commands.describe(voice_2=f'Voices {voice_options1_label.upper()}')
        @app_commands.choices(voice_2=voice_options1)
        @app_commands.rename(voice_3=f'voices_{voice_options2_label}')
        @app_commands.describe(voice_3=f'Voices {voice_options2_label.upper()}')
        @app_commands.choices(voice_3=voice_options2)
        @app_commands.choices(lang=lang_options)
        async def speak(ctx: discord.ext.commands.Context, input_text: str, voice_1: typing.Optional[app_commands.Choice[str]], voice_2: typing.Optional[app_commands.Choice[str]], voice_3: typing.Optional[app_commands.Choice[str]], lang: typing.Optional[app_commands.Choice[str]], voice_input: typing.Optional[discord.Attachment]):
            if sum(1 for v in (voice_1, voice_2, voice_3) if v) > 1:
                await ctx.send("A voice was picked from two separate menus. Using the first selection.", ephemeral=True)
            selected_voice = ((voice_1 or voice_2 or voice_3) and (voice_1 or voice_2 or voice_3).value) or ''
            voice_input = voice_input if voice_input is not None else ''
            lang = lang.value if lang is not None else ''
            await process_speak(ctx, input_text, selected_voice, lang, voice_input)

#################################################################
####################### DEFAULT SETTINGS ########################
#################################################################
# Sub-classes under a main class 'Settings'
class Behavior:
    def __init__(self):
        self.reply_to_itself = 0.0
        self.chance_to_reply_to_other_bots = 0.5
        self.reply_to_bots_when_adressed = 0.3
        self.only_speak_when_spoken_to = True
        self.ignore_parentheses = True
        self.go_wild_in_channel = True
        self.conversation_recency = 600
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
        if i.mention_everyone or (i.author == client.user and not self.probability_to_reply(self.reply_to_itself)):
            return False
        # Whether to reply to other bots
        if i.author.bot and bot_settings.database.last_character.lower() in text.lower() and i.channel.id in bot_settings.database.main_channels:
            if 'bye' in text.lower(): # don't reply if another bot is saying goodbye
                return False
            return self.probability_to_reply(self.reply_to_bots_when_adressed)
        # Whether to reply when text is nested in parentheses
        if self.ignore_parentheses and (i.content.startswith('(') and i.content.endswith(')')) or (i.content.startswith('<:') and i.content.endswith(':>')):
            return False
        # Whether to reply if only speak when spoken to
        if (self.only_speak_when_spoken_to and (client.user.mentioned_in(i) or any(word in i.content.lower() for word in bot_settings.database.last_character.lower().split()))) \
            or (self.in_active_conversation(i.author.id) and i.channel.id in bot_settings.database.main_channels):
            return True
        reply = False
        # few more conditions
        if i.author.bot and i.channel.id in bot_settings.database.main_channels:
            reply = self.probability_to_reply(self.chance_to_reply_to_other_bots)
        if self.go_wild_in_channel and i.channel.id in bot_settings.database.main_channels:
            reply = True
        if reply:
            self.update_user_dict(i.author.id)
        return reply

    def probability_to_reply(self, probability):
        # Determine if the bot should reply based on a probability
        return random.random() < probability

class ImgModel:
    def __init__(self):
        self.imgmodel_name = '' # label used for /imgmodel command
        self.override_settings = {}
        self.img_payload = {
            'alwayson_scripts': {
                'controlnet': {
                    'args': [{'enabled': False, 'image': None, 'mask_image': None, 'model': 'None', 'module': 'None', 'weight': 1.0, 'processor_res': 64, 'pixel_perfect': True, 'guidance_start': 0.0, 'guidance_end': 1.0, 'threshold_a': 64, 'threshold_b': 64, 'control_mode': 0, 'resize_mode': 1, 'lowvram': False, 'save_detected_map': False}]
                },
                'layerdiffuse': {
                    'args': {'enabled': False, 'method': '(SDXL) Only Generate Transparent Image (Attention Injection)', 'weight': 1.0, 'stop_at': 1.0, 'foreground': None, 'background': None, 'blending': None, 'resize_mode': 'Crop and Resize', 'output_mat_for_i2i': False, 'fg_prompt': '', 'bg_prompt': '', 'blended_prompt': ''}
                },
                'forge_couple': {
                    'args': {'enable': False, 'mode': 'Basic', 'sep': 'SEP', 'direction': 'Horizontal', 'global_effect': 'First Line', 'global_weight': 0.5, 'maps': [['0:0.5', '0.0:1.0', '1.0'],['0.5:1.0', '0.0:1.0', '1.0']]}
                },                
                'reactor': {
                    'args': {'image': '', 'enabled': False, 'source_faces': '0', 'target_faces': '0', 'model': 'inswapper_128.onnx', 'restore_face': 'CodeFormer', 'restore_visibility': 1, 'restore_upscale': True, 'upscaler': '4x_NMKD-Superscale-SP_178000_G', 'scale': 1.5, 'upscaler_visibility': 1, 'swap_in_source_img': False, 'swap_in_gen_img': True, 'log_level': 1, 'gender_detect_source': 0, 'gender_detect_target': 0, 'save_original': False, 'codeformer_weight': 0.8, 'source_img_hash_check': False, 'target_img_hash_check': False, 'system': 'CUDA', 'face_mask_correction': True, 'source_type': 0, 'face_model': '', 'source_folder': '', 'multiple_source_images': None, 'random_img': True, 'force_upscale': True, 'threshold': 0.6, 'max_faces': 2}
                }
            }
        }
        self.tags = []

class LLMContext:
    def __init__(self):
        self.context = 'The following is a conversation with an AI Large Language Model. The AI has been trained to answer questions, provide recommendations, and help with decision making. The AI follows user requests. The AI thinks outside the box.'
        self.extensions = {}
        self.greeting = '' # 'How can I help you today?'
        self.name = 'AI'
        self.use_voice_channel = False
        self.bot_in_character_menu = True

class LLMState:
    def __init__(self):
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
            'custom_system_message': '',
            'custom_token_bans': '',
            'do_sample': True,
            'dynamic_temperature': False,
            'dynatemp_low': 1,
            'dynatemp_high': 1,
            'dynatemp_exponent': 1,
            'encoder_repetition_penalty': 1,
            'epsilon_cutoff': 0,
            'eta_cutoff': 0,
            'frequency_penalty': 0,
            'greeting': '',
            'guidance_scale': 1,
            'history': {'internal': [], 'visible': []},
            'max_new_tokens': 512,
            'max_tokens_second': 0,
            'max_updates_second': 0,
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
            'penalty_alpha': 0,
            'presence_penalty': 0,
            'prompt_lookup_num_tokens': 0,
            'repetition_penalty': 1.18,
            'repetition_penalty_range': 1024,
            'sampler_priority': [],
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
            'user_bio': '',
            'chat_template_str': '''{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{- name1 + ': ' + message['content'] + '\\n'-}}\n        {%- else -%}\n            {{- name2 + ': ' + message['content'] + '\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}''',
            'instruction_template_str': '''{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not ns.found -%}\n    {{- '' + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + '\\n\\n' -}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- '' + message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{-'### Instruction:\\n' + message['content'] + '\\n\\n'-}}\n        {%- else -%}\n            {{-'### Response:\\n' + message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{-'### Response:\\n'-}}\n{%- endif -%}''',
            'chat-instruct_command': '''Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>'''
            }
        self.regenerate = False
        self._continue = False

# Holds the default value of all sub-classes.
class Settings:
    def __init__(self):
        self.behavior = Behavior()
        self.imgmodel = ImgModel()
        self.llmcontext = LLMContext()
        self.llmstate = LLMState()

    # Returns the value of Settings as a dictionary
    def settings_to_dict(self):
        return {
            'behavior': vars(self.behavior),
            'imgmodel': vars(self.imgmodel),
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
        self.last_character = self.initialize_last_setting('last_character')
        self.last_change = self.initialize_last_time('last_change')
        self.last_user_msg = self.initialize_last_time('last_user_msg')
        self.main_channels = self.initialize_main_channels()
        self.warned_once = self.initialize_warned_once()

    def initialize_first_run(self):
        with sqlite3.connect('bot.db') as conn:
            c = conn.cursor()
            c.execute('''SELECT name FROM sqlite_master WHERE type='table' AND name='first_run' ''')
            is_first_run_table_exists = c.fetchone()
            if not is_first_run_table_exists:
                c.execute('''CREATE TABLE IF NOT EXISTS first_run (is_first_run BOOLEAN)''')
                c.execute('''INSERT INTO first_run (is_first_run) VALUES (1)''')
                conn.commit()
                return True
            c.execute('''SELECT COUNT(*) FROM first_run''')
            is_first_run_exists = c.fetchone()[0]
            return is_first_run_exists == 0

    def initialize_last_setting(self, location):
        try:
            with sqlite3.connect('bot.db') as conn:
                c = conn.cursor()
                c.execute(f'''CREATE TABLE IF NOT EXISTS {location} (setting TEXT)''')
        except Exception as e:
            logging.error(f"Error initializing {location}: {e}")

    def update_last_setting(self, value, location):
        try:
            with sqlite3.connect('bot.db') as conn:
                c = conn.cursor()
                c.execute(f'''UPDATE {location} SET setting = ?''', (value,))
                conn.commit()
            setattr(self, location, value)
        except Exception as e:
            logging.error(f"An error occurred while logging '{value}' to '{location}' in bot.db: {e}")

    def initialize_last_time(self, location='last_change'):
        try:
            with sqlite3.connect('bot.db') as conn:
                c = conn.cursor()
                c.execute(f'''CREATE TABLE IF NOT EXISTS {location} (timestamp TEXT)''')
                c.execute(f'''SELECT timestamp FROM {location}''')
                timestamp = c.fetchone()
                if timestamp is None or timestamp[0] is None:
                    now = datetime.now()
                    formatted_now = now.strftime('%Y-%m-%d %H:%M:%S')
                    if timestamp is None:
                        c.execute(f'''INSERT INTO {location} (timestamp) VALUES (?)''', (formatted_now,))
                    else:
                        c.execute(f'''UPDATE {location} SET timestamp = ?''', (formatted_now,))
                    conn.commit()
                    return formatted_now
            return timestamp[0] if timestamp else None
        except Exception as e:
            logging.error(f"Error initializing {location}: {e}")

    def update_last_time(self, location='last_change'):
        try:
            with sqlite3.connect('bot.db') as conn:
                c = conn.cursor()
                now = datetime.now()
                formatted_now = now.strftime('%Y-%m-%d %H:%M:%S')
                c.execute(f'''UPDATE {location} SET timestamp = ?''', (formatted_now,))
                conn.commit()
            setattr(self, location, formatted_now)
        except Exception as e:
            logging.error(f"An error occurred while logging time of profile update to bot.db: {e}")

    def initialize_main_channels(self):
        with sqlite3.connect('bot.db') as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS main_channels (channel_id TEXT UNIQUE)''')
            c.execute('''SELECT channel_id FROM main_channels''')
            result = [int(row[0]) for row in c.fetchall()]
            return result if result else []

    def initialize_warned_once(self):
        with sqlite3.connect('bot.db') as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS warned_once (flag_name TEXT UNIQUE, value INTEGER)''')
            flags_to_insert = [('loractl', 0), ('char_tts', 0), ('no_llmmodel', 0), ('forgecouple', 0), ('layerdiffuse', 0), ('dynaprompt', 0)]
            for flag_name, value in flags_to_insert:
                c.execute('''INSERT OR REPLACE INTO warned_once (flag_name, value) VALUES (?, ?)''', (flag_name, value))
            conn.commit()

    def was_warned(self, flag_name):
        with sqlite3.connect('bot.db') as conn:
            c = conn.cursor()
            c.execute('''SELECT value FROM warned_once WHERE flag_name = ?''', (flag_name,))
            result = c.fetchone()
            if result:
                return result[0]
            else:
                return None

    def update_was_warned(self, flag_name, value):
        with sqlite3.connect('bot.db') as conn:
            c = conn.cursor()
            c.execute('''UPDATE warned_once SET value = ? WHERE flag_name = ?''', (value, flag_name))
            conn.commit()

class BotSettings:
    def __init__(self):
        self.defaults = Settings() # Instance of the default settings
        self.defaults = self.defaults.settings_to_dict() # Convert instance to dict
        self.active_settings = load_file('ad_discordbot/activesettings.yaml')
        self.active_settings = dict(self.active_settings)
        self.settings = fix_dict(self.active_settings, self.defaults)
        self.behavior = Behavior()
        self.behavior.update_behavior_dict(self.active_settings['behavior'])
        self.database = Database()

    def get_settings_dict(self):
        return self.settings

bot_settings = BotSettings()

class ChatHistoryManager:
    def __init__(self):
        self.limit_history = config.get('textgenwebui', {}).get('chat_history', {}).get('limit_history', True)
        self.truncation = int(bot_settings.settings['llmstate']['state']['truncation_length'] * 4) if self.limit_history else None
        self.autosave_history = config.get('textgenwebui', {}).get('chat_history', {}).get('autosave_history', False)
        self.unique_id = None
        self.session_history = {'internal': [], 'visible': []}
        self.recent_messages = {'user': [], 'llm': []}
        self.collected_prompt = ''

    # Loads most recent history for current character
    def load_history(self, state):
        self.session_history = load_latest_history(state)
        last_exchange = self.session_history['visible'][-1] if self.session_history['visible'] else None
        if last_exchange:
            last_user_message = last_exchange[0]
            last_assistant_message = last_exchange[1]
            logging.info(f'Loaded most recent chat history. Last message exchange:\n User: "{last_user_message}"\n {bot_settings.database.last_character}: "{last_assistant_message}"')
        else:
            logging.info("Starting new conversation.")
        all_histories = find_all_histories(state)
        if len(all_histories) > 0:
            self.unique_id = all_histories[0]

    # Save history to a new file
    def save_history(self, auto_id=None):
        state = bot_settings.settings['llmstate']['state']
        if not auto_id:
            self.unique_id = datetime.now().strftime('%Y%m%d-%H-%M-%S')
            logging.info(f'''Chat history was saved to "/logs/{state['mode']}/{bot_settings.database.last_character}/{self.unique_id}.json"''')
        save_history(self.session_history, self.unique_id, state['character_menu'], state['mode'])

    # Retain most recent 10 elements or 10,000 characters from user prompts and bot replies
    def manage_recent_messages(self, prompt, reply=None):
        if prompt:
            self.recent_messages['user'].insert(0, prompt)
            while len(self.recent_messages['user']) > 10 or sum(len(message) for message in self.recent_messages['user']) > 5000:
                oldest_message = self.recent_messages['user'].pop()
        if reply:
            self.recent_messages['llm'].insert(0, reply)
            while len(self.recent_messages['llm']) > 10 or sum(len(message) for message in self.recent_messages['llm']) > 5000:
                oldest_message = self.recent_messages['llm'].pop()
    
    # Manage the session history for textgen-webui
    def manage_prompt_history(self, prompt, reply=None, save_to_history=True):
        if reply is None:                       # Only prompt is received
            if self.collected_prompt:           # If there are previously collected prompts
                self.collected_prompt += '\n\n' + prompt
            else:
                self.collected_prompt = prompt
        else:                                   # Both prompt and reply are received
            if self.collected_prompt:           # If there are previously collected prompts
                prompt = self.collected_prompt + '\n\n' + prompt
                self.collected_prompt = ''      # Reset collected prompts
            if save_to_history:
                self.session_history['internal'].append([prompt, reply])
                self.session_history['visible'].append([prompt, reply])
        if self.truncation:
            while sum(len(message) for exchange in self.session_history['internal'] for message in exchange) > self.truncation:
                oldest_exchange = self.session_history['internal'].pop(0)
            while sum(len(message) for exchange in self.session_history['visible'] for message in exchange) > self.truncation:
                oldest_exchange = self.session_history['visible'].pop(0)
        if self.autosave_history:
            self.save_history(auto_id=self.unique_id)

    def manage_history(self, prompt, reply=None, save_to_history=True):
        self.manage_recent_messages(prompt, reply)
        self.manage_prompt_history(prompt, reply, save_to_history)

    def reset_session_history(self):
        self.session_history = {'internal': [], 'visible': []}
        logging.info("Starting new conversation.")

history_manager = ChatHistoryManager()

client.run(bot_token, root_logger=True, log_handler=handler)