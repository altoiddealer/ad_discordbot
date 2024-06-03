from ad_discordbot.modules.logs import import_track, log, get_logger, log_file_handler, log_file_formatter; import_track(__file__, fp=True)
logging = get_logger(__name__)
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import random
import json
import re
import glob
import os
import warnings
import discord
from discord.ext import commands
from discord import app_commands, File
import typing
import io
import base64
import yaml
from PIL import Image, PngImagePlugin
import requests
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

from typing import Union

sys.path.append("ad_discordbot")

from ad_discordbot.modules.database import Database, ActiveSettings, Config, StarBoard, Statistics
from ad_discordbot.modules.utils_shared import task_semaphore, shared_path, patterns
from ad_discordbot.modules.utils_misc import fix_dict, update_dict, sum_update_dict, update_dict_matched_keys, format_time
from ad_discordbot.modules.utils_discord import ireply, send_long_message, SelectedListItem, SelectOptionsView, CtxInteraction, get_user_ctx_inter
from ad_discordbot.modules.utils_files import load_file, merge_base, save_yaml_file
from ad_discordbot.modules.utils_aspect_ratios import round_to_precision, res_to_model_fit, dims_from_ar, avg_from_dims, get_aspect_ratio_parts, calculate_aspect_ratio_sizes

# Databases
bot_active_settings = ActiveSettings()
starboard = StarBoard()
bot_database = Database()
bot_statistics = Statistics()
config = Config()

#################################################################
#################### DISCORD / BOT STARTUP ######################
#################################################################

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

# Set Discord bot token from config, or args, or prompt for it, or exit
TOKEN = config['discord'].get('TOKEN', None)

bot_token = bot_args.token if bot_args.token else TOKEN
if not bot_token:
    print('\nA Discord bot token is required. Please enter it below.\n \
          For help, refer to Install instructions on the project page\n \
          (https://github.com/altoiddealer/ad_discordbot)')

    print('\nDiscord bot token (enter "0" to exit):\n')
    bot_token = (input().strip())
    print()
    if bot_token == '0':
        logging.error("Discord bot token is required. Exiting.")
        sys.exit(2)
    elif bot_token:
        config['discord']['TOKEN'] = bot_token
        config.save()
        logging.info("Discord bot token saved to 'config.yaml'")
    else:
        logging.error("Discord bot token is required. Exiting.")
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
SD_CLIENT = None

if sd_enabled:
    SD_URL = config['sd'].get('SD_URL', None) # Get the URL from config.yaml
    if SD_URL is None:
        SD_URL = config['sd'].get('A1111', 'http://127.0.0.1:7860')

    async def sd_api(endpoint:str, method='get', json=None, retry=True):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method.lower(), url=f'{SD_URL}{endpoint}', json=json) as response:
                    if response.status == 200:
                        r = await response.json()
                        if SD_CLIENT is None and endpoint != '/sdapi/v1/cmd-flags':
                           await get_sd_sysinfo()
                           bot_settings.imgmodel.refresh_enabled_extensions()
                        return r
                    else:
                        logging.error(f'{SD_URL}{endpoint} response: {response.status} "{response.reason}"')
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
        global SD_CLIENT
        try:
            r = await sd_api(endpoint='/sdapi/v1/cmd-flags', method='get', json=None, retry=False)
            if not r:
                raise Exception(f'Failed to connect to SD api, make sure to start it or disable the api in your "{shared_path.config}"')

            ui_settings_file = r.get("ui_settings_file", "")
            if "webui-forge" in ui_settings_file:
                SD_CLIENT = 'SD WebUI Forge'
            elif "webui" in ui_settings_file:
                SD_CLIENT = 'A1111 SD WebUI'
            else:
                SD_CLIENT = 'SD WebUI'
        except Exception as e:
            logging.error(f"Error getting SD sysinfo API: {e}")
            SD_CLIENT = None
        
    # Set Stable Diffusion client name to use in messages, warnings, etc
    asyncio.run(get_sd_sysinfo())

    # Function to attempt restarting the SD WebUI Client in the event it gets stuck
    @client.hybrid_command(description=f"Immediately Restarts the {SD_CLIENT} server. Requires '--api-server-stop' SD WebUI launch flag.")
    async def restart_sd_client(ctx: commands.Context):
        try:
            system_embed = None
            await ctx.send(f"**`/restart_sd_client` __will not work__ unless {SD_CLIENT} was launched with flag: `--api-server-stop`**", delete_after=10)
            await sd_api(endpoint='/sdapi/v1/server-restart', method='post', json=None, retry=False)
            title = f"{ctx.author.display_name} used '/restart_sd_client'. Restarting {SD_CLIENT} ..."
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
        logging.info(f"SD WebUI currently offline. Image commands/features will function when client is active and accessible via API.'")

#################################################################
##################### TEXTGENWEBUI STARTUP ######################
#################################################################
if not 'textgenwebui' in config:
    logging.warning("'config.yaml' is missing a new dictionary 'textgenwebui'. Enabling TGWUI by default.")
    textgenwebui_enabled = True
else:
    textgenwebui_enabled = config['textgenwebui'].get('enabled', True)

if textgenwebui_enabled:
    import modules.extensions as extensions_module
    from modules.chat import chatbot_wrapper, load_character, save_history, get_history_file_path, find_all_histories
    from modules import shared
    from modules import chat, utils
    from modules.LoRA import add_lora_to_model
    from modules.models import load_model, unload_model
    from modules.models_settings import get_model_metadata, update_model_parameters, get_fallback_settings, infer_loader
    from modules.prompts import count_tokens

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
        logging.info(f"Loading text-generation-webui settings from {settings_file}...")
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
                if not bot_database.was_warned(name):
                    bot_database.update_was_warned(name)
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
    tts_settings = config.get('discord', {}).get('tts_settings', {})

supported_tts_clients = ['alltalk_tts', 'coqui_tts', 'silero_tts', 'elevenlabs_tts']

def init_textgenwebui_extensions():
    # monkey patch load_extensions behavior from pre-commit b3fc2cd
    extensions_module.load_extensions = load_extensions

    shared.args.extensions = []
    extensions_module.available_extensions = utils.get_available_extensions()

    tts_client = ''

    # Initialize shared args extensions
    for extension in shared.settings['default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)

    # Get any supported TTS client found in TGWUI CMD_FLAGS
    for extension in shared.args.extensions:
        if extension in supported_tts_clients:
            tts_client = extension
            break

    # If any TTS extension defined in config.yaml, set tts bot vars and add extension to shared.args.extensions
    tts_client = tts_settings.get('extension') or tts_client or '' # tts client
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
        elif tts_client == 'coqui_tts':
            tts_voice_key = 'voice'
            tts_lang_key = 'language'
        elif tts_client == 'silero_tts':
            tts_voice_key = 'speaker'
            tts_lang_key = 'language'
        elif tts_client == 'elevenlabs_tts':
            tts_voice_key = 'selected_voice'
            tts_lang_key = ''
            
        if tts_client not in shared.args.extensions:
            shared.args.extensions.append(tts_client)

    # Activate the extensions
    if shared.args.extensions and len(shared.args.extensions) > 0:
        extensions_module.load_extensions(extensions_module.extensions, extensions_module.available_extensions)

    return tts_client, tts_api_key, tts_voice_key, tts_lang_key

def init_textgenwebui_llmmodels():
    all_llmmodels = utils.get_available_models()

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
########################## BOT STARTUP ##########################
#################################################################
## Function to automatically change image models
# Select imgmodel based on mode, while avoid repeating current imgmodel
async def auto_select_imgmodel(current_imgmodel_name, mode='random'):
    try:
        all_imgmodels = await fetch_imgmodels()
        all_imgmodel_names = [imgmodel.get('imgmodel_name', '') for imgmodel in all_imgmodels]

        current_index = None
        if current_imgmodel_name and current_imgmodel_name in all_imgmodel_names:
            current_index = all_imgmodel_names.index(current_imgmodel_name)

        if mode == 'random':
            if current_index is not None and len(all_imgmodels) > 1:
                all_imgmodels.pop(current_index)

            return random.choice(all_imgmodels)

        elif mode == 'cycle':
            if current_index is not None:
                next_index = (current_index + 1) % len(all_imgmodel_names)  # Cycle to the beginning if at the end
                return all_imgmodels[next_index]

            else:
                logging.info("The previous imgmodel name was not matched in list of fetched imgmodels, so cannot 'cycle'. New imgmodel was instead picked at random.")
                return random.choice(all_imgmodels) # If no image model set yet, select randomly

    except Exception as e:
        logging.error(f"Error automatically selecting image model: {e}")

# Task to auto-select an imgmodel at user defined interval
async def auto_update_imgmodel_task(mode, duration):
    while True:
        await asyncio.sleep(duration)
        try:
            current_imgmodel_name = bot_settings.settings['imgmodel'].get('imgmodel_name', '')
            # Select an imgmodel automatically
            selected_imgmodel = await auto_select_imgmodel(current_imgmodel_name, mode)

            async with task_semaphore:
                # offload to ai_gen queue
                params = {'imgmodel': selected_imgmodel}
                await change_imgmodel_task('Automatically', channel=None, params=params, ictx=None)
                logging.info("Automatically updated imgmodel settings")

        except Exception as e:
            logging.error(f"Error automatically updating image model: {e}")
        #await asyncio.sleep(duration)

imgmodel_update_task = None # Global variable allows process to be cancelled and restarted (reset sleep timer)

if sd_enabled:
    # Register command for helper function to toggle auto-select imgmodel
    @client.hybrid_command(description='Toggles the automatic Img model changing task')
    async def toggle_auto_change_imgmodels(ctx: commands.Context):
        global imgmodel_update_task
        if imgmodel_update_task and not imgmodel_update_task.done():
            imgmodel_update_task.cancel()
            await ctx.send("Auto-change Imgmodels task was cancelled.", ephemeral=True, delete_after=5)
            logging.info("Auto-change Imgmodels task was cancelled via '/toggle_auto_change_imgmodels_task'")
            
        else:
            await bg_task_queue.put(start_auto_change_imgmodels())
            await ctx.send(f"Auto-change Img models task was started.", ephemeral=True, delete_after=5)

# helper function to begin auto-select imgmodel task
async def start_auto_change_imgmodels():
    try:
        global imgmodel_update_task
        imgmodels_data = load_file(shared_path.img_models, {})
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
        # This will be either the char name found in activesettings, or the default char name
        source = bot_settings.settings['llmcontext']['name']
        # If name doesn't match the bot's discord username, try to figure out best char data to initialize with
        if source != bot_database.last_character:
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
        bot_database.set('first_run', False)

# Unpack tag presets and add global tag keys
async def update_tags(tags:list) -> list:
    if not isinstance(tags, list):
        logging.warning(f'''One or more "tags" are improperly formatted. Please ensure each tag is formatted as a list item designated with a hyphen (-)''')
        return tags
    try:
        tags_data = load_file(shared_path.tags, {})
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

#################################################################
########################### ON READY ############################
#################################################################
@client.event
async def on_ready():
    try:
        # If first time running bot
        if bot_database.first_run:
            await first_run()
        if textgenwebui_enabled:
            char_name = get_character() # Try loading character data regardless of mode (chat/instruct)
            if char_name:
                await character_loader(char_name)
            # Load history or set empty history
            if bot_history.autoload_history and (bot_history.change_char_history_method == 'keep'):
                bot_history.load_bot_history()
            else:
                bot_history.reset_session_history()
        # Create background task processing queue
        client.loop.create_task(process_tasks_in_background())
        # Start background task to sync the discord client tree
        await bg_task_queue.put(client.tree.sync())
        # Start background task to to change image models automatically
        if sd_enabled:
            imgmodels_data = load_file(shared_path.img_models, {})
            if imgmodels_data and imgmodels_data.get('settings', {}).get('auto_change_imgmodels', {}).get('enabled', False):
                await bg_task_queue.put(start_auto_change_imgmodels())
        logging.info("Bot is ready")
    except Exception as e:
        logging.error(f"Error with on_ready: {e}")
        traceback.print_exc()

#################################################################
####################### DISCORD FEATURES ########################
#################################################################
# Starboard feature
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
        if target_channel and message.id not in starboard.messages:
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
            starboard.messages.append(message.id)
            starboard.save()

# Post settings to a dedicated channel
async def post_active_settings():
    target_channel_id = config['discord']['post_active_settings'].get('target_channel_id', None)
    if target_channel_id == 11111111111111111111:
        target_channel_id = None

    if target_channel_id:
        target_channel = await client.fetch_channel(target_channel_id)
        if target_channel:
            settings_content = yaml.dump(bot_active_settings.get_vars(), default_flow_style=False)

            async for message in target_channel.history(limit=None):
                await message.delete()
                await asyncio.sleep(0.5)  # minimum delay for discord limit
            # Send the entire settings content as a single message
            await send_long_message(target_channel, f"Current settings:\n```yaml\n{settings_content}\n```")
        else:
            logging.error(f"Target channel with ID {target_channel_id} not found.")
    else:
        logging.warning("Channel ID must be specified in config.yaml")

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
                    logging.warning(f'Bot launched with {tts_client}, but no voice channel is specified in config.yaml')
            else:
                if not bot_database.was_warned('char_tts'):
                    bot_database.update_was_warned('char_tts')
                    logging.warning(f'Character "use_voice_channel" = True, and "voice channel" is specified in config.yaml, but no "tts_client" is specified in config.yaml')
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
        # Add tts API key if one is provided in config.yaml
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
    mp3_filename = os.path.splitext(filename)[0] + '.mp3'
    
    bit_rate = int(tts_settings.get('mp3_bit_rate', 128))
    with io.BytesIO() as buffer:
        audio = AudioSegment.from_wav(tts_resp)
        audio.export(buffer, format="mp3", bitrate=f"{bit_rate}k")
        mp3_file = File(buffer, filename=mp3_filename)
        await channel.send(file=mp3_file)
    

async def process_tts_resp(channel, tts_resp, i=None):
    play_mode = int(tts_settings.get('play_mode', 0))
    # Upload to interaction channel
    if play_mode > 0:
        await upload_tts_file(channel, tts_resp)
    # Play in voice channel
    if play_mode != 1 and (voice_client == i.guild.voice_client):
        await bg_task_queue.put(play_in_voice_channel(tts_resp)) # run task in background

#################################################################
########################### ON MESSAGE ##########################
#################################################################
async def fix_llm_payload(llm_payload):
    # Fix llm_payload by adding any missing required settings
    defaults = bot_settings.settings_to_dict() # Get default settings as dict
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

async def swap_llm_character(char_name:str, user_name:str, llm_payload:dict):
    try:
        char_data = await load_character_data(char_name)
        name1 = user_name
        if char_data.get('state', {}):
            llm_payload['state'] = char_data['state']
            llm_payload['state']['name1'] = name1
        llm_payload['state']['name2'] = char_data.get('name', 'AI')
        llm_payload['state']['character_menu'] = char_data.get('name', 'AI')
        llm_payload['state']['context'] = char_data.get('context', '')
        llm_payload = await fix_llm_payload(llm_payload) # Add any missing required information
        return llm_payload
    except Exception as e:
        logging.error(f"An error occurred while loading the file for swap_character: {e}")
        return llm_payload

def format_prompt_with_recent_output(user_name:str, prompt:str):
    try:
        formatted_prompt = prompt
        # Find all matches of {user_x} and {llm_x} in the prompt
        matches = patterns.recent_msg_roles.findall(prompt)
        # Iterate through the matches
        for match in matches:
            prefix, index = match
            index = int(index)
            if prefix in ['user', 'llm'] and 0 <= index <= 10:
                message_list = bot_history.recent_messages[prefix]
                if not message_list or index >= len(message_list):
                    continue
                matched_syntax = f"{prefix}_{index}"
                formatted_prompt = formatted_prompt.replace(f"{{{matched_syntax}}}", message_list[index])
            elif prefix == 'history' and 0 <= index <= 10:
                user_message = bot_history.recent_messages['user'][index] if index < len(bot_history.recent_messages['user']) else ''
                llm_message = bot_history.recent_messages['llm'][index] if index < len(bot_history.recent_messages['llm']) else ''
                formatted_history = f'"{user_name}:" {user_message}\n"{bot_database.last_character}:" {llm_message}\n'
                matched_syntax = f"{prefix}_{index}"
                formatted_prompt = formatted_prompt.replace(f"{{{matched_syntax}}}", formatted_history)
        formatted_prompt = formatted_prompt.replace('{last_image}', '__temp/temp_img_0.png')
        return formatted_prompt
    except Exception as e:
        logging.error(f'An error occurred while formatting prompt with recent messages: {e}')
        return prompt

def process_tag_formatting(user_name:str, prompt:str, formatting:dict):
    try:
        updated_prompt = prompt
        format_prompt = formatting.get('format_prompt', [])
        time_offset = formatting.get('time_offset', None)
        time_format = formatting.get('time_format', None)
        date_format = formatting.get('date_format', None)
        # Tag handling for prompt formatting
        if format_prompt:
            for fmt_prompt in format_prompt:
                updated_prompt = fmt_prompt.replace('{prompt}', updated_prompt)
        # format prompt with any defined recent messages
        updated_prompt = format_prompt_with_recent_output(user_name, updated_prompt)
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

async def process_llm_payload_tags(ictx: CtxInteraction, llm_payload:dict, llm_prompt:str, mods:dict, params={}):
    try:
        user_name = get_user_ctx_inter(ictx).display_name
        char_params = {}
        flow = mods.get('flow', None)
        load_history = mods.get('load_history', None)
        param_variances = mods.get('param_variances', {})
        state = mods.get('state', {})
        prefix_context = mods.get('prefix_context', None)
        suffix_context = mods.get('suffix_context', None)
        change_character = mods.get('change_character', None)
        swap_character = mods.get('swap_character', None)
        change_llmmodel = mods.get('change_llmmodel', None)
        swap_llmmodel = mods.get('swap_llmmodel', None)
        # Flow handling
        if flow is not None and not flow_event.is_set():
            await build_flow_queue(flow)
        # History handling
        if load_history is not None:
            chankey = str(ictx.channel.id)
            if load_history < 0:
                llm_payload['state']['history']['internal'] = []
                llm_payload['state']['history']['visible'] = []
                logging.info("[TAGS] History is being ignored")
            elif load_history > 0:
                i_list, v_list = bot_history.get_history_iv_lists_keys(chankey)
                # Calculate the number of items to retain (up to the length of session_history)
                num_to_retain = min(load_history, len(i_list))
                llm_payload['state']['history']['internal'] = i_list[-num_to_retain:]
                llm_payload['state']['history']['visible'] = v_list[-num_to_retain:]
                logging.info(f'[TAGS] History is being limited to previous {load_history} exchanges')
        if param_variances:
            processed_params = process_param_variances(param_variances)
            logging.info(f'[TAGS] LLM Param Variances: {processed_params}')
            sum_update_dict(llm_payload['state'], processed_params) # Updates dictionary while adding floats + ints
        if state:
            update_dict(llm_payload['state'], state)
            logging.info(f'[TAGS] LLM State was modified')
        # Context insertions
        if prefix_context:
            prefix_str = "\n".join(str(item) for item in prefix_context)
            if prefix_str:
                llm_payload['state']['context'] = f"{prefix_str}\n{llm_payload['state']['context']}"
                logging.info(f'[TAGS] Prefixed context with text.')
        if suffix_context:
            suffix_str = "\n".join(str(item) for item in suffix_context)
            if suffix_str:
                llm_payload['state']['context'] = f"{llm_payload['state']['context']}\n{suffix_str}"
                logging.info(f'[TAGS] Suffixed context with text.')
        # Character handling
        char_params = change_character or swap_character or {} # 'character_change' will trump 'character_swap'
        if char_params:
            # Error handling
            all_characters, _ = get_all_characters()
            if not any(char_params == char['name'] for char in all_characters):
                logging.error(f'Character not found: {char_params}')
            else:
                if char_params == change_character:
                    verb = 'Changing'
                    char_params = {'character': {'char_name': char_params, 'mode': 'change', 'verb': verb}}
                    await change_char_task(ictx, 'Tags', char_params)
                else:
                    verb = 'Swapping'
                    llm_payload = await swap_llm_character(swap_character, user_name, llm_payload)
                logging.info(f'[TAGS] {verb} Character: {char_params}')
        # LLM model handling
        model_change = change_llmmodel or swap_llmmodel or None # 'llmmodel_change' will trump 'llmmodel_swap'
        if model_change:
            if model_change == shared.model_name:
                logging.info(f'[TAGS] LLM model was triggered to change, but it is the same as current ("{shared.model_name}").')
            else:
                mode = 'change' if model_change == change_llmmodel else 'swap'
                verb = 'Changing' if mode == 'change' else 'Swapping'
                # Error handling
                all_llmmodels = utils.get_available_models()
                if not any(model_change == model for model in all_llmmodels):
                    logging.error(f'LLM model not found: {model_change}')
                else:
                    logging.info(f'[TAGS] {verb} LLM Model: {model_change}')
                    params['llmmodel'] = {'llmmodel_name': params, 'mode': mode, 'verb': verb}
        return llm_payload, llm_prompt, params
    except Exception as e:
        logging.error(f"Error processing LLM tags: {e}")
        return llm_payload, llm_prompt, {}

def collect_llm_tag_values(tags, params):
    llm_payload_mods = {}
    formatting = {}
    try:
        for tag in tags['matches']:
            # Values that will only apply from the first tag matches
            if 'flow' in tag and not llm_payload_mods.get('flow'):
                llm_payload_mods['flow'] = tag.pop('flow')
            if 'save_history' in tag and not params.get('save_to_history'):
                params['save_to_history'] = bool(tag.pop('save_history'))
            if 'load_history' in tag and not llm_payload_mods.get('load_history'):
                llm_payload_mods['load_history'] = int(tag.pop('load_history'))
                
            # change_character is higher priority, if added ignore swap_character
            if 'change_character' in tag and not (llm_payload_mods.get('change_character') or llm_payload_mods.get('swap_character')):
                llm_payload_mods['change_character'] = str(tag.pop('change_character'))
            if 'swap_character' in tag and not (llm_payload_mods.get('change_character') or llm_payload_mods.get('swap_character')):
                llm_payload_mods['swap_character'] = str(tag.pop('swap_character'))
                
            # change_llmmodel is higher priority, if added ignore swap_llmmodel
            if 'change_llmmodel' in tag and not (llm_payload_mods.get('change_llmmodel') or llm_payload_mods.get('swap_llmmodel')):
                llm_payload_mods['change_llmmodel'] = str(tag.pop('change_llmmodel'))
            if 'swap_llmmodel' in tag and not (llm_payload_mods.get('change_llmmodel') or llm_payload_mods.get('swap_llmmodel')):
                llm_payload_mods['swap_llmmodel'] = str(tag.pop('swap_llmmodel'))
                
            # Values that may apply repeatedly
            if 'prefix_context' in tag:
                llm_payload_mods.setdefault('prefix_context', [])
                llm_payload_mods['prefix_context'].append(tag.pop('prefix_context'))
            if 'suffix_context' in tag:
                llm_payload_mods.setdefault('suffix_context', [])
                llm_payload_mods['suffix_context'].append(tag.pop('suffix_context'))
            if 'send_user_image' in tag:
                user_image_file = tag.pop('send_user_image')
                user_image_args = get_image_tag_args('User image', str(user_image_file), key=None, set_dir=None)
                user_image = discord.File(user_image_args)
                params.setdefault('send_user_image', [])
                params['send_user_image'].append(user_image)
                logging.info(f'[TAGS] Sending user image.')
            if 'format_prompt' in tag:
                formatting.setdefault('format_prompt', [])
                formatting['format_prompt'].append(str(tag.pop('format_prompt')))
            if 'time_offset' in tag:
                formatting['time_offset'] = float(tag.pop('time_offset'))
            if 'time_format' in tag:
                formatting['time_format'] = str(tag.pop('time_format'))
            if 'date_format' in tag:
                formatting['date_format'] = str(tag.pop('date_format'))
            if 'llm_param_variances' in tag:
                llm_param_variances = dict(tag.pop('llm_param_variances'))
                llm_payload_mods.setdefault('llm_param_variances', {})
                try:
                    llm_payload_mods['param_variances'].update(llm_param_variances) # Allow multiple to accumulate.
                except:
                    logging.warning("Error processing a matched 'llm_param_variances' tag; ensure it is a dictionary.")
            if 'state' in tag:
                state = dict(tag.pop('state'))
                llm_payload_mods.setdefault('state', {})
                try:
                    llm_payload_mods['state'].update(state) # Allow multiple to accumulate.
                except:
                    logging.warning("Error processing a matched 'state' tag; ensure it is a dictionary.")
    except Exception as e:
        logging.error(f"Error collecting LLM tag values: {e}")
    return llm_payload_mods, formatting, params

def process_tag_insertions(prompt:str, tags:dict):
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

def process_tag_trumps(matches:list, trump_params:list=[]):
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

def match_tags(search_text:str, tags:dict, phase='llm') -> dict:
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

def sort_tags(all_tags: list) -> Union[list, dict]:
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


def _expand_value(value:str) -> str:
    # Split the value on commas
    parts = value.split(',')
    expanded_values = []
    for part in parts:
        # Check if the part contains curly brackets
        if '{' in part and '}' in part:
            # Use regular expression to find all curly bracket groups
            group_matches = patterns.in_curly_brackets.findall(part)
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

async def expand_triggers(all_tags:list) -> list:
    try:
        for tag in all_tags:
            if 'trigger' in tag:
                tag['trigger'] = _expand_value(tag['trigger'])

    except Exception as e:
        logging.error(f"Error expanding tags: {e}")

    return all_tags

# Function to convert string values to bool/int/float
def extract_value(value_str:str) -> Union[bool, int, float]:
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

def parse_tag_from_text_value(value_str:str) -> str:
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
                sublist_strings = patterns.brackets.findall(inner_text)
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
        matches = patterns.instant_tags.findall(text)
        detagged_text = patterns.instant_tags.sub('', text)
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
        base_tags = bot_settings.base_tags # base tags
        imgmodel_tags = bot_settings.settings['imgmodel'].get('tags', []) # imgmodel specific tags
        char_tags = bot_settings.settings['llmcontext'].get('tags', []) # character specific tags
        detagged_text, tags_from_text = get_tags_from_text(text)
        all_tags = tags_from_text + flow_step_tags + char_tags + imgmodel_tags + base_tags  # merge tags to one dictionary
        sorted_tags = sort_tags(all_tags) # sort tags into phases (user / llm / userllm)
        return detagged_text, sorted_tags
    except Exception as e:
        logging.error(f"Error getting tags: {e}")
        return text, []

async def init_llm_payload(ictx: CtxInteraction, user_name:str, text:str) -> dict:
    llm_payload = copy.deepcopy(bot_settings.settings['llmstate'])
    llm_payload['text'] = text
    name1 = user_name
    name2 = bot_settings.settings['llmcontext']['name']
    context = bot_settings.settings['llmcontext']['context']
    llm_payload['state']['name1'] = name1
    llm_payload['state']['name2'] = name2
    llm_payload['state']['name1_instruct'] = name1
    llm_payload['state']['name2_instruct'] = name2
    llm_payload['state']['character_menu'] = name2
    llm_payload['state']['context'] = context
    ictx_history = bot_history.get_channel_history(ictx)
    llm_payload['state']['history'] = ictx_history
    return llm_payload

def get_wildcard_value(matched_text, dir_path=None):
    dir_path = dir_path or os.path.join('ad_discordbot', 'wildcards')
    selected_option = None
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
        braces_match = patterns.braces.search(selected_option)
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
            nested_dir = os.path.join('ad_discordbot', 'wildcards', os.path.dirname(nested_dir))
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
        wildcard_match = patterns.wildcard.search(option)
        if wildcard_match:
            wildcard_phrase = wildcard_match.group()
            dir_path = os.path.join('ad_discordbot', 'wildcards')
            wildcard_value = get_wildcard_value(matched_text=wildcard_phrase, dir_path=dir_path)
            if wildcard_value:
                chosen_options[index] = wildcard_value
    chosen_options = [option for option in chosen_options if option is not None]
    if separator:
        replaced_text = separator.join(chosen_options)
    else:
        replaced_text = ', '.join(chosen_options) if num_choices > 1 else chosen_options[0]
    return replaced_text

async def dynamic_prompting(user_name:str, text:str, i=None):
    if not config.get('dynamic_prompting_enabled', True):
        if not bot_database.was_warned('dynaprompt'):
            bot_database.update_was_warned('dynaprompt')
            logging.warning(f"'{shared_path.config}' is missing a new parameter 'dynamic_prompting_enabled'. Defaulting to 'True' (enabled) ")
    if not dynamic_prompting:
        return text

    # copy text for adding comments
    text_with_comments = text
    # Process braces patterns
    braces_start_indexes = []
    braces_matches = patterns.braces.finditer(text)
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
    wildcard_matches = patterns.wildcard.finditer(text)
    wildcard_matches = sorted(wildcard_matches, key=lambda x: -x.start())  # Sort matches in reverse order by their start indices
    for match in wildcard_matches:
        matched_text = match.group()
        replaced_text = get_wildcard_value(matched_text=matched_text, dir_path=shared_path.dir_wildcards)
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
        await i.reply(content=f"__Text with **[Dynamic Prompting](<https://github.com/altoiddealer/ad_discordbot/wiki/dynamic-prompting>)**__:\n>>> **{user_name}**: {text_with_comments}", mention_author=False, silent=True)
    return text

@client.event
async def on_message(message: discord.Message):
    try:
        text = message.clean_content # primarly converts @mentions to actual user names
        if textgenwebui_enabled and not bot_behavior.bot_should_reply(message, text): 
            return # Check that bot should reply or not
        # Store the current time. The value will save locally to database.yaml at another time
        bot_database.update_last_user_msg(message.channel.id, save_now=False)
        # if @ mentioning bot, remove the @ mention from user prompt
        if text.startswith(f"@{bot_database.last_character} "):
            text = text.replace(f"@{bot_database.last_character} ", "", 1)
        # apply wildcards
        text = await dynamic_prompting(message.author.display_name, text, message)

        async with task_semaphore:
            async with message.channel.typing():
                logging.info(f'reply requested: {message.author.display_name} said: "{text}"')
                await on_message_task(message, 'on_message', text)
                await run_flow_if_any(message, 'on_message', text)

    except Exception as e:
        logging.error(f"An error occurred in on_message: {e}")

#################################################################
#################### QUEUED FROM ON MESSAGE #####################
#################################################################
async def on_message_task(ictx: CtxInteraction, source:str, text:str):
    log.debug(f"on_message_task {len(bot_history.recent_messages.get('user',[]))}, {len(bot_history.recent_messages.get('llm',[]))}, {bot_history.recent_messages}")

    try:
        user_name = get_user_ctx_inter(ictx).display_name
        channel = ictx.channel
        params = {} # dictionary to pass parameters through the event
        # collects all tags, sorted into sub-lists by phase (user / llm / userllm)
        text, tags = await get_tags(text)
        # match tags labeled for user / userllm.
        tags = match_tags(text, tags, phase='llm')
        # check what bot should do
        bot_will_do = bot_should_do(tags)
        params['bot_will_do'] = bot_will_do
        # do what bot should do
        if bot_will_do['should_gen_text']:
            # build llm_payload with defaults
            llm_payload = await init_llm_payload(ictx, user_name, text)
            # make working copy of user's request (without @ mention)
            llm_prompt = text
            # apply tags to prompt
            llm_prompt, tags = process_tag_insertions(llm_prompt, tags)
            # collect matched tag values
            llm_payload_mods, formatting, params = collect_llm_tag_values(tags, params)
            # apply tags relevant to LLM payload
            llm_payload, llm_prompt, params = await process_llm_payload_tags(ictx, llm_payload, llm_prompt, llm_payload_mods, params)
            # apply formatting tags to LLM prompt
            llm_prompt = process_tag_formatting(user_name, llm_prompt, formatting)
            # offload to ai_gen queue
            llm_payload['text'] = llm_prompt

            await hybrid_llm_img_gen(ictx, source, text, tags, llm_payload, params)
        elif bot_will_do['should_gen_image']:
            if await sd_online(channel):
                await channel.send(f'Bot was triggered by Tags to not respond with text.\n**Processing image generation using your input as the prompt ...**', delete_after=5) # msg for if LLM model is unloaded
            await img_gen_task(source, text, params, ictx, tags)

    except Exception as e:
        logging.error(f"An error occurred processing on_message request: {e}")

async def hybrid_llm_img_gen(ictx: CtxInteraction, source:str, text:str, tags:dict, llm_payload:dict, params:dict):
    try:
        user_name = get_user_ctx_inter(ictx).display_name
        channel = ictx.channel
        bot_will_do = params['bot_will_do']
        change_embed = None
        img_gen_embed = None
        tts_resp = None

        # Check params to see if an LLM model change/swap was triggered by Tags
        llmmodel_params = params.get('llmmodel', {})
        send_user_image = params.get('send_user_image', [])
        mode = llmmodel_params.get('mode', 'change') # default to 'change' unless a tag was triggered with 'swap'
        if llmmodel_params:
            orig_llmmodel = shared.model_name                       # copy current LLM model name
            change_embed = await change_llmmodel_task(ictx, params)    # Change LLM model
            if mode == 'swap' and change_embed:                     # Delete embed before the second call
                await change_embed.delete()

        # make a 'Prompting...' embed when generating text for an image response
        if bot_will_do['should_gen_image'] and textgenwebui_enabled:
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
                if not bot_database.was_warned('no_llmmodel'):
                    bot_database.update_was_warned('no_llmmodel')
                    await channel.send(f'(Cannot process text request: No LLM model is currently loaded. Use "/llmmodel" to load a model.)', delete_after=10)
                    logging.warning(f'Bot tried to generate text for {user_name}, but no LLM model was loaded')
            # Check to apply Server Mode
            llm_payload = apply_server_mode(llm_payload, ictx)
            # Only generate TTS for the server conntected to Voice Channel
            tts_sw = None
            if (not bot_will_do['should_send_text']) or (voice_client and (voice_client != ictx.guild.voice_client) and int(tts_settings.get('play_mode', 0)) == 0):
                tts_sw = await toggle_tts(toggle='off')
            # generate text with text-gen-webui
            last_resp, tts_resp = await llm_gen(llm_payload, source, params, ictx, tts_sw)
            # If no text was generated, treat user input at the response
            if last_resp is not None:
                logging.info("reply sent: \"" + user_name + ": {'text': '" + llm_payload["text"] + "', 'response': '" + last_resp + "'}\"")
            else:
                if bot_will_do['should_gen_image'] and sd_enabled:
                    last_resp = text
                else:
                    return

            # if LLM model swapping was triggered
            if mode == 'swap':
                params['llmmodel']['llmmodel_name'] = orig_llmmodel
                change_embed = await change_llmmodel_task(ictx, params)    # Swap LLM Model back
                if change_embed:
                    await change_embed.delete()                         # Delete embed again after the second call

        # process image generation (A1111 / Forge)
        if sd_enabled:
            tags = match_img_tags(last_resp, tags)
            bot_will_do = bot_should_do(tags, bot_will_do) # check for updates from tags
            if bot_will_do['should_gen_image']:
                if img_gen_embed:
                    await img_gen_embed.delete()
                params['bot_will_do'] = bot_will_do
                await img_gen_task(source, last_resp, params, ictx, tags)
        if tts_resp:
            await process_tts_resp(channel, tts_resp, ictx)
        mention_resp = update_mention(get_user_ctx_inter(ictx).mention, last_resp) # @mention non-consecutive users
        if bot_will_do['should_send_text']:
            await send_long_message(channel, mention_resp)
        if send_user_image:
            await channel.send(file=send_user_image) if len(send_user_image) == 1 else await channel.send(files=send_user_image)
        return
    except Exception as e:
        logging.error(f'An error occurred while processing "{source}" request: {e}')
        if img_gen_embed_info:
            img_gen_embed_info.title = f'An error occurred while processing "{source}" request'
            img_gen_embed_info.description = e
            if img_gen_embed:
                await img_gen_embed.edit(embed=img_gen_embed_info)
            else:
                await channel.send(embed=img_gen_embed_info)

        if change_embed:
            await change_embed.delete()

#################################################################
##################### QUEUED LLM GENERATION #####################
#################################################################
# Update LLM Gen Statistics
def update_llm_gen_statistics(last_resp:str):
    try:
        total_gens = bot_statistics.llm.get('generations_total', 0)
        total_gens += 1
        bot_statistics.llm['generations_total'] = total_gens
        # Update tokens statistics
        last_tokens = int(count_tokens(last_resp))
        bot_statistics.llm['num_tokens_last'] = last_tokens
        total_tokens = bot_statistics.llm.get('num_tokens_total', 0)
        total_tokens += last_tokens
        bot_statistics.llm['num_tokens_total'] = total_tokens
        # Update time statistics
        total_time = bot_statistics.llm.get('time_total', 0)
        total_time += (time.time() - bot_statistics._llm_gen_time_start_last)
        bot_statistics.llm['time_total'] = round(total_time, 4)
        # Update averages
        bot_statistics.llm['tokens_per_gen_avg'] = total_tokens/total_gens
        bot_statistics.llm['tokens_per_sec_avg'] = round((total_tokens/total_time), 4)
        bot_statistics.save()
    except Exception as e:
        logging.error(f'An error occurred while saving LLM gen statistics: {e}')

# Add guild data
def apply_server_mode(llm_payload:dict, i=None):
    if i and config.get('textgenwebui', {}).get('server_mode', False):
        try:
            name1 = f'Server: {i.guild}'
            llm_payload['state']['name1'] = name1
            llm_payload['state']['name1_instruct'] = name1
        except Exception as e:
            logging.error(f'An error occurred while applying Server Mode: {e}')
    return llm_payload

# Add dynamic stopping strings
def extra_stopping_strings(llm_payload:dict):
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
    except Exception as e:
        logging.error(f'An error occurred while updating stopping strings: {e}')
    return llm_payload

# Only generate TTS for the server conntected to Voice Channel
async def toggle_tts(toggle='on', tts_sw=None):
    try:
        extensions = copy.deepcopy(bot_settings.settings['llmcontext'].get('extensions', {}))
        if toggle == 'off' and extensions.get(tts_client, {}).get('activate'):
            extensions[tts_client]['activate'] = False
            await update_extensions(extensions)
            # Return True if subsequent toggle_tts() should enable TTS
            return True
        if tts_sw:
            extensions[tts_client]['activate'] = True
            await update_extensions(extensions)
    except Exception as e:
        logging.error(f'An error occurred while toggling the TTS on/off in llm_gen(): {e}')
    return None

# Send LLM Payload - get response
async def llm_gen(llm_payload:dict, source:str, params:dict={}, i=None, tts_sw=None):
    try:
        if shared.model_name == 'None':
            return None, None
        llm_payload = extra_stopping_strings(llm_payload)

        loop = asyncio.get_event_loop()

        # Store time for statistics
        bot_statistics._llm_gen_time_start_last = time.time()

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
                        audio_format_match = patterns.audio_src.search(last_vis_resp)
                        if audio_format_match:
                            tts_resp = audio_format_match.group(1)
            return last_resp, tts_resp  # bot's reply

        # Offload the synchronous task to a separate thread using run_in_executor
        last_resp, tts_resp = await loop.run_in_executor(None, process_responses)

        if last_resp and source != 'speak':
            update_llm_gen_statistics(last_resp) # Update statistics
            save_to_history = params.get('save_to_history', True)
            bot_history.manage_history(prompt=llm_payload['text'], reply=last_resp, save_to_history=save_to_history, chankey=str(i.channel.id))

        # Toggle TTS back on if it was toggled off
        await toggle_tts(toggle='on', tts_sw=tts_sw)

        return last_resp, tts_resp
    except Exception as e:
        logging.error(f'An error occurred in llm_gen(): {e}')
        if str(e).startswith('list index out of range'):
            logging.warning(f'Note (this may not be the cause of error): "regen" and "continue" commands only work if bot sent message during current session.')
        traceback.print_exc()
        return None, None

async def cont_regen_task(inter:discord.Interaction, source:str, text:str, message:discord.Message):
    cmd = ''
    try:
        user_name = get_user_ctx_inter(inter).display_name # just incase this function is used elsewhere later
        channel = inter.channel
        system_embed = None
        llm_payload = await init_llm_payload(inter, user_name, text)
        params = {'save_to_history': False}
        if source == 'cont':
            cmd = 'Continuing'
            llm_payload['_continue'] = True
        else:
            cmd = 'Regenerating'
            llm_payload['regenerate'] = True
        if shared.model_name == 'None':
            await channel.send('(Cannot process text request: No LLM model is currently loaded. Use "/llmmodel" to load a model.)', delete_after=5)
            logging.warning(f'{user_name} used {cmd} but no LLM model was loaded')
            return
        if system_embed_info:
            system_embed_info.title = f'{cmd} ... '
            system_embed_info.description = f'{cmd} text for {user_name}'
            system_embed = await channel.send(embed=system_embed_info)
        # Check to apply Server Mode
        llm_payload = apply_server_mode(llm_payload, inter)
        # Only generate TTS for the server conntected to Voice Channel
        tts_sw = None
        if voice_client and (voice_client != inter.guild.voice_client) and int(tts_settings.get('play_mode', 0)) == 0:
            tts_sw = await toggle_tts(toggle='off')
        # generate text with text-gen-webui
        last_resp, tts_resp = await llm_gen(llm_payload, source, params, inter, tts_sw)
        if system_embed:
            await system_embed.delete()
        if last_resp is None:
            return
        logging.info("reply sent: \"" + user_name + ": {'text': '" + llm_payload["text"] + "', 'response': '" + last_resp + "'}\"")
        fetched_message = await channel.fetch_message(message)
        await fetched_message.delete()
        if tts_resp:
            await process_tts_resp(channel, tts_resp, inter)
        if source == 'regen':
            await inter.followup.send('__Regenerated text:__', silent=True)
        await send_long_message(channel, last_resp)
    except Exception as e:
        e_msg = f'An error occurred while processing "{cmd}"'
        logging.error(f'{e_msg}: {e}')
        if str(e).startswith('cannot unpack non-iterable NoneType object'):
            none_msg = f'Error: {cmd} only works on messages sent from the bot during current session.'
            logging.error(none_msg)
            await inter.followup.send(none_msg, silent=True)
        else:
            await inter.followup.send(e_msg, silent=True)
        if system_embed:
            await system_embed.delete()

async def speak_task(ctx: commands.Context, text:str, params:dict):
    user_name = ctx.author.display_name
    channel = ctx.channel
    try:
        system_embed = None
        if shared.model_name == 'None':
            await channel.send('Cannot process "/speak" request: No LLM model is currently loaded. Use "/llmmodel" to load a model.)', delete_after=5)
            logging.warning(f'Bot tried to generate tts for {user_name}, but no LLM model was loaded')
            return
        if system_embed_info:
            system_embed_info.title = f'{user_name} requested tts ... '
            system_embed_info.description = ''
            system_embed = await channel.send(embed=system_embed_info)
        llm_payload = await init_llm_payload(ctx, user_name, text)
        llm_payload['_continue'] = True
        llm_payload['state']['max_new_tokens'] = 1
        llm_payload['state']['history'] = {'internal': [[text, text]], 'visible': [[text, text]]}
        params['save_to_history'] = False
        tts_args = params.get('tts_args', {})
        await update_extensions(tts_args)
        _, tts_resp = await llm_gen(llm_payload, 'speak', params)
        if system_embed:
            await system_embed.delete()
        if tts_resp is None:
            return
        await process_tts_resp(channel, tts_resp, ctx)
        # remove api key (don't want to share this to the world!)
        for sub_dict in tts_args.values():
            if 'api_key' in sub_dict:
                sub_dict.pop('api_key')
        if system_embed_info:
            system_embed_info.title = f'{user_name} requested tts:'
            system_embed_info.description = f"**Params:** {tts_args}\n**Text:** {text}"
            system_embed = await channel.send(embed=system_embed_info)
        await update_extensions(bot_settings.settings['llmcontext'].get('extensions', {})) # Restore character specific extension settings
        if params.get('user_voice'): os.remove(params['user_voice'])
    except Exception as e:
        logging.error(f"An error occurred while generating tts for '/speak': {e}")
        if system_embed_info:
            system_embed_info.title = "An error occurred while generating tts for '/speak'"
            system_embed_info.description = e
            if system_embed:
                await system_embed.edit(embed=system_embed_info)

#################################################################
###################### QUEUED MODEL CHANGE ######################
#################################################################
# Process selected Img model
async def change_imgmodel_task(user_name:str, channel, params:dict, ictx=None):
    try:
        if ictx:
            user_name = get_user_ctx_inter(ictx).display_name
            channel = ictx.channel
        change_embed = None
        await sd_online(channel) # Can't change Img model if not online!

        imgmodel_params = params.get('imgmodel', {})
        imgmodel_name = imgmodel_params.get('imgmodel_name', '')
        mode = imgmodel_params.get('mode', 'change')    # default to 'change
        verb = imgmodel_params.get('verb', 'Changing')  # default to 'Changing'

        # Was not 'None' and did not match any known model names/checkpoints
        if len(imgmodel_params) < 3:
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

        # Swap Image model
        if mode == 'swap' or mode == 'swap_back':
            current_model_settings = bot_settings.settings['imgmodel'].get('override_settings') or bot_settings.settings['imgmodel']['payload'].get('override_settings')
            new_model_settings = copy.deepcopy(current_model_settings)
            new_model_settings['sd_model_checkpoint'] = imgmodel_params['sd_model_checkpoint']
            _ = await sd_api(endpoint='/sdapi/v1/options', method='post', json=new_model_settings, retry=True)
            if change_embed:
                await change_embed.delete()
            return True

        # Change Image model
        await change_imgmodel(imgmodel_params)
        # if imgmodel_name != 'None': ### IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
        if channel and change_embed:
            await change_embed.delete()
            if bot_database.announce_channels:
                # Send embeds to announcement channels
                await bg_task_queue.put(announce_changes(ictx, 'Img model', imgmodel_name))
            else:
                # Send change embed to interaction channel
                change_embed_info.title = f"{user_name} changed Img model:"
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
            if change_embed:
                await change_embed.edit(embed=change_embed_info)
            else:
                if channel:
                    await channel.send(embed=change_embed_info)
        return False

# Process selected LLM model
async def change_llmmodel_task(ictx, params:dict):
    try:
        user_name = get_user_ctx_inter(ictx).display_name
        channel = ictx.channel
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
                    bot_database.update_was_warned('no_llmmodel', False) # Reset warning message
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
            if change_embed:
                await change_embed.delete()
                # Send embeds to announcement channels
                if bot_database.announce_channels:
                    await bg_task_queue.put(announce_changes(ictx, 'LLM model', llmmodel_name))
                else:
                    # Send change embed to interaction channel
                    if llmmodel_name == 'None':
                        change_embed_info.title = f"{user_name} unloaded the LLM model"
                        change_embed_info.description = 'Use "/llmmodel" to load a new one'
                    else:
                        change_embed_info.title = f"{user_name} changed LLM model:"
                        change_embed_info.description = f'**{llmmodel_name}**'
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
async def send_char_greeting_or_history(ictx: CtxInteraction, char_name:str):
    channel = ictx.channel
    try:
        # Send message to channel
        message = ''
        if bot_history.greeting_or_history == 'history':
            last_exchange = bot_history.session_history['visible'][-1] if bot_history.session_history.get('visible') else None
            if last_exchange:
                last_user_message = last_exchange[0]
                last_assistant_message = last_exchange[1]
                message = f'__**Last message exchange**__:\n>>> **User**: "{last_user_message}"\n **{bot_database.last_character}**: "{last_assistant_message}"'
        if not message:
            greeting = bot_settings.settings['llmcontext']['greeting']
            if greeting:
                message = greeting.replace('{{user}}', 'user')
                message = message.replace('{{char}}', char_name)
            else:
                message = f'**{char_name}** has entered the chat"'
        await send_long_message(channel, message)
    except Exception as e:
        logging.error(f'An error occurred while sending greeting or history for "{char_name}": {e}')

async def announce_changes(ictx: CtxInteraction, change_label:str, change_name:str):
    user_name = get_user_ctx_inter(ictx).display_name if ictx else 'Automatically'
    change_embed_info.title = f"{user_name} changed {change_label}:"
    change_embed_info.description = f'**{change_name}**'
    try:
        # adjust delay depending on how many channels there are to prevent being rate limited
        delay = math.floor(len(bot_database.announce_channels)/2)
        for channel_id in bot_database.announce_channels:
            await asyncio.sleep(delay)
            channel = await client.fetch_channel(channel_id)
            # if Automatic imgmodel change (no interaction object)
            if ictx is None:
                await channel.send(embed=change_embed_info)
            # Channel is interaction channel
            elif channel_id == ictx.channel.id:
                continue # already sent
            # Channel in interaction server
            elif channel_id in [channel.id for channel in ictx.guild.channels]:
                await channel.send(embed=change_embed_info)
            # Channel is in another server
            else:
                change_embed_info.title = f"A user changed {change_label} in another bot server:"
                await channel.send(embed=change_embed_info)
    except Exception as e:
        logging.error(f'An error occurred while announcing changes to announce channels: {e}')

async def change_char_task(ictx: CtxInteraction, source:str, params:dict):
    user_name = get_user_ctx_inter(ictx).display_name
    channel = ictx.channel
    change_embed = None
    try:
        char_params = params.get('character', {})
        char_name = char_params.get('char_name', {})
        verb = char_params.get('verb', 'Changing')
        mode = char_params.get('mode', 'change')
        if change_embed_info:
            change_embed_info.title = f'{verb} character ... '
            change_embed_info.description = f'{user_name} requested character {mode}: "{char_name}"'
            change_embed = await channel.send(embed=change_embed_info)
        # Change character
        await change_character(char_name, channel, source)
        # Set history
        if bot_history.autoload_history and (bot_history.change_char_history_method == 'keep' and source != 'reset'):
            bot_history.load_bot_history()
        else:
            if source == 'reset':
                bot_history.reset_session_history(ictx)
            else:
                bot_history.reset_session_history()
        if change_embed:
            await change_embed.delete()
            # Send embeds to announcement channels
            if bot_database.announce_channels:
                await bg_task_queue.put(announce_changes(ictx, 'character', char_name))
            else:
                # Send change embed to interaction channel
                change_embed_info.title = f"{user_name} changed character:"
                change_embed_info.description = f'**{char_name}**'
                await channel.send(embed=change_embed_info)
        if not bot_history.per_channel_history_enabled:
            await send_char_greeting_or_history(ictx, char_name)
        logging.info(f"Character loaded: {char_name}")
    except Exception as e:
        logging.error(f'An error occurred while loading character for "{source}": {e}')
        if change_embed_info:
            change_embed_info.title = "An error occurred while loading character"
            change_embed_info.description = e
            if change_embed:
                await change_embed.edit(embed=change_embed_info)

#################################################################
######################## MAIN TASK QUEUE ########################
#################################################################
def bot_should_do(tags={}, prior_set={}):
    # Defaults
    bot_will_do = {'should_gen_text': True,
                   'should_send_text': True,
                   'should_gen_image': False,
                   'should_send_image': True}
    # Update defaults with anything previously set
    bot_will_do.update(prior_set)
    try:
        # iterate through matched tags and update
        matches = tags.get('matches', {})
        if matches:
            for item in matches:
                if isinstance(item, tuple):
                    tag, _, _ = item
                else:
                    tag = item
                for key, value in tag.items():
                    if key in bot_will_do:
                        bot_will_do[key] = value
        # Disable things as set by config
        if not textgenwebui_enabled:
            bot_will_do['should_gen_text'] = False
            bot_will_do['should_send_text'] = False
        if not sd_enabled:
            bot_will_do['should_gen_image'] = False
            bot_will_do['should_send_image'] = False
    except Exception as e:
        logging.error(f"An error occurred while checking if bot should do '{key}': {e}")
    return bot_will_do

# For @ mentioning users who were not last replied to
previous_user_mention = ''

def update_mention(user_mention: str, last_resp:str=''):
    global previous_user_mention
    mention_resp = last_resp

    if user_mention != previous_user_mention:
        mention_resp = f"{user_mention} {last_resp}"
    previous_user_mention = user_mention
    return mention_resp

flow_event = asyncio.Event()
flow_queue = asyncio.Queue()

#################################################################
########################## QUEUED FLOW ##########################
#################################################################
async def format_next_flow(next_flow, user_name:str, text:str):
    flow_name = ''
    formatted_flow_tags = {}
    for key, value in next_flow.items():
        # get name for message embed
        if key == 'flow_step':
            flow_name = f": {value}"
        # format prompt before feeding it back into on_message_task()
        elif key == 'format_prompt':
            formatting = {'format_prompt': [value]}
            text = process_tag_formatting(user_name, text, formatting)
        # see if any tag values have dynamic formatting (user prompt, LLM reply, etc)
        elif key != 'format_prompt' and isinstance(value, str):
            formatted_value = format_prompt_with_recent_output(user_name, value)       # output will be a string
            if formatted_value != value:                                        # if the value changed,
                formatted_value = parse_tag_from_text_value(formatted_value)    # convert new string to correct value type
            formatted_flow_tags[key] = formatted_value
        # apply wildcards
        text = await dynamic_prompting(user_name, text, i=None)
    next_flow.update(formatted_flow_tags) # commit updates
    return flow_name, text

# function to get a copy of the next queue item while maintaining the original queue
async def peek_flow_queue(queue, user_name:str, text:str):
    temp_queue = asyncio.Queue()
    total_queue_size = queue.qsize()
    first_flow = None
    while queue.qsize() > 0:
        if queue.qsize() == total_queue_size:
            item = await queue.get()
            flow_name, formatted_text = await format_next_flow(item, user_name, text)
        else:
            item = await queue.get()
        await temp_queue.put(item)
    # Enqueue the items back into the original queue
    while temp_queue.qsize() > 0:
        item_to_put_back = await temp_queue.get()
        await queue.put(item_to_put_back)
    return flow_name, formatted_text

async def flow_task(ictx: CtxInteraction, source:str, text:str):
    user_name = get_user_ctx_inter(ictx).display_name
    channel = ictx.channel
    try:
        global flow_event
        flow_embed = None
        total_flow_steps = flow_queue.qsize()
        if flow_embed_info:
            flow_embed_info.title = f'Processing Flow for {user_name} with {total_flow_steps} steps'
            flow_embed_info.description = ''
            flow_embed = await channel.send(embed=flow_embed_info)
        while flow_queue.qsize() > 0:   # flow_queue items are removed in get_tags()
            flow_name, text = await peek_flow_queue(flow_queue, user_name, text)
            remaining_flow_steps = flow_queue.qsize()
            if flow_embed_info:
                flow_embed_info.description = flow_embed_info.description.replace("**Processing", ":white_check_mark: **")
                flow_embed_info.description += f'**Processing Step {total_flow_steps + 1 - remaining_flow_steps}/{total_flow_steps}**{flow_name}\n'
                if flow_embed: await flow_embed.edit(embed=flow_embed_info)
            await on_message_task(ictx, source, text)
        if flow_embed_info:
            flow_embed_info.title = f"Flow completed for {user_name}"
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


async def run_flow_if_any(ictx: CtxInteraction, source:str, text:str):
    if flow_queue.qsize() > 0:
        # flows are activated in process_llm_payload_tags(), and is where the flow queue is populated
        await flow_task(ictx, source, text)

#################################################################
#################### QUEUED IMAGE GENERATION ####################
#################################################################
async def sd_online(channel: discord.TextChannel):
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

async def sd_img_gen(channel, temp_dir:str, img_payload:dict, endpoint:str):
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

async def process_image_gen(img_payload:dict, channel, params:dict):
    try:
        bot_will_do = params.get('bot_will_do', {})
        img_censoring = params.get('img_censoring', 0)
        endpoint = params.get('endpoint', '/sdapi/v1/txt2img')
        default_save_path = os.path.join('ad_discordbot', 'sd_outputs')
        sd_output_dir = params.get('sd_output_dir', default_save_path)
        # Ensure the necessary directories exist
        os.makedirs(sd_output_dir, exist_ok=True)
        temp_dir = os.path.join('ad_discordbot', 'user_images', '__temp')
        os.makedirs(temp_dir, exist_ok=True)
        # Generate images, save locally
        images = await sd_img_gen(channel, temp_dir, img_payload, endpoint)
        if not images:
            return
        # Send images to discord
        # If the censor mode is 1 (blur), prefix the image file with "SPOILER_"
        file_prefix = 'temp_img_'
        if img_censoring == 1:
            file_prefix = 'SPOILER_temp_img_'
        image_files = [discord.File(f'{temp_dir}/temp_img_{idx}.png', filename=f'{file_prefix}{idx}.png') for idx in range(len(images))]
        if bot_will_do['should_send_image']:
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

        # Clean up extension keys
        extensions = config.get('sd', {}).get('extensions', {})
        alwayson_scripts = img_payload.get('alwayson_scripts', {})
        # Clean ControlNet
        if alwayson_scripts.get('controlnet'):
            # Delete all 'controlnet' keys if disabled by config
            if not extensions.get('controlnet_enabled'):
                del alwayson_scripts['controlnet']
        # Clean Forge Couple
        if alwayson_scripts.get('forge_couple'):
            # Delete all 'forge_couple' keys if disabled by config
            if not extensions.get('forgecouple_enabled') or img_payload.get('init_images'):
                del alwayson_scripts['forge_couple']
            else:
                img_payload['alwayson_scripts']['forge_couple']['args'] = list(img_payload['alwayson_scripts']['forge_couple']['args'].values()) # convert dictionary to list
                img_payload['alwayson_scripts']['forge couple'] = img_payload['alwayson_scripts'].pop('forge_couple') # Add the required space between "forge" and "couple" ("forge couple")
        # Clean layerdiffuse
        if alwayson_scripts.get('layerdiffuse'):
            # Delete all 'layerdiffuse' keys if disabled by config
            if not extensions.get('layerdiffuse_enabled'):
                del alwayson_scripts['layerdiffuse']
            else:
                img_payload['alwayson_scripts']['layerdiffuse']['args'] = list(img_payload['alwayson_scripts']['layerdiffuse']['args'].values()) # convert dictionary to list
        # Clean ReActor
        if alwayson_scripts.get('reactor'):
            # Delete all 'reactor' keys if disabled by config
            if not extensions.get('reactor_enabled'):
                del alwayson_scripts['reactor']
            else:
                img_payload['alwayson_scripts']['reactor']['args'] = list(img_payload['alwayson_scripts']['reactor']['args'].values()) # convert dictionary to list

        # Workaround for denoising strength bug
        if not img_payload.get('enable_hr', False) and not img_payload.get('init_images', False):
            img_payload['denoising_strength'] = None

        # Fix SD Client compatibility for sampler names / schedulers
        sampler_name = img_payload.get('sampler_name', '')
        if sampler_name:
            known_schedulers = [' uniform', ' karras', ' exponential', ' polyexponential', ' sgm uniform']
            for value in known_schedulers:
                if sampler_name.lower().endswith(value):
                    if not bot_database.was_warned('sampler_name'):
                        bot_database.update_was_warned('sampler_name')
                        # Extract the value (without leading space) and set it to the 'scheduler' key
                        img_payload['scheduler'] = value.strip()
                        if SD_CLIENT == 'A1111 SD WebUI':
                            logging.warning(f'Img payload value "sampler_name": "{sampler_name}" is incompatible with current version of "{SD_CLIENT}". "{value}" must be omitted from "sampler_name", and instead used for the "scheduler" parameter. This is being corrected automatically. To avoid this warning, please update "sampler_name" parameter wherever present in your settings.')
                            # Remove the matched part from sampler_name
                            start_index = sampler_name.lower().rfind(value)
                            fixed_sampler_name = sampler_name[:start_index].strip()
                            img_payload['sampler_name'] = fixed_sampler_name
                            bot_settings.settings['imgmodel']['payload']['sampler_name'] = fixed_sampler_name
                            bot_settings.settings['imgmodel']['payload']['scheduler'] = value.strip()
                        else:
                            logging.warning(f'Img payload value "sampler_name": "{sampler_name}" may cause an error due to the scheduler ("{value}") being part of the value. The scheduler may be expected as a separate parameter in current version of "{SD_CLIENT}".')
                        break

        # Delete all empty keys
        keys_to_delete = []
        for key, value in img_payload.items():
            if value == "":
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del img_payload[key]
    except Exception as e:
        logging.error(f"An error occurred when cleaning img_payload: {e}")
    return img_payload

def apply_loractl(tags):
    try:
        if SD_CLIENT != 'A1111 SD WebUI':
            if not bot_database.was_warned('loractl'):
                bot_database.update_was_warned('loractl')
                logging.warning(f'loractl is not known to be compatible with "{SD_CLIENT}". Not applying loractl...')
            return tags
        scaling_settings = [v for k, v in config['sd'].get('extensions', {}).get('lrctl', {}).items() if 'scaling' in k]
        scaling_settings = scaling_settings if scaling_settings else ['']
        # Flatten the matches dictionary values to get a list of all tags (including those within tuples)
        matched_tags = [tag if isinstance(tag, dict) else tag[0] for tag in tags['matches']]
        # Filter the matched tags to include only those with certain patterns in their text fields
        lora_tags = [tag for tag in matched_tags if any(patterns.sd_lora.findall(text) for text in (tag.get('positive_prompt', ''), tag.get('positive_prompt_prefix', ''), tag.get('positive_prompt_suffix', '')))]
        if len(lora_tags) >= config['sd']['extensions']['lrctl']['min_loras']:
            for index, tag in enumerate(lora_tags):
                # Determine the key with a non-empty value among the specified keys
                used_key = next((key for key in ['positive_prompt', 'positive_prompt_prefix', 'positive_prompt_suffix'] if tag.get(key, '')), None)
                if used_key:  # If a key with a non-empty value is found
                    positive_prompt = tag[used_key]
                    lora_matches = patterns.sd_lora.findall(positive_prompt)
                    if lora_matches:
                        for lora_match in lora_matches:
                            lora_weight_match = patterns.sd_lora_weight.search(lora_match) # Extract lora weight
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

def process_img_prompt_tags(img_payload:dict, tags:dict) -> dict:
    try:
        img_prompt, tags = process_tag_insertions(img_payload['prompt'], tags)
        updated_positive_prompt = img_prompt
        updated_negative_prompt = img_payload['negative_prompt']
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

def convert_lists_to_tuples(dictionary:dict) -> dict:
    for key, value in dictionary.items():
        if isinstance(value, list) and len(value) == 2 and all(isinstance(item, (int, float)) for item in value) and not any(isinstance(item, bool) for item in value):
            dictionary[key] = tuple(value)
    return dictionary

def process_param_variances(param_variances: dict) -> dict:
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

async def process_img_payload_tags(img_payload:dict, mods:dict, params:dict):
    try:
        flow = mods.pop('flow', None)
        change_imgmodel = mods.pop('change_imgmodel', None)
        swap_imgmodel = mods.pop('swap_imgmodel', None)
        payload = mods.pop('payload', None)
        aspect_ratio = mods.pop('aspect_ratio', None)
        param_variances = mods.pop('param_variances', {})
        controlnet = mods.pop('controlnet', [])
        forge_couple = mods.pop('forge_couple', {})
        layerdiffuse = mods.pop('layerdiffuse', {})
        reactor = mods.pop('reactor', {})
        img2img = mods.pop('img2img', {})
        img2img_mask = mods.pop('img2img_mask', {})
        # Process the tag matches
        if flow or change_imgmodel or swap_imgmodel or payload or aspect_ratio or param_variances or controlnet or forge_couple or layerdiffuse or reactor or img2img or img2img_mask:
            # Flow handling
            if flow is not None and not flow_event.is_set():
                await build_flow_queue(flow)
            # Imgmodel handling
            new_imgmodel = change_imgmodel or swap_imgmodel or None
            if new_imgmodel:
                    ## IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
                    ## if not change_imgmodel and swap_imgmodel and swap_imgmodel == 'None':
                        # _ = await sd_api(endpoint='/sdapi/v1/unload-checkpoint', method='post', json=None, retry=True)
                params['imgmodel'] = await get_selected_imgmodel_data(new_imgmodel) # {sd_model_checkpoint, imgmodel_name, filename}
                current_sd_model_checkpoint = bot_settings.settings['imgmodel'].get('override_settings', {}).get('sd_model_checkpoint') or bot_settings.settings['imgmodel']['payload'].get('override_settings', {}).get('sd_model_checkpoint') or ''
                current_imgmodel_name = bot_settings.settings['imgmodel'].get('imgmodel_name')
                # Check if new model same as current model
                if current_imgmodel_name == params['imgmodel'].get('imgmodel_name', ''):
                    logging.info(f'[TAGS] Img model was triggered to change, but it is the same as current ("{current_imgmodel_name}").')
                else:
                    params['imgmodel']['current_imgmodel_name'] = current_imgmodel_name
                    params['imgmodel']['current_sd_model_checkpoint'] = current_sd_model_checkpoint
                    params['imgmodel']['mode'] = 'change' if new_imgmodel == change_imgmodel else 'swap'
                    params['imgmodel']['verb'] = 'Changing' if params['imgmodel']['mode'] == 'change' else 'Swapping'
                    logging.info(f'[TAGS] {params["imgmodel"]["verb"]} Img model: "{params["imgmodel"].get("imgmodel_name", "")}"')
            # Payload handling
            if payload:
                if isinstance(payload, dict):
                    logging.info(f"[TAGS] Updated payload: '{payload}'")
                    update_dict(img_payload, payload)
                else:
                    logging.warning("A tag was matched with invalid 'payload'; must be a dictionary.")
            # Aspect Ratio
            if aspect_ratio:
                try:
                    current_avg = get_current_avg_from_dims()
                    n, d = get_aspect_ratio_parts(aspect_ratio)
                    w, h = dims_from_ar(current_avg, n, d)
                    img_payload['width'], img_payload['height'] = w, h
                    logging.info(f'[TAGS] Applied aspect ratio "{aspect_ratio}" (Width: "{w}", Height: "{h}").')
                except:
                    pass
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
                img_payload['init_images'] = [str(img2img)]
                params['endpoint'] = '/sdapi/v1/img2img'
            # Inpaint Mask handling
            if img2img_mask:
                img_payload['mask'] = str(img2img_mask)
        return img_payload, params
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

def collect_img_tag_values(tags, params):
    img_payload_mods = {}
    payload_order_hack = {}
    controlnet_args = {}
    forge_couple_args = {}
    layerdiffuse_args = {}
    reactor_args = {}
    extensions = config.get('sd', {}).get('extensions', {})
    accept_only_first = ['flow', 'aspect_ratio', 'img2img', 'img2img_mask']
    try:
        for tag in tags['matches']:
            if isinstance(tag, tuple):
                tag = tag[0] # For tags with prompt insertion indexes
            for key, value in tag.items():
                # Accept only the first occurance
                if key in accept_only_first and not img_payload_mods.get(key):
                    img_payload_mods[key] = value
                elif key == 'sd_output_dir' and not params.get('sd_output_dir'):
                    params['sd_output_dir'] = str(value)
                elif key == 'img_censoring' and not params.get('img_censoring'):
                    params['img_censoring'] = int(value)
                    logging.info(f"[TAGS] Censoring: {'Image Blurred' if value == 1 else 'Generation Blocked'}")
                # Accept only first 'change' or 'swap'
                elif key == 'change_imgmodel' or key == 'swap_imgmodel' and not (img_payload_mods.get('change_imgmodel') or img_payload_mods.get('swap_imgmodel')):
                    img_payload_mods[key] = str(value)
                # Allow multiple to accumulate
                elif key == 'payload':
                    try:
                        if img_payload_mods.get('payload'):
                            payload_order_hack = dict(value)
                            update_dict(payload_order_hack, img_payload_mods['payload'])
                            img_payload_mods['payload'] = payload_order_hack
                        else:
                            img_payload_mods['payload'] = dict(value)
                    except:
                        logging.warning("Error processing a matched 'payload' tag; ensure it is a dictionary.")
                elif key == 'img_param_variances':
                    img_payload_mods.setdefault('param_variances', {})
                    try:
                        update_dict(img_payload_mods['param_variances'], dict(value))
                    except:
                        logging.warning("Error processing a matched 'img_param_variances' tag; ensure it is a dictionary.")
                # get any ControlNet extension params
                elif key.startswith('controlnet') and extensions.get('controlnet_enabled'):
                    index = int(key[len('controlnet'):]) if key != 'controlnet' else 0  # Determine the index (cnet unit) for main controlnet args
                    controlnet_args.setdefault(index, {}).update({'image': value, 'enabled': True})         # Update controlnet args at the specified index
                elif key.startswith('cnet') and extensions.get('controlnet_enabled'):
                    # Determine the index for controlnet_args sublist
                    if key.startswith('cnet_'):
                        index = int(key.split('_')[0][len('cnet'):]) if not key.startswith('cnet_') else 0  # Determine the index (cnet unit) for additional controlnet args
                    controlnet_args.setdefault(index, {}).update({key.split('_', 1)[-1]: value})   # Update controlnet args at the specified index
                # get any layerdiffuse extension params
                elif key == 'layerdiffuse' and extensions.get('layerdiffuse_enabled'):
                    img_payload_mods['layerdiffuse']['method'] = str(value)
                elif key.startswith('laydiff_') and extensions.get('layerdiffuse_enabled'):
                    laydiff_key = key[len('laydiff_'):]
                    layerdiffuse_args[laydiff_key] = value
                # get any ReActor extension params
                elif key == 'reactor' and extensions.get('reactor_enabled'):
                    img_payload_mods['reactor']['image'] = value
                elif key.startswith('reactor_') and extensions.get('reactor_enabled'):
                    reactor_key = key[len('reactor_'):]
                    reactor_args[reactor_key] = value
                # get any Forge Couple extension params
                elif key == 'forge_couple' and extensions.get('forgecouple_enabled'):
                    if value.startswith('['):
                        img_payload_mods['forge_couple']['maps'] = list(value)
                    else: img_payload_mods['forge_couple']['direction'] = str(value)
                elif key.startswith('couple_') and extensions.get('forgecouple_enabled'):
                    forge_couple_key = key[len('couple_'):]
                    if value.startswith('['):
                        forge_couple_args[forge_couple_key] = list(value)
                    else:
                        forge_couple_args[forge_couple_key] = str(value)
                # get any user image(s)
                elif key == 'send_user_image':
                    user_image_args = get_image_tag_args('User image', str(value), key=None, set_dir=None)
                    user_image = discord.File(user_image_args)
                    user_image = discord.File(user_image_args)
                    params.setdefault('send_user_image', [])
                    params['send_user_image'].append(user_image)
                    logging.info(f'[TAGS] Sending user image.')
        # Add the collected SD WebUI extension args to the img_payload_mods dict
        if controlnet_args:
            img_payload_mods.setdefault('controlnet', [])
            for index in sorted(set(controlnet_args.keys())):   # This flattens down any gaps between collected ControlNet units (ensures lowest index is 0, next is 1, and so on)
                cnet_basesettings = copy.copy(bot_settings.settings['imgmodel']['payload']['alwayson_scripts']['controlnet']['args'][0])  # Copy of required dict items
                cnet_unit_args = controlnet_args.get(index, {})
                cnet_unit = update_dict(cnet_basesettings, cnet_unit_args)
                img_payload_mods['controlnet'].append(cnet_unit)
        if forge_couple_args:
            img_payload_mods.setdefault('forge_couple', {})
            img_payload_mods['forge_couple'].update(forge_couple_args)
        if layerdiffuse_args:
            img_payload_mods.setdefault('layerdiffuse', {})
            img_payload_mods['layerdiffuse'].update(layerdiffuse_args)
        if reactor_args:
            img_payload_mods.setdefault('reactor', {})
            img_payload_mods['reactor'].update(reactor_args)

        img_payload_mods = collect_img_extension_mods(img_payload_mods)
    except Exception as e:
        logging.error(f"Error collecting Img tag values: {e}")
    return img_payload_mods, params

def init_img_payload(img_prompt:str, neg_prompt:str) -> dict:
    try:
        # Initialize img_payload settings
        img_payload = {"prompt": img_prompt, "negative_prompt": neg_prompt}
        # Apply settings from imgmodel configuration
        imgmodel_img_payload = copy.deepcopy(bot_settings.settings['imgmodel'].get('payload', {}))
        img_payload.update(imgmodel_img_payload)
        img_payload['override_settings'] = copy.deepcopy(bot_settings.settings['imgmodel'].get('override_settings', {}))
        return img_payload

    except Exception as e:
        logging.error(f"Error initializing img payload: {e}")

def match_img_tags(img_prompt:str, tags:dict) -> dict:
    try:
        # Unmatch any previously matched tags which try to insert text into the img_prompt
        for tag in tags['matches'][:]:  # Iterate over a copy of the list
            tag:dict

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

    except Exception as e:
        logging.error(f"Error matching tags for img phase: {e}")

    return tags

async def img_gen_task(source:str, img_prompt:str, params:dict, ictx:CtxInteraction, tags={}):
    user_name = get_user_ctx_inter(ictx).display_name or None
    channel = ictx.channel
    bot_will_do = params.get('bot_will_do', {})
    img_censoring = params.get('img_censoring', 0)
    try:
        check_key = bot_settings.settings['imgmodel'].get('override_settings', {}) or bot_settings.settings['imgmodel'].get('payload', {}).get('override_settings', {})
        if check_key.get('sd_model_checkpoint', '') == 'None': # Model currently unloaded
            await channel.send("**Cannot process image request:** No Img model is currently loaded")
            logging.warning(f'Bot tried to generate image for {user_name}, but no Img model was loaded')
        if not tags:
            img_prompt, tags = await get_tags(img_prompt)
            tags = match_img_tags(img_prompt, tags)
            bot_will_do = bot_should_do(tags)
        # Initialize img_payload
        neg_prompt = params.get('neg_prompt', '')
        img_payload = init_img_payload(img_prompt, neg_prompt)
        # collect matched tag values
        img_payload_mods, params = collect_img_tag_values(tags, params)
        send_user_image = img_payload_mods.pop('send_user_image', [])
        # Apply tags relevant to Img gen
        img_payload, params = await process_img_payload_tags(img_payload, img_payload_mods, params)
        # Check censoring
        if img_censoring == 2:
            if img_send_embed_info:
                img_send_embed_info.title = "Image prompt was flagged as inappropriate."
                img_send_embed_info.description = ""
                await channel.send(embed=img_send_embed_info)
            return
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
        imgmodel_params = params.get('imgmodel', {})
        if imgmodel_params:
            # Add new checkpoint to payload
            img_payload['override_settings']['sd_model_checkpoint'] = imgmodel_params.get('sd_model_checkpoint', '')
            swap_params = {'imgmodel': {}}
            swap_params['imgmodel']['imgmodel_name'] = imgmodel_params.pop('current_imgmodel_name', '')
            swap_params['imgmodel']['sd_model_checkpoint'] = imgmodel_params.pop('current_sd_model_checkpoint', '')
            should_swap = await change_imgmodel_task(user_name, channel, params, ictx)
        # Generate and send images
        params['bot_will_do'] = bot_will_do
        await process_image_gen(img_payload, channel, params)
        if (source == 'image' or (bot_will_do['should_send_text'] and not bot_will_do['should_gen_text'])) and img_send_embed_info:
            img_send_embed_info.title = f"{user_name} requested an image:"
            img_send_embed_info.description = params.get('message', img_prompt)
            if ictx:
                if hasattr(ictx, 'followup'): await ictx.followup.reply(embed=img_send_embed_info)
                else: await ictx.reply(embed=img_send_embed_info)
            else: await channel.send(embed=img_send_embed_info)
        if send_user_image:
            await channel.send(file=send_user_image) if len(send_user_image) == 1 else await channel.send(files=send_user_image)
        # If switching back to original Img model
        if should_swap:
            swap_params['imgmodel']['mode'] = 'swap_back'
            swap_params['imgmodel']['verb'] = 'Swapping back to'
            await change_imgmodel_task(user_name, channel, swap_params, ictx)
        return
    except Exception as e:
        logging.error(f"An error occurred in img_gen_task(): {e}")

#################################################################
######################## /IMAGE COMMAND #########################
#################################################################
if sd_enabled:

    # Updates size options for /image command
    async def update_size_options(average):
        global size_choices
        options = load_file(shared_path.cmd_options, {})
        sizes = options.get('sizes', [])
        aspect_ratios = [size.get("ratio") for size in sizes.get('ratios', [])]
        size_choices.clear()  # Clear the existing list
        ratio_options = calculate_aspect_ratio_sizes(average, aspect_ratios)
        static_options = sizes.get('static_sizes', [])
        size_options = (ratio_options or []) + (static_options or [])
        size_choices.extend(
            app_commands.Choice(name=option['name'], value=option['name'])
            for option in size_options)
        await client.tree.sync()

    async def get_imgcmd_choices(size_options, style_options) -> tuple[list[app_commands.Choice], list[app_commands.Choice]]:
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

    def get_current_avg_from_dims():
        w = bot_active_settings.get('imgmodel', {}).get('payload', {}).get('width', 512)
        h = bot_active_settings.get('imgmodel', {}).get('payload', {}).get('height', 512)
        return avg_from_dims(w, h)

    async def get_imgcmd_options():
        try:
            options = load_file(shared_path.cmd_options, {})
            options = dict(options)
            # Get sizes and aspect ratios from 'dict_cmdoptions.yaml'
            sizes = options.get('sizes', {})
            aspect_ratios = [size.get("ratio") for size in sizes.get('ratios', [])]
            # Calculate the average and aspect ratio sizes
            current_avg = get_current_avg_from_dims()
            ratio_options = calculate_aspect_ratio_sizes(current_avg, aspect_ratios)
            # Collect any defined static sizes
            static_options = sizes.get('static_sizes', [])
            # Merge dynamic and static sizes
            size_options = (ratio_options or []) + (static_options or [])
            # Get style and controlnet options
            style_options = options.get('styles', {})
            return size_options, style_options

        except Exception as e:
            logging.error(f"An error occurred while building options for /image: {e}")

    async def get_cnet_data() -> dict:

        async def check_cnet_online(endpoint):
            if config['sd']['extensions'].get('controlnet_enabled', False):
                try:
                    online = await sd_api(endpoint='/controlnet/model_list', method='get', json=None, retry=False)
                    if online: return True
                    else: return False
                except:
                    logging.warning(f"ControlNet is enabled in config.yaml, but was not responsive from {SD_CLIENT} API.")
            return False

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
                cnet_online = await check_cnet_online()
                if cnet_online:
                    logging.warning("ControlNet is both enabled in config.yaml and detected. However, ad_discordbot relies on the '/controlnet/control_types' \
                        API endpoint which is missing. See here: (https://github.com/altoiddealer/ad_discordbot/wiki/troubleshooting).")
        return filtered_cnet_data

    # Get size and style options for /image command
    size_options, style_options = asyncio.run(get_imgcmd_options())
    size_choices, style_choices = asyncio.run(get_imgcmd_choices(size_options, style_options))

    # Check if extensions enabled in config
    cnet_enabled = config.get('sd', {}).get('extensions', {}).get('controlnet_enabled', False)
    reactor_enabled = config.get('sd', {}).get('extensions', {}).get('reactor_enabled', False)

    if cnet_enabled and reactor_enabled:
        @client.hybrid_command(name="image", description=f'Generate an image using {SD_CLIENT}')
        @app_commands.describe(style='Applies a positive/negative prompt preset')
        @app_commands.describe(img2img='Diffuses from an input image instead of pure latent noise.')
        @app_commands.describe(img2img_mask='Masks the diffusion strength for the img2img input. Requires img2img.')
        @app_commands.describe(face_swap='For best results, attach a square (1:1) cropped image of a face, to swap into the output.')
        @app_commands.describe(controlnet='Guides image diffusion using an input image or map.')
        @app_commands.choices(size=size_choices)
        @app_commands.choices(style=style_choices)
        async def image(ctx: commands.Context, prompt: str, size: typing.Optional[app_commands.Choice[str]], style: typing.Optional[app_commands.Choice[str]], neg_prompt: typing.Optional[str], img2img: typing.Optional[discord.Attachment], img2img_mask: typing.Optional[discord.Attachment],
            face_swap: typing.Optional[discord.Attachment], controlnet: typing.Optional[discord.Attachment]):
            user_selections = {"prompt": prompt, "size": size.value if size else None, "style": style.value if style else None, "neg_prompt": neg_prompt, "img2img": img2img if img2img else None, "img2img_mask": img2img_mask if img2img_mask else None,
            "face_swap": face_swap if face_swap else None, "cnet": controlnet if controlnet else None}
            await process_image(ctx, user_selections)
    elif cnet_enabled and not reactor_enabled:
        @client.hybrid_command(name="image", description=f'Generate an image using {SD_CLIENT}')
        @app_commands.describe(style='Applies a positive/negative prompt preset')
        @app_commands.describe(img2img='Diffuses from an input image instead of pure latent noise.')
        @app_commands.describe(img2img_mask='Masks the diffusion strength for the img2img input. Requires img2img.')
        @app_commands.describe(controlnet='Guides image diffusion using an input image or map.')
        @app_commands.choices(size=size_choices)
        @app_commands.choices(style=style_choices)
        async def image(ctx: commands.Context, prompt: str, size: typing.Optional[app_commands.Choice[str]], style: typing.Optional[app_commands.Choice[str]], neg_prompt: typing.Optional[str], img2img: typing.Optional[discord.Attachment], img2img_mask: typing.Optional[discord.Attachment],
            controlnet: typing.Optional[discord.Attachment]):
            user_selections = {"prompt": prompt, "size": size.value if size else None, "style": style.value if style else None, "neg_prompt": neg_prompt, "img2img": img2img if img2img else None, "img2img_mask": img2img_mask if img2img_mask else None,
            "cnet": controlnet if controlnet else None}
            await process_image(ctx, user_selections)
    elif reactor_enabled and not cnet_enabled:
        @client.hybrid_command(name="image", description=f'Generate an image using {SD_CLIENT}')
        @app_commands.describe(style='Applies a positive/negative prompt preset')
        @app_commands.describe(img2img='Diffuses from an input image instead of pure latent noise.')
        @app_commands.describe(img2img_mask='Masks the diffusion strength for the img2img input. Requires img2img.')
        @app_commands.describe(face_swap='For best results, attach a square (1:1) cropped image of a face, to swap into the output.')
        @app_commands.choices(size=size_choices)
        @app_commands.choices(style=style_choices)
        async def image(ctx: commands.Context, prompt: str, size: typing.Optional[app_commands.Choice[str]], style: typing.Optional[app_commands.Choice[str]], neg_prompt: typing.Optional[str], img2img: typing.Optional[discord.Attachment], img2img_mask: typing.Optional[discord.Attachment],
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
        async def image(ctx: commands.Context, prompt: str,  size: typing.Optional[app_commands.Choice[str]], style: typing.Optional[app_commands.Choice[str]], neg_prompt: typing.Optional[str], img2img: typing.Optional[discord.Attachment], img2img_mask: typing.Optional[discord.Attachment]):
            user_selections = {"prompt": prompt, "size": size.value if size else None, "style": style.value if style else None, "neg_prompt": neg_prompt, "img2img": img2img if img2img else None, "img2img_mask": img2img_mask if img2img_mask else None}
            await process_image(ctx, user_selections)

    async def process_image(ctx: commands.Context, selections):
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
            prompt = await dynamic_prompting(ctx.author.display_name, prompt, i=None)
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
                # Get filtered ControlNet data
                cnet_data = await get_cnet_data()
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
                    logging.info(f'{ctx.author.display_name} used "/image": "{prompt}"')
                    await img_gen_task('image', prompt, params, ctx, tags={})
                    await run_flow_if_any(ctx, 'image', prompt)

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

    @client.hybrid_command(description="Display performance statistics")
    async def statistics_llm_gen(ctx):
        statistics_dict = bot_statistics.llm.data
        description_lines = []
        for key, value in statistics_dict.items():
            if key == 'time_total' or key == 'tokens_per_sec_avg':
                formatted_value, label = format_time(value)
                description_lines.append(f"{key}: {formatted_value} {label}")
            else:
                description_lines.append(f"{key}: {value}")
        formatted_description = "\n".join(description_lines)
        system_embed_info.title = "Bot LLM Gen Statistics:"
        system_embed_info.description = f">>> {formatted_description}"
        await ctx.send(embed=system_embed_info)

@client.hybrid_command(description="Toggle current channel as an announcement channel for the bot (model changes)")
async def announce(ctx: commands.Context):
    try:
        if ctx.channel.id in bot_database.announce_channels:
            bot_database.announce_channels.remove(ctx.channel.id) # If the channel is already in the announce channels, remove it
            action_message = f'Removed {ctx.channel.mention} from announce channels. Use "/announce" again if you want to add it back.'
        else:
            # If the channel is not in the announce channels, add it
            bot_database.announce_channels.append(ctx.channel.id)
            action_message = f'Added {ctx.channel.mention} to announce channels. Use "/announce" again to remove it.'

        bot_database.save()
        await ctx.reply(action_message)
    except Exception as e:
        logging.error(f"Error toggling announce channel setting: {e}")

@client.hybrid_command(description="Toggle current channel as main channel for bot to auto-reply without needing to be called")
async def main(ctx: commands.Context):
    try:
        if ctx.channel.id in bot_database.main_channels:
            bot_database.main_channels.remove(ctx.channel.id) # If the channel is already in the main channels, remove it
            action_message = f'Removed {ctx.channel.mention} from main channels. Use "/main" again if you want to add it back.'
        else:
            # If the channel is not in the main channels, add it
            bot_database.main_channels.append(ctx.channel.id)
            action_message = f'Added {ctx.channel.mention} to main channels. Use "/main" again to remove it.'

        bot_database.save()
        await ctx.reply(action_message)
    except Exception as e:
        logging.error(f"Error toggling main channel setting: {e}")

@client.hybrid_command(description="Update dropdown menus without restarting bot script.")
async def sync(ctx: commands.Context):
    try:
        await ctx.reply('Syncing client tree. Note: Menus may not update instantly.', ephemeral=True, delete_after=10)
        logging.info(f"{ctx.author.display_name} used '/sync' to sync the client.tree (refresh commands).")
        await bg_task_queue.put(client.tree.sync()) # Process this in the background
    except Exception as e:
        logging.error(f"Error syncing client.tree with '/sync': {e}")

#################################################################
######################### LLM COMMANDS ##########################
#################################################################
if textgenwebui_enabled:
    # /reset command - Resets current character
    @client.hybrid_command(description="Reset the conversation with current character")
    async def reset_conversation(ctx: commands.Context):
        try:
            shared.stop_everything = True
            await ireply(ctx, 'character reset') # send a response msg to the user
            bot_history.reset_session_history(ctx)

            async with task_semaphore:
                # offload to ai_gen queue
                logging.info(f'{ctx.author.display_name} used "/reset": "{bot_database.last_character}"')
                params = {'character': {'char_name': bot_database.last_character, 'verb': 'Resetting', 'mode': 'reset'}}
                await change_char_task(ctx, 'reset', params)

        except Exception as e:
            logging.error(f"Error with /reset: {e}")

    # /reset command - Resets current character
    @client.hybrid_command(description="Saves the current conversation to a new file in text-generation-webui/logs/")
    async def save_conversation(ctx: commands.Context):
        try:
            await ctx.reply('Saved current conversation history', ephemeral=True)
            bot_history.save_history()
        except Exception as e:
            logging.error(f"Error with /reset: {e}")

    # Context menu command to Regenerate last reply
    @client.tree.context_menu(name="regenerate")
    async def regen_llm_gen(inter: discord.Interaction, message:discord.Message):
        text = message.content
        await inter.response.defer(thinking=False)

        async with task_semaphore:
            async with inter.channel.typing():
                # offload to ai_gen queue
                logging.info(f'{inter.user.display_name} used "Regenerate"')
                await cont_regen_task(inter, 'regen', text, message.id)
                await run_flow_if_any(inter, 'regen', text)

    # Context menu command to Continue last reply
    @client.tree.context_menu(name="continue")
    async def continue_llm_gen(inter: discord.Interaction, message:discord.Message):
        text = message.content
        await inter.response.defer(thinking=False)

        async with task_semaphore:
            async with inter.channel.typing():
                # offload to ai_gen queue
                logging.info(f'{inter.user.display_name} used "Continue"')
                await cont_regen_task(inter, 'cont', text, message.id)
                await run_flow_if_any(inter, 'cont', text)

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
async def character_loader(char_name, channel=None, source=None):
    try:
        # Get data using textgen-webui native character loading function
        _, name, _, greeting, context = load_character(char_name, '', '')
        missing_keys = [key for key, value in {'name': name, 'greeting': greeting, 'context': context}.items() if not value]
        if any (missing_keys):
            logging.warning(f'Note that character "{char_name}" is missing the following info:"{missing_keys}".')
        textgen_data = {'name': name, 'greeting': greeting, 'context': context}
        # Check for extra bot data
        char_data = await load_character_data(char_name)
        char_instruct = char_data.get('instruction_template_str', None)
        # Merge with basesettings
        char_data = merge_base(char_data, 'llmcontext')
        # Reset warning for character specific TTS
        bot_database.update_was_warned('char_tts', False)
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
        char_llmstate['character_menu'] = char_name
        # Commit the character data to bot_settings.settings
        bot_settings.settings['llmcontext'] = dict(char_llmcontext) # Replace the entire dictionary key
        state_dict = bot_settings.settings['llmstate']['state']
        update_dict(state_dict, dict(char_llmstate))
        bot_behavior.update_behavior(dict(char_behavior))
        # Print mode in cmd
        logging.info(f"Initializing in {state_dict['mode']} mode")
        # Check for any char defined or model defined instruct_template
        update_instruct = char_instruct or instruction_template_str or None # 'instruction_template_str' is global variable
        if update_instruct:
            state_dict['instruction_template_str'] = update_instruct
        # Update stored database value for character
        bot_database.set('last_character', char_name)
        # Update discord username / avatar
        await update_client_profile(char_name, channel)
        # Mirror the changes in bot_active_settings
        bot_active_settings['llmcontext'] = char_llmcontext
        bot_active_settings['behavior'] = char_behavior
        bot_active_settings['llmstate']['state'] = char_llmstate
        bot_active_settings.save()
    except Exception as e:
        logging.error(f"Error loading character. Check spelling and file structure. Use bot cmd '/character' to try again. {e}")

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
        bot_database.set('last_change', time.time())  # Store the current time in bot_database_v2.yaml
    except Exception as e:
        logging.error(f"Error while changing character username or avatar: {e}")

async def update_client_profile(char_name, channel=None):
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
        last_change = bot_database.last_change
        last_cooldown = last_change + timedelta(minutes=10).seconds
        if time.time() >= last_cooldown:
            # Apply changes immediately if outside 10 minute cooldown
            delayed_profile_update_task = asyncio.create_task(delayed_profile_update(char_name, avatar, 0))
        else:
            remaining_cooldown = last_cooldown - time.time()
            seconds = int(remaining_cooldown)
            if channel:
                await channel.send(f'**Due to Discord limitations, character name/avatar will update in {seconds} seconds.**', delete_after=10)
            logging.info(f"Due to Discord limitations, character name/avatar will update in {remaining_cooldown} seconds.")
            delayed_profile_update_task = asyncio.create_task(delayed_profile_update(char_name, avatar, seconds))
    except Exception as e:
        logging.error(f"An error occurred while updating Discord profile: {e}")

# Apply character changes
async def change_character(char_name, channel, source):
    try:
        # Load the character
        await character_loader(char_name, channel, source)
        # Update all settings
        bot_settings.update_settings()
        await bot_settings.update_base_tags()
    except Exception as e:
        await channel.send(f"An error occurred while changing character: {e}")
        logging.error(f"An error occurred while changing character: {e}")

async def process_character(ctx, selected_character_value):
    try:
        if not selected_character_value:
            await ctx.reply('**No character was selected**.', ephemeral=True, delete_after=5)
            return
        char_name = Path(selected_character_value).stem
        await ireply(ctx, 'character change') # send a response msg to the user

        async with task_semaphore:
            # offload to ai_gen queue
            logging.info(f'{ctx.author.display_name} used "/character": "{char_name}"')
            params = {'character': {'char_name': char_name, 'verb': 'Changing', 'mode': 'change'}}
            await change_char_task(ctx, 'character', params)

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

                char_data = load_file(file, {})
                if not char_data:
                    continue

                char_data = dict(char_data) # TODO does yaml loader behave weird that we need to convert it to a dict here?
                if char_data.get('bot_in_character_menu', True):
                    filtered_characters.append(character)

    except Exception as e:
        logging.error(f"An error occurred while getting all characters: {e}")
    return all_characters, filtered_characters

if textgenwebui_enabled:
    # Command to change characters
    @client.hybrid_command(description="Choose a character")
    async def character(ctx: commands.Context):
        try:
            _, filtered_characters = get_all_characters()
            if filtered_characters:
                items_for_character = [i['name'] for i in filtered_characters]
                warned_too_many_character = False # TODO use the warned_once feature?
                characters_view = SelectOptionsView(items_for_character,
                                                custom_id_prefix='characters',
                                                placeholder_prefix='Characters: ',
                                                unload_item=None,
                                                warned=warned_too_many_character)
                view_message = await ctx.send('### Select a Character.', view=characters_view, ephemeral=True)
                await characters_view.wait()

                selected_item = characters_view.get_selected()
                await view_message.delete()
                await process_character(ctx, selected_item)
            else:
                await ctx.send('There are no characters available', ephemeral=True)
        except Exception as e:
            logging.error(f"An error occurred while selecting a Character from '/characters' command: {e}")

#################################################################
####################### /IMGMODEL COMMAND #######################
#################################################################
# Apply user defined filters to imgmodel list
async def filter_imgmodels(imgmodels:list) -> list:
    try:
        imgmodels_data = load_file(shared_path.img_models, {})
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

# Get current list of imgmodels from API
async def fetch_imgmodels() -> list:
    try:
        imgmodels = await sd_api(endpoint='/sdapi/v1/sd-models', method='get', json=None, retry=False)
        # Replace key names for easier management
        for imgmodel in imgmodels:
            if 'title' in imgmodel:
                imgmodel['sd_model_checkpoint'] = imgmodel.pop('title')
            if 'model_name' in imgmodel:
                imgmodel['imgmodel_name'] = imgmodel.pop('model_name')
        imgmodels = await filter_imgmodels(imgmodels)
        return imgmodels

    except Exception as e:
        logging.error(f"Error fetching image models: {e}")
        return []

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

async def change_imgmodel(selected_imgmodel:dict):
    # Merge selected imgmodel/tag data with base settings
    async def merge_new_imgmodel_data(selected_imgmodel:dict):
        try:
            selected_imgmodel_name = selected_imgmodel.get('imgmodel_name')
            ### IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
            # if selected_imgmodel_name == 'None': # Unloading model
            #     selected_imgmodel_tags = []
            #     return selected_imgmodel, selected_imgmodel_name, selected_imgmodel_tags
            # Get tags if defined
            selected_imgmodel_tags = None
            imgmodel_settings = {'payload': {}, 'override_settings': {}}
            imgmodels_data = load_file(shared_path.img_models, {})
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
            return {}

    # Save new Img model data
    async def save_new_imgmodel_settings(selected_imgmodel, selected_imgmodel_tags):
        try:
            # get current/new average width/height for '/image' cmd size options
            current_avg = get_current_avg_from_dims()
            new_avg = avg_from_dims(selected_imgmodel.get('payload', {}).get('width', 512), selected_imgmodel.get('payload', {}).get('height', 512))
            bot_active_settings['imgmodel'] = selected_imgmodel
            bot_active_settings['imgmodel']['tags'] = selected_imgmodel_tags
            bot_active_settings.save()
            # Update all settings
            bot_settings.update_settings()
            await bot_settings.update_base_tags()

            ### IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
            # if selected_imgmodel['imgmodel_name'] == 'None':
            # _ = await sd_api(endpoint='/sdapi/v1/unload-checkpoint', method='post', json=None)
            #     change_embed.title = 'Unloaded Img model'
            #     change_embed.description = ''
            #     await channel.send(embed=change_embed)
            #     return
            # Load the imgmodel and VAE via API
            model_data = bot_settings.settings['imgmodel'].get('override_settings') or bot_settings.settings['imgmodel']['payload'].get('override_settings')
            _ = await sd_api(endpoint='/sdapi/v1/options', method='post', json=model_data, retry=True)
            # Update size options for /image command if old/new averages are different
            if current_avg != new_avg:
                await bg_task_queue.put(update_size_options(new_avg))
        except Exception as e:
            logging.error(f"Error updating settings with the selected imgmodel data: {e}")

    selected_imgmodel, selected_imgmodel_name, selected_imgmodel_tags = await merge_new_imgmodel_data(selected_imgmodel)
    await save_new_imgmodel_settings(selected_imgmodel, selected_imgmodel_tags)

async def get_selected_imgmodel_data(selected_imgmodel_value:str) -> dict:
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
        all_imgmodels = await fetch_imgmodels()
        for imgmodel in all_imgmodels:
            # check that the value matches a valid checkpoint
            if selected_imgmodel_value == (imgmodel.get('imgmodel_name') or imgmodel.get('sd_model_checkpoint')):
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
            logging.info(f'{ctx.author.display_name} used "/imgmodel": "{selected_imgmodel_value}"')
            params = {}
            params['imgmodel'] = await get_selected_imgmodel_data(selected_imgmodel_value) # {sd_model_checkpoint, imgmodel_name, filename}
            await change_imgmodel_task(ctx.author.display_name, ctx.channel, params, ctx)

    except Exception as e:
        logging.error(f"Error processing selected imgmodel from /imgmodel command: {e}")

if sd_enabled:

    @client.hybrid_command(description="Choose an Img Model")
    async def imgmodel(ctx: commands.Context):
        try:
            all_imgmodels = await fetch_imgmodels()
            if all_imgmodels:
                ### IF API IMG MODEL UNLOADING GETS EVER DEBUGGED
                # unload_options = [app_commands.Choice(name="Unload Model", value="None"),
                # app_commands.Choice(name="Do Not Unload Model", value="Exit")]
                items_for_img_model = [i["imgmodel_name"] for i in all_imgmodels]
                warned_too_many_img_model = False # TODO use the warned_once feature?
                imgmodels_view = SelectOptionsView(items_for_img_model,
                                                custom_id_prefix='imgmodels',
                                                placeholder_prefix='ImgModels: ',
                                                unload_item=None,
                                                warned=warned_too_many_img_model)
                view_message = await ctx.send('### Select an Image Model.', view=imgmodels_view, ephemeral=True)
                await imgmodels_view.wait()
                selected_item = imgmodels_view.get_selected()
                await view_message.delete()
                await process_imgmodel(ctx, selected_item)
            else:
                await ctx.send('There are no Img models available', ephemeral=True)
        except Exception as e:
            logging.error(f"An error occurred while selecting an Img model from '/imgmodel' command: {e}")

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
            logging.info(f'{ctx.author.display_name} used "/llmmodel": "{selected_llmmodel}"')
            params = {'llmmodel': {'llmmodel_name': selected_llmmodel, 'verb': 'Changing', 'mode': 'change'}}
            await change_llmmodel_task(ctx, params)

    except Exception as e:
        logging.error(f"Error processing /llmmodel command: {e}")

if textgenwebui_enabled:

    @client.hybrid_command(description="Choose an LLM Model")
    async def llmmodel(ctx: commands.Context):
        try:
            all_llmmodels = utils.get_available_models()
            if all_llmmodels:
                items_for_llm_model = [i for i in all_llmmodels]
                unload_llmmodel = items_for_llm_model.pop(0)
                warned_too_many_llm_model = False # TODO use the warned_once feature?
                llmmodels_view = SelectOptionsView(items_for_llm_model,
                                                custom_id_prefix='llmmodels',
                                                placeholder_prefix='LLMModels: ',
                                                unload_item=unload_llmmodel,
                                                warned=warned_too_many_llm_model)
                view_message = await ctx.send('### Select an LLM Model.', view=llmmodels_view, ephemeral=True)
                await llmmodels_view.wait()
                selected_item = llmmodels_view.get_selected()
                await view_message.delete()
                await process_llmmodel(ctx, selected_item)
            else:
                await ctx.send('There are no LLM models available', ephemeral=True)
        except Exception as e:
            logging.error(f"An error occurred while selecting an LLM model from '/llmmodel' command: {e}")

#################################################################
####################### /SPEAK COMMAND #######################
#################################################################
async def process_speak_silero_non_eng(ctx: commands.Context, lang):
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

async def process_speak_args(ctx: commands.Context, selected_voice=None, lang=None, user_voice=None):
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

async def process_user_voice(ctx: commands.Context, voice_input=None):
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

async def process_speak(ctx: commands.Context, input_text, selected_voice=None, lang=None, voice_input=None):
    try:
        user_voice = await process_user_voice(ctx, voice_input)
        tts_args = await process_speak_args(ctx, selected_voice, lang, user_voice)
        await ireply(ctx, 'tts') # send a response msg to the user

        async with task_semaphore:
            async with ctx.channel.typing():
                # offload to ai_gen queue
                logging.info(f'{ctx.author.display_name} used "/speak": "{input_text}"')
                params = {'tts_args': tts_args, 'user_voice': user_voice}
                await speak_task(ctx, input_text, params)
                await run_flow_if_any(ctx, 'speak', input_text)

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

    _voice_hash_dict = {str(hash(voice_name)):voice_name for voice_name in all_voices}

    voice_options = [app_commands.Choice(name=voice_name.replace('_', ' ').title(), value=str(hash(voice_name))) for voice_name in all_voices[:25]]
    voice_options_label = f'{voice_options[0].name[0]}-{voice_options[-1].name[0]}'.lower()
    if len(all_voices) > 25:
        voice_options1 = [app_commands.Choice(name=voice_name.replace('_', ' ').title(), value=str(hash(voice_name))) for voice_name in all_voices[25:50]]
        voice_options1_label = f'{voice_options1[0].name[0]}-{voice_options1[-1].name[0]}'.lower()
        if voice_options1_label == voice_options_label:
            voice_options1_label = f'{voice_options1_label}_1'
        if len(all_voices) > 50:
            voice_options2 = [app_commands.Choice(name=voice_name.replace('_', ' ').title(), value=str(hash(voice_name))) for voice_name in all_voices[50:75]]
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
        async def speak(ctx: commands.Context, input_text: str, voice: typing.Optional[app_commands.Choice[str]], lang: typing.Optional[app_commands.Choice[str]], voice_input: typing.Optional[discord.Attachment]):
            # Only generate TTS for the server conntected to Voice Channel
            if voice_client and (voice_client != ctx.guild.voice_client) and int(tts_settings.get('play_mode', 0)) == 0:
                await ctx.send('Voice Channel is not enabled on this server', ephemeral=True, delete_after=5)
                return
            selected_voice = voice.value if voice is not None else ''
            if selected_voice:
                selected_voice = _voice_hash_dict[selected_voice]
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
        async def speak(ctx: commands.Context, input_text: str, voice_1: typing.Optional[app_commands.Choice[str]], voice_2: typing.Optional[app_commands.Choice[str]], lang: typing.Optional[app_commands.Choice[str]], voice_input: typing.Optional[discord.Attachment]):
            # Only generate TTS for the server conntected to Voice Channel
            if voice_client and (voice_client != ctx.guild.voice_client) and int(tts_settings.get('play_mode', 0)) == 0:
                await ctx.send('Voice Channel is not enabled on this server', ephemeral=True, delete_after=5)
                return
            if voice_1 and voice_2:
                await ctx.send("A voice was picked from two separate menus. Using the first selection.", ephemeral=True)
            selected_voice = ((voice_1 or voice_2) and (voice_1 or voice_2).value) or ''
            if selected_voice:
                selected_voice = _voice_hash_dict[selected_voice]
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
        async def speak(ctx: commands.Context, input_text: str, voice_1: typing.Optional[app_commands.Choice[str]], voice_2: typing.Optional[app_commands.Choice[str]], voice_3: typing.Optional[app_commands.Choice[str]], lang: typing.Optional[app_commands.Choice[str]], voice_input: typing.Optional[discord.Attachment]):
            # Only generate TTS for the server conntected to Voice Channel
            if voice_client and (voice_client != ctx.guild.voice_client) and int(tts_settings.get('play_mode', 0)) == 0:
                await ctx.send('Voice Channel is not enabled on this server', ephemeral=True, delete_after=5)
                return
            if sum(1 for v in (voice_1, voice_2, voice_3) if v) > 1:
                await ctx.send("A voice was picked from two separate menus. Using the first selection.", ephemeral=True)
            selected_voice = ((voice_1 or voice_2 or voice_3) and (voice_1 or voice_2 or voice_3).value) or ''
            if selected_voice:
                selected_voice = _voice_hash_dict[selected_voice]
            voice_input = voice_input if voice_input is not None else ''
            lang = lang.value if lang is not None else ''
            await process_speak(ctx, input_text, selected_voice, lang, voice_input)

#################################################################
####################### DEFAULT SETTINGS ########################
#################################################################
class Behavior:
    def __init__(self):
        self.reply_to_itself = 0.0
        self.chance_to_reply_to_other_bots = 0.5
        self.reply_to_bots_when_addressed = 0.3
        self.only_speak_when_spoken_to = True
        self.ignore_parentheses = True
        self.go_wild_in_channel = True
        self.conversation_recency = 600
        self.user_conversations = {}
        # New Behaviors
        self.maximum_typing_speed = -1
        self.responsiveness = 1.0
        self.max_reply_delay = 30.0
        self.msg_size_affects_delay = False
        self.spontaneous_msg_chance = 0.0
        self.spontaneous_msg_max_consecutive = -1
        self.spontaneous_msg_min_wait = 10.0
        self.spontaneous_msg_max_wait = 60.0
        self.spontaneous_msg_prompts = {}

    def update_behavior(self, behavior):
        for key, value in behavior.items():
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

    def bot_should_reply(self, message:discord.Message, text:str) -> bool:
        # Don't reply to @everyone or to itself
        if message.mention_everyone or (message.author == client.user and not self.probability_to_reply(self.reply_to_itself)):
            return False
        # Whether to reply to other bots
        if message.author.bot and bot_database.last_character.lower() in text.lower() and message.channel.id in bot_database.main_channels:
            if 'bye' in text.lower(): # don't reply if another bot is saying goodbye
                return False
            return self.probability_to_reply(self.reply_to_bots_when_addressed)
        # Whether to reply when text is nested in parentheses
        if self.ignore_parentheses and (message.content.startswith('(') and message.content.endswith(')')) or (message.content.startswith('<:') and message.content.endswith(':>')):
            return False
        # Whether to reply if only speak when spoken to
        if (self.only_speak_when_spoken_to and (client.user.mentioned_in(message) or any(word in message.content.lower() for word in bot_database.last_character.lower().split()))) \
            or (self.in_active_conversation(message.author.id) and message.channel.id in bot_database.main_channels):
            return True
        reply = False
        # few more conditions
        if message.author.bot and message.channel.id in bot_database.main_channels:
            reply = self.probability_to_reply(self.chance_to_reply_to_other_bots)
        if self.go_wild_in_channel and message.channel.id in bot_database.main_channels:
            reply = True
        if reply:
            self.update_user_dict(message.author.id)
        return reply

    def probability_to_reply(self, probability):
        # Determine if the bot should reply based on a probability
        return random.random() < probability

# Sub-classes under a main class 'Settings'
class ImgModel:
    def __init__(self):
        self.tags = []
        self.imgmodel_name = '' # label used for /imgmodel command
        self.override_settings = {}
        self.payload = {'alwayson_scripts': {}}
        self.init_sd_extensions()

    def refresh_enabled_extensions(self):
        self.init_sd_extensions()
        new_payload = merge_base(self.payload, 'imgmodel,payload')
        update_dict(bot_active_settings['imgmodel']['payload'], new_payload)
        bot_active_settings.save()

    def init_sd_extensions(self):
        extensions = config.get('sd', {}).get('extensions', {})
        # Initialize ControlNet defaults
        if extensions.get('controlnet_enabled'):
            self.payload['alwayson_scripts']['controlnet'] = {'args': [{
                'enabled': False, 'image': None, 'mask_image': None, 'model': 'None', 'module': 'None', 'weight': 1.0, 'processor_res': 64, 'pixel_perfect': True,
                'guidance_start': 0.0, 'guidance_end': 1.0, 'threshold_a': 64, 'threshold_b': 64, 'control_mode': 0, 'resize_mode': 1, 'lowvram': False, 'save_detected_map': False}]}
            if SD_CLIENT:
                logging.info(f'"ControlNet" extension support is enabled and active.')
        # Initialize Forge Couple defaults
        if extensions.get('forgecouple_enabled'):
            self.payload['alwayson_scripts']['forge_couple'] = {'args': {
                'enable': False, 'mode': 'Basic', 'sep': 'SEP', 'direction': 'Horizontal', 'global_effect': 'First Line',
                'global_weight': 0.5, 'maps': [['0:0.5', '0.0:1.0', '1.0'],['0.5:1.0', '0.0:1.0', '1.0']]}}
            if SD_CLIENT:
                logging.info(f'"Forge Couple" extension support is enabled and active.')
            # Warn Non-Forge:
            if SD_CLIENT and SD_CLIENT != 'SD WebUI Forge':
                logging.warning(f'"Forge Couple" is not known to be compatible with "{SD_CLIENT}". If you experience errors, disable this extension in config.yaml')
        # Initialize layerdiffuse defaults
        if extensions.get('layerdiffuse_enabled'):
            self.payload['alwayson_scripts']['layerdiffuse'] = {'args': {
                'enabled': False, 'method': '(SDXL) Only Generate Transparent Image (Attention Injection)', 'weight': 1.0, 'stop_at': 1.0, 'foreground': None, 'background': None,
                'blending': None, 'resize_mode': 'Crop and Resize', 'output_mat_for_i2i': False, 'fg_prompt': '', 'bg_prompt': '', 'blended_prompt': ''}}
            if SD_CLIENT:
                logging.info(f'"layerdiffuse" extension support is enabled and active.')
            if SD_CLIENT and SD_CLIENT != 'SD WebUI Forge':
                logging.warning(f'"layerdiffuse" is not known to be compatible with "{SD_CLIENT}". If you experience errors, disable this extension in config.yaml')
        # Initialize ReActor defaults
        if extensions.get('reactor_enabled'):
            self.payload['alwayson_scripts']['reactor'] = {'args': {
                'image': '', 'enabled': False, 'source_faces': '0', 'target_faces': '0', 'model': 'inswapper_128.onnx', 'restore_face': 'CodeFormer', 'restore_visibility': 1,
                'restore_upscale': True, 'upscaler': '4x_NMKD-Superscale-SP_178000_G', 'scale': 1.5, 'upscaler_visibility': 1, 'swap_in_source_img': False, 'swap_in_gen_img': True, 'log_level': 1,
                'gender_detect_source': 0, 'gender_detect_target': 0, 'save_original': False, 'codeformer_weight': 0.8, 'source_img_hash_check': False, 'target_img_hash_check': False, 'system': 'CUDA',
                'face_mask_correction': True, 'source_type': 0, 'face_model': '', 'source_folder': '', 'multiple_source_images': None, 'random_img': True, 'force_upscale': True, 'threshold': 0.6, 'max_faces': 2}}
            if SD_CLIENT:
                logging.info(f'"ReActor" extension support is enabled and active.')
            
class LLMContext:
    def __init__(self):
        self.context = 'The following is a conversation with an AI Large Language Model. The AI has been trained to answer questions, provide recommendations, and help with decision making. The AI follows user requests. The AI thinks outside the box.'
        self.extensions = {}
        self.greeting = '' # 'How can I help you today?'
        self.name = 'AI'
        self.use_voice_channel = False
        self.bot_in_character_menu = True
        self.tags = []

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

class Settings:
    def __init__(self, bot_behavior):
        self.bot_behavior = bot_behavior
        self.imgmodel = ImgModel()
        self.llmcontext = LLMContext()
        self.llmstate = LLMState()
        self.settings = {}
        self.base_tags = []
        # Initialize main settings and base tags
        self.update_settings()
        asyncio.run(self.update_base_tags())

    async def update_base_tags(self):
        try:
            tags_data = load_file(shared_path.tags, {})
            base_tags_data = tags_data.get('base_tags', [])
            base_tags = copy.deepcopy(base_tags_data)
            base_tags = await update_tags(base_tags)
            self.base_tags = base_tags

        except Exception as e:
            logging.error(f"Error updating client base tags: {e}")

    # Returns the value of Settings as a dictionary
    def settings_to_dict(self):
        return {
            'imgmodel': vars(self.imgmodel),
            'llmcontext': vars(self.llmcontext),
            'llmstate': vars(self.llmstate)
        }

    def update_settings(self):
        defaults = self.settings_to_dict()
        # Current user custom settings
        active_settings = copy.deepcopy(bot_active_settings.get_vars())
        behavior = active_settings.pop('behavior', {})
        # Add any missing required settings
        self.settings = fix_dict(active_settings, defaults)
        bot_behavior.update_behavior(behavior)

    # Allows printing default values of Settings
    def __str__(self):
        attributes = ", ".join(f"{attr}={getattr(self, attr)}" for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__"))
        return f"{self.__class__.__name__}({attributes})"
    
class History:
    def __init__(self):
        # History settings
        chat_history = config.get('textgenwebui', {}).get('chat_history', {})
        self.limit_history = chat_history.get('limit_history', True)
        self.autosave_history = chat_history.get('autosave_history', False)
        self.autoload_history = chat_history.get('autoload_history', False)
        self.change_char_history_method = chat_history.get('change_char_history_method', 'new')
        self.greeting_or_history = chat_history.get('greeting_or_history', 'history')
        self.per_channel_history_enabled = chat_history.get('per_channel_history', True)
        # History management
        self.unique_id = None
        self.session_history = {}
        self.recent_messages = {}
        self.collected_prompts = {}

    # Modified version of TGQUI function
    def load_history(self, unique_id, character, mode):
        type = 'single'
        p = get_history_file_path(unique_id, character, mode)
        f = json.loads(open(p, 'rb').read())
        if 'internal' in f and 'visible' in f:
            history = f
        elif 'data' in f and 'data_visible' in f:
            history = {'internal': f['data'],
                       'visible': f['data_visible']}
        else:
            history = f
            type = 'multichan'
        return history, type

    # Modified version of TGQUI function
    def load_latest_history(self, state):
        type = 'single'
        histories = find_all_histories(state)
        if len(histories) > 0:
            history, type = self.load_history(histories[0], state['character_menu'], state['mode'])
        else:
            history = {}
        return history, type

    # Loads most recent history for current character
    def load_bot_history(self):
        state_dict = bot_settings.settings['llmstate']['state']
        values_to_load_history = {'character_menu': state_dict['character_menu'],
                                'mode': state_dict['mode']}
        latest_history, history_type = self.load_latest_history(values_to_load_history)
        # Only load history if most recent history type matches current history mode
        matched = None
        if self.per_channel_history_enabled:
            if history_type == 'multichan':
                matched = 'multichan'
        else:
            if history_type == 'single':
                matched = 'single'
        if matched:
            self.session_history = dict(latest_history)
            # print recent exchange if per-channel history
            if matched == 'single':
                last_exchange = self.session_history['visible'][-1] if self.session_history.get('visible') else None
                if last_exchange:
                    last_user_message = last_exchange[0]
                    last_assistant_message = last_exchange[1]
                    logging.info(f'Loaded most recent chat history. Last message exchange:\n User: "{last_user_message}"\n {bot_database.last_character}: "{last_assistant_message}"')
                else:
                    logging.info("Starting new conversation.")
            if matched == 'multichan':
                logging.info("Loaded most recent chat history for all channels.")
            all_histories = find_all_histories(values_to_load_history)
            if len(all_histories) > 0:
                self.unique_id = all_histories[0]

    # Save history to a new file
    def save_bot_history(self):
        state_dict = bot_settings.settings['llmstate']['state']
        mode = state_dict['mode']
        character_menu = state_dict["character_menu"]
        if not self.unique_id:
            if self.per_channel_history_enabled:
                self.unique_id = f"{datetime.now().strftime('%Y%m%d-%H-%M-%S')}_multiple-history"
            else:
                self.unique_id = datetime.now().strftime('%Y%m%d-%H-%M-%S')
            logging.info(f'''Chat history was saved to "/logs/{mode}/{character_menu}/{self.unique_id}.json"''')
        save_history(self.session_history, self.unique_id, character_menu, mode)
    
    # Truncates history to approx. same length as 'truncation_length' state parameter
    def limit_prompt_history(self, i_list:list, v_list:list):
        truncation = int(bot_settings.settings['llmstate']['state']['truncation_length'] * 4) #approx tokens
        while (sum(len(message) for exchange in i_list for message in exchange) > truncation) \
            or (sum(len(message) for exchange in v_list for message in exchange) > truncation):
            if i_list and v_list:
                i_list.pop(0) # pop oldest exchanges
                v_list.pop(0)

    # Manage the session history for textgen-webui
    def manage_prompt_history(self, cp_list:list, i_list:list, v_list:list, prompt:str, reply:str=None, save_to_history:bool=True):
        # Only prompt is received (no 'reply)
        if reply is None:
            # If there are previously collected prompts, join with '\n\n'
            if cp_list:
                cp_list += '\n\n' + prompt
            else:
                cp_list = prompt
        # Both prompt and reply are received
        else:
            # If there are previously collected prompts, merge and reset collected prompts
            if cp_list:
                prompt = cp_list + '\n\n' + prompt
                # Reset collected prompts
                cp_list = []
            if save_to_history:
                i_list.append([prompt, reply])
                # Do not append Visible list for per-channel history
                if self.per_channel_history_enabled:
                    v_list.append(['', ''])
                else:
                    v_list.append([prompt, reply])
        #TODO return prompt

    def get_history_iv_lists_keys(self, chankey:str = None):
        if self.per_channel_history_enabled:
            # internal and visible lists
            i_list = self.session_history.setdefault(chankey, {}).setdefault('internal', [])
            v_list = self.session_history[chankey].setdefault('visible', [])
        else:
            # internal and visible lists
            i_list = self.session_history.setdefault('internal', [])
            v_list = self.session_history.setdefault('visible', [])
        return i_list, v_list

    def get_history_ul_lists_keys(self, chankey:str = None):
        if self.per_channel_history_enabled:
            # user and llm lists
            u_list = self.recent_messages.setdefault(chankey, {}).setdefault('user', [])
            l_list = self.recent_messages[chankey].setdefault('llm', [])
        else:
            # user and llm lists
            u_list = self.recent_messages.setdefault('user', [])
            l_list = self.recent_messages.setdefault('llm', [])
        return u_list, l_list

    def get_history_cp_list_key(self, chankey:str = None):
        if self.per_channel_history_enabled:
            # collected prompts lists
            cp_list = self.collected_prompts.setdefault(chankey, [])
        else:
            # collected prompts lists
            cp_list = self.collected_prompts
        return cp_list

    def get_history_lists_keys(self, chankey:str = None):
        i_list, v_list = self.get_history_iv_lists_keys(chankey)
        u_list, l_list = self.get_history_ul_lists_keys(chankey)
        cp_list = self.get_history_cp_list_key(chankey)

        return i_list, v_list, u_list, l_list, cp_list

    # Retain most recent elements or characters from user prompts and bot replies (mainly for Flows feature)
    def manage_recent_messages(self, u_list:list, l_list:list, prompt:str, reply:str=None):
        if self.per_channel_history_enabled:
            elem_limit = 5
            character_limit = 3000
        else:
            elem_limit = 10
            character_limit = 10000

        if prompt:
            u_list.insert(0, prompt)
            while len(u_list) > elem_limit or sum(len(message) for message in u_list) > character_limit:
                oldest_message = u_list.pop()
        if reply:
            l_list.insert(0, reply)
            while len(l_list) > elem_limit or sum(len(message) for message in l_list) > character_limit:
                oldest_message = l_list.pop()

    def manage_history(self, prompt:str, reply:str=None, save_to_history:bool=True, chankey:str=None):
        # Get list keys to simplify code in further steps
        i_list, v_list, u_list, l_list, cp_list = self.get_history_lists_keys(chankey)
        # Update recent user/LLM messages (separate from chat history)
        self.manage_recent_messages(u_list, l_list, prompt, reply)
        # Update history
        self.manage_prompt_history(cp_list, i_list, v_list, prompt, reply, save_to_history)
        # Truncate history
        if self.limit_history:
            self.limit_prompt_history(i_list, v_list)
        # Save history
        if self.autosave_history:
            self.save_bot_history()

    def set_history_key_defaults(self, ictx=None):
        # If per-channel history
        if self.per_channel_history_enabled and ictx:
            chkey = str(ictx.channel.id)
            self.session_history[chkey] = {
                'guild_name': str(ictx.guild),
                'channel_name': str(ictx.channel),
                'internal': [],
                'visible': []
                }
            self.collected_prompts[chkey] = []
        # If only one history
        else:
            self.session_history = {'internal': [], 'visible': []}
            self.collected_prompts = []

    def get_channel_history(self, ictx=None):
        # If per-channel history
        if self.per_channel_history_enabled:
            chkey = str(ictx.channel.id)
            if not self.session_history.get(chkey):
                self.set_history_key_defaults(ictx)
            return self.session_history[chkey]
        # If only one history
        else:
            if not self.session_history:
                self.set_history_key_defaults()
            return self.session_history

    def reset_session_history(self, ictx=None):
        # If per-channel history
        if self.per_channel_history_enabled:
            # if no interaction, all history will be reset
            if ictx:
                chkey = str(ictx.channel.id)
                guild_chan = f'{ictx.guild} - {ictx.channel}'
                logging.info(f"Starting new conversation in: {guild_chan}.")
                # If channel has history
                if self.session_history.get(chkey):
                    self.set_history_key_defaults(ictx)
                # if channel does not have history
                else:
                    self.get_channel_history(ictx) # will initialize channel keys
                return
            else:
                logging.info("Starting new conversation in all channels.")
        # If only one history
        else:
            logging.info("Starting new conversation.")
        # Reset everything
        self.session_history = {}
        self.collected_prompts = {}

bot_behavior = Behavior() # needs to be loaded before settings
bot_settings = Settings(bot_behavior=bot_behavior)
bot_history = History()

client.run(bot_token, log_handler=log_file_handler, log_formatter=log_file_formatter)