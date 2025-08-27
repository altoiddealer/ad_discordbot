# pyright: reportOptionalMemberAccess=false
import logging as _logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import Any, Optional, Tuple, Union, Literal
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
from discord import app_commands, File, abc
import typing
import io
import base64
import yaml
from PIL import Image
import requests
import aiohttp
import math
import time
from threading import Lock
from pydub import AudioSegment
import copy
from shutil import copyfile
import sys
import traceback
from modules.typing import ChannelID, UserID, MessageID, CtxInteraction, FILE_INPUT, APIRequestCancelled  # noqa: F401
import signal
from functools import partial
import inspect
import types

from modules.utils_files import load_file, merge_base, save_yaml_file  # noqa: F401
from modules.utils_shared import client, TOKEN, is_tgwui_integrated, shared_path, bg_task_queue, task_event, flows_queue, flows_event, patterns, bot_emojis, config, bot_database, get_api
from modules.database import StarBoard, Statistics, BaseFileMemory
from modules.utils_misc import check_probability, fix_dict, set_key, deep_merge, update_dict, sum_update_dict, random_value_from_range, convert_lists_to_tuples, \
    consolidate_prompt_strings, get_time, format_time, format_time_difference, get_normalized_weights, get_pnginfo_from_image, is_base64, valueparser # noqa: F401
from modules.utils_processing import resolve_placeholders, collect_content_to_send, send_content_to_discord, comfy_delete_and_reroute_nodes
from modules.utils_discord import Embeds, guild_only, guild_or_owner_only, configurable_for_dm_if, custom_commands_check_dm, is_direct_message, ireply, sleep_delete_message, send_long_message, \
    EditMessageModal, SelectedListItem, SelectOptionsView, get_user_ctx_inter, get_message_ctx_inter, apply_reactions_to_messages, replace_msg_in_history_and_discord, MAX_MESSAGE_LENGTH, muffled_send  # noqa: F401
from modules.utils_aspect_ratios import ar_parts_from_dims, dims_from_ar, avg_from_dims, get_aspect_ratio_parts, calculate_aspect_ratio_sizes  # noqa: F401
from modules.utils_chat import custom_load_character, load_character_data
from modules.history import HistoryManager, History, HMessage, cnf
from modules.typing import AlertUserError, TAG
from modules.utils_asyncio import generate_in_executor
from modules.tags import base_tags, persistent_tags, Tags, TAG_LIST

from discord.ext.commands.errors import HybridCommandError, CommandError
from discord.errors import DiscordException
from discord.app_commands import AppCommandError, CommandInvokeError
from modules.logs import import_track, get_logger, log_file_handler, log_file_formatter; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

from modules.apis import API, APIClient, Endpoint, ImgGenEndpoint, ImgGenClient, TTSGenClient, TextGenClient
api:API = asyncio.run(get_api())
from modules.stepexecutor import call_stepexecutor

imggen_enabled = config.imggen.get('enabled', True)

# Databases
starboard = StarBoard()
bot_statistics = Statistics()

#################################################################
#################### DISCORD / BOT STARTUP ######################
#################################################################
bot_embeds = Embeds()

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="You have modified the pretrained model configuration to control generation")

# Method to check if an API is enabled and available
async def api_online(client_type:str|None=None, client_name:str='', strict=False, ictx:CtxInteraction|None=None) -> bool:
    api_client:Optional[APIClient] = api.get_client(client_type=client_type, client_name=client_name, strict=strict)
    if not api_client:
        return False

    api_client_online, emsg = await api_client.is_online()
    if not api_client_online and emsg and ictx:
        await bot_embeds.send('system', f"{api_client.name} is not running at: {api_client.url}", emsg, channel=ictx.channel, delete_after=10)
        return False
    elif not api_client_online:
        return False
    else:
        return True

#################################################################
##################### TEXTGENWEBUI STARTUP ######################
#################################################################
if is_tgwui_integrated:
    log.info('The bot is installed with text-generation-webui integration. Loading applicable modules and features.')
    sys.path.append(shared_path.dir_tgwui)

    from modules.utils_tgwui import tgwui, tgwui_shared_module, tgwui_utils_module, tgwui_extensions_module, get_tgwui_functions
    
else:
    log.warning('The bot is NOT installed with text-generation-webui integration.')
    log.warning('Features related to text generation and TTS will not be available.')
    log.warning('To integrate the bot with TGWUI, please refer to the github.')

# Must be TGWUI integrated install, and also enabled in config
tgwui_enabled = is_tgwui_integrated and tgwui.enabled

#################################################################
##################### BACKGROUND QUEUE TASK #####################
#################################################################
async def process_tasks_in_background():
    while True:
        task = await bg_task_queue.get()
        await task

#################################################################
########################## BOT STARTUP ##########################
#################################################################
def disable_unsupported_features():
    if callable(api.imggen):
        if config.controlnet_enabled() and not api.imggen.is_sdwebui_variant():
            log.warning('ControlNet is enabled in config.yaml, but is currently only supported by A1111-like WebUIs (A1111/Forge/ReForge). Disabling.')
            config.imggen['extensions']['controlnet_enabled'] = False
        if config.reactor_enabled() and not api.imggen.is_sdwebui_variant():
            log.warning('ReActor is enabled in config.yaml, but is currently only supported by A1111-like WebUIs (A1111/Forge/ReForge). Disabling.')
            config.imggen['extensions']['reactor_enabled'] = False
        if config.forgecouple_enabled() and not api.imggen.is_forge():
            log.warning('Forge Couple is enabled in config.yaml, but is currently only supported by SD Forge. Disabling.')
            config.imggen['extensions']['forgecouple_enabled'] = False
        if config.layerdiffuse_enabled() and not api.imggen.is_forge():
            log.warning('Layerdiffuse is enabled in config.yaml, but is currently only supported by SD Forge. Disabling.')
            config.imggen['extensions']['layerdiffuse_enabled'] = False
        if config.imggen.get('extensions', {}).get('loractl', {}).get('enabled', False) and not api.imggen.supports_loractrl():
            log.warning('Loractl-scaling feature is enabled in config.yaml, but is currently only supported by SD Forge/ReForge. Disabling.')
            config.imggen['extensions']['loractl']['enabled'] = False

# Feature to automatically change imgmodels periodically
async def init_auto_change_imgmodels():
    imgmodels_data:dict = load_file(shared_path.img_models, {})
    if imgmodels_data and imgmodels_data.get('settings', {}).get('auto_change_imgmodels', {}).get('enabled', False):
        if config.is_per_server_imgmodels():
            for guild_id, settings in guild_settings.items():
                settings.imgmodel._guild_id = guild_id # store Guild ID
                settings.imgmodel._guild_name = guild_id # store Guild Name
                await bg_task_queue.put(settings.imgmodel.start_auto_change_imgmodels())
        else:
            await bg_task_queue.put(bot_settings.imgmodel.start_auto_change_imgmodels())

# Try getting a valid character file source
def get_character(guild_id:int|None=None, guild_settings=None):
    # Determine applicable Settings()
    settings:Settings = guild_settings if guild_settings is not None else bot_settings
    joined_msg = 'has joined the chat'
    if guild_settings:
        joined_msg = f'has joined {settings._guild_name}'
    try:
        # Try loading last known character with fallback sources
        sources = [bot_settings.get_last_setting_for("last_character", guild_id=guild_id),
                   settings.llmcontext.name,
                   client.user.display_name]
        char_name = None
        for source_name in sources:
            log.debug(f'Trying to load character "{source_name}"...')
            _, char_name, _, _, _ = custom_load_character(source_name, '', '', try_tgwui=tgwui_enabled)
            if char_name:
                log.info(f'"{source_name}" {joined_msg}.')
                return source_name
        if not char_name:
            log.error(f"Character not found. Tried files: {sources}.")          
            return None
    except Exception as e:
        log.error(f"Error trying to load character data: {e}")
        return None

# Try loading character data regardless of mode (chat/instruct)
async def init_characters():
    # Per-server characters
    if config.is_per_character():
        for guild_id, settings in guild_settings.items():
            log.info("----------------------------------------------")
            log.info(f"Initializing {settings._guild_name}...")
            char_name = get_character(guild_id, settings)
            await character_loader(char_name, settings, guild_id=guild_id)
    # Not per-server characters
    else:
        char_name = get_character()
        await character_loader(char_name, bot_settings)
    bot_status.build_idle_weights()

def update_base_tags_modified():
    mod_time = os.path.getmtime(shared_path.tags)
    last_mod_time = bot_database.last_base_tags_modified
    updated = (mod_time > last_mod_time) if last_mod_time else False
    bot_database.set("last_base_tags_modified", mod_time, save_now=updated) # update value
    return updated

# Creates instances of Settings() for all guilds the bot is in
async def init_guilds():
    global guild_settings
    per_server_settings = config.is_per_server()
    post_settings = config.discord['post_active_settings'].get('enabled', True)
    if per_server_settings or post_settings:
        # check/update last time modified for dict_tags.yaml
        tags_updated = update_base_tags_modified()
        # iterate over guilds
        for guild in client.guilds:
            # create Settings()
            if per_server_settings:
                guild_settings[guild.id] = Settings(guild)
            # post Tags settings
            previously_sent_settings = bot_database.settings_sent.get(guild.id) # must have sent settings before
            previously_sent_tags = bool(bot_database.get_settings_msgs_for(guild.id, 'tags'))
            # post tags if file was updated, or if guild has not yet posted them
            if post_settings and previously_sent_settings and (tags_updated or not previously_sent_tags):
                await bg_task_queue.put(post_active_settings(guild, ['tags']))

# If first time bot script is run
async def first_run():
    try:
        log.info('Welcome to ad_discordbot!')
        log.info('• Use "/character" for changing characters.')
        log.info('• Use "/helpmenu" to see other useful commands.')
        log.info('• Visit https://github.com/altoiddealer/ad_discordbot for more info.')
        log.info('• Learn about command permissions in the Wiki section: "Commands".')
        for guild in client.guilds:  # Iterate over all guilds the bot is a member of
            if guild.text_channels and bot_embeds.enabled('system'):
                # Find the 'general' channel, if it exists
                default_channel = None
                for channel in guild.text_channels:
                    if channel.name == 'general':
                        default_channel = channel
                        break
                # If 'general' channel is not found, use the first text channel
                if default_channel is None:
                    default_channel = guild.text_channels[0]
                
                async with muffled_send(default_channel):
                    await default_channel.send(embed = bot_embeds.helpmenu())
                    
                break  # Exit the loop after sending the message to the first guild
            
        log.info('Welcome to ad_discordbot! Use "/helpmenu" to see main commands. (https://github.com/altoiddealer/ad_discordbot) for more info.')
        
    except Exception as e: # muffled send will not catch all errors, only specific ones we can ignore.
        log.error(f"An error occurred while welcoming user to the bot: {e}")
        
    finally:
        bot_database.set('first_run', False)

async def in_any_guilds():
    if len(client.guilds) == 0:
        log.error("The bot is not a member of any guilds. Please invite it to a server and try again.")
        log.error("Shutting down in 5 seconds...")
        await asyncio.sleep(5)
        await client.close()
        sys.exit(3)

#################################################################
########################### ON READY ############################
#################################################################
@client.event
async def on_ready():
    try:
        await in_any_guilds()

        # If first time running bot
        if bot_database.first_run:
            await first_run()
        
        # Ensure startup tasks do not re-execute if bot's discord connection status fluctuates
        if client.is_first_on_ready: # type: ignore
            client.is_first_on_ready = False # type: ignore

            # Setup API clients
            await api.setup_all_clients()

            # womp womp
            disable_unsupported_features()

            # Build options for /speak command
            await speak_cmd_options.build_options(sync=False)

            # Enforce only one TTS method enabled
            enforce_one_tts_method()

            # Create background task processing queue
            client.loop.create_task(process_tasks_in_background())
            # Start the Task Manager
            client.loop.create_task(task_manager.start())

            # Run guild startup tasks
            await init_guilds()

            # Load character(s)
            await init_characters()

            # Start background task to to change image models automatically
            if imggen_enabled:
                await init_auto_change_imgmodels()
            
            log.info("----------------------------------------------")
            log.info("                Bot is ready")
            log.info("    Use Ctrl+C to shutdown the bot cleanly")
            log.info("----------------------------------------------")
        # Run only on discord reconnections
        else:
            await voice_clients.restore_state()
    except Exception as e:
        print(traceback.format_exc())
        log.critical(e)
    
    ######################
    # Run every on_ready()
    
    # Start background task to sync the discord client tree
    await bg_task_queue.put(client.tree.sync())
    
    # Schedule bot to go idle, if configured
    await bot_status.schedule_go_idle()

#################################################################
################### DISCORD EVENTS/FEATURES #####################
#################################################################
@client.event
async def on_message(message: discord.Message):
    text = message.clean_content # primarily converts @mentions to actual user names
    settings:Settings = get_settings(message)
    last_character = bot_settings.get_last_setting_for("last_character", message)
    if not settings.behavior.bot_should_reply(message, text, last_character): 
        return # Check that bot should reply or not
    # Store the current time. The value will save locally to database.yaml at another time
    bot_database.update_last_msg_for(message.channel.id, 'user', save_now=False)
    # if @ mentioning bot, remove the @ mention from user prompt
    if text.startswith(f"@{last_character} "):
        text = text.replace(f"@{last_character} ", "", 1)
    # apply wildcards
    text = await dynamic_prompting(text, message, message.author.display_name)
    # Create Task from message
    task = Task('on_message', message, text=text)
    # Send to to MessageManager()
    await message_manager.queue_message_task(task, settings)

# Starboard feature
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
        bot_emojis_list = bot_emojis.get_emojis()
        for reaction in message.reactions:
            if reaction.emoji in bot_emojis_list:
                continue
            total_reaction_count += reaction.count
    if total_reaction_count >= config.discord['starboard'].get('min_reactions', 2):

        target_channel_id = config.discord['starboard'].get('target_channel_id', None)
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

#################################################################
##################### POST ACTIVE SETTINGS ######################
#################################################################
async def delete_old_settings_messages(channel:discord.TextChannel, old_msg_ids:list):
    for msg_id in old_msg_ids:
        message = None
        try:
            message = await channel.fetch_message(msg_id)
        except Exception as e:
            log.warning(f"[Post Active Settings] Tried deleting an old settings message which was not found: {e}")
        if message:
            await message.delete()

# Post settings to all servers
async def post_active_settings_to_all(key_str_list:Optional[list[str]]=None):
    for guild in client.guilds:
        await post_active_settings(guild, key_str_list)

# Post settings to a dedicated channel
async def post_active_settings(guild:discord.Guild, key_str_list:Optional[list[str]]=None, channel:discord.TextChannel|None=None):
    # List of settings keys to update
    key_str_list = key_str_list if key_str_list is not None else ['character', 'behavior', 'tags', 'imgmodel', 'llmstate']
    # For configured keys: Delete old settings messages -> Send new settings
    managed_keys = config.discord['post_active_settings'].get('post_settings_for', [])

    # get settings channel if not provided to function
    if channel is None:
        channel_id = bot_database.get_settings_channel_id_for(guild.id)
        # Warn (once) if no ID set for server while setting is enabled
        if not channel_id:
            if not bot_database.was_warned(f'{guild.id}_chan'):
                bot_database.update_was_warned(f'{guild.id}_chan')
                log.warning(f"[Post Active Settings] This feature is enabled, but a channel is not yet set for server '{guild.name}'.")
                log.warning("[Post Active Settings] Use command '/set_server_settings_channel' to designate a 'settings channel'.")
            return
        try:
            channel = await guild.fetch_channel(channel_id)
        except:
            log.error(f"[Post Active Settings] Failed to fetch channel from stored id '{channel_id}'")
            log.info("[Post Active Settings] Use command '/set_server_settings_channel' to set a new channel.")
            return

    log.info(f"[Post Active Settings] Posting updated settings for '{guild.name}': {key_str_list}.")

    # Collect current settings
    settings:"Settings" = guild_settings.get(guild.id, bot_settings)
    settings_copy = copy.deepcopy(settings.get_vars(public_only=True))
    # Extract tags for Tags message
    char_tags = settings_copy['llmcontext'].pop('tags', [])
    imgmodel_tags = settings_copy['imgmodel'].pop('tags', [])

    # Manage settings messages for the provided list of settings keys, as configured
    for key_name in key_str_list:
        if key_name not in managed_keys:
            continue # skip keys not configured

        # Get stored IDs for old settings messages
        old_msg_ids_list = bot_database.get_settings_msgs_for(guild.id, key_name)
        # Fetch and delete the old messages
        if old_msg_ids_list:
            await delete_old_settings_messages(channel, old_msg_ids_list)

        # Set empty defaults
        tags_ids = []
        tags_key = None
        custom_prefix = ''
        
        # Determine settings sources for current key
        if key_name == 'character':
            custom_prefix = f'name: {bot_settings.get_last_setting_for("last_character", guild_id=guild.id)}\n'
            # resolve alias
            settings_key = settings_copy.get('llmcontext', {})
            # check if updating character tags
            if 'tags' in managed_keys:
                tags_key = char_tags
        elif key_name == 'imgmodel':
            custom_prefix = f'name: {settings.imgmodel.last_imgmodel_name}\n'
            settings_key = settings_copy.get('imgmodel', {})
            # check if updating imgmodel tags
            if 'tags' in managed_keys:
                tags_key = imgmodel_tags
        elif key_name == 'tags':
            settings_key = base_tags.tags
        else:
            settings_key = settings_copy.get(key_name, {})

        # Convert dictionary to yaml for message
        settings_content = yaml.dump(settings_key, default_flow_style=False)

        # Send the updated settings content to the settings channel
        new_settings_ids, _ = await send_long_message(channel, f"## {key_name.upper()}:\n```yaml\n{custom_prefix}{settings_content}\n────────────────────────────────```")
        # Also send relevant Tags for the item in a second process
        if tags_key is not None:
            # Convert tags list to yaml for message
            tags_content = yaml.dump(tags_key, default_flow_style=False)
            tags_ids, _ = await send_long_message(channel, f"### {key_name.upper()} TAGS:\n```yaml\n{tags_content}\n────────────────────────────────```")
        # Merge all sent message IDs while retaining them to database
        new_settings_ids = new_settings_ids + tags_ids

        # Update the database with the new message ID(s)
        bot_database.update_settings_key_msgs_for(guild.id, key_name, new_settings_ids)

async def switch_settings_channels(guild:discord.Guild, channel:discord.TextChannel):
    # Get old settings dict (if any)
    old_settings_dict = bot_database.get_settings_msgs_for(guild.id)
    # Get old settings channel ID (if any)
    old_channel_id = bot_database.get_settings_channel_id_for(guild.id)
    # Update the guild settings channel
    bot_database.update_settings_channel(guild.id, channel.id)
    # Delete messages from old settings channel
    if old_channel_id and old_settings_dict:
        log.info("[Post Active Settings] Trying to delete old messages from previous settings channel...")
        old_channel = None
        try:
            old_channel = await guild.fetch_channel(old_channel_id)
        except Exception as e:
            log.error(f"[Post Active Settings] Failed to fetch old settings channel from ID '{old_channel_id}': {e}")
        if old_channel:
            for old_msg_ids in old_settings_dict.values():
                await delete_old_settings_messages(old_channel, old_msg_ids)
    # Post all current settings to new settings channel
    await post_active_settings(guild, channel=channel)

if config.discord['post_active_settings'].get('enabled', True):
    # Command to set settings channels (post_active_settings feature)
    @client.hybrid_command(name="set_server_settings_channel", description="Assign a channel as the settings channel for this server.")
    @app_commands.describe(channel='Begin typing the channel name and it should appear in the menu.')
    @app_commands.checks.has_permissions(manage_channels=True)
    @guild_only()
    async def set_server_settings_channel(ctx: commands.Context, channel: Optional[discord.TextChannel]=None):
        if channel is None:
            raise AlertUserError("Please select a text channel to set (select from 'channel').")
        if channel not in ctx.guild.channels:
            raise AlertUserError('Selected channel not found in this server.')
        
        # Skip update process if same channel as previously set
        old_channel_id = bot_database.get_settings_channel_id_for(ctx.guild.id)
        if channel.id == old_channel_id:
            await ctx.send("New settings channel is the same as the previously set one.", delete_after=5)
            return

        log.info(f'[Post Active Settings] {ctx.author.display_name} used "/set_server_settings_channel".')
        log.info(f"[Post Active Settings] Settings channel for '{ctx.guild.name}' was set to '{channel.name}'")

        # Reply to interaction
        await ctx.send(f"Settings channel for **{ctx.guild.name}** set to **{channel.name}**.", delete_after=5)

        # Process message updates in the background
        await bg_task_queue.put(switch_settings_channels(ctx.guild, channel))          

#################################################################
######################## TTS PROCESSING #########################
#################################################################
async def toggle_any_tts(settings, tts_to_toggle:str='api', force:str|None=None) -> str:
    """
    Parameters:
    - tts_to_toggle (str): 'api' or 'tgwui'
    - force (str): 'enabled' or 'disabled' - forces the TTS enabled or disabled
    """
    message = force
    # Toggle TGWUI TTS
    if tts_to_toggle == 'tgwui':
        if force is not None:
            await tgwui.tts.toggle_tts_extension(settings, toggle=force)
            return message
        else:
            return await tgwui.tts.apply_toggle_tts(settings) # returns 'enabled' or 'disabled'

    # Toggle API TTS
    if force is None:
        message = 'disabled' if api.ttsgen.enabled else 'enabled'
    else:
        force = True if force == 'enabled' else False
    if tts_to_toggle == 'api':
        api.ttsgen.enabled = (force) or (not api.ttsgen.enabled)
    return message


def tts_is_enabled(and_online:bool=False, for_mode:str='any') -> bool | Tuple[bool, bool]:
    """
    Check if TTS is available and optionally online.

    Parameters:
    - and_online (bool): If True, also require TTS to be currently enabled (online).
    - for_mode (str): One of 'api', 'tgwui', 'both', or 'any'. Determines which TTS mode(s) to check.
    """
    if not config.tts_enabled():
        if for_mode == 'both':
            return False, False
        return False

    api_tts_ready = api.ttsgen and (not and_online or api.ttsgen.enabled)
    tgwui_tts_ready = tgwui_enabled and tgwui.tts.extension and (not and_online or tgwui.tts.enabled)

    if for_mode == 'api':
        return api_tts_ready
    elif for_mode == 'tgwui':
        return tgwui_tts_ready
    elif for_mode == 'both':
        return api_tts_ready, tgwui_tts_ready

    return api_tts_ready or tgwui_tts_ready

def enforce_one_tts_method():
    api_tts_on, tgwui_tts_on = tts_is_enabled(and_online=True, for_mode='both')
    if api_tts_on and tgwui_tts_on:
        if client.is_first_on_ready:
            log.warning("Bot was initialized with both API and TGWUI extension TTS methods enabled. Disabling TGWUI extension.")
        toggle_any_tts(bot_settings, 'tgwui', force='off')
        tgwui.tts.enabled = False

@client.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
    if member.id != client.user.id:
        return
    guild_id = member.guild.id
    # Skip if this change was initiated by the script
    if guild_id in voice_clients._internal_change:
        return
    # Bot disconnected
    if before.channel and not after.channel:
        voice_clients._update_state(guild_id, False)
    # Bot connected or moved
    elif after.channel:
        vc = after.channel.guild.voice_client
        voice_clients._update_state(guild_id, True, vc)

class VoiceClients:
    def __init__(self):
        self.guild_vcs:dict = {}
        self.expected_state:dict = {}
        self._internal_change: set[int] = set()  # Tracks guilds where the bot initiated a VC change
        self.queued_audio:list = []

    def is_connected(self, guild_id):
        if self.guild_vcs.get(guild_id):
            return self.guild_vcs[guild_id].is_connected()
        return False
    
    def should_be_connected(self, guild_id):
        return self.expected_state.get(guild_id, False)

    def _update_state(self, guild_id, connected: bool, vc: Optional[discord.VoiceClient] = None):
        if connected:
            self.guild_vcs[guild_id] = vc
            self.expected_state[guild_id] = True
        else:
            self.guild_vcs.pop(guild_id, None)
            self.expected_state[guild_id] = False

    # Try loading character data regardless of mode (chat/instruct)
    async def restore_state(self):
        for guild_id, should_be_connected in self.expected_state.items():
            try:
                if should_be_connected and not self.is_connected(guild_id):
                    voice_channel = client.get_channel(bot_database.voice_channels[guild_id])
                    self.guild_vcs[guild_id] = await voice_channel.connect()
                elif not should_be_connected and self.is_connected(guild_id):
                    await self.guild_vcs[guild_id].disconnect()
            except Exception as e:
                log.error(f'[Voice Clients] An error occurred while restoring voice channel state for guild ID "{guild_id}": {e}')

    async def toggle_voice_client(self, guild_id, toggle: str = None):
        try:
            self._internal_change.add(guild_id)
            if toggle == 'enabled' and not self.is_connected(guild_id):
                if bot_database.voice_channels.get(guild_id):
                    voice_channel = client.get_channel(bot_database.voice_channels[guild_id])
                    vc = await voice_channel.connect()
                    self._update_state(guild_id, True, vc)
                else:
                    log.warning('[Voice Clients] TTS Gen is enabled, but no VC is set.')
            elif toggle == 'disabled' and self.is_connected(guild_id):
                await self.guild_vcs[guild_id].disconnect()
                self._update_state(guild_id, False)
        except Exception as e:
            log.error(f'[Voice Clients] Error toggling VC for guild {guild_id}: {e}')
        finally:
            self._internal_change.discard(guild_id)

    async def voice_channel(self, guild_id:int, vc_setting:bool=True):
        try:
            # Start voice client if configured, and not explicitly deactivated in settings
            if config.tts_enabled() and vc_setting == True and int(config.ttsgen.get('play_mode', 0)) != 1 and not self.guild_vcs.get(guild_id):
                try:
                    if tts_is_enabled(and_online=True):
                        await self.toggle_voice_client(guild_id, 'enabled')
                    else:
                        if not bot_database.was_warned('char_tts'):
                            bot_database.update_was_warned('char_tts')
                            log.warning('[Voice Clients] TTS is enabled in config, but no TTS clients are available/enabled.')
                except Exception as e:
                    log.error(f"[Voice Clients] An error occurred while connecting to voice channel: {e}")
            # Stop voice client if explicitly deactivated in character settings
            if self.guild_vcs.get(guild_id) and self.guild_vcs[guild_id].is_connected():
                if vc_setting is False:
                    log.info("[Voice Clients] New context has setting to disconnect from voice channel. Disconnecting...")
                    await self.toggle_voice_client(guild_id, 'disabled')
        except Exception as e:
            log.error(f"[Voice Clients] An error occurred while managing channel settings: {e}")

    def after_playback(self, guild_id, file, error):
        if error:
            log.info(f'[Voice Clients] Message from audio player: {error}, output: {error.stderr.decode("utf-8")}')
        # Check save mode setting
        if int(config.ttsgen.get('save_mode', 0)) > 0:
            try:
                os.remove(file)
            except Exception:
                pass
        # Check if there are queued tasks
        if self.queued_audio:
            # Pop the first task from the queue and play it
            next_file = self.queued_audio.pop(0)
            source = discord.FFmpegPCMAudio(next_file)
            self.guild_vcs[guild_id].play(source, after=lambda e: self.after_playback(guild_id, next_file, e))

    async def play_in_voice_channel(self, guild_id, file):
        if not self.guild_vcs.get(guild_id):
            log.warning(f"[Voice Clients] Tried playing an audio file, but voice channel not connected for guild ID {guild_id}")
            return
        # Queue the task if audio is already playing
        if self.guild_vcs[guild_id].is_playing():
            self.queued_audio.append(file)
        else:
            # Otherwise, play immediately
            source = discord.FFmpegPCMAudio(file)
            self.guild_vcs[guild_id].play(source, after=lambda e: self.after_playback(guild_id, file, e))

    async def toggle_playback_in_voice_channel(self, guild_id, action='stop'):
        if self.guild_vcs.get(guild_id):          
            guild_vc:discord.VoiceClient = self.guild_vcs[guild_id]
            if action == 'stop' and guild_vc.is_playing():
                guild_vc.stop()
                log.info(f"Audio playback was stopped for guild {guild_id}")
            elif (action == 'pause' or action == 'toggle') and guild_vc.is_playing():
                guild_vc.pause()
                log.info(f"Audio playback was paused in guild {guild_id}")
            elif (action == 'resume' or action == 'toggle') and guild_vc.is_paused():
                guild_vc.resume()
                log.info(f"Audio playback resumed in guild {guild_id}")

    async def upload_audio_file(self, ictx:CtxInteraction, audio_fp: str):
        bit_rate = int(config.ttsgen.get('mp3_bit_rate', 128))
        ext = os.path.splitext(audio_fp)[1].lower()
        buffer = io.BytesIO()
        mp3_filename = os.path.splitext(os.path.basename(audio_fp))[0] + '.mp3'

        try:
            if ext == '.wav':
                audio = AudioSegment.from_wav(audio_fp)
                audio.export(buffer, format="mp3", bitrate=f"{bit_rate}k")
                buffer.seek(0)

            elif ext == '.mp3':
                with open(audio_fp, 'rb') as f:
                    buffer.write(f.read())
                buffer.seek(0)

            else:
                log.error(f"Unsupported audio format for upload: {audio_fp}")
                return

            # Make a normalized file dict
            file_info:FILE_INPUT = {"file_obj": buffer,
                                    "filename": mp3_filename,
                                    "mime_type": "audio/mpeg",
                                    "file_size": len(buffer.getbuffer()),
                                    "should_close": False}

            await send_content_to_discord(ictx=ictx, text=None, files=[file_info], vc=None, normalize=False)

        except Exception as e:
            log.error(f"Failed to upload audio file '{audio_fp}': {e}")

    async def process_audio_file(self, ictx:CtxInteraction, audio_fp:str, bot_hmessage:Optional[HMessage]=None):
        play_mode = int(config.ttsgen.get('play_mode', 0))
        # Upload to interaction channel
        if play_mode > 0:
            await self.upload_audio_file(ictx, audio_fp)
        # Play in voice channel
        is_connected = self.guild_vcs.get(ictx.guild.id)
        if is_connected and play_mode != 1 and not is_direct_message(ictx):
            await bg_task_queue.put(self.play_in_voice_channel(ictx.guild.id, audio_fp)) # run task in background
        if bot_hmessage:
            bot_hmessage.update(spoken=True)

voice_clients = VoiceClients()

# Command to set voice channels
@client.hybrid_command(name="set_server_voice_channel", description="Assign a channel as the voice channel for this server")
@app_commands.checks.has_permissions(manage_channels=True)
@guild_only()
async def set_server_voice_channel(ctx: commands.Context, channel: Optional[discord.VoiceChannel]=None):
    if isinstance(ctx.author, discord.Member) and ctx.author.voice:
        channel = channel or ctx.author.voice.channel # type: ignore
        
    if channel is None:
        raise AlertUserError('Please select or join a voice channel to set.')

    log.info(f'{ctx.author.display_name} used "/set_server_voice_channel".')
    
    bot_database.update_voice_channels(ctx.guild.id, channel.id)
    await ctx.send(f"Voice channel for **{ctx.guild}** set to **{channel.name}**.", delete_after=5)

if tts_is_enabled():
    # Register command for helper function to toggle TTS
    @client.hybrid_command(description='Toggles TTS on/off')
    @guild_only()
    async def toggle_tts(ctx: commands.Context):
        await ireply(ctx, 'toggle TTS') # send a response msg to the user
        # offload to TaskManager() queue
        log.info(f'{ctx.author.display_name} used "/toggle_tts"')
        toggle_tts_task = Task('toggle_tts', ctx)
        await task_manager.queue_task(toggle_tts_task, 'normal_queue')

# Define context menu app command
@client.tree.context_menu(name="Bot Join VC")
@guild_or_owner_only()
async def bot_join_voice_channel(inter: discord.Interaction, user: discord.User):
    if user != client.user or not client.is_owner(user):
        await inter.response.send_message(f"'Bot Join VC' is only for admins to manually join the Bot to VC.", ephemeral=True, delete_after=5)
        return
    try:
        await voice_clients.toggle_voice_client(inter.guild.id, 'enabled')
        await inter.response.send_message(f"Joined {user.display_name} to Voice Channel.", ephemeral=True, delete_after=5)
    except Exception as e:
        await inter.response.send_message(f"Failed to connect bot to VC: {e}", ephemeral=True, delete_after=5)
        log.error(f"[Bot Join VC] Error joining bot to VC: {e}")

#################################################################
###################### DYNAMIC PROMPTING ########################
#################################################################
def get_wildcard_value(matched_text:str, dir_path: Optional[str] = None) -> Optional[str]:
    dir_path = dir_path or shared_path.dir_user_wildcards
    selected_option: Optional[str] = None
    search_phrase = matched_text[2:] if matched_text.startswith('##') else matched_text
    search_path = f"{search_phrase}.txt"
    # List files in the directory
    txt_files = glob.glob(os.path.join(dir_path, search_path))
    if txt_files:
        selected_file = random.choice(txt_files)
        with open(selected_file, 'r') as file:
            lines = file.readlines()
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
            # Get the last component of the nested directory path
            search_phrase = os.path.split(nested_dir)[-1]
            # Remove the last component from the nested directory path
            nested_dir = os.path.join(shared_path.dir_user_wildcards, os.path.dirname(nested_dir))
            # Recursively check filenames in the nested directory
            selected_option = get_wildcard_value(search_phrase, nested_dir)
    return selected_option

def process_dynaprompt_options(options:list[str]) -> list[tuple[float, str]]:
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

def choose_dynaprompt_option(options: list[tuple[float, str]], num_choices: int = 1) -> list[str]:
    chosen_values = random.choices(options, weights=[weight for weight, _ in options], k=num_choices)
    return [value for _, value in chosen_values]

def get_braces_value(matched_text:str) -> str:
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
            wildcard_value = get_wildcard_value(matched_text=wildcard_phrase, dir_path=shared_path.dir_user_wildcards)
            if wildcard_value:
                chosen_options[index] = wildcard_value
    chosen_options = [option for option in chosen_options if option is not None]
    if separator:
        replaced_text = separator.join(chosen_options)
    else:
        replaced_text = ', '.join(chosen_options) if num_choices > 1 else chosen_options[0]
    return replaced_text

async def dynamic_prompting(text: str, ictx: Optional[CtxInteraction] = None, user_name=None) -> str:
    try:
        if not config.get('dynamic_prompting_enabled', True):
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
            replaced_text = get_wildcard_value(matched_text=matched_text, dir_path=shared_path.dir_user_wildcards)
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
        if ictx and user_name and (braces_matches or wildcard_matches):
            await ictx.reply(content=f"__Text with **[Dynamic Prompting](<https://github.com/altoiddealer/ad_discordbot/wiki/dynamic-prompting>)**__:\n>>> **{user_name}**: {text_with_comments}", mention_author=False, silent=True)
    except Exception as e:
        log.error(f'An error occurred while processing Dynamic Prompting: {e}')
    return text

#################################################################
######################### ANNOUNCEMENTS #########################
#################################################################
async def announce_changes(change_label:str, change_name:str, ictx: CtxInteraction|None=None):
    user_name = get_user_ctx_inter(ictx).display_name if ictx else 'Automatically'
    try:
        # adjust delay depending on how many channels there are to prevent being rate limited
        delay = math.floor(len(bot_database.announce_channels)/2)
        for channel_id in bot_database.announce_channels:
            await asyncio.sleep(delay)
            channel = await client.fetch_channel(channel_id)
            # if Automatic imgmodel change (no interaction object)
            if ictx is None:
                await bot_embeds.send('change', f"{user_name} {change_label}:", f'**{change_name}**', channel=channel)
            # If private channel
            elif ictx.channel.overwrites_for(ictx.guild.default_role).read_messages is False:
                continue
            # Public channels in interaction server
            elif any(channel_id == channel.id for channel in ictx.guild.channels):
                await bot_embeds.send('change', f"{user_name} {change_label} in <#{ictx.channel.id}>:", f'**{change_name}**', channel=channel)
            # Channel is in another server
            elif channel_id not in [channel.id for channel in ictx.guild.channels]:
                if change_label != 'reset the conversation':
                    await bot_embeds.send('change', f"A user {change_label} in another bot server:", f'**{change_name}**', channel=channel)
    except Exception as e:
        log.error(f'An error occurred while announcing changes to announce channels: {e}')

#################################################################
########################## BOT VARS #############################
#################################################################
class BotVars():
    def __init__(self):
        self._loras_index = 1
        # General
        self.prompt:Optional[str] = None
        self.character:Optional[str] = None
        # Image related
        self.neg_prompt:Optional[str] = None
        self.width:Optional[int] = None
        self.height:Optional[int] = None
        self.ckpt_name:Optional[str] = None
        self.seed:Optional[int] = None
        self.i2i_image:Optional[str] = None
        self.i2i_mask:Optional[str] = None
        self.denoising_strength:Optional[float] = None
        self.cnet_image:Optional[str] = None
        self.cnet_mask:Optional[str] = None
        self.cnet_model:Optional[str] = None
        self.cnet_module:Optional[str] = None
        self.cnet_weight:Optional[float] = None
        self.cnet_processor_res:Optional[int] = None
        self.cnet_guidance_start:Optional[float] = None
        self.cnet_guidance_end:Optional[float] = None
        self.cnet_threshold_a:Optional[int] = None
        self.cnet_threshold_b:Optional[int] = None
        self.face_image:Optional[str] = None
    
    def update(self, ictx:Optional[CtxInteraction]=None):
        self.character = bot_settings.get_last_setting_for("last_character", ictx)
        self.seed = random.randint(10**14, 10**15 - 1)
        imgmodel_settings:ImgModel = get_imgmodel_settings(ictx)
        self.ckpt_name = imgmodel_settings.last_imgmodel_value

    def update_from_dict(self, input:dict):
        for k, v in input.items():
            setattr(self, k, v)
    
    def add_lora(self, name:str, strength:float):
        index_str = f"{self._loras_index:02}"  # Format index with leading zeros (2 digits)
        # Dynamically set incrementing attributes like lora1_name, lora1_weight, etc.
        setattr(self, f"lora_{index_str}", name)
        setattr(self, f"strength_{index_str}", strength)
        self._loras_index += 1

    def get_vars(self, return_copy=False):
        attrs = {k: v for k, v in vars(self).items() if not k.startswith('_')}
        return copy.deepcopy(attrs) if return_copy else attrs

bot_vars = BotVars()

#################################################################
########################### PARAMS ##############################
#################################################################
class Params:
    def __init__(self, **kwargs):
        '''
        kwargs:
        save_to_history, should_gen_text, should_send_text, should_gen_image, should_send_image,
        imgcmd, img_censoring, endpoint, sd_output_dir, ref_message, regenerated,
        skip_create_user_hmsg, skip_create_bot_hmsg, bot_hmsg_hidden, bot_hmessage_to_update,
        target_discord_msg_id, character, llmmodel, imgmodel, tts_args, user_voice
        '''
        self.save_to_history: bool      = kwargs.get('save_to_history', True)

        # Behavior
        self.should_gen_text: bool      = kwargs.get('should_gen_text', True)
        self.should_send_text: bool     = kwargs.get('should_send_text', True)
        self.should_gen_image: bool     = kwargs.get('should_gen_image', False)
        self.should_send_image: bool    = kwargs.get('should_send_image', True)
        self.should_tts: bool           = kwargs.get('should_tts', True)

        # Image command params
        self.imgcmd: dict = kwargs.get('imgcmd', {
            'size': None,
            'neg_prompt': '',
            'style': {},
            'face_swap': None,
            'controlnet': None,
            'img2img': {}
        })

        # Image related params
        self.img_censoring: int = kwargs.get('img_censoring', 0)
        self.mode: str      = kwargs.get('mode', None)
        self.sd_output_dir: str = kwargs.get('sd_output_dir', '')

        # discord/HMessage related params
        self.ref_message                 = kwargs.get('ref_message', None)
        self.continued: bool             = kwargs.get('continued', False)
        self.regenerated: bool           = kwargs.get('regenerated', False)
        self.skip_create_user_hmsg: bool = kwargs.get('skip_create_user_hmsg', False)
        self.skip_create_bot_hmsg: bool  = kwargs.get('skip_create_bot_hmsg', False)
        self.bot_hmsg_hidden: bool       = kwargs.get('bot_hmsg_hidden', False)
        self.bot_hmessage_to_update      = kwargs.get('bot_hmessage_to_update', None)
        self.target_discord_msg_id       = kwargs.get('target_discord_msg_id', None)
        self.chunk_msg_ids               = kwargs.get('chunk_msg_ids', [])

        # Model/Character Change params
        self.character: dict = kwargs.get('character', {})
        self.llmmodel: dict  = kwargs.get('llmmodel', {})
        self.imgmodel: dict  = kwargs.get('imgmodel', {})

        # /Speak cmd
        self.tts_args: dict        = kwargs.get('tts_args', {})
        self.user_voice: str       = kwargs.get('user_voice', None)

    def get_active_imggen_mode(self) -> str:
        return 'img2img' if self.imgcmd.get('img2img') else 'txt2img'

    def get_active_imggen_ep(self) -> Optional[ImgGenEndpoint]:
        mode = self.get_active_imggen_mode()
        return getattr(api.imggen, f'post_{mode}', None)

    def update_bot_should_do(self, tags:Tags):
        actions = ['should_gen_text', 'should_send_text', 'should_gen_image', 'should_send_image']
        try:
            # iterate through matched tags and update
            matches = getattr(tags, 'matches')
            if matches and isinstance(matches, list):
                # in reverse, to maintain tag priority
                for tag in reversed(matches):
                    tag_dict:TAG = tags.untuple(tag)
                    for key, value in tag_dict.items():
                        if key in actions:
                            setattr(self, key, value)
            # Disable things as set by config
            if not tgwui_enabled:
                self.should_gen_text = False
                self.should_send_text = False
            if not imggen_enabled:
                self.should_gen_image = False
                self.should_send_image = False
        except Exception as e:
            log.error(f"An error occurred while checking if bot should do '{key}': {e}")

#################################################################
########################## MENTIONS #############################
#################################################################
# For @ mentioning users who were not last replied to
class Mentions:
    def __init__(self):
        self.previous_user_mention = ''

    def update_mention(self, user_mention:str, llm_resp:str='') -> str:
        mention_resp = llm_resp

        if user_mention != self.previous_user_mention:
            mention_resp = f"{user_mention} {llm_resp}"
        self.previous_user_mention = user_mention
        return mention_resp
    
mentions = Mentions()

#################################################################
####################### TASK PROCESSING #########################
#################################################################
class TaskAttributes():
    name: str
    ictx: CtxInteraction
    channel: discord.TextChannel
    user: Union[discord.User, discord.Member]
    user_name: str
    settings: "Settings"
    embeds: Embeds
    text: str
    prompt: str
    payload: dict
    params: Params
    tags: Tags
    llm_resp: str
    tts_resps: list
    user_hmessage: HMessage
    bot_hmessage: HMessage
    local_history: History
    istyping: "IsTyping"

class TaskProcessing(TaskAttributes):
    ####################### MOSTLY TEXT GEN PROCESSING #########################
    async def reset_behaviors(self:Union["Task","Tasks"]):
        # Reschedule time to go idle
        await bot_status.schedule_go_idle()
        # reset timer for spontaneous message from bot, as configured
        await spontaneous_messaging.set_for_channel(self.ictx)
        # TODO Message Manager may prioritize queued messages differently if from same channel
        #message_manager.last_channel = self.channel.id

        if self.name == 'spontaneous_messaging':
            task, tally = spontaneous_messaging.tasks[self.channel.id]
            spontaneous_messaging.tasks[self.channel.id] = (task, tally + 1)

    async def send_response_chunk(self:Union["Task","Tasks"], resp_chunk:str):
        # Process most recent TTS response (if any)
        if self.tts_resps:
            await voice_clients.process_audio_file(self.ictx, self.tts_resps[-1], self.bot_hmessage)
        # @mention non-consecutive users
        mention_resp = mentions.update_mention(self.user.mention, resp_chunk)
        # send responses to channel - reference a message if applicable
        sent_chunk_msg_ids, _ = await send_long_message(self.channel, mention_resp, self.params.ref_message)
        self.params.ref_message = None
        # Add sent message IDs to collective message attribute
        self.params.chunk_msg_ids.extend(sent_chunk_msg_ids)
        bot_database.update_last_msg_for(self.channel.id, 'bot', save_now=False)

    async def send_responses(self:Union["Task","Tasks"]):
        # Process any TTS response
        streamed_tts = getattr(self.params, 'streamed_tts', False)
        if self.tts_resps and not streamed_tts:
            await voice_clients.process_audio_file(self.ictx, self.tts_resps[0], self.bot_hmessage)
        # Send text responses
        if self.bot_hmessage and self.params.should_send_text:
            # Send single reply if message was not already streamed in chunks
            was_chunked = getattr(self.params, 'was_chunked', False)
            if not was_chunked:
                # @mention non-consecutive users
                mention_resp = mentions.update_mention(self.user.mention, self.llm_resp)
                # send responses to channel - reference a message if applicable
                sent_msg_ids, sent_msg = await send_long_message(self.channel, mention_resp, self.params.ref_message)
                # Update IDs for Bot HMessage
                sent_msg_ids:list
                last_msg_id = sent_msg_ids.pop(-1)
                self.bot_hmessage.update(id=last_msg_id, related_ids=sent_msg_ids)
                # Update last messages
                bot_database.update_last_msg_for(self.channel.id, 'bot', save_now=False)
                if self.params.should_gen_image:
                    setattr(self, 'img_ref_message', sent_msg)
            # Manage IDs for chunked message handling
            msg_ids_to_react = None
            if self.params.chunk_msg_ids:
                msg_ids_to_react = copy.deepcopy(self.params.chunk_msg_ids)
                if not self.bot_hmessage.id:
                    self.bot_hmessage.id = self.params.chunk_msg_ids.pop(-1)
                self.bot_hmessage.related_ids.extend(self.params.chunk_msg_ids)
            # Apply any reactions applicable to message
            if config.discord['history_reactions'].get('enabled', True):
                await bg_task_queue.put(apply_reactions_to_messages(self.ictx, self.bot_hmessage, msg_ids_to_react))

    def collect_extra_content(self: Union["Task", "Tasks"], results):
        processed_results = collect_content_to_send(results)
        self.extra_text.extend(processed_results['text'])
        self.extra_audio.extend(processed_results['audio'])
        self.extra_files.extend(processed_results['files'])

    async def fix_llm_payload(self:Union["Task","Tasks"]):
        # Fix llmgen payload by adding any missing required settings
        default_llmstate = vars(LLMState())
        default_state = default_llmstate['state']
        current_state = self.payload['state']
        self.payload['state'], _ = fix_dict(current_state, default_state)

    async def swap_llm_character(self:Union["Task","Tasks"], char_name:str):
        try:
            char_data = await load_character_data(char_name, try_tgwui=tgwui_enabled)
            if char_data.get('state', {}):
                self.payload['state'] = char_data['state']
                self.payload['state']['name1'] = self.user_name
            self.payload['state']['name2'] = char_data.get('name', 'AI')
            self.payload['state']['character_menu'] = char_data.get('name', 'AI')
            self.payload['state']['context'] = char_data.get('context', '')
            setattr(self.params, 'impersonated_by', char_name)
            await self.fix_llm_payload() # Add any missing required information
        except Exception as e:
            log.error(f"An error occurred while loading the file for swap_character: {e}")

    async def process_llm_payload_tags(self:Union["Task","Tasks"], mods:dict):
        try:
            char_params: dict            = {}
            begin_reply_with: str        = mods.get('begin_reply_with', None)
            save_history: bool           = mods.get('save_history', None)
            filter_history_for: list     = mods.get('filter_history_for', None)
            load_history: bool           = mods.get('load_history', None)
            include_hidden_history: bool = mods.get('include_hidden_history', None)
            param_variances: dict        = mods.get('param_variances', {})
            state: dict                  = mods.get('state', {})
            prefix_context: str          = mods.get('prefix_context', None)
            suffix_context: str          = mods.get('suffix_context', None)
            change_character: str        = mods.get('change_character', None)
            swap_character: str          = mods.get('swap_character', None)
            change_llmmodel: str         = mods.get('change_llmmodel', None)
            swap_llmmodel: str           = mods.get('swap_llmmodel', None)
            should_tts: bool             = mods.get('should_tts', None)

            # Character handling
            char_params = change_character or swap_character or {} # 'character_change' will trump 'character_swap'
            if char_params:
                # Error handling
                all_characters, _ = get_all_characters()
                if not any(char_params == char['name'] for char in all_characters):
                    log.error(f'Character not found: {char_params}')
                else:
                    if char_params == change_character:
                        verb = 'Changing'
                        # RUN A SUBTASK
                        self.params.character = {'char_name': char_params, 'mode': 'change', 'verb': verb}
                        await self.run_subtask('change_char')
                    else:
                        verb = 'Swapping'
                        await self.swap_llm_character(swap_character)
                    log.info(f'[TAGS] {verb} Character: {char_params}')

            # Tags applicable to TGWUI
            if tgwui_enabled:
                # Begin reply with handling
                if begin_reply_with is not None:
                    setattr(self.params, 'begin_reply_with', begin_reply_with)
                    self.apply_begin_reply_with()
                    log.info(f"[TAGS] Reply is being continued from: '{begin_reply_with}'")
                # TTS Adjustments
                if should_tts is not None:
                    self.params.should_tts = should_tts
                # History handling
                if save_history is not None:
                    self.params.save_to_history = save_history # save_to_history
                if filter_history_for is not None or load_history is not None or include_hidden_history is not None:
                    history_to_render = self.local_history
                    include_hidden = False
                    # Filter history
                    if filter_history_for is not None:
                        history_to_render = self.local_history.get_filtered_history_for(names_list=filter_history_for)
                        log.info(f"[TAGS] History is being filtered for: {filter_history_for}")
                    # Include hidden history
                    if include_hidden_history:
                        include_hidden = True
                        log.info(f"[TAGS] Any 'hidden' History is being included.")
                    # Render history for payload
                    i_list, v_list = history_to_render.render_to_tgwui_tuple(include_hidden)
                    # Factor load history tag
                    if load_history is not None:
                        if load_history <= 0:
                            i_list, v_list = [], []
                            log.info("[TAGS] History is being ignored")
                        else:
                            # Calculate the number of items to retain (up to the length of history)
                            num_to_retain = min(load_history, len(i_list))
                            i_list, v_list = i_list[-num_to_retain:], v_list[-num_to_retain:]
                            log.info(f'[TAGS] History is being limited to previous {load_history} exchanges')
                    # Apply history changes
                    self.payload['state']['history']['internal'] = i_list
                    self.payload['state']['history']['visible'] = v_list
                # Payload param variances
                if param_variances:
                    processed_params = self.process_param_variances(param_variances)
                    log.info(f'[TAGS] LLM Param Variances: {processed_params}')
                    sum_update_dict(self.payload['state'], processed_params) # Updates dictionary while adding floats + ints
                if state:
                    update_dict(self.payload['state'], state)
                    log.info('[TAGS] LLM State was modified')
                # Context insertions
                if prefix_context:
                    prefix_str = "\n".join(str(item) for item in prefix_context)
                    if prefix_str:
                        self.payload['state']['context'] = f"{prefix_str}\n{self.payload['state']['context']}"
                        log.info('[TAGS] Prefixed context with text.')
                if suffix_context:
                    suffix_str = "\n".join(str(item) for item in suffix_context)
                    if suffix_str:
                        self.payload['state']['context'] = f"{self.payload['state']['context']}\n{suffix_str}"
                        log.info('[TAGS] Suffixed context with text.')
                # LLM model handling
                model_change = change_llmmodel or swap_llmmodel or None # 'llmmodel_change' will trump 'llmmodel_swap'
                if model_change:
                    if model_change == tgwui_shared_module.model_name:
                        log.info(f'[TAGS] LLM model was triggered to change, but it is the same as current ("{tgwui_shared_module.model_name}").')
                    else:
                        mode = 'change' if model_change == change_llmmodel else 'swap'
                        verb = 'Changing' if mode == 'change' else 'Swapping'
                        # Error handling
                        all_llmmodels = tgwui_utils_module.get_available_models()
                        if not any(model_change == model for model in all_llmmodels):
                            log.error(f'LLM model not found: {model_change}')
                        else:
                            log.info(f'[TAGS] {verb} LLM Model: {model_change}')
                            self.params.llmmodel = {'llmmodel_name': model_change, 'mode': mode, 'verb': verb}

        except TaskCensored:
            raise
        except Exception as e:
            log.error(f"Error processing LLM tags: {e}")

    async def collect_llm_tag_values(self:Union["Task","Tasks"]) -> tuple[dict, dict]:
        llm_payload_mods = {}
        formatting = {}
        try:
            for tag in self.tags.matches:
                tag:TAG
                tag_name, tag_print = self.tags.get_name_print_for(tag)

                # Values that will only apply from the first tag matches
                if 'begin_reply_with' in tag and not llm_payload_mods.get('begin_reply_with'):
                    llm_payload_mods['begin_reply_with'] = str(tag.pop('begin_reply_with'))
                if 'save_history' in tag and not llm_payload_mods.get('save_history'):
                    llm_payload_mods['save_history'] = bool(tag.pop('save_history'))
                if 'load_history' in tag and not llm_payload_mods.get('load_history'):
                    llm_payload_mods['load_history'] = int(tag.pop('load_history'))
                if 'should_tts' in tag and not llm_payload_mods.get('should_tts'):
                    llm_payload_mods['should_tts'] = bool(tag.pop('should_tts'))
                if 'include_hidden_history' in tag and not llm_payload_mods.get('include_hidden_history'):
                    llm_payload_mods['include_hidden_history'] = bool(tag.pop('include_hidden_history'))
                    
                # change_character is higher priority, if added ignore swap_character
                if 'change_character' in tag and not is_direct_message(self.ictx) and not (llm_payload_mods.get('change_character') or llm_payload_mods.get('swap_character')):
                    llm_payload_mods['change_character'] = str(tag.pop('change_character'))
                if 'swap_character' in tag and not (llm_payload_mods.get('change_character') or llm_payload_mods.get('swap_character')):
                    llm_payload_mods['swap_character'] = str(tag.pop('swap_character'))
                    
                # change_llmmodel is higher priority, if added ignore swap_llmmodel
                if 'change_llmmodel' in tag and not is_direct_message(self.ictx) and not (llm_payload_mods.get('change_llmmodel') or llm_payload_mods.get('swap_llmmodel')):
                    llm_payload_mods['change_llmmodel'] = str(tag.pop('change_llmmodel'))
                if 'swap_llmmodel' in tag and not (llm_payload_mods.get('change_llmmodel') or llm_payload_mods.get('swap_llmmodel')):
                    llm_payload_mods['swap_llmmodel'] = str(tag.pop('swap_llmmodel'))
                    
                # Values that may apply repeatedly
                if 'filter_history_for' in tag:
                    llm_payload_mods.setdefault('filter_history_for', [])
                    llm_payload_mods['filter_history_for'].append(str(tag.pop('filter_history_for')))
                if 'prefix_context' in tag:
                    llm_payload_mods.setdefault('prefix_context', [])
                    llm_payload_mods['prefix_context'].append(str(tag.pop('prefix_context')))
                if 'suffix_context' in tag:
                    llm_payload_mods.setdefault('suffix_context', [])
                    llm_payload_mods['suffix_context'].append(str(tag.pop('suffix_context')))
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
                    except Exception:
                        log.warning(f"[TAGS] Error processing a matched 'llm_param_variances' {tag_print}; ensure it is a dictionary.")
                if 'state' in tag:
                    state = dict(tag.pop('state'))
                    llm_payload_mods.setdefault('state', {})
                    try:
                        llm_payload_mods['state'].update(state) # Allow multiple to accumulate.
                    except Exception:
                        log.warning(f"[TAGS] Error processing a matched 'state' {tag_print}; ensure it is a dictionary.")

        except TaskCensored:
            raise
        except Exception as e:
            log.error(f"Error collecting LLM tag values: {e}")
        return llm_payload_mods, formatting
    
    async def apply_generic_tag_matches(self:Union["Task","Tasks"], phase='llm'):
        prevent_multiple = []
        postponed_processing = []
        try:
            for tag in self.tags.matches:
                tag_dict:TAG = self.tags.untuple(tag)
                tag_name, tag_print = self.tags.get_name_print_for(tag_dict)
                # Check if censored
                if phase == 'llm' and 'llm_censoring' in tag_dict and bool(tag_dict['llm_censoring']) == True:
                    censor_text = tag_dict.get('matched_trigger', '')
                    censor_message = f' (text match: {censor_text})'
                    log.info(f"[TAGS] Censoring: LLM generation was blocked{censor_message if censor_text else ''}")
                    self.embeds.create('censor', "Text prompt was flagged as inappropriate", "Text generation task has been cancelled.")
                    await self.embeds.send('censor', delete_after=5)
                    raise TaskCensored
                if 'flow' in tag_dict and not 'flow' in prevent_multiple:
                    prevent_multiple.append('flow')
                    if not flows_event.is_set():
                        await flows.build_queue(tag_dict.pop('flow'))
                if 'toggle_vc_playback' in tag_dict and not 'toggle_vc_playback' in prevent_multiple:
                    prevent_multiple.append('toggle_vc_playback')
                    if not is_direct_message(self.ictx):
                        await voice_clients.toggle_playback_in_voice_channel(self.ictx.guild.id, str(tag_dict.pop('toggle_vc_playback')))
                if 'send_user_image' in tag_dict:
                    user_image_file = str(tag_dict.pop('send_user_image'))
                    user_image_args = await self.get_image_tag_args('User image', user_image_file, key=None, set_dir=None)
                    user_image = discord.File(user_image_args)
                    self.extra_files.append(user_image)
                    log.info(f'[TAGS] Sending user image for matched {tag_print}')
                if 'persist' in tag_dict:
                    if not tag_name:
                        log.warning(f"[TAGS] A persistent {tag_print} was matched, but it is missing a required 'name' parameter. Cannot make tag persistent.")
                    else:
                        persist = int(tag_dict.pop('persist'))
                        log.info(f'[TAGS] A persistent {tag_print} was matched, which will be auto-applied for the next ({persist}) tag matching phases (pre-{phase.upper()} gen).')
                        persistent_tags.append_tag_name_to(phase, self.channel.id, persist, tag_name)
                # Postponed handling
                if 'call_api' in tag_dict:
                    postponed_processing.append( {'call_api': tag_dict.pop('call_api')} )
                if 'run_workflow' in tag_dict:
                    postponed_processing.append( {'run_workflow': tag_dict.pop('run_workflow')} )
        except TaskCensored:
            raise
        except Exception as e:
            log.error(f"Error processing generic tag matches: {e}")
        # Save postponed tag processing for later
        setattr(self, 'postponed_tags', postponed_processing)

    async def process_postponed_tags(self:Union["Task","Tasks"]):
        postponed:list[dict] = getattr(self, 'postponed_tags', None)
        if postponed:
            base_context = self.vars.get_vars(return_copy=True)
            for tag_dict in postponed:
                if 'call_api' in tag_dict:
                    api_config = tag_dict.pop('call_api')
                    if not isinstance(api_config, dict):
                        log.error('[TAGS] A "call_api" tag was triggered, but it must be in a dict format.')
                    else:
                        queue_to = api_config.pop('queue_to', 'gen_queue')
                        if queue_to:
                            queue_to = 'gen_queue' if queue_to == 'message_queue' else queue_to # Do not allow to go to Message queue
                            log.info('An API task was triggered, created and queued.')
                            await self.embeds.send('system', title='Processing an API Request', description='An API task was triggered, created and queued.', delete_after=5)
                            api_task = self.clone('api', self.ictx, ignore_list=['payload'])
                            setattr(api_task, 'api_config', api_config)
                            await task_manager.queue_task(api_task, queue_to)
                        else:
                            endpoint, updated_api_config = api.get_endpoint_from_config(api_config)
                            if not isinstance(endpoint, Endpoint):
                                log.warning(f'[TAGS] Endpoint not found for triggered "call_api"')
                            else:
                                await self.run_api_task(endpoint, updated_api_config)
                if 'run_workflow' in tag_dict:
                    workflow_config = tag_dict.pop('run_workflow')
                    if not isinstance(workflow_config, dict):
                        log.error('[TAGS] A "run_workflow" tag was triggered, but it must be in a dict format.')
                    else:
                        queue_to = workflow_config.pop('queue_to', 'gen_queue')
                        # Inject task context to the workflow
                        wf_context = workflow_config.pop('context', {})
                        workflow_config['context'] = deep_merge(base_context, wf_context)
                        if queue_to:
                            queue_to = 'gen_queue' if queue_to == 'message_queue' else queue_to # Do not allow to go to Message queue
                            log.info('A Workflow task was triggered, created and queued.')
                            await self.embeds.send('system', title='Processing a Workflow Request', description='A Workflow task was triggered, created and queued.', delete_after=5)
                            workflow_task = self.clone('workflow', self.ictx, ignore_list=['payload'])
                            setattr(workflow_task, 'workflow_config', workflow_config)
                            await task_manager.queue_task(workflow_task, queue_to)
                        else:
                            await self.run_workflow_task(workflow_config)

    def apply_begin_reply_with(self:Union["Task","Tasks"]):
        # Continue from value of 'begin_reply_with'
        begin_reply_with = getattr(self.params, 'begin_reply_with', None)
        if begin_reply_with:
            self.payload['state']['history']['internal'].append([self.text, begin_reply_with])
            self.payload['state']['history']['visible'].append([self.text, begin_reply_with])
            self.payload['_continue'] = True
            setattr(self.params, "include_continued_text", True)

    def apply_prompt_params(self:Union["Task","Tasks"]):
        mode = getattr(self.params, 'mode', None)
        if mode:
            self.payload['state']['mode'] = mode
        system_message = getattr(self.params, 'system_message', None)
        if system_message:
            self.payload['state']['system_message'] = system_message
        load_history = getattr(self.params, 'prompt_load_history', None)
        if load_history is not None:
            i_list, v_list = load_history
            self.payload['state']['history']['internal'] = i_list
            self.payload['state']['history']['visible'] = v_list
        save_to_history = getattr(self.params, 'prompt_save_to_history', None)
        if save_to_history is not None:
            self.params.save_to_history = save_to_history
        response_type = getattr(self.params, 'prompt_response_type', None)
        if response_type is not None:
            self.params.should_send_text = True if 'Text' in response_type else False
            if 'Image' in response_type:
                self.params.should_gen_image = True
        self.apply_begin_reply_with()

    async def collect_images_for_llm(self: Union["Task", "Tasks"]):
        if not self.ictx or not hasattr(self.ictx, 'attachments'):
            return

        IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
        image_paths = []

        save_dir = shared_path.dir_internal_cache

        for attachment in self.ictx.attachments:
            if (
                attachment.filename.lower().endswith(IMAGE_EXTENSIONS)
                or (attachment.content_type and attachment.content_type.startswith("image/"))
            ):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                name, ext = os.path.splitext(attachment.filename)
                unique_filename = f"{name}_{timestamp}{ext}"
                file_path = os.path.join(save_dir, unique_filename)

                if hasattr(attachment, "read"):
                    data = await attachment.read()
                    with open(file_path, "wb") as f:
                        f.write(data)
                elif hasattr(attachment, "save"):
                    await attachment.save(file_path)

                image_paths.append(str(file_path))

        return image_paths

    async def init_llm_payload(self:Union["Task","Tasks"]):
        self.payload = copy.deepcopy(vars(self.settings.llmstate))
        self.payload['text'] = self.text
        self.payload['state']['name1'] = self.user_name
        self.payload['state']['name2'] = self.settings.name
        self.payload['state']['name1_instruct'] = self.user_name
        self.payload['state']['name2_instruct'] = self.settings.name
        self.payload['state']['character_menu'] = self.settings.name
        self.payload['state']['context'] = self.settings.llmcontext.context
        self.payload['state']['history'] = self.local_history.render_to_tgwui()
        if tgwui.is_multimodal:
            image_paths = await self.collect_images_for_llm()
            if image_paths:
                self.payload['state']['image_paths'] = image_paths

    async def message_img_gen(self:Union["Task","TaskProcessing"]):
        await self.tags.match_img_tags(self.prompt, self.settings.get_vars())
        await self.apply_generic_tag_matches(phase='img')
        self.params.update_bot_should_do(self.tags) # check for updates from tags
        if self.params.should_gen_image and await api_online(client_type='imggen', ictx=self.ictx):
            # CLONE CURRENT TASK AND QUEUE IT
            log.info('An image task was triggered, created and queued.')
            await self.embeds.send('system', title='Generating an image', description='An image task was triggered, created and queued.', delete_after=5)
            img_gen_task = self.clone('img_gen', self.ictx, ignore_list=['payload']) # Allow payload to be rebuilt. Keep all other task attributes (matched Tags, params, etc)
            await task_manager.queue_task(img_gen_task, 'gen_queue')

    async def check_tts_before_llm_gen(self:Union["Task","Tasks"]) -> bool|str:
        '''Returns 'api' or 'tgwui' if it toggled one off. Else False'''
        toggle = False

        api_tts_on, tgwui_tts_on = tts_is_enabled(and_online=True, for_mode='both')
        if api_tts_on or tgwui_tts_on:
            # Toggle TTS off if not sending text, or if triggered by Tags
            if (not self.params.should_send_text) or (self.params.should_tts == False):
                toggle = True
            # If guild interaction, and guild not enabled for VC playback
            elif hasattr(self.ictx, 'guild') and getattr(self.ictx.guild, 'voice_client', None) \
                and not voice_clients.guild_vcs.get(self.ictx.guild.id) and int(config.ttsgen.get('play_mode', 0)) == 0:
                toggle = True

            if toggle:
                toggle = 'api' if api_tts_on else 'tgwui'
                await toggle_any_tts(self.settings, toggle, force='off')

        return toggle

    async def message_llm_gen(self:Union["Task","Tasks"]):
        # if no LLM model is loaded, notify that no text will be generated
        if tgwui_shared_module.model_name == 'None':
            if not bot_database.was_warned('no_llmmodel'):
                bot_database.update_was_warned('no_llmmodel')
                await self.channel.send('(Cannot process text request: No LLM model is currently loaded. Use "/llmmodel" to load a model.)', delete_after=10)
                log.warning(f'Bot tried to generate text for {self.user_name}, but no LLM model was loaded')
        ## Finalize payload, generate text via TGWUI, and process responses
        # Toggle TTS if necessary
        tts_sw = await self.check_tts_before_llm_gen()
        # Check to apply Server Mode
        self.apply_server_mode()
        # Update names in stopping strings
        self.extra_stopping_strings()
        # generate text with text-generation-webui
        await self.llm_gen()
        # Toggle TTS back on if it was toggled off
        if tts_sw:
            await toggle_any_tts(self.settings, tts_sw, force='on')

    async def process_user_prompt(self:Union["Task","Tasks"]):
        # Update an existing LLM payload (Flows), or initialize with defaults
        if tgwui_enabled:
            if self.payload:
                self.payload['text'] = self.text
            else:
                await self.init_llm_payload()
        # apply previously matched tags
        await self.apply_generic_tag_matches(phase='llm')
        self.prompt = self.tags.process_tag_insertions(self.prompt)
        # collect matched tag values
        llm_payload_mods, formatting = await self.collect_llm_tag_values()
        # apply tags relevant to LLM payload
        await self.process_llm_payload_tags(llm_payload_mods)
        # apply formatting tags to LLM prompt
        self.process_prompt_formatting(self.prompt, **formatting)

        # Update bot vars from self (Task)
        self.update_vars()
        # apply params from /prompt command
        self.apply_prompt_params()
        # Apply vars overrides
        self.override_payload()
        # assign finalized prompt to payload
        self.payload['text'] = self.prompt

    def apply_server_mode(self:Union["Task","Tasks"]):
        # TODO Server Mode
        if self.ictx and config.textgen.get('server_mode', False):
            try:
                name1 = f'Server: {self.ictx.guild}'
                self.payload['state']['name1'] = name1
                self.payload['state']['name1_instruct'] = name1
            except Exception as e:
                log.error(f'An error occurred while applying Server Mode: {e}')

    # Add dynamic stopping strings
    def extra_stopping_strings(self: Union["Task", "Tasks"]):
        try:
            name1_value = self.payload['state']['name1']
            name2_value = self.payload['state']['name2']

            # Replace "name1" and "name2" in custom_stopping_strings list
            custom_stopping_strings = self.payload['state'].get('custom_stopping_strings', [])
            if custom_stopping_strings:
                if not isinstance(custom_stopping_strings, list):
                    log.warning("'custom_stopping_strings' must be a list (the value will be ignored)")
                else:
                    custom_stopping_strings = [
                        s.replace("name1", name1_value).replace("name2", name2_value)
                        for s in custom_stopping_strings
                    ]
                    self.payload['state']['custom_stopping_strings'] = custom_stopping_strings

            # Replace "name1" and "name2" in stopping_strings list
            stopping_strings = self.payload['state'].get('stopping_strings', [])
            if stopping_strings:
                if not isinstance(stopping_strings, list):
                    log.warning("'stopping_strings' must be a list (the value will be ignored)")
                else:
                    stopping_strings = [
                        s.replace("name1", name1_value).replace("name2", name2_value)
                        for s in stopping_strings
                    ]
                    self.payload['state']['stopping_strings'] = stopping_strings

        except Exception as e:
            log.error(f'An error occurred while updating stopping strings: {e}')

    # Process responses from text-generation-webui
    async def create_bot_hmessage(self:Union["Task","Tasks"]) -> HMessage:
        try:
            # custom handlings, mainly from 'regenerate'
            self.bot_hmessage = self.local_history.new_message(self.settings.name, self.llm_resp, 'assistant', self.settings._bot_id, text_visible=self.tts_resps)
            if self.user_hmessage:
                self.bot_hmessage.mark_as_reply_for(self.user_hmessage)
            if self.params.regenerated:
                self.bot_hmessage.mark_as_regeneration_for(self.params.regenerated)
            if self.params.bot_hmsg_hidden or self.params.save_to_history == False:
                self.bot_hmessage.update(hidden=True)
            imposter_name = getattr(self.params, 'impersonated_by', None)
            if imposter_name:
                self.bot_hmessage.update(impersonated_by=imposter_name)

            if is_direct_message(self.ictx):
                self.bot_hmessage.dont_save()

            if self.llm_resp:
                truncation = int(self.settings.llmstate.state['truncation_length'] * 4) #approx tokens
                self.bot_hmessage.history.truncate(truncation)
                client.loop.create_task(self.bot_hmessage.history.save())

            return self.bot_hmessage
        except Exception as e:
            log.error(f'An error occurred while creating Bot HMessage: {e}')
            return None

    # Creates User HMessage in HManager
    async def create_user_hmessage(self:Union["Task","Tasks"]) -> HMessage:
        try:
            # Add User HMessage before processing bot reply.
            # this gives time for other messages to accrue before the bot's response, as in realistic chat scenario.
            message = get_message_ctx_inter(self.ictx)
            self.user_hmessage = self.local_history.new_message(self.payload['state']['name1'], self.payload['text'], 'user', self.user.id)
            self.user_hmessage.id = message.id if hasattr(message, 'id') else None
            # set history flag
            if self.params.save_to_history == False:
                self.user_hmessage.update(hidden=True)
            if is_direct_message(self.ictx):
                self.user_hmessage.dont_save()
                await self.warn_direct_channel()
            return self.user_hmessage
        except Exception as e:
            log.error(f'An error occurred while creating User HMessage: {e}')
            return None

    async def create_hmessages(self:Union["Task","Tasks"]) -> tuple[HMessage, HMessage]:
        # Create user message in HManager
        if not self.params.skip_create_user_hmsg:
            await self.create_user_hmessage()

        # Create Bot HMessage in HManager
        if not self.params.skip_create_bot_hmsg:
            # Replacing original Bot HMessage via "regenerate replace"
            if self.params.bot_hmessage_to_update:
                apply_reactions = config.discord['history_reactions'].get('enabled', True)
                self.bot_hmessage = await replace_msg_in_history_and_discord(self.ictx, self.params, self.llm_resp, self.tts_resps, apply_reactions)
                self.params.should_send_text = False
            else:
                await self.create_bot_hmessage()

        return self.user_hmessage, self.bot_hmessage
    

    # Get responses from LLM Payload
    async def llm_gen(self:Union["Task","Tasks"]) -> tuple[str, str]:
        if tgwui_shared_module.model_name == 'None':
            return
        if tgwui.lazy_load_llm:
            try:
                await self.embeds.send('change', 'Loading LLM model ... ', tgwui_shared_module.model_name)
                await tgwui.load_llm_model()
                await self.embeds.delete('change')
            except Exception as e:
                log.error(f"Error lazy-loading LLM Model: {e}")
                await self.embeds.edit_or_send('change', "Error loading LLM Model", str(e))
                return

        try:

            # Stream message chunks
            async def process_chunk(resp_chunk:str):
                # Immediately send message chunks (Do not queue)
                if self.settings.behavior.responsiveness == 1.0 or self.name in ['regenerate', 'continue']:
                    await self.send_response_chunk(resp_chunk)
                # Queue message chunks to MessageManager()
                else:
                    chunk_message = self.message.create_chunk_message(resp_chunk)
                    chunk_message.factor_typing_speed()
                    # Assign some values to the task. 'llm_resp' used later if queued.
                    chunk_task = Task('chunk_message',
                                    self.ictx,
                                    channel=self.channel,
                                    user=self.user,
                                    user_name=self.user_name,
                                    llm_resp=resp_chunk,
                                    message=chunk_message,
                                    params=self.params,
                                    local_history=self.local_history,
                                    istyping=IsTyping(self.channel))
                    # Schedule typing timing for the chunk message
                    updated_istyping_time = await chunk_task.message.update_timing()
                    if updated_istyping_time != chunk_task.message.istyping_time:
                        chunk_task.istyping.start(start_time=updated_istyping_time)
                    chunk_task.message.istyping_time = updated_istyping_time
                    # check if chunk message should be queued
                    if chunk_task.message.send_time is not None:
                        await message_manager.queue_delayed_message(chunk_task)
                    else:
                        await self.send_response_chunk(resp_chunk)

            async def check_censored(search_text):
                for tag in self.tags.llm_censor_tags:
                    trigger_keys = [key for key in tag if key.startswith('trigger')]
                    trigger_match = None
                    for key in trigger_keys:
                        triggers = [t.strip() for t in tag[key].split(',')]
                        for trigger in triggers:
                            trigger_regex = r'\b[^\w]*{}\b'.format(re.escape(trigger))
                            trigger_match = re.search(trigger_regex, search_text, flags=re.IGNORECASE)
                            if trigger_match:
                                censor_text = str(trigger)
                                censor_message = f' (text match: {censor_text})'
                                log.info(f"[TAGS] Censoring: LLM response was blocked{censor_message if censor_text else ''}")
                                self.embeds.create('censor', "LLM response was flagged as inappropriate", "Further task processing has been cancelled.")
                                await self.embeds.send('censor', delete_after=5)
                                raise TaskCensored

            class StreamReplies:
                def __init__(self, task:"Task"):
                    # Only try chunking responses if sending to channel, configured to chunk, and not '/speak' command
                    self.can_chunk:bool          = task.params.should_send_text and (task.settings.behavior.chance_to_stream_reply > 0) and (task.name not in ['speak'])
                    # Behavior values
                    self.chance_to_chunk:float   = task.settings.behavior.chance_to_stream_reply
                    # Chunk syntax
                    self.chunk_syntax:list[str]  = task.settings.behavior.stream_reply_triggers # ['\n\n', '\n', '.']
                    self.longest_syntax_len:int  = max(len(syntax) for syntax in self.chunk_syntax)
                    # For if shorter syntax is initially matched
                    self.retry_counter:int       = 0
                    # Sum of all previously sent message chunks
                    self.already_chunked:str     = ''
                    # Prevents re-checking same string after it fails a random probability check
                    self.last_checked:str        = ''
                    # TTS streaming
                    self.stream_tts:bool         = config.tts_enabled() and self.can_chunk and config.ttsgen.get('tts_streaming', True) and task.params.should_tts
                    if self.stream_tts and tgwui_enabled and tgwui.tts.extension:
                        self.stream_tts = False
                        if not bot_database.was_warned('tgwui_tts_streaming'):
                            log.error(f"TTS Streaming is only supported for API method - not TGWUI extension method")
                            bot_database.update_was_warned('tgwui_tts_streaming')
                    self.streamed_tts:bool       = False

                async def try_chunking(self, resp:str):
                    # Strip previously sent string
                    partial_resp = resp[len(self.already_chunked):]
                    # Strip last checked string
                    check_resp: str = partial_resp[len(self.last_checked):]
                    # Must be enough characters to compare against
                    if len(check_resp) < self.longest_syntax_len:
                        return None

                    # Compare each chunk_syntax to check_resp
                    for syntax in self.chunk_syntax:
                        syntax_len = len(syntax)

                        # Create a window of characters to check for the syntax
                        check_window = check_resp[-(self.longest_syntax_len + 2):]

                        # Check if the syntax is found within this window
                        if syntax in check_window:
                            chance_to_chunk = self.chance_to_chunk

                            # Ensure markdown syntax is not cut off
                            if not patterns.check_markdown_balanced(self.last_checked):
                                return None
                            
                            # Get the match index from the full (potential) response chunk
                            match_start = partial_resp.rfind(syntax)
                            match_end = match_start + syntax_len
                            # No tiny chunks
                            if match_start < 2:
                                return None

                            # Check if less than longest syntax
                            if syntax_len != self.longest_syntax_len:
                                # Allow longer syntax to have a chance to be matched, if possible
                                if not (self.retry_counter + syntax_len) >= self.longest_syntax_len:
                                    self.retry_counter += syntax_len
                                    return None

                            # Increase chance to chunk if double newlines
                            if syntax == '\n\n':
                                chance_to_chunk = chance_to_chunk * 1.5

                            # Special handling for sentence completion
                            elif syntax == '.':
                                # Check if the character before '.' is a digit
                                if partial_resp[match_start - 1].isdigit():
                                    return None  # Avoid chunking on numerical lists
                                # Check if there's a newline before '.'
                                # if '\n' in partial_resp[match_start - 2:match_start]:
                                #     return None  # Avoid chunking on lists with newlines

                                # Reduce chance to chunk
                                chance_to_chunk = chance_to_chunk * 0.5

                            # Update for next iteration
                            self.last_checked += check_resp
                            self.retry_counter = 0

                            # Roll random probability and return message chunk if successful
                            if check_probability(chance_to_chunk):
                                chunk = partial_resp[:match_end] # split at correct index
                                await check_censored(chunk)      # check for censored text
                                self.last_checked = ''           # reset for next iteration
                                self.already_chunked += chunk    # add chunk to already chunked
                                await apply_tts(chunk)           # generate TTS if configured

                                return chunk

                    return None

            # Easier to manage this as a class
            stream_replies = StreamReplies(self)

            async def apply_tts(resp_chunk:str, was_streamed=True):
                if tts_is_enabled(and_online=True, for_mode='api') and api.ttsgen.post_generate:
                    ep = api.ttsgen.post_generate
                    tts_payload:dict = ep.get_payload()
                    tts_payload[ep.text_input_key] = resp_chunk
                    audio_fp = await api.ttsgen.post_generate.call(input_data=tts_payload, main=True)
                    if audio_fp:
                        stream_replies.streamed_tts = was_streamed
                        setattr(self.params, 'streamed_tts', was_streamed)
                        self.tts_resps.append(audio_fp)

            # Sends LLM Payload and processes the generated text
            async def process_responses():

                regenerate = self.payload.get('regenerate', False)

                continued_from = ''
                _continue = self.payload.get('_continue', False)
                if _continue:
                    continued_from = self.payload['state']['history']['internal'][-1][-1]
                include_continued_text = getattr(self.params, "include_continued_text", False)
                continue_condition = _continue and not include_continued_text

                text = self.payload['text']
                image_paths = self.payload['state'].pop('image_paths', None)
                if image_paths:
                    text = {'text': self.payload['text'],
                            'files': image_paths}
                    
                chatbot_wrapper = get_tgwui_functions('chatbot_wrapper')

                # Send payload and get responses
                func = partial(chatbot_wrapper,
                               text = text,
                               state = self.payload['state'],
                               regenerate = regenerate,
                               _continue = _continue,
                               loading_message = True,
                               for_ui = False)

                async for streaming_response in generate_in_executor(func):
                    # Capture response internally as it is generating
                    i_resp_stream = streaming_response.get('internal', [])
                    if len(i_resp_stream) > 0:
                        i_resp_stream:str = i_resp_stream[len(i_resp_stream) - 1][1]
                    
                    # Yes response chunking
                    if stream_replies.can_chunk:
                        # Omit continued text from response processing
                        base_resp:str = i_resp_stream
                        if continue_condition and (len(base_resp) > len(continued_from)):
                            base_resp = base_resp[len(continued_from):]
                        # Check current iteration to see if it meets criteria
                        chunk = await stream_replies.try_chunking(base_resp)
                        # process message chunk
                        if chunk:                          
                            yield chunk                   

                # Check for an unsent chunk
                if stream_replies.already_chunked:
                    # Flag that the task sent chunked responses
                    setattr(self.params, 'was_chunked', True)
                    # Handle last chunk
                    resp_chunk = base_resp[len(stream_replies.already_chunked):].strip()
                    if resp_chunk:
                        # Check last reply chunk for censored text
                        await check_censored(resp_chunk)
                        # generate TTS if configured
                        await apply_tts(resp_chunk)
                        yield resp_chunk
                # Check complete response for censored text
                else:
                    await check_censored(i_resp_stream)

                full_llm_resp = i_resp_stream

                # look for unprocessed tts response after all text generated
                if not stream_replies.streamed_tts:
                    # generate TTS if configured
                    await apply_tts(full_llm_resp, was_streamed=False)

                # Save the complete response
                self.llm_resp = full_llm_resp

            ####################################
            ## RUN ALL HELPER FUNCTIONS ABOVE ##
            ####################################

            # Store time for statistics
            bot_statistics._llm_gen_time_start_last = time.time()

            # Runs chatbot_wrapper(), gets responses
            async for resp_chunk in process_responses():
                await process_chunk(resp_chunk)

            if self.llm_resp:
                self.update_llm_gen_statistics() # Update statistics

        except TaskCensored:
            raise
        except Exception as e:
            log.error(f'An error occurred in llm_gen(): {e}')
            traceback.print_exc()


    # Warn anyone direct messaging the bot
    async def warn_direct_channel(self:Union["Task","Tasks"]):
        warned_id = f'dm_{self.user.id}'
        if not bot_database.was_warned(warned_id):
            bot_database.update_was_warned(warned_id)
            if self.embeds.enabled('system'):
                await self.embeds.send('system', "This conversation will not be saved, ***however***:", "Your interactions will be included in the bot's general logging.")
            else:
                await self.ictx.channel.send("This conversation will not be saved. ***However***, your interactions will be included in the bot's general logging.")

    def format_prompt_with_recent_output(self:Union["Task","Tasks"], prompt:str) -> str:
        try:
            local_history = self.local_history
            formatted_prompt = prompt
            # Find all matches of {user_x} and {llm_x} in the prompt
            matches = patterns.recent_msg_roles.findall(prompt)
            if matches:
                user_msgs = [user_hmsg.text for user_hmsg in local_history.role_messages('user')[-10:]][::-1]
                bot_msgs = [bot_hmsg.text for bot_hmsg in local_history.role_messages('assistant')[-10:]][::-1]
                recent_messages = {'user': user_msgs, 'llm': bot_msgs}
                log.debug(f"format_prompt_with_recent_output {len(user_msgs)}, {len(bot_msgs)}, {repr(user_msgs[0])}, {repr(bot_msgs[0])}")
            # Iterate through the matches
            for match in matches:
                prefix, index = match
                index = int(index)
                if prefix in ['user', 'llm'] and 0 <= index <= 10:
                    message_list = recent_messages[prefix]
                    if not message_list or index >= len(message_list):
                        continue
                    matched_syntax = f"{prefix}_{index}"
                    formatted_prompt = formatted_prompt.replace(f"{{{matched_syntax}}}", message_list[index])
                elif prefix == 'history' and 0 <= index <= 10:
                    user_hmessage = recent_messages['user'][index] if index < len(recent_messages['user']) else ''
                    llm_message = recent_messages['llm'][index] if index < len(recent_messages['llm']) else ''
                    last_character = bot_settings.get_last_setting_for("last_character", self.ictx)
                    formatted_history = f'"{self.user_name}:" {user_hmessage}\n"{last_character}:" {llm_message}\n'
                    matched_syntax = f"{prefix}_{index}"
                    formatted_prompt = formatted_prompt.replace(f"{{{matched_syntax}}}", formatted_history)
            # If {last_image} is a value for any image key
            last_image_fp = os.path.join(shared_path.old_user_images, '__temp/temp_img_0.png')
            formatted_prompt = formatted_prompt.replace('{last_image}', f'{last_image_fp}')
            return formatted_prompt
        except Exception as e:
            log.error(f'An error occurred while formatting prompt with recent messages: {e}')
            return prompt

    def process_prompt_formatting(self:Union["Task","Tasks"], prompt:str|None=None, format_prompt:list|None=None, **kwargs) -> str:
        updated_prompt = prompt if prompt else ''
        try:
            time_offset = kwargs.get('time_offset', None)
            time_format = kwargs.get('time_format', None)
            date_format = kwargs.get('date_format', None)

            # Tag handling for prompt formatting
            if format_prompt:
                for fmt_prompt in format_prompt:
                    updated_prompt = fmt_prompt.replace('{prompt}', updated_prompt)
            # format prompt with any defined recent messages
            updated_prompt = self.format_prompt_with_recent_output(updated_prompt)
            # format prompt with last time
            time_since_last_msg = bot_database.get_last_msg_for(self.channel.id)
            if time_since_last_msg:
                time_since_last_msg = format_time_difference(time.time(), time_since_last_msg)
            else:
                time_since_last_msg = ''
            updated_prompt = updated_prompt.replace('{time_since_last_msg}', time_since_last_msg)
            # Format time if defined
            new_time, new_date = get_time(time_offset, time_format, date_format)
            updated_prompt = updated_prompt.replace('{time}', new_time)
            updated_prompt = updated_prompt.replace('{date}', new_date)
            if updated_prompt != prompt:
                log.info(f'Prompt was formatted: {updated_prompt}')
        except Exception as e:
            log.error(f"Error formatting LLM prompt: {e}")
        return updated_prompt

    def update_llm_gen_statistics(self:Union["Task","Tasks"]):
        try:
            total_gens = bot_statistics.llm.get('generations_total', 0)
            total_gens += 1
            bot_statistics.llm['generations_total'] = total_gens
            # Update tokens statistics
            count_tokens_func = get_tgwui_functions('count_tokens')
            last_tokens = int(count_tokens_func(self.llm_resp))
            bot_statistics.llm['num_tokens_last'] = last_tokens
            total_tokens = bot_statistics.llm.get('num_tokens_total', 0)
            total_tokens += last_tokens
            bot_statistics.llm['num_tokens_total'] = total_tokens
            # Update time statistics
            total_time = bot_statistics.llm.get('time_total', 0)
            llm_gen_time = time.time() - bot_statistics._llm_gen_time_start_last
            total_time += llm_gen_time

            bot_statistics.llm['time_total'] = round(total_time, 4)
            # Update averages
            bot_statistics.llm['tokens_per_gen_avg'] = total_tokens/total_gens
            bot_statistics.llm['tokens_per_sec_avg'] = round((total_tokens/total_time), 4)
            bot_statistics.save()

            # Set self attributes
            if hasattr(self, 'message') and self.message is not None:
                self.message.last_tokens = last_tokens
                self.message.llm_gen_time = llm_gen_time

        except Exception as e:
            log.error(f'An error occurred while saving LLM gen statistics: {e}')

    async def send_char_greeting(self:Union["Task","Tasks"], char_name:str):
        try:
            greeting_msg = ''
            bot_text = None
            greeting:str = self.settings.llmcontext.greeting
            if greeting:
                greeting_msg = greeting.replace('{{user}}', 'user')
                greeting_msg = greeting_msg.replace('{{char}}', char_name)
            else:
                greeting_msg = f'**{char_name}** has entered the chat"'
            bot_text = greeting_msg
            await send_long_message(self.channel, greeting_msg)
            # Play TTS Greeting
            if tts_is_enabled(and_online=True) and config.ttsgen.get('tts_greeting', False):
                self.text = bot_text
                self.embeds.enabled_embeds = {'system': False}
                self.params.tts_args = self.settings.llmcontext.extensions
                await self.run_subtask('speak')
        except TaskCensored:
            raise
        except Exception as e:
            print(traceback.format_exc())
            log.error(f'An error occurred while sending greeting for "{char_name}": {e}')

####################### MOSTLY IMAGE GEN PROCESSING #########################

    async def apply_reactor_mask(self: Union["Task", "Tasks"], images: list[FILE_INPUT], reactor_mask_b64: str) -> list[FILE_INPUT]:
        try:
            # Load reactor mask from base64, convert to L mode for alpha mask
            reactor_mask = Image.open(io.BytesIO(base64.b64decode(reactor_mask_b64))).convert('L')

            # Open original and face-swapped image from their respective file_obj streams
            orig_image_dict = images[0]
            face_image_dict = images.pop(1)
            orig_image = Image.open(orig_image_dict["file_obj"])
            pnginfo = get_pnginfo_from_image(orig_image)

            face_image = Image.open(face_image_dict["file_obj"])
            face_image.putalpha(reactor_mask)                # Apply the mask as alpha channel to face image
            orig_image.paste(face_image, (0, 0), face_image) # Paste the face image (with mask) onto the original

            # Save the modified image back into orig_image_dict's BytesIO buffer
            new_buf = io.BytesIO()
            orig_image.save(new_buf, format="PNG", pnginfo=pnginfo)
            new_buf.seek(0)
            new_buf.name = orig_image_dict["filename"]

            # Update original dict in place
            orig_image_dict["file_obj"] = new_buf
            orig_image_dict["file_size"] = new_buf.getbuffer().nbytes

            # Replace modified dict in the list
            images[0] = orig_image_dict
        except Exception as e:
            log.error(f'Error masking ReActor output images: {e}')
        return images

    async def img_gen(self:Union["Task","Tasks"]) -> list[FILE_INPUT]:
        reactor_args = self.payload.get('alwayson_scripts', {}).get('reactor', {}).get('args', [])
        last_item = reactor_args[-1] if reactor_args else None
        reactor_mask = reactor_args.pop() if is_base64(last_item) else None
        images = await api.imggen._main_imggen(self)
        # Apply ReActor mask
        reactor = self.payload.get('alwayson_scripts', {}).get('reactor', {})
        if len(images) > 1 and reactor and reactor_mask:
            images = await self.apply_reactor_mask(images, reactor_mask)
        return images
    
    def resolve_img_output_dir(self:Union["Task","Tasks"]):
        output_dir = shared_path.output_dir
        try:
            # Accept value whether it is relative or absolute
            if os.path.isabs(self.params.sd_output_dir):
                output_dir = self.params.sd_output_dir
            else:
                output_dir = os.path.join(shared_path.output_dir, self.params.sd_output_dir)
            # backwards compatibility (user defined output dir was originally expected to include the base directory)
            if not config.path_allowed(output_dir):
                log.warning(f"Tried saving Imggen output results to a path which is not allowed: {output_dir}. Defaulting to '/output'.")
                output_dir = shared_path.output_dir
            # Create custom output dir if not already existing
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            log.error(f"An error occurred preparing the imggen output dir: {e}")
            return shared_path.output_dir
        return output_dir

    async def process_image_gen(self: Union["Task", "Tasks"]):
        output_dir = self.resolve_img_output_dir()
        try:
            # Generate images and get file dicts
            images: list[FILE_INPUT] = await self.img_gen()
            if not images:
                return
            
            # Save images locally
            save_all = config.imggen.get('save_all_outputs', False)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for i, image in enumerate(images):
                img_idx = f'_{i}' if save_all else ''
                image_path = os.path.join(output_dir, f"{timestamp}{img_idx}.png")
                with open(image_path, "wb") as f:
                    f.write(image["file_obj"].getbuffer())
                if i == 0 and self.bot_hmessage:
                    self.bot_hmessage.sent_image = os.path.relpath(image_path, shared_path.output_dir)
                if not save_all:
                    break

            # Send images to Discord
            if self.params.should_send_image:
                await send_content_to_discord(ictx=self.ictx, files=images, normalize=False)

        except Exception as e:
            log.error(f"An error occurred when processing image generation: {e}")

    def apply_loractl(self:Union["Task","Tasks"]):
        matched_tags: list = self.tags.matches
        try:
            if not api.imggen.supports_loractrl():
                if not bot_database.was_warned('loractl'):
                    bot_database.update_was_warned('loractl')
                    log.warning(f'loractl integration is enabled in config.yaml, but is not known to be compatible with "{api.imggen.name}".')
                return
            if api.imggen.is_reforge():
                self.payload.setdefault('alwayson_scripts', {}).setdefault('dynamic lora weights (reforge)', {}).setdefault('args', []).append({'Enable Dynamic Lora Weights': True})
            scaling_settings = [v for k, v in config.imggen['extensions'].get('loractl', {}).items() if 'scaling' in k]
            scaling_settings = scaling_settings if scaling_settings else ['']
            # Flatten the matches dictionary values to get a list of all tags (including those within tuples)
            matched_tags.sort(key=lambda x: (isinstance(x, tuple), x[1] if isinstance(x, tuple) else float('inf')))
            all_matched_tags = [tag if isinstance(tag, dict) else tag[0] for tag in matched_tags]
            # Filter the matched tags to include only those with certain patterns in their text fields
            lora_tags = [tag for tag in all_matched_tags if any(patterns.sd_lora.findall(text) for text in (tag.get('positive_prompt', ''), tag.get('positive_prompt_prefix', ''), tag.get('positive_prompt_suffix', '')))]
            if len(lora_tags) >= config.imggen['extensions']['loractl']['min_loras']:
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
                                    scaling_values = config.imggen['extensions']['loractl'].get(scaling_key, '')
                                    if scaling_values:
                                        scaling_factors = [round(float(factor.split('@')[0]) * lora_weight, 2) for factor in scaling_values.split(',')]
                                        scaling_steps = [float(step.split('@')[1]) for step in scaling_values.split(',')]
                                        # Construct/apply the calculated lora-weight string
                                        new_lora_weight_str = f'{",".join(f"{factor}@{step}" for factor, step in zip(scaling_factors, scaling_steps))}'
                                        updated_lora_match = lora_match.replace(str(lora_weight), new_lora_weight_str)
                                        new_positive_prompt = positive_prompt.replace(lora_match, updated_lora_match)
                                        # Update the appropriate key in the tag dictionary
                                        tag[used_key] = new_positive_prompt
                                        log.info(f'''[TAGS] loractl applied: "{lora_match}" > "{updated_lora_match}"''')
        except Exception as e:
            log.error(f"Error processing loractl: {e}")

    def process_img_prompt_tags(self:Union["Task","Tasks"]):
        try:
            self.prompt = self.tags.process_tag_insertions(self.prompt)
            updated_positive_prompt = self.prompt
            updated_negative_prompt = self.neg_prompt

            for tag in self.tags.matches:
                join = tag.get('img_text_joining', ' ')
                if 'imgtag_uninserted' in tag: # was flagged as a trigger match but not inserted
                    log.info(f'''[TAGS] "{tag['matched_trigger']}" not found in the image prompt. Appending rather than inserting.''')
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
            
            self.prompt = updated_positive_prompt
            self.neg_prompt = updated_negative_prompt

            if api.imggen._lora_names:
                self.prompt = self.imgmodel_settings.handle_loras(self.prompt, task=self)
                self.neg_prompt = self.imgmodel_settings.handle_loras(self.neg_prompt, task=self)

        except Exception as e:
            log.error(f"Error processing Img prompt tags: {e}")

    def process_param_variances(self:Union["Task","Tasks"], param_variances: dict) -> dict:
        try:
            param_variances = convert_lists_to_tuples(param_variances) # Only converts lists containing ints and floats (not strings or bools)
            processed_params = copy.deepcopy(param_variances)
            for key, value in param_variances.items():
                # unpack dictionaries assuming they contain variances
                if isinstance(value, dict):
                    processed_params[key] = self.process_param_variances(value)
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
                        log.warning(f'Invalid params "{key}", "{value}" will not be applied.')
                        processed_params.pop(key)  # Remove invalid key
                else:
                    log.warning(f'Invalid params "{key}", "{value}" will not be applied.')
                    processed_params.pop(key)  # Remove invalid key
            return processed_params

        except Exception as e:
            log.error(f"Error processing param variances: {e}")
            return {}

    def select_random_image_or_subdir(self:Union["Task","Tasks"], directory=None, root_dir=None, key=None):
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
            image_file_path, method = self.select_random_image_or_subdir(directory=root_dir, root_dir=None, key=None)
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
            return self.select_random_image_or_subdir(directory=subdir_path, root_dir=root_dir, key=key)
        # If neither image files nor subdirectories found, return None
        return None, None

    async def get_image_tag_args(self:Union["Task","Tasks"], extension, value, key=None, set_dir=None):
        args = {}
        image_file_path = ''
        method = ''
        try:
            home_path = shared_path.dir_user_images
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
                    image_file_path, method = self.select_random_image_or_subdir(directory=os_path, root_dir=root_dir, key=key)
                    if image_file_path:
                        break  # Break the loop if an image is found and selected
                    else:
                        if not os.listdir(os_path):
                            log.warning(f'Valid file not found in a "{home_path}" or any subdirectories: "{value}"')
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
                    return image_file_path # user image does not need to be handled uniquely

                filename = os.path.basename(image_file_path)
                if image_file_path.endswith(".txt"):
                    with open(image_file_path, "r") as txt_file:
                        args['image'] = txt_file.read()
                        method = 'base64 from .txt'
                else:
                    with open(image_file_path, "rb") as image_file:
                        file_bytes = image_file.read()
                        args['image'] = await self.imgmodel_settings.handle_image_input(file_bytes, filename=filename, file_type='image')
                        if not method: # will already have value if random img picked from dir
                            method = 'Image file'
            if method:
                log.info(f'[TAGS] {extension}: "{value}" ({method}).')
                if method == 'Random from folder':
                    args['selected_folder'] = os.path.dirname(image_file_path)
            return args
        except Exception as e:
            log.error(f"[TAGS] Error processing {extension} tag: {e}")
            return {}

    async def process_img_payload_tags(self:Union["Task","Tasks"], mods:dict):
        try:
            change_imgmodel: str  = mods.pop('change_imgmodel', None)
            swap_imgmodel: str    = mods.pop('swap_imgmodel', None)
            payload_updates: dict = mods.pop('payload', None)
            aspect_ratio: str     = mods.pop('aspect_ratio', None)
            param_variances: dict = mods.pop('param_variances', {})
            controlnet: list      = mods.pop('controlnet', [])
            forge_couple: dict    = mods.pop('forge_couple', {})
            layerdiffuse: dict    = mods.pop('layerdiffuse', {})
            reactor: dict         = mods.pop('reactor', {})
            img2img: str          = mods.pop('img2img', '')
            img2img_mask: str     = mods.pop('img2img_mask', '')
            self.params.sd_output_dir = (mods.pop('sd_output_dir', self.params.sd_output_dir)).lstrip('/')  # Remove leading slash if it exists
            self.params.img_censoring = mods.pop('img_censoring', self.params.img_censoring)

            # Imgmodel handling
            new_imgmodel = change_imgmodel or swap_imgmodel or None
            if new_imgmodel:
                current_imgmodel_name = self.imgmodel_settings.last_imgmodel_name
                current_imgmodel_value = self.imgmodel_settings.last_imgmodel_value
                mode = 'change' if change_imgmodel else 'swap'
                verb = 'Changing' if change_imgmodel else 'Swapping'
                # Check if new model same as current model
                if (new_imgmodel == current_imgmodel_name) or (new_imgmodel == current_imgmodel_value):
                    log.info(f'[TAGS] Img model was triggered to {mode}, but it is the same as current.')
                else:
                    # Add values to Params. Will trigger model change
                    self.params.imgmodel = await self.imgmodel_settings.get_model_params(imgmodel=new_imgmodel, mode=mode, verb=verb)
                    log.info(f'[TAGS] {verb} Img model: "{new_imgmodel}"')
            # Payload handling
            if payload_updates:
                if isinstance(payload_updates, dict):
                    log.info(f"[TAGS] Updated payload: '{payload_updates}'")
                    self.imgmodel_settings.handle_payload_updates(payload_updates, self)
                else:
                    log.warning("[TAGS] A tag was matched with invalid 'payload'; must be a dictionary.")
            # Aspect Ratio
            if aspect_ratio:
                try:
                    current_avg = self.imgmodel_settings.last_imgmodel_res
                    # Use AR from input image, while adhering to current model res
                    if img2img and aspect_ratio.lower() in ['use img2img', 'from img2img']:
                        from io import BytesIO
                        base64_string = str(img2img)
                        image_data = base64.b64decode(base64_string)
                        image_bytes = BytesIO(image_data)
                        with Image.open(image_bytes) as img:
                            img_w, img_h = img.size
                        n, d = ar_parts_from_dims(img_w, img_h)
                    else:
                        n, d = get_aspect_ratio_parts(aspect_ratio)
                    w, h = dims_from_ar(current_avg, n, d)
                    size_update = {'width': w, 'height': h}
                    self.imgmodel_settings.handle_payload_updates(size_update, self)
                    log.info(f'[TAGS] Applied aspect ratio "{aspect_ratio}" (Width: "{w}", Height: "{h}").')
                except Exception as e:
                    log.error(f"[TAGS] Error applying aspect ratio: {e}")
            # Param variances handling
            if param_variances:
                processed_params = self.process_param_variances(param_variances)
                log.info(f"[TAGS] Applied Param Variances: '{processed_params}'")
                self.imgmodel_settings.apply_payload_param_variances(processed_params, self)
            # Controlnet handling
            if controlnet and config.controlnet_enabled():
                self.imgmodel_settings.apply_controlnet(controlnet, self)
            # forge_couple handling
            if forge_couple and config.forgecouple_enabled():
                self.imgmodel_settings.apply_forge_couple(forge_couple, self)
            # layerdiffuse handling
            if layerdiffuse and config.layerdiffuse_enabled():
                self.imgmodel_settings.apply_layerdiffuse(layerdiffuse, self)
            # ReActor face swap handling
            if reactor and config.reactor_enabled():
                self.imgmodel_settings.apply_reactor(reactor, self)

            # Img2Img handling
            if img2img and (api.imggen.is_sdwebui_variant() or api.imggen.is_swarm()):
                self.params.mode = 'img2img'
                if api.imggen.is_swarm():
                    self.payload['initimage'] = str(img2img)
                else:
                    self.payload['init_images'] = [str(img2img)]
                # Inpaint Mask handling
                if img2img_mask:
                    if api.imggen.is_swarm():
                        self.payload['maskimage'] = str(img2img_mask)
                    else:
                        self.payload['mask'] = str(img2img_mask)
        except Exception as e:
            log.error(f"[TAGS] Error processing Img tags: {e}")
            traceback.print_exc()

    # The methods of this function allow multiple extensions with an identical "select image from random folder" value to share the first selected folder.
    # The function will first try to find a specific image file based on the extension's key name (ex: 'canny.png' or 'img2img_mask.jpg')
    async def collect_img_extension_mods(self, mods):
        controlnet = mods.get('controlnet', [])
        reactor = mods.get('reactor', None)
        img2img = mods.get('img2img', None)
        img2img_mask = mods.get('img2img_mask', None)
        set_dir = None
        if img2img:
            try:
                img2img_args = await self.get_image_tag_args('Img2Img', img2img, key='img2img', set_dir=set_dir)
                mods['img2img'] = img2img_args.get('image', '')
                if img2img_args:
                    if set_dir is None:
                        set_dir = img2img_args.get('selected_folder', None)
                    if img2img_mask:
                        img2img_mask_args = await self.get_image_tag_args('Img2Img Mask', img2img_mask, key='img2img_mask', set_dir=set_dir)
                        mods['img2img_mask'] = img2img_mask_args.get('image', '')
                        if img2img_mask_args:
                            if set_dir is None:
                                set_dir = img2img_mask_args.get('selected_folder', None)
            except Exception as e:
                log.error(f"[TAGS] Error collecting img2img tag values: {e}")
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
                        cnet_args = await self.get_image_tag_args('ControlNet Image', image, key=prefix, set_dir=set_dir)
                        if not cnet_args:
                            controlnet[idx] = {}
                        else:
                            if set_dir is None:
                                set_dir = cnet_args.pop('selected_folder', None)
                            else:
                                cnet_args.pop('selected_folder', None)
                            controlnet[idx].update(cnet_args)
                            controlnet[idx]['enabled'] = True
                            # Update controlnet item with mask_image information
                            if mask_image:
                                key = f'{prefix}_mask' if prefix else None
                                cnet_mask_args = await self.get_image_tag_args('ControlNet Mask', mask_image, key=key, set_dir=set_dir)
                                controlnet[idx]['mask_image'] = cnet_mask_args.get('image', None)
                                if cnet_mask_args:
                                    if set_dir is None:
                                        set_dir = cnet_mask_args.get('selected_folder', None)
                mods['controlnet'] = controlnet
            except Exception as e:
                log.error(f"[TAGS] Error collecting ControlNet tag values: {e}")
        if reactor:
            try:
                image = reactor.get('image', None)
                mask_image = reactor.get('mask', None)
                if image:
                    reactor_args = await self.get_image_tag_args('ReActor Enabled', image, key='reactor', set_dir=None)
                    if reactor_args:
                        reactor_args.pop('selected_folder', None)
                        mods['reactor'].update(reactor_args)
                        mods['reactor']['enabled'] = True
                        if mask_image:
                            reactor_mask_args = await self.get_image_tag_args('ReActor Mask', mask_image, key='reactor_mask', set_dir=set_dir)
                            mods['reactor']['mask'] = reactor_mask_args.get('image', '')
                            if reactor_mask_args and set_dir is None:
                                set_dir = reactor_mask_args.get('selected_folder', None)
            except Exception as e:
                log.error(f"[TAGS] Error collecting ReActor tag values: {e}")
        return mods

    async def collect_img_tag_values(self:Union["Task","Tasks"]):
        img_payload_mods = {}
        payload_order_hack = {}
        controlnet_args = {}
        forge_couple_args = {}
        layerdiffuse_args = {}
        reactor_args = {}
        accept_only_first = ['aspect_ratio', 'img2img', 'img2img_mask', 'sd_output_dir']
        try:
            for tag in self.tags.matches:
                tag_dict:TAG = self.tags.untuple(tag)
                tag_name, tag_print = self.tags.get_name_print_for(tag_dict)
                for key, value in tag_dict.items():
                    # Check censoring
                    if key == 'img_censoring' and value != 0:
                        img_payload_mods['img_censoring'] = int(value)
                        censor_text = tag_dict.get('matched_trigger', '')
                        censor_message = f' (text match: {censor_text})'
                        log.info(f"[TAGS] Censoring: {'Image will be blurred' if value == 1 else 'Image generation blocked'}{censor_message if censor_text else ''}")
                        if value == 2:
                            await self.embeds.send('img_send', "Image prompt was flagged as inappropriate.", "Image generation task has been cancelled.", delete_after=5)
                            raise TaskCensored
                    # Accept only the first occurance
                    elif key in accept_only_first and not img_payload_mods.get(key):
                        img_payload_mods[key] = value
                    # Accept only first 'change' or 'swap'
                    elif (key == 'change_imgmodel' and not is_direct_message(self.ictx)) or key == 'swap_imgmodel' and not (img_payload_mods.get('change_imgmodel') or img_payload_mods.get('swap_imgmodel')):
                        img_payload_mods[key] = str(value)
                    # Allow multiple to accumulate
                    elif key == 'payload':
                        try:
                            if img_payload_mods.get('payload'):
                                payload_order_hack = dict(value)
                                img_payload_mods['payload'] = update_dict(payload_order_hack, img_payload_mods['payload'], in_place=False)
                            else:
                                img_payload_mods['payload'] = dict(value)
                        except Exception:
                            log.warning(f"[TAGS] Error processing a matched 'payload' {tag_print}; ensure it is a dictionary.")
                    elif key == 'img_param_variances':
                        img_payload_mods.setdefault('param_variances', {})
                        try:
                            update_dict(img_payload_mods['param_variances'], dict(value))
                        except Exception:
                            log.warning(f"[TAGS] Error processing a matched 'img_param_variances' {tag_print}; ensure it is a dictionary.")
                    # get any ControlNet extension params
                    elif key.startswith('controlnet') and config.controlnet_enabled():
                        index = int(key[len('controlnet'):]) if key != 'controlnet' else 0  # Determine the index (cnet unit) for main controlnet args
                        controlnet_args.setdefault(index, {}).update({'image': value, 'enabled': True})         # Update controlnet args at the specified index
                    elif key.startswith('cnet') and config.controlnet_enabled():
                        # Determine the index for controlnet_args sublist
                        if key.startswith('cnet_'):
                            index = int(key.split('_')[0][len('cnet'):]) if not key.startswith('cnet_') else 0  # Determine the index (cnet unit) for additional controlnet args
                        controlnet_args.setdefault(index, {}).update({key.split('_', 1)[-1]: value})   # Update controlnet args at the specified index
                    # get any layerdiffuse extension params
                    elif key == 'layerdiffuse' and config.layerdiffuse_enabled():
                        layerdiffuse_args['method'] = str(value)
                    elif key.startswith('laydiff_') and config.layerdiffuse_enabled():
                        laydiff_key = key[len('laydiff_'):]
                        layerdiffuse_args[laydiff_key] = value
                    # get any ReActor extension params
                    elif key == 'reactor' and config.reactor_enabled():
                        reactor_args['image'] = value
                    elif key.startswith('reactor_') and config.reactor_enabled():
                        reactor_key = key[len('reactor_'):]
                        reactor_args[reactor_key] = value
                    # get any Forge Couple extension params
                    elif key == 'forge_couple' and config.forgecouple_enabled():
                        if value.startswith('['):
                            forge_couple_args['maps'] = list(value)
                        else: 
                            forge_couple_args['direction'] = str(value)
                    elif key.startswith('couple_') and config.forgecouple_enabled():
                        forge_couple_key = key[len('couple_'):]
                        if value.startswith('['):
                            forge_couple_args[forge_couple_key] = list(value)
                        else:
                            forge_couple_args[forge_couple_key] = str(value)

            # Add the collected SD WebUI extension args to the img_payload_mods dict
            if controlnet_args:
                img_payload_mods.setdefault('controlnet', [])
                for index in sorted(set(controlnet_args.keys())):   # This flattens down any gaps between collected ControlNet units (ensures lowest index is 0, next is 1, and so on)
                    alwayson = self.payload.setdefault('alwayson_scripts', {})
                    controlnet = alwayson.setdefault('controlnet', {})
                    args = controlnet.setdefault('args', [])
                    # Ensure at least one element exists
                    if len(args) == 0:
                        args.append({})
                    user_default_cnet_unit = copy.copy(args[0])
                    cnet_unit_args = controlnet_args.get(index, {})
                    cnet_unit = update_dict(user_default_cnet_unit, cnet_unit_args)
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

            img_payload_mods = await self.collect_img_extension_mods(img_payload_mods)
        except TaskCensored:
            raise
        except Exception as e:
            log.error(f"Error collecting Img tag values: {e}")
        return img_payload_mods

    def init_img_payload(self:Union["Task","Tasks"]):
        try:
            # Apply values set by /image command to prompt/neg_prompt (Additional /image cmd values are applied later)
            imgcmd_params   = self.params.imgcmd
            self.params.mode = imgcmd_params['img2img'].get('mode', 'txt2img')
            neg_prompt: str = imgcmd_params['neg_prompt']
            style: dict     = imgcmd_params['style']
            positive_style: str = style.get('positive', "{}")
            negative_style: str = style.get('negative', '')

            self.prompt     = positive_style.format(self.prompt)
            self.neg_prompt = f"{neg_prompt}, {negative_style}" if negative_style else neg_prompt

            # Get endpoint for mode
            imggen_ep = self.params.get_active_imggen_ep()

            if isinstance(imggen_ep, ImgGenEndpoint):
                self.payload = imggen_ep.get_payload()
            else:
                raise RuntimeError(f"Error initializing img payload: No valid endpoint available for main imggen task.")

            # Apply settings from imgmodel configuration
            self.settings.imgmodel.override_payload(task=self)

        except Exception as e:
            log.error(f"Error initializing img payload: {e}")

#################################################################
############################ TASKS ##############################
#################################################################
class TaskCensored(Exception):
    """Custom exception to abort censored text generation tasks"""
    pass

class Tasks(TaskProcessing):
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Task management framework which could be improved by further subclassing.
    Instances of Task() are run/queued in TasksManager() queue.
    The value for the Task() "name" attribute is used to run the appropriate task.
    Each Task() is processed by Tasks() with the methods of TaskProcessing().
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    #################################################################
    ######################### MESSAGE TASK ##########################
    #################################################################
    # message_task is split into two separate functions to apply any intentional delays after text is generated (behavior settings)
    async def message_llm_task(self:"Task"):
        try:
            # make working copy of user's request
            self.prompt = self.text

            # Stop any pending spontaneous message task for current channel
            await spontaneous_messaging.reset_for_channel(task_name=self.name, ictx=self.ictx)

            # match tags labeled for user / userllm.
            await self.tags.match_tags(self.text, self.settings.get_vars(), phase='llm')

            # Updates prompt / LLM payload based on previously matched tags.
            await self.process_user_prompt()

            # check what bot should do
            self.params.update_bot_should_do(self.tags)

            # Bot should generate text...
            if tgwui_enabled and self.params.should_gen_text:

                # Process LLM model change if any
                if self.params.llmmodel:
                    # RUN CHANGE LLMMODEL AS SUBTASK
                    self.run_subtask('change_llmmodel')

                # generate text with TGWUI
                await self.message_llm_gen()

        except TaskCensored:
            await self.embeds.delete('img_gen')
            await self.embeds.delete('change')
            raise
        except Exception as e:
            print(traceback.format_exc())
            log.error(f'An error occurred while processing "{self.name}" request: {e}')
            await self.embeds.edit_or_send('system', f'An error occurred while processing "{self.name}" request', e)
            await self.embeds.delete('img_gen')
            await self.embeds.delete('change')


    # Parked Message Task may be resumed from here
    async def message_post_llm_task(self:"Task") -> tuple[HMessage, HMessage]:
        try:
            # set response to prompt, then pre-process responses
            if self.llm_resp:
                self.prompt = self.llm_resp

                # Log message exchange
                log.info(f'''{self.user_name}: "{self.payload['text']}"''')
                log.info(f'''{self.payload['state']['name2']}: "{self.llm_resp.strip()}"''')

                # Create messages in History
                await self.create_hmessages()

                # add history reactions to user message
                if config.discord['history_reactions'].get('enabled', True):
                    await bg_task_queue.put(apply_reactions_to_messages(self.ictx, self.user_hmessage))

                # Swap LLM Model back if triggered
                if self.params.llmmodel and self.params.llmmodel.get('mode', 'change') == 'swap':
                    # CREATE TASK AND QUEUE IT
                    change_llmmodel_task = Task('change_llmmodel', self.ictx, params=self.params) # Only needs current params
                    await task_manager.queue_task(change_llmmodel_task, 'gen_queue')

            # Stop typing if typing
            if hasattr(self, 'istyping') and self.istyping is not None:
                self.istyping.stop()

            # send responses (text, TTS, images)
            await self.send_responses()

            # schedule idle, set Sponantaneous Message, etc
            await self.reset_behaviors()

            # Create an img gen task if triggered to
            if imggen_enabled:
                await self.message_img_gen()

            return self.user_hmessage, self.bot_hmessage

        except TaskCensored:
            await self.embeds.delete('img_gen')
            await self.embeds.delete('change')
            raise
        except Exception as e:
            print(traceback.format_exc())
            log.error(f'An error occurred while processing "{self.name}" request: {e}')
            await self.embeds.edit_or_send('system', f'An error occurred while processing "{self.name}" request', e)
            await self.embeds.delete('img_gen')
            await self.embeds.delete('change')
            return None, None
        
    async def check_message(self:"Task") -> bool:
        proceed = True

        if hasattr(self, "message") and self.message is not None:
            self.message.factor_typing_speed()
            updated_istyping_time = await self.message.update_timing()
            if updated_istyping_time != self.message.istyping_time:
                self.istyping.start(start_time=updated_istyping_time)
            self.message.istyping_time = updated_istyping_time
            send_time = getattr(self.message, 'send_time', None)
            if send_time:
                self.name = 'message_post_llm' # Change self task name
                if message_manager.send_msg_queue.empty():
                    writing_message = '' if self.settings.behavior.maximum_typing_speed < 0 else f' (seconds to write: {round(self.message.seconds_to_write, 2)})'
                    log.info(f'Response to message #{self.message.num} will send in {round(send_time - time.time(), 2)} seconds.{writing_message}')
                proceed = False

        return proceed

    #################################################################
    ################### MESSAGE TASK VARIATIONS #####################
    #################################################################
    # Generic message task
    async def message_task(self:"Task"):
        await self.message_llm_task()
        return await self.message_post_llm_task()

    # From normal discord message
    async def on_message_task(self:"Task"):
        await self.message_llm_task()
        # check if task should have special handling
        proceed = await self.check_message()
        if proceed:
            await self.message_post_llm_task()

    # From Spontaneous Message feature
    async def spontaneous_message_task(self:"Task"):
        await self.message_llm_task()
        # replace the previous discord message ID with a randomly generated one
        self.ictx.id = int(''.join([str(random.randint(0, 9)) for _ in range(19)]))
        # check if task should have special handling
        proceed = await self.check_message()
        if proceed:
            await self.message_post_llm_task()

    # From /image command (use_llm = True)
    async def msg_image_cmd_task(self:"Task"):
        await self.message_llm_task()
        await self.message_post_llm_task()

    # From Flows
    async def flows_task(self:"Task"):
        await self.message_llm_task()
        await self.message_post_llm_task()

    #################################################################
    ############################ API TASK ###########################
    #################################################################
    async def run_api_task(self:"Task", endpoint:Endpoint, config:dict):
        # update default endpoint payload with any provided input
        input_data = config.pop('input_data', {}) # pop input
        base_payload = endpoint.get_payload()     # get default payload for endpoint
        if base_payload and isinstance(base_payload, dict):
            input_data = deep_merge(base_payload, input_data)
        config['input_data'] = input_data         # put payload back in before formatting
        # formats bot syntax like '{prompt}', {llm_0}, etc
        formatted_payload = self.format_api_payload(config)
        # Call and collect results
        api_results = await endpoint.call(ictx=self.ictx, **formatted_payload)
        if api_results:
            self.collect_extra_content(api_results)

    def format_api_payload(self: "Task", api_payload: dict):
        def recursive_format(value):
            if isinstance(value, str):
                formatted = self.process_prompt_formatting(format_prompt=[value])
                if formatted != value:
                    formatted = valueparser.parse_value(formatted)
                return formatted
            elif isinstance(value, dict):
                return {k: recursive_format(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [recursive_format(item) for item in value]
            elif isinstance(value, tuple):
                return tuple(recursive_format(item) for item in value)
            else:
                return value

        return recursive_format(api_payload)

    async def api_task(self:"Task"):
        api_config = getattr(self, 'api_config')
        endpoint, config = api.get_endpoint_from_config(api_config)
        if not isinstance(endpoint, Endpoint):
            log.warning(f'Endpoint not found for triggered "call_api"')
        else:
            await self.run_api_task(endpoint, config)

    #################################################################
    ######################## WORKFLOW TASK ##########################
    #################################################################
    async def run_workflow_task(self:"Task", config:dict):
        # formats bot syntax like '{prompt}', {llm_0}, etc
        formatted_payload = self.format_api_payload(config)
        # Run workflow and collect results
        workflow_results = await call_stepexecutor(task=self, **formatted_payload)
        if workflow_results:
            self.collect_extra_content(workflow_results)

    async def workflow_task(self:"Task"):
        workflow_config = getattr(self, 'workflow_config')
        await self.run_workflow_task(workflow_config)

    #################################################################
    ###################### USER COMMAND TASK ########################
    #################################################################
    async def custom_command_task(self:"Task"):
        try:
            custom_cmd_config:dict = getattr(self, 'custom_cmd_config')
            # Unpack config
            cmd_name:dict = custom_cmd_config['custom_cmd_name']
            selections:dict = custom_cmd_config['custom_cmd_selections']
            options_meta:dict = custom_cmd_config['custom_cmd_option_meta']
            main_steps:dict = custom_cmd_config['custom_cmd_steps']

            # Process each option value with StepExecutor if steps are defined
            processed_params = {}
            option_names = [] # names for logging
            for meta in options_meta:
                name = meta["name"]
                value = selections.get(name)
                if value is None:
                    continue  # Skip optional inputs not provided
                # Decode Attachments and retain filename for if needed.
                if isinstance(value, discord.Attachment):
                    file_name = Path(value.filename).stem
                    fn_key = f"{name}_file_name"
                    processed_params[fn_key] = file_name
                    value = await value.read()
                option_names.append(name)
                steps = meta.get("steps")
                if steps:
                    value = await call_stepexecutor(steps=steps, input_data=value, context=processed_params, task=self, prefix=f'Pre-processing results of cmd option "{name}" with ')
                processed_params[name] = value

            # Run the command's main steps if defined
            if main_steps:
                cmd_results = await call_stepexecutor(steps=main_steps, task=self, input_data=processed_params, context=processed_params, prefix=f'Processing command "{cmd_name}" with ')
                await self.embeds.send('img_send', f'{self.user_name} used "/{cmd_name}"', f'Options: {", ".join(optname for optname in option_names)}')
                if cmd_results:
                    self.collect_extra_content(cmd_results)
            else:
                await self.embeds.send('img_send', f'{self.user_name} used "/{cmd_name}"', f'Options: {", ".join(optname for optname in option_names)}')
        except Exception as e:
            e_msg = f'An error occurred while processing "/{cmd_name}"'
            log.error(f'{e_msg}: {e}')
            await self.ictx.followup.send(f'{e_msg} \n> {e}', ephemeral=True)
            raise

    #################################################################
    ######################### CONTINUE TASK #########################
    #################################################################
    async def continue_task(self:"Task"):
        # Attributes set in Task
        self.local_history:History
        self.target_discord_msg:discord.Message # set as kwargs
        try:
            original_user_hmessage, original_bot_hmessage = self.local_history.get_history_pair_from_msg_id(self.target_discord_msg.id)
            # Requires finding original bot HMessage in history
            if not original_bot_hmessage:
                await self.ictx.followup.send('Message not found in current chat history. Try using "continue" on a response from the character.', ephemeral=True)
                return

            # To continue, both messages must be visible
            temp_reveal_user_hmsg = True if original_user_hmessage.hidden else False
            if temp_reveal_user_hmsg:
                original_user_hmessage.update(hidden=False)
            temp_reveal_bot_hmsg = True if original_bot_hmessage.hidden else False
            if temp_reveal_bot_hmsg:
                original_bot_hmessage.update(hidden=False)
            # Prepare payload. 'text' parameter unimportant (only used for logging)
            self.text = original_user_hmessage.text if original_user_hmessage else (self.target_discord_msg.clean_content or '')
            await self.init_llm_payload()
            sliced_history = original_bot_hmessage.new_history_end_here()
            sliced_i, _ = sliced_history.render_to_tgwui_tuple()
            # using original 'visible' produces wonky TTS responses combined with "Continue" function. Using 'internal' for both.
            self.payload['state']['history']['internal'] = copy.deepcopy(sliced_i)
            self.payload['state']['history']['visible'] = copy.deepcopy(sliced_i)
            self.payload['_continue'] = True # let TGWUI handle the continue function
            # Restore hidden status
            if temp_reveal_user_hmsg:
                original_user_hmessage.update(hidden=True)
            if temp_reveal_bot_hmsg:
                original_bot_hmessage.update(hidden=True)

            self.embeds.create('continue', 'Continuing ... ', f'Continuing text for {self.user_name}')
            await self.embeds.send('continue')

            # Get a possible message to reply to
            ref_message = self.target_discord_msg
            if original_bot_hmessage.id != self.target_discord_msg.id:
                ref_message = await self.channel.fetch_message(original_bot_hmessage.id)

            # Assign ref message to params. Will become 'None' if message responses are streamed (chunked)
            self.params.ref_message = ref_message

            ## Finalize payload, generate text via TGWUI.
            # Does not create hmessages. Responses may be streamed.
            await self.message_llm_gen()

            await self.embeds.delete('continue') # delete embed

            # Return if failed
            if not self.llm_resp:
                await self.ictx.followup.send('Failed to continue text.', silent=True)
                return

            # Extract the continued text from previous text
            continued_text = self.llm_resp[len(original_bot_hmessage.text):]

            # Return if nothing new generated
            if not continued_text.strip():
                no_gen_msg = await self.ictx.followup.send(':warning: Generation was continued, but nothing new was added.')
                await bg_task_queue.put(sleep_delete_message(no_gen_msg))
                return

            # Log message exchange
            log.info(f'''{self.user_name}: "{self.payload['text']}"''')
            log.info('Continued text:')
            log.info(f'''{self.payload['state']['name2']}: "{self.llm_resp}"''')

            # Update the original message in history manager
            updated_bot_hmessage = original_bot_hmessage
            updated_bot_hmessage.is_continued = True                         # Mark message as continued
            updated_bot_hmessage.update(text=self.llm_resp, text_visible=self.tts_resps) # replace responses
            updated_bot_hmessage.related_ids.insert(0, updated_bot_hmessage.id) # Insert previous last message id into related ids

            new_discord_msg = None

            was_chunked = getattr(self.params, 'was_chunked', False)

            if was_chunked and self.params.chunk_msg_ids:
                updated_bot_hmessage.id = self.params.chunk_msg_ids.pop(-1)
                updated_bot_hmessage.related_ids.extend(self.params.chunk_msg_ids)

            else:
                # Send new response to discord, get IDs and Message()
                sent_msg_ids, new_discord_msg = await send_long_message(self.channel, continued_text, self.params.ref_message)
                # Update IDs for the new Bot HMessage
                sent_msg_ids:list
                last_msg_id = sent_msg_ids.pop(-1)
                updated_bot_hmessage.update(id=last_msg_id, related_ids=sent_msg_ids)

            # Apply any reactions applicable to HMessage
            msg_ids_to_edit = [updated_bot_hmessage.id] + updated_bot_hmessage.related_ids
            if config.discord['history_reactions'].get('enabled', True):
                await bg_task_queue.put(apply_reactions_to_messages(self.ictx, updated_bot_hmessage, msg_ids_to_edit, new_discord_msg))

            # process any tts resp
            # if self.tts_resps:
            #     await voice_clients.process_audio_file(self.ictx, self.tts_resps[0], updated_bot_hmessage)

        except TaskCensored:
            raise
        except Exception as e:
            e_msg = 'An error occurred while processing "Continue"'
            log.error(f'{e_msg}: {e}')
            await self.ictx.followup.send(f'{e_msg} \n> {e}', silent=True)
            await self.embeds.delete('continue') # delete embed
            raise

    #################################################################
    ####################### REGENERATE TASK #########################
    #################################################################
    async def regenerate_task(self:"Task"):
        # Attributes set in Task
        self.local_history:History
        self.target_discord_msg:discord.Message # set as kwargs
        self.target_hmessage:HMessage # set as kwargs
        self.mode:str # set as kwargs
        try:
            self.user_hmessage, self.bot_hmessage = self.local_history.get_history_pair_from_msg_id(self.target_discord_msg.id, user_hmsg_attr='regenerated_from', bot_hmsg_list_attr='regenerations')

            # Replace method requires finding original bot HMessage in history
            if self.mode == 'replace' and not self.bot_hmessage:
                await self.ictx.followup.send("Message not found in current chat history.", ephemeral=True)
                return
            '''''''''''''''''''''''''''''''''''
            Original user text is needed for both 'create' and 'replace'
            For create, `target_bot_hmessage` will hide the currently revealed message.
            For replace, `target_bot_hmessage` will replace the currently revealed message.
            '''''''''''''''''''''''''''''''''''
            all_bot_regens = self.user_hmessage.regenerations
            # Update attributes
            if not all_bot_regens and self.mode == 'create':
                self.bot_hmessage.mark_as_regeneration_for(self.user_hmessage)

            original_discord_msg = self.target_discord_msg

            # if command used on user's own message
            if self.user == self.target_discord_msg.author:
                self.text = self.target_discord_msg.clean_content   # get the message contents for prompt
                target_bot_hmessage = self.bot_hmessage                 # set to most recent bot regeneration
                if all_bot_regens:
                    for i in range(len(all_bot_regens)):
                        if not all_bot_regens[i].hidden:
                            target_bot_hmessage = all_bot_regens[i] # set the target message to a non-hidden bot HMessage
                            break
            # if command used on a bot regen
            elif client.user == self.target_discord_msg.author:
                if not self.user_hmessage or isinstance(self.user_hmessage, str): # will be uuid text string if message failed to be found
                    await self.ictx.followup.send("Original user prompt is required, which could not be found from the selected message or in current chat history. Please try again, using the command on your own message.", ephemeral=True)
                    return
                # default the target hmessage to the one associated with the message selected via cmd
                target_bot_hmessage = self.target_hmessage
                # get the user's message contents for prompt
                original_discord_msg = await self.channel.fetch_message(self.user_hmessage.id)
                self.text = original_discord_msg.clean_content
                # If other regens, change target to the unhidden regenerated message
                if target_bot_hmessage.hidden and all_bot_regens and self.mode == 'create':
                    for i in range(len(all_bot_regens)):
                        if not all_bot_regens[i].hidden:
                            target_bot_hmessage = all_bot_regens[i] # set to first non-hidden bot HMessage
                            break
            else:
                return # invalid user
            
            # To regenerate, both messages must be visible
            temp_reveal_msgs = True if (self.user_hmessage.hidden and target_bot_hmessage.hidden) else False
            if temp_reveal_msgs:
                self.user_hmessage.update(hidden=False)
                target_bot_hmessage.update(hidden=False)

            # Initialize payload with sliced history
            hmessage_for_slicing = self.user_hmessage or self.bot_hmessage
            await self.init_llm_payload()
            sliced_history = hmessage_for_slicing.new_history_end_here(include_self=False) # Exclude the original exchange pair
            sliced_i, _ = sliced_history.render_to_tgwui_tuple()
            self.payload['state']['history']['internal'] = copy.deepcopy(sliced_i)
            self.payload['state']['history']['visible'] = copy.deepcopy(sliced_i)

            self.embeds.create('regenerate', 'Regenerating ... ', f'Regenerating text for {self.user_name}')
            await self.embeds.send('regenerate')

            # Flags to skip message logging/special message handling
            regen_params = Params()
            regen_params.skip_create_user_hmsg = True
            regen_params.ref_message = original_discord_msg
            regen_params.bot_hmsg_hidden = temp_reveal_msgs # Hides new bot HMessage if regenerating from hidden exchange
            if self.mode == 'create':
                regen_params.regenerated = self.user_hmessage
            if self.mode == 'replace':
                regen_params.bot_hmessage_to_update = target_bot_hmessage
                regen_params.target_discord_msg_id = target_bot_hmessage.id

            # CLONE CURRENT TASK AS A MESSAGE TASK, RUN IT AS SUBTASK
            regenerate_message_task: Task = self.clone('message', self.ictx)
            regenerate_message_task.params = regen_params # Update params for the Task
            _, new_bot_hmessage = await regenerate_message_task.run_subtask()
    
            del regenerate_message_task # Delete the task
            new_bot_hmessage:HMessage

            # Mark as reply
            new_bot_hmessage.mark_as_reply_for(self.user_hmessage)

            # Update the messages hidden statuses
            self.user_hmessage.update(hidden=new_bot_hmessage.hidden)
            # If set to toggle, make hidden again regardless of what was set during regeneration
            if temp_reveal_msgs:
                self.user_hmessage.update(hidden=True)

            if self.mode == 'create':
                new_bot_hmessage.mark_as_regeneration_for(self.user_hmessage)
                # Adjust attributes/reactions for prior active bot reply/regeneration
                target_bot_hmessage.update(hidden=True) # always hide previous regen when creating

                target_bot_hmessage_ids = [target_bot_hmessage.id] + target_bot_hmessage.related_ids
                if config.discord['history_reactions'].get('enabled', True):
                    await bg_task_queue.put(apply_reactions_to_messages(self.ictx, target_bot_hmessage, target_bot_hmessage_ids, self.target_discord_msg))

            # Update reactions for user message
            if config.discord['history_reactions'].get('enabled', True):
                await bg_task_queue.put(apply_reactions_to_messages(self.ictx, self.user_hmessage))

            await self.embeds.delete('regenerate')

        except TaskCensored:
            await self.embeds.delete('regenerate')
            raise
        except Exception as e:
            e_msg = 'An error occurred while processing "Regenerate"'
            log.error(f'{e_msg}: {e}')
            await self.ictx.followup.send(f'{e_msg} \n> {e}', silent=True)
            await self.embeds.delete('regenerate')
            raise

    #################################################################
    ################# HIDE OR REVEAL HISTORY TASK ###################
    #################################################################
    async def hide_or_reveal_history_task(self:"Task"):
        # Attributes set in Task
        self.local_history:History
        self.target_discord_msg:discord.Message # set as kwargs
        self.target_hmessage:HMessage # set as kwargs
        try:
            user_hmessage, bot_hmessage = self.local_history.get_history_pair_from_msg_id(self.target_discord_msg.id)
            user_hmessage:HMessage|None
            bot_hmessage:HMessage|None

            # determine outcome
            if user_hmessage is None or bot_hmessage is None:
                undeletable_message = await self.ictx.followup.send("A valid message pair could not be found for the target message.", ephemeral=True)
                await bg_task_queue.put(sleep_delete_message(undeletable_message))
                return
            verb = 'hidden' if not self.target_hmessage.hidden else 'revealed'

            # Helper function to toggle two messages (bot/user or bot/bot)
            def toggle_hmessages(bot_hmessage:HMessage, other_hmessage:HMessage, verb:str, stagger:bool=False):
                if verb == 'hidden':
                    bot_hmessage.update(hidden=True)
                    other_hmessage.update(hidden=True if not stagger else False)
                else:
                    bot_hmessage.update(hidden=False)
                    other_hmessage.update(hidden=False if not stagger else True)

            # List of HMessages to update 'associated' discord msg IDs for (only applies to bot HMessages)
            bot_hmsgs_to_react = [self.target_hmessage] if client.user == self.target_discord_msg.author else []

            # Get all bot regenerations
            all_bot_regens = user_hmessage.regenerations

            # If 0-1 regenerations (1 should not be possible...), toggle with user message
            if len(all_bot_regens) <= 1:
                toggle_hmessages(bot_hmessage, user_hmessage, verb)
                bot_hmsgs_to_react = [bot_hmessage]

            # If multiple regenerations, get next regen and toggle with it
            elif len(all_bot_regens) > 1:
                next_reply = None
                # If command was used on user message
                if self.user == self.target_discord_msg.author:
                    if verb == 'hidden':
                        for i in range(len(all_bot_regens)):
                            if not all_bot_regens[i].hidden: # get revealed bot HMessage
                                next_reply = all_bot_regens[i]
                                break
                    elif verb == 'revealed':
                        next_reply = all_bot_regens[-1] # get last regen

                # If command was used on a bot reply
                elif client.user == self.target_discord_msg.author:
                    # if user message is hidden, then do not toggle any other bot HMessages
                    if user_hmessage.hidden:
                        bot_hmsgs_to_react = [self.target_hmessage]
                    # if user message is revealed, determine next bot HMessage to toggle
                    else:
                        if verb == 'hidden':
                            for i in range(len(all_bot_regens)):
                                if all_bot_regens[i] == bot_hmessage:
                                    # There's a more recent reply
                                    if i + 1 < len(all_bot_regens):
                                        next_reply = all_bot_regens[i + 1]
                                    # Selected reply is the last one
                                    else:
                                        next_reply = all_bot_regens[i - 1]
                                    break
                        elif verb == 'revealed':
                            for i in range(len(all_bot_regens)):
                                if all_bot_regens[i] == bot_hmessage:
                                    for j in range(i + 1, len(all_bot_regens)):
                                        if not all_bot_regens[j].hidden:
                                            next_reply = all_bot_regens[j]
                                            break
                                    if not next_reply and i > 0:
                                        for j in range(i - 1, -1, -1):
                                            if not all_bot_regens[j].hidden:
                                                next_reply = all_bot_regens[j]
                                                break
                                    break
                else:
                    return # invalid user

                if next_reply:
                    # If cmd was used on a bot reply, toggle next bot HMessage (opposite value than targeted bot msg)
                    if client.user == self.target_discord_msg.author:
                        toggle_hmessages(bot_hmessage, next_reply, verb, stagger=True)
                    # If cmd was used on a user reply, toggle appropriate bot reply with it
                    else:
                        toggle_hmessages(user_hmessage, next_reply, verb)
                    # Add HMessages to update
                    bot_hmsgs_to_react.append(next_reply)
                # if user message is hidden and being revealed - this will toggle most recent bot msg by default
                else:
                    toggle_hmessages(bot_hmessage, user_hmessage, verb)

            # Apply reaction to user message
            if config.discord['history_reactions'].get('enabled', True):
                await bg_task_queue.put(apply_reactions_to_messages(self.ictx, user_hmessage))

            # Process all messages that need label updates
            for target_hmsg in bot_hmsgs_to_react:
                msg_ids_to_edit = []
                msg_ids_to_edit.append(target_hmsg.id)
                if target_hmsg.related_ids:
                    msg_ids_to_edit.extend(target_hmsg.related_ids)
                # Process reactions for all affected messages
                if config.discord['history_reactions'].get('enabled', True):
                    await bg_task_queue.put(apply_reactions_to_messages(self.ictx, target_hmsg, msg_ids_to_edit, self.target_discord_msg))
            
            result = f"**Modified message exchange pair in history for {self.user_name}** (messages {verb})."
            log.info(result)
            undeletable_message = await self.channel.send(result)
            await bg_task_queue.put(sleep_delete_message(undeletable_message))

        except Exception as e:
            log.error(f'An error occured while toggling "hidden" attribute for "Hide History": {e}')

    #################################################################
    ###################### EDIT HISTORY TASK ########################
    #################################################################
    async def edit_history_task(self:"Task"):
        # Attributes set in Task as kwargs
        self.target_discord_msg:discord.Message
        self.matched_hmessage:HMessage
        # Send modal which handles all further processing
        modal = EditMessageModal(self.ictx, matched_hmessage = self.matched_hmessage, target_discord_msg = self.target_discord_msg, params=self.params)
        await self.ictx.response.send_modal(modal)

    #################################################################
    ####################### TOGGLE TTS TASK #########################
    #################################################################
    async def toggle_tts_task(self:"Task"):
        try:
            message = 'toggled'
            api_tts_on, tgwui_tts_on = tts_is_enabled(for_mode='both')
            if not api_tts_on and not tgwui_tts_on:
                log.warning('Tried to toggle TTS but no client is available to toggle.')
                return
            toggle = 'api' if api_tts_on else 'tgwui'
            message = await toggle_any_tts(self.settings, toggle)
            vc_guild_ids = [self.ictx.guild.id] if config.is_per_server() else [guild.id for guild in client.guilds]
            for vc_guild_id in vc_guild_ids:
                await voice_clients.toggle_voice_client(vc_guild_id, message)
            if self.embeds.enabled('change'):
                # Send change embed to interaction channel
                await self.embeds.send('change', f"{self.user_name} {message} TTS.", 'Note: Does not load/unload the TTS model.', channel=self.channel)
                if bot_database.announce_channels:
                    # Send embeds to announcement channels
                    await bg_task_queue.put(announce_changes(f'{message} TTS', ' ', self.ictx))
            log.info(f"TTS was {message}.")
        except Exception as e:
            log.error(f'Error when toggling TTS to "{message}": {e}')

    #################################################################
    ########################## SPEAK TASK ###########################
    #################################################################
    async def speak_task(self:"Task"):
        try:
            api_tts_on, tgwui_tts_on = tts_is_enabled(and_online=True, for_mode='both')
            if tgwui_tts_on and tgwui_shared_module.model_name == 'None':
                await self.channel.send('Cannot process "/speak" request: No LLM model is currently loaded. Use "/llmmodel" to load a model.)', delete_after=5)
                log.warning(f'Bot tried to generate tts for {self.user_name}, but no LLM model was loaded')
                return
            await self.embeds.send('system', f'{self.user_name} requested tts ... ', '')
            
            await self.init_llm_payload()
            self.payload['state']['history'] = {'internal': [[self.text, self.text]], 'visible': [[self.text, self.text]]}
            self.params.save_to_history = False
            tts_args = self.params.tts_args
            if tgwui_tts_on:
                await tgwui.update_extensions(tts_args)

            # Check to apply Server Mode
            self.apply_server_mode()
            # Get history for interaction channel
            await self.create_user_hmessage()

            if api_tts_on:
                if not api.ttsgen.post_generate:
                    raise RuntimeError(f"No 'post_generate' endpoint available for TTS Client {api.ttsgen.name}")
                ep = api.ttsgen.post_generate
                tts_payload:dict = ep.get_payload()
                tts_payload.update(tts_args) # update with selected voice and lang
                if ep.text_input_key:
                    tts_payload[ep.text_input_key] = self.text # update with input text
                audio_file = await api.ttsgen.post_generate.call(input_data=tts_payload, main=True)
                if audio_file:
                    self.tts_resps.append(audio_file)
            elif tgwui_tts_on:
                loop = asyncio.get_event_loop()
                vis_resp_chunk:str = await loop.run_in_executor(None, tgwui_extensions_module.apply_extensions, 'output', self.text, self.payload['state'], True)
                audio_format_match = patterns.audio_src.search(vis_resp_chunk)
                if audio_format_match:
                    self.tts_resps.append(audio_format_match.group(1))

            # Process responses
            await self.create_bot_hmessage()
            await self.embeds.delete('system') # delete embed
            if not self.bot_hmessage:
                return
            await voice_clients.process_audio_file(self.ictx, self.tts_resps[0], self.bot_hmessage)
            # remove api key (don't want to share this to the world!)
            for sub_dict in tts_args.values():
                if 'api_key' in sub_dict:
                    sub_dict.pop('api_key')
            await self.embeds.send('system', f'{self.user_name} requested tts:', f"**Params:** {tts_args}\n**Text:** {self.text}")
            if tgwui_tts_on:
                await tgwui.update_extensions(self.settings.llmcontext.extensions) # Restore character specific extension settings
            if self.params.user_voice:
                os.remove(self.params.user_voice)
        except Exception as e:
            log.error(f"An error occurred while generating tts for '/speak': {e}")
            traceback.print_exc()
            await self.embeds.edit_or_send('system', "An error occurred while generating tts for '/speak'", e)

    #################################################################
    #################### CHANGE IMG MODEL TASK ######################
    #################################################################
    async def change_imgmodel_task(self:"Task"):
        # delegate to ImgModel()
        await self.imgmodel_settings.change_imgmodel_task(task=self)

    #################################################################
    #################### CHANGE LLM MODEL TASK ######################
    #################################################################
    # Process selected LLM model
    async def change_llmmodel_task(self:"Task"):
        try:
            llmmodel_params = self.params.llmmodel
            llmmodel_name = llmmodel_params.get('llmmodel_name')
            mode = llmmodel_params.get('mode', 'change')
            verb = llmmodel_params.get('verb', 'Changing')
            # Load the new model if it is different from the current one
            if tgwui_shared_module.model_name != llmmodel_name:
                await self.embeds.send('change', f'{verb} LLM model ... ', f"{verb} to {llmmodel_name}")
                # Retain current model name to swap back to
                if mode == 'swap':
                    previous_llmmodel = tgwui_shared_module.model_name
                # If an LLM model is loaded, unload it
                if tgwui_shared_module.model_name != 'None':
                    unload_model_func = get_tgwui_functions('unload_model')
                    unload_model_func()
                try:
                    tgwui_shared_module.model_name = llmmodel_name   # set to new LLM model
                    if tgwui_shared_module.model_name != 'None':
                        bot_database.update_was_warned('no_llmmodel') # Reset warning message
                        loader = tgwui.get_llm_model_loader(llmmodel_name)    # Try getting loader from user-config.yaml to prevent errors
                        await tgwui.load_llm_model(loader)                    # Load an LLM model if specified
                except Exception as e:
                    await self.embeds.delete('change')
                    await self.embeds.send('change', "An error occurred while changing LLM Model. No LLM Model is loaded.", e)

                await self.embeds.delete('change') # delete embed after model changed
                if mode == 'swap':
                    self.params.llmmodel['llmmodel_name'] = previous_llmmodel
                    return
                if self.embeds.enabled('change'):
                    # Send change embed to interaction channel
                    title = f"{self.user_name} unloaded the LLM model" if llmmodel_name == 'None' else f"{self.user_name} changed LLM model:"
                    description = 'Use "/llmmodel" to load a new one' if llmmodel_name == 'None' else f'**{llmmodel_name}**'
                    await self.embeds.send('change', title, description)
                    # Send embeds to announcement channels
                    if bot_database.announce_channels:
                        await bg_task_queue.put(announce_changes('changed LLM model', llmmodel_name, self.ictx))
                log.info(f"LLM model changed to: {llmmodel_name}")
        except Exception as e:
            log.error(f"An error occurred while changing LLM Model from '/llmmodel': {e}")
            traceback.print_exc()
            await self.embeds.edit_or_send('change', "An error occurred while changing LLM model", e)

    #################################################################
    #################### CHANGE CHARACTER TASK ######################
    #################################################################
    async def change_char_task(self:"Task"):
        try:
            char_params = self.params.character
            char_name = char_params.get('char_name', {})
            verb = char_params.get('verb', 'Changing')
            mode = char_params.get('mode', 'change')
            await self.embeds.send('change', f'{verb} character ... ', f'{self.user_name} requested character {mode}: "{char_name}"')

            # Change character
            await change_character(char_name, self.ictx)
            # Set history
            if not bot_history.autoload_history or bot_history.change_char_history_method == 'new': # if we don't keep history...
                # create a clone of History with same settings but empty, and replace it in the manager
                history_char, history_mode = get_char_mode_for_history(settings=self.settings)
                history = bot_history.get_history_for(self.ictx.channel.id, history_char, history_mode, cached_only=True)
                if history is None:
                    history_char, history_mode = get_char_mode_for_history(self.ictx)
                    bot_history.new_history_for(self.ictx.channel.id, history_char, history_mode)
                else:
                    history.fresh().replace()
            log.info(f"Character loaded: {char_name}")
            await self.send_char_greeting(char_name)

            # Announce change
            if self.embeds.enabled('change'):
                await self.embeds.delete('change')
                change_message = 'reset the conversation' if mode == 'reset' else 'changed character'
                # Send change embed to interaction channel
                await self.embeds.send('change', f'**{char_name}**', f"{self.user_name} {change_message}.")
                # Send embeds to announcement channels
                if bot_database.announce_channels and not is_direct_message(self.ictx):
                    await bg_task_queue.put(announce_changes(change_message, char_name, self.ictx))

            # Post settings
            if config.discord['post_active_settings'].get('enabled', True):
                settings_keys = ['character']
                if config.is_per_character():
                    await bg_task_queue.put(post_active_settings(self.ictx.guild, settings_keys))
                else:
                    await bg_task_queue.put(post_active_settings_to_all(settings_keys))

            # Reset Spontaneous Messaging tasks
            if config.is_per_character():
                await spontaneous_messaging.reset_for_server(self.ictx)
            else:
                await spontaneous_messaging.reset_all()

        except Exception as e:
            log.error(f'An error occurred while loading character for "{self.name}": {e}')
            await self.embeds.edit_or_send('change', "An error occurred while loading character", e)

    async def reset_task(self:"Task"):
        if tgwui_enabled:
            tgwui_shared_module.stop_everything = True
        # Create a new instance of the history and set it to active
        history_char, history_mode = get_char_mode_for_history(self.ictx)
        bot_history.get_history_for(self.channel.id, history_char, history_mode).fresh().replace()
        await self.change_char_task()

    #################################################################
    ######################## IMAGE GEN TASK #########################
    #################################################################
    async def img_gen_task(self:"Task"):
        try:
            if not self.tags:
                self.tags = Tags(self.ictx)
                await self.tags.match_img_tags(self.prompt, self.settings.get_vars())
                await self.apply_generic_tag_matches(phase='img')
                self.params.update_bot_should_do(self.tags)
            # Initialize imggen payload
            self.init_img_payload()
            # collect matched tag values
            img_payload_mods = await self.collect_img_tag_values()
            # Apply tags relevant to Img gen
            await self.process_img_payload_tags(img_payload_mods)
            # Process loractl
            if config.loractl_enabled():
                self.apply_loractl()
            # Apply tags relevant to Img prompts
            self.process_img_prompt_tags()

            # Update vars
            self.update_vars()
            # Apply menu selections from /image command
            self.imgmodel_settings.apply_imgcmd_params(task=self)
            # Apply updated prompt/negative prompt to payload
            self.imgmodel_settings.apply_final_prompts_for_task(self)
            # Apply vars overrides
            self.override_payload()
            # Clean anything up that gets messy
            self.imgmodel_settings.clean_payload(self.payload)

            # Change imgmodel if triggered by tags
            swap_imgmodel_params = None
            if self.params.imgmodel:
                if getattr(self.params.imgmodel, 'mode', 'change') == 'swap':
                    # Prepare params for the secondary model change
                    current_imgmodel = self.imgmodel_settings.last_imgmodel_value
                    swap_imgmodel_params = await self.imgmodel_settings.get_model_params(imgmodel=current_imgmodel, mode='swap_back', verb='Swapping back to')
                # RUN A CHANGE IMGMODEL SUBTASK
                new_options = await self.run_subtask('change_imgmodel')
                self.payload = deep_merge(self.payload, new_options)

            # Generate images
            await self.process_image_gen()

            # Send images, user's prompt, and any params from "/image" cmd
            imgcmd_task = getattr(self, 'imgcmd_task', False)
            if imgcmd_task or (self.params.should_send_text and not self.params.should_gen_text):
                imgcmd_message = self.params.imgcmd.get('message')
                original_prompt = f'-# {self.text}'[:2000]
                if imgcmd_message:
                    await self.embeds.send('img_send', f"{self.user_name} requested an image:", '', footer=imgcmd_message, nonembed_text=original_prompt)
                else:
                    await self.channel.send(original_prompt)

            # If switching back to original Img model
            if swap_imgmodel_params:
                # RUN A CHANGE IMGMODEL SUBTASK
                self.params.imgmodel = swap_imgmodel_params
                await self.run_subtask('change_imgmodel')
        except TaskCensored:
            raise
        except Exception as e:
            log.error(f"An error occurred in img_gen_task(): {e}")
            traceback.print_exc()

    # Task created from /image command
    async def image_cmd_task(self:"Task"):
        self.tags = None
        await self.img_gen_task()

#################################################################
################# ISTYPING (INSTANCE IN TASK) ###################
#################################################################
class IsTyping:
    def __init__(self, channel: discord.TextChannel):
        self.channel = channel
        self.typing_event = asyncio.Event()
        self.scheduled_typing_task: Optional[asyncio.Task] = None
        self.typing_task = asyncio.create_task(self.typing())

    async def typing(self):
        while True:
            await self.typing_event.wait()
            async with self.channel.typing():
                await asyncio.sleep(1)

    async def start_typing_after(self, start_time, end_time):
        try:
            if start_time is None:
                start_time = time.time()
            await asyncio.sleep(max(0, start_time - time.time()))
            self.typing_event.set()
            if end_time is not None:
                await asyncio.sleep(max(0, end_time - time.time()))
                self.typing_event.clear()
        except asyncio.CancelledError:
            log.debug("start_typing_after task was cancelled")
            self.typing_event.clear()
            self.scheduled_typing_task = None

    def start(self, start_time=None, end_time=None):
        if self.scheduled_typing_task is not None:
            self.scheduled_typing_task.cancel()
        if start_time is None and end_time is None:
            self.typing_event.set()
        else:
            self.scheduled_typing_task = asyncio.create_task(self.start_typing_after(start_time, end_time))

    def stop(self):
        if hasattr(self, 'scheduled_typing_task') and self.scheduled_typing_task is not None:
            self.scheduled_typing_task.cancel()
        if hasattr(self, 'typing_event'):
            self.typing_event.clear()

    async def close(self):
        self.stop()
        if self.typing_task:
            self.typing_task.cancel()
            # try:
            #     await self.typing_task
            # except asyncio.CancelledError:
            #     log.debug("Typing task cancelled.")

#################################################################
################## MESSAGE (INSTANCE IN TASK) ###################
#################################################################
class Message:
    def __init__(self, settings:"Settings", num:int, received_time:float, response_delay:float, read_text_delay:float=0.0, **kwargs):
        # Values set initially
        self.parent_settings = settings
        self.num = num
        self.received_time = received_time
        self.response_delay = response_delay
        self.read_text_delay = read_text_delay
        # Ensure delays do not exceed 'max_reply_delay'
        self.scale_delays()
        self.come_online_time = self.received_time + self.response_delay
        self.istyping_time = self.come_online_time + self.read_text_delay

        # Set while unqueuing
        self.unqueue_time = None

        # Updated after LLM Gen
        self.seconds_to_write = 0
        self.last_tokens = kwargs.pop('last_tokens', None)
        self.llm_gen_time = 0
        self.send_time = None

        self.num_chunks = 0
    
    def create_chunk_message(self, chunk_str):
        response_delay = self.response_delay if self.num_chunks == 0 else 0
        read_text_delay = self.read_text_delay if self.num_chunks == 0 else 0
        self.num_chunks += 1
        num = self.num + (self.num_chunks/1000)
        count_tokens_func = get_tgwui_functions('count_tokens')
        last_tokens = int(count_tokens_func(chunk_str))
        chunk_message = Message(self.parent_settings, num, self.received_time, response_delay, read_text_delay, last_tokens=last_tokens)
        return chunk_message

    def scale_delays(self):
        total_delay = self.response_delay + self.read_text_delay
        # Check if the total delay exceeds the maximum allowed delay
        if total_delay > self.parent_settings.behavior.max_reply_delay:
            # Calculate the scaling factor
            scaling_factor = self.parent_settings.behavior.max_reply_delay / total_delay
            # Scale the delays proportionally
            self.response_delay *= scaling_factor
            self.read_text_delay *= scaling_factor

    def factor_typing_speed(self):
        # Calculate and apply effect of "typing speed", if configured
        if self.parent_settings.behavior.maximum_typing_speed > 0 and self.last_tokens is not None:
            words_generated = self.last_tokens*0.75
            words_per_second = self.parent_settings.behavior.maximum_typing_speed / 60
            # update seconds_to_write
            self.seconds_to_write = (words_generated / words_per_second)

    # Offloads task to "parked queue" if not ready to send
    async def update_timing(self) -> float:
        current_time = time.time()

        # If bot currently online, ommit the response delay
        if bot_status.online:
            self.response_delay = 0

        # Update come online timing
        updated_come_online_time = self.received_time + self.response_delay
        if updated_come_online_time != self.come_online_time:
            await bot_status.schedule_come_online(updated_come_online_time)
        self.come_online_time = updated_come_online_time

        updated_istyping_time = self.received_time + self.response_delay + self.read_text_delay
        updated_send_time = updated_istyping_time + self.seconds_to_write

        # Determine whether to delay the responses (or update an existing 'send_time')
        if (updated_send_time > current_time) or (self.send_time is not None):
            # Only messages with delayed responses will have value for 'send_time
            self.send_time = updated_send_time

        #self.debug_timing(current_time, base_time, typing_offset, updated_delay, updated_istyping_time, updated_send_time, updated_online_time)

    # def debug_timing(self, current_time, base_time, typing_offset, updated_delay, updated_istyping_time, updated_send_time, updated_online_time):
    #     print("Original schedule:")
    #     print("Received time:", 0)
    #     print("Come Online time: Not Set")
    #     print("IsTyping time:", round(self.response_time - self.received_time))
    #     print("Received time:", round(self.received_time - current_time, 2))
    #     print("Come Online time:", round(self.response_time - current_time, 2))
    #     print("IsTyping time:", round(self.response_time - current_time, 2))
    #     print("Send Time: Not Set")
    #     print("")
    #     print("Unqueued time:", self.unqueue_time - self.received_time)

    #     print("Come Online time: Not Set")
    #     print("Updated schedule:")
    #     print("Unqueued time:", round(self.unqueue_time - current_time, 2))
    #     print("Current time:", 0)
    #     print("Updated delay:", round(updated_delay, 2))
    #     print("Updated come online time:", round(updated_online_time - current_time, 2))
    #     print("Typing offset:", typing_offset)
    #     print("Seconds to write:", self.seconds_to_write)
    #     print("Updated IsTyping time:", round(updated_istyping_time - current_time, 2))
    #     print("Updated send time:", round(updated_send_time - current_time, 2))


        return updated_istyping_time

#################################################################
############################# TASK ##############################
#################################################################
class Task(Tasks):
    def __init__(self, name:str, ictx:CtxInteraction|None=None, **kwargs): # text:str='', payload:dict|None=None, params:Params|None=None):
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        TaskManager.run() will use the Task() name to dynamically call a Tasks() method.

        Valid Task names:
        'on_message' / 'message_llm' / 'message_post_llm' / 'spontaneous_message' / 'flows'
        'edit_history' / 'continue' / 'regenerate' / 'hide_or_reveal_history' / 'toggle_tts'
        'change_imgmodel' / 'change_llmmodel' / 'change_char' / 'img_gen' / 'msg_image_cmd' / 'speak'

        Default kwargs:
        channel, user, user_name, embeds, text, prompt, payload, 
        params, tags, user_hmessage, bot_hmessage, local_history
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.name: str = name
        self.ictx: CtxInteraction = ictx
        self._semaphore:Optional[asyncio.Semaphore] = None
        self._semaphore_released: bool = False
        # TaskQueue() will initialize the Task's values before it is processed
        self.channel: discord.TextChannel = kwargs.pop('channel', None)
        self.user: Union[discord.User, discord.Member] = kwargs.pop('user', None)
        self.user_name: str          = kwargs.pop('user_name', None)
        self.embeds: Embeds          = kwargs.pop('embeds', None)
        self.text: str               = kwargs.pop('text', None)
        self.prompt: str             = kwargs.pop('prompt', None)
        self.neg_prompt: str         = kwargs.pop('neg_prompt', None)
        self.payload: dict           = kwargs.pop('payload', None)

        self.params: Params          = kwargs.pop('params', None)
        self.vars: BotVars           = kwargs.pop('vars', None)
        self.tags: Tags              = kwargs.pop('tags', None)
        self.llm_resp: str           = kwargs.pop('llm_resp', None)
        self.tts_resps: list         = kwargs.pop('tts_resps', None)
        self.user_hmessage: HMessage = kwargs.pop('user_hmessage', None)
        self.bot_hmessage: HMessage  = kwargs.pop('bot_hmessage', None)
        self.local_history           = kwargs.pop('local_history', None)
        self.settings: Settings      = kwargs.pop('settings', None)
        self.imgmodel_settings:ImgModel = kwargs.pop('imgmodel_settings', None)
        self.istyping: IsTyping      = kwargs.pop('istyping', None)
        self.message: Message        = kwargs.pop('message', None)
        self.extra_text: list        = kwargs.pop('extra_text', None)
        self.extra_audio: list       = kwargs.pop('extra_audio', None)
        self.extra_files: list       = kwargs.pop('extra_files', None)

        # Dynamically assign custom keyword arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    # Assigns defaults for attributes which are not already set
    def init_self_values(self):
        self.initialized = True # Flag that Task has been initialized (skip if using same Task object for subtasks, etc)
        self.channel: discord.TextChannel = self.ictx.channel if self.ictx else None
        self.user: Union[discord.User, discord.Member] = get_user_ctx_inter(self.ictx) if self.ictx else None
        self.user_name: str          = self.user.display_name if self.user else ""
        self.embeds: Embeds          = self.embeds if self.embeds else Embeds(self.ictx)
        # The original input text
        self.text: str               = self.text if self.text else ""
        # for updating prompt key in gen tasks
        self.prompt: str             = self.prompt if self.prompt else ''
        self.neg_prompt: str         = self.neg_prompt if self.neg_prompt else ''
        # payload for TGWUI / API call
        self.payload: dict           = self.payload if self.payload else {}
        # Misc parameters
        self.params: Params          = self.params if self.params else Params()
        self.vars: BotVars           = self.vars if self.vars else BotVars()
        self.tags: Tags              = self.tags if self.tags else Tags(self.ictx)
        # Bot response attributes
        self.llm_resp: str           = self.llm_resp if self.llm_resp else ''
        self.tts_resps: list         = self.tts_resps if self.tts_resps else []
        # History attributes
        self.user_hmessage: HMessage = self.user_hmessage if self.user_hmessage else None
        self.bot_hmessage: HMessage  = self.bot_hmessage if self.bot_hmessage else None
        # Get history for interaction channel
        non_history_tasks = ['change_char', 'change_imgmodel', 'change_llmmodel', 'toggle_tts'] # tasks that definitely do not need history
        # Establish correct Settings instance and Local History values
        self.settings: Settings      = self.settings if self.settings else None
        self.imgmodel_settings: ImgModel = self.imgmodel_settings if self.imgmodel_settings else get_imgmodel_settings(self.ictx)
        if self.ictx:
            is_dm = is_direct_message(self.ictx)
            if not self.settings and not is_dm:
                # Assign guild-specific settings
                self.settings = guild_settings.get(self.ictx.guild.id, bot_settings)
            if self.name not in non_history_tasks:
                history_char, history_mode = get_char_mode_for_history(settings=self.settings)
                if is_dm:
                    self.local_history = self.local_history if self.local_history else bot_history.get_history_for(self.ictx.channel.id, history_char, history_mode).dont_save()
                else:
                    self.local_history = self.local_history if self.local_history else bot_history.get_history_for(self.ictx.channel.id, history_char, history_mode)
        # Fall back to main bot settings
        if not self.settings:
            self.settings = bot_settings
        # Typing Task
        self.istyping:IsTyping  = self.istyping if self.istyping else None
        # Extra attributes/methods for regular message requests
        self.message:Message    = self.message if self.message else None
        self.extra_text:list    = self.extra_text if self.extra_text else []
        self.extra_audio:list   = self.extra_audio if self.extra_audio else []
        self.extra_files:list   = self.extra_files if self.extra_files else []

    def print_name(self) -> str:
        return f'{self.name.replace("_", " ").title()} Task'

    def get_channel_id(self, default=None) -> int|None:
        channel = getattr(self.ictx, 'channel', None)
        return getattr(channel, 'id', default)

    def override_payload(self, payload:dict|None = None):
        # Use self.payload if none is passed
        is_self = payload is None
        if is_self:
            payload = self.payload

        if "__overrides__" not in payload:
            return payload if not is_self else None

        # Extract and remove overrides from the payload
        overrides = payload.pop("__overrides__")
        # Update the default overrides with current vars
        updated_overrides = update_dict(overrides, vars(self.vars), in_place=False, skip_none=True)
        # Replace placeholders like {prompt} with overrides["prompt"]
        resolved = resolve_placeholders(payload, updated_overrides, log_prefix=f'[{self.print_name()}]', log_suffix='into payload')
        if is_self:
            self.payload = resolved
            return None
        return resolved

    def update_vars_from_imgcmd(self):
        imgcmd_params = self.params.imgcmd
        if imgcmd_params.get('size'):
            self.vars.width = imgcmd_params['size'].get('width')
            self.vars.height = imgcmd_params['size'].get('height')
        if imgcmd_params.get('img2img'):
            self.vars.i2i_image = imgcmd_params['img2img'].get('image')
            self.vars.i2i_mask = imgcmd_params['img2img'].get('mask')
            self.vars.denoising_strength = imgcmd_params['img2img'].get('denoising_strength')
        if imgcmd_params.get('cnet_dict'):
            cnet_dict = imgcmd_params['cnet_dict']
            for key, value in cnet_dict.items():
                attr_name = f'cnet_{key}'
                if hasattr(self, attr_name):
                    setattr(self, attr_name, value)
        self.vars.face_image = imgcmd_params['face_swap']

    def update_vars(self):
        self.vars.update(self.ictx)
        self.vars.prompt = self.prompt
        self.vars.neg_prompt = self.neg_prompt

    def get_extra_content(self, keep=False) -> dict:
        extra_content = {}
        for key in ('text', 'audio', 'files'):
            value = getattr(self, f'extra_{key}', None)
            if value:
                extra_content[key] = value
                if not keep:
                    setattr(self, f'extra_{key}', None)
        return extra_content
    
    async def send_extra_content_if_any(self):
        extra_content = self.get_extra_content()
        if extra_content:
            await bg_task_queue.put(send_content_to_discord(ictx=self.ictx, vc=voice_clients, **extra_content))

    def release_semaphore(self):
        if self._semaphore and not self._semaphore_released:
            self._semaphore_released = True
            self._semaphore.release()

    def init_typing(self, start_time=None, end_time=None):
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        Creates a IsTyping() instance, and begins typing in channel.
        Automatically factors any response_delay from character behavior
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        if not self.channel:
            log.warning(f"Tried initializing IsTyping() in Task() {self.name}, but the task has no channel.")
            return
        if not self.istyping:
            self.istyping = IsTyping(self.channel)
            # Self schedule typing for message requests
            if self.message is not None and self.message.istyping_time:
                start_time = self.message.istyping_time
            self.istyping.start(start_time=start_time, end_time=end_time)

    async def stop_typing(self):
        if self.istyping:
            await self.istyping.close()
            self.istyping = None

    def clone(self, name:str='', ictx:CtxInteraction|None=None, ignore_list:list|None=None, init_now:Optional[bool]=False, keep_typing:Optional[bool]=False) -> "Task":
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        Returns a new Task() with all of the same attributes.
        Can optionally clone a running 'istyping'
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ignore_list = ignore_list if ignore_list is not None else []
        always_ignore = ['name', 'ictx', 'message', 'embeds']
        ignore_list = ignore_list + always_ignore

        deepcopy_list = ['payload']

        current_attributes = {}
        for key, value in vars(self).items():
            if key in ignore_list:
                continue
            elif value is None or (key.startswith('__') and not callable(value)):
                continue
            elif key in deepcopy_list:
                current_attributes[key] = copy.deepcopy(value)
            elif isinstance(value, IsTyping):
                current_attributes[key] = self._clone_istyping(value) if keep_typing else IsTyping(self.channel)
            elif key == "postponed_tags":
                current_attributes[key] = value
                setattr(self, 'postponed_tags', None)
            # shallow copy remaining items
            else:
                current_attributes[key] = value
            
        new_task = Task(name=name, ictx=ictx, embeds=Embeds(ictx), **current_attributes)
        
        if init_now:
            new_task.init_self_values()

        return new_task

    def _clone_istyping(self, istyping:IsTyping):
        # Clone the IsTyping manually to ensure a new instance
        new_istyping = IsTyping(istyping.channel)
        # If the typing event is set, ensure the new instance starts typing immediately
        if istyping.typing_event.is_set():
            new_istyping.typing_event.set()
        return new_istyping
    

    async def run(self, task_name: str|None=None) -> Any:
        try:
            if not hasattr(self, 'initialized'):
                self.init_self_values()
            # Dynamically get the method and call it
            task_name = task_name if task_name is not None else self.name
            method_name = f'{task_name}_task'
            method = getattr(self, method_name, None)
            if method is not None and callable(method):
                return await method()
            else:
                logging.error(f"No such method: {method_name}")
        except TaskCensored:
            pass
        except Exception as e:
            logging.error(f"An error occurred while processing task {self.name}: {e}")
            traceback.print_exc()


    async def run_task(self) -> Any:
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        run_task() should only be called by the TaskManager()
        Runs a method of Tasks()
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        return await self.run()


    async def run_subtask(self, subtask:Optional[str]=None) -> Any:
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        run_subtask() should only be called from a method of Tasks() (a main task)
        Can be called from a new Task() instance.
        Can also be called from the current main Task() instance while providing subtask name.
        Runs a method of Tasks()
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        try:
            return await self.run(subtask)
        except TaskCensored:
            raise

#################################################################
######################## TASK MANAGEMENT ########################
#################################################################
class QueuedTask:
    def __init__(self, task: 'Task', queue_name: Literal['message_queue', 'history_queue', 'normal_queue', 'gen_queue'], priority: int = 0):
        self.task = task
        self.queue_name = queue_name
        self.priority = priority
        self.enqueue_time = time.time()

    def __lt__(self, other: 'QueuedTask'):
        return self.enqueue_time < other.enqueue_time


class TaskManager:
    def __init__(self, max_workers: int = 4):
        # Prevent concurrent message/history tasks in same channel
        self.locked_channels: set[int] = set()
        # Task source queues
        self.message_queue = asyncio.PriorityQueue()
        self.history_queue = asyncio.Queue()
        self.normal_queue = asyncio.Queue()
        self.gen_queue = asyncio.Queue()

        # Global scheduler (for fairness)
        self.scheduler_queue = asyncio.PriorityQueue()

        # Max total concurrency
        self.max_workers = max_workers
        self.active_workers = 0

        queue_config = config.task_queues

        # Per-queue concurrency limits
        self.queue_concurrency_limits = {
            'message_queue': queue_config.get('message_queue_concurrency', 1),
            'history_queue': queue_config.get('history_queue_concurrency', 1),
            'gen_queue': queue_config.get('gen_queue_concurrency', 1),
            'normal_queue': queue_config.get('normal_queue_concurrency', 3)
        }

        self.queue_semaphores = {
            queue_name: asyncio.Semaphore(limit)
            for queue_name, limit in self.queue_concurrency_limits.items()
        }

        # Queues that must not run concurrently
        self.MUTEX_GROUPS = [{'message_queue', 'history_queue'}]
        self.active_queues: set[str] = set()

        self.worker_tasks = []
        self._running = False

    def can_run(self, queue_name: str, channel_id: Optional[int] = None) -> bool:
        # Only enforce per-channel locking for mutually exclusive queues
        if queue_name in {"message_queue", "history_queue"} and channel_id is not None:
            if channel_id in self.locked_channels:
                return False
        return True

    async def queue_task(self, task: 'Task', queue_name: str, priority: int = 0):
        qt = QueuedTask(task, queue_name, priority)

        # Maintain existing behavior
        if queue_name == 'message_queue':
            await self.message_queue.put((priority, task))
        elif queue_name == 'history_queue':
            await self.history_queue.put(task)
        elif queue_name == 'normal_queue':
            await self.normal_queue.put(task)
        elif queue_name == 'gen_queue':
            await self.gen_queue.put(task)

        # Add to centralized scheduler
        await self.scheduler_queue.put((qt.enqueue_time, qt))

    async def start(self):
        if not self._running:
            self._running = True
            for _ in range(self.max_workers):
                worker = asyncio.create_task(self.process_task_worker())
                self.worker_tasks.append(worker)

    async def stop(self):
        self._running = False
        for task in self.worker_tasks:
            task.cancel()
        self.worker_tasks.clear()

    async def _run_with_semaphore(self, qt: QueuedTask):
        queue_name = qt.queue_name
        task:Task = qt.task
        channel_id = task.get_channel_id()

        # Lock this channel if applicable
        if queue_name in {"message_queue", "history_queue"} and channel_id is not None:
            self.locked_channels.add(channel_id)

        sem = self.queue_semaphores[queue_name]
        await sem.acquire()
        self.active_queues.add(queue_name)

        asyncio.create_task(self.run_and_cleanup(qt, sem, channel_id))

    async def process_task_worker(self):
        while self._running:
            try:
                # Step 1: Get the first task from the scheduler
                first_time, first_qt = await self.scheduler_queue.get()
                first_task: Task = first_qt.task
                first_queue = first_qt.queue_name
                first_sem = self.queue_semaphores[first_queue]
                first_channel_id = first_task.get_channel_id()

                can_run_first = self.can_run(first_queue, first_channel_id)
                try:
                    await asyncio.wait_for(first_sem.acquire(), timeout=0.01)
                    first_sem.release()
                    if can_run_first:
                        await self._run_with_semaphore(first_qt)
                        continue
                except asyncio.TimeoutError:
                    pass

                # Step 2: Look ahead to one more task
                try:
                    second_time, second_qt = await asyncio.wait_for(self.scheduler_queue.get(), timeout=0.1)
                    second_task: Task = second_qt.task
                    second_queue = second_qt.queue_name
                    second_sem = self.queue_semaphores[second_queue]
                    second_channel_id = second_task.get_channel_id()

                    can_run_second = self.can_run(second_queue, second_channel_id)
                    try:
                        await asyncio.wait_for(second_sem.acquire(), timeout=0.01)
                        second_sem.release()
                        if can_run_second:
                            await self.scheduler_queue.put((first_time, first_qt))
                            await self._run_with_semaphore(second_qt)
                            continue
                    except asyncio.TimeoutError:
                        pass

                    # Neither task can run; requeue both
                    await self.scheduler_queue.put((first_time, first_qt))
                    await self.scheduler_queue.put((second_time, second_qt))
                    await asyncio.sleep(0.25)

                except asyncio.TimeoutError:
                    # No second task; requeue the first
                    await self.scheduler_queue.put((first_time, first_qt))
                    await asyncio.sleep(0.25)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Worker error: {e}")
                traceback.print_exc()

    async def run_and_cleanup(self, qt: QueuedTask, sem: asyncio.Semaphore, channel_id: Optional[int] = None):
        task = qt.task
        task._semaphore = sem
        queue_name = qt.queue_name
        sem_released = False

        try:
            task.init_self_values()
            await bot_status.come_online()

            if 'message' in task.name or task.name in ['flows', 'regenerate', 'msg_image_cmd', 'speak', 'continue']:
                task.init_typing()

            log.info(f"Running task '{task.name}' from queue '{queue_name}'")
            await self.run_task(task)
            sem_released = getattr(task, '_semaphore_released', False)

            if task.message is not None and getattr(task.message, 'send_time', None):
                await message_manager.queue_delayed_message(task)
            else:
                await task.stop_typing()
                await task.send_extra_content_if_any()
                del task
                await bot_status.schedule_go_idle()

        except TaskCensored:
            pass
        except APIRequestCancelled as e:
            if e.cancel_event:
                e.cancel_event.clear()
            log.info(e)
            await task.embeds.edit_or_send('img_gen', str(e), " ")
        except Exception as e:
            logging.error(f"Error running task {task.name}: {e}")
            traceback.print_exc()
        finally:
            if not sem_released:
                self.active_queues.discard(queue_name)
                sem.release()
                if channel_id is not None:
                    self.locked_channels.discard(channel_id)
            task_event.clear()

    async def run_task(self, task: 'Task'):
        await task.run_task()
        if getattr(task, 'postponed_tags', None):
            await task.process_postponed_tags()
        if flows_queue.qsize() > 0:
            await flows.run_flow_if_any(task.text, task.ictx)

task_manager = TaskManager(max_workers=config.task_queues.get('maximum_concurrency', 3))

#################################################################
########################## QUEUED FLOW ##########################
#################################################################
class Flows(TaskProcessing):
    def __init__(self, ictx:CtxInteraction|None=None):
        self.ictx: CtxInteraction = ictx
        self.user_name: Optional[str] = get_user_ctx_inter(ictx).display_name if ictx else None
        self.channel: Optional[discord.TextChannel] = ictx.channel if ictx else None
        self.local_history = None

    async def build_queue(self, input_flow):
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
                if not step: # ignore possible empty dict
                    continue
                flow_step = copy.copy(flow_base) # use flow base
                flow_step.update(step)           # update with flow step tags
                # duplicate flow step depending on 'flow_loops'
                counter = 1
                flow_step_loops = flow_step.pop('flow_step_loops', 0)
                counter += (flow_step_loops - 1) if flow_step_loops else 0
                total_flows += counter
                while counter > 0:
                    counter -= 1
                    await flows_queue.put(flow_step)
            flows_event.set() # flag that a flow is being processed. Check with 'if flow_event.is_set():'
        except Exception as e:
            log.error(f"Error building Flow: {e}")
            traceback.print_exc()

    async def format_next_flow(self, next_flow:dict, text:str) -> tuple[str, str]:
        flow_name = ''
        formatted_flow_tags = {}
        for key, value in next_flow.items():
            # get name for message embed
            if key == 'flow_step':
                flow_name = f": {value}"
            # format prompt before feeding it back into message_task()
            elif key == 'format_prompt':
                text = self.process_prompt_formatting(text, format_prompt=[value])
            # see if any tag values have dynamic formatting (user prompt, LLM reply, etc)
            elif isinstance(value, str):
                formatted_value = self.format_prompt_with_recent_output(value) # output will be a string
                # if the value changed...
                if formatted_value != value:         
                    formatted_value = valueparser.parse_value(formatted_value) # convert new string to correct value type
                formatted_flow_tags[key] = formatted_value
            # apply wildcards
            text = await dynamic_prompting(text)
        next_flow.update(formatted_flow_tags) # commit updates
        return flow_name, text

    # function to get a copy of the next queue item while maintaining the original queue
    async def peek_flow_queue(self, text:str):
        temp_queue = asyncio.Queue()
        total_queue_size = flows_queue.qsize()
        while flows_queue.qsize() > 0:
            if flows_queue.qsize() == total_queue_size:
                item = await flows_queue.get()
                flow_name, formatted_text = await self.format_next_flow(item, text)
            else:
                item = await flows_queue.get()
            await temp_queue.put(item)
        # Enqueue the items back into the original queue
        while temp_queue.qsize() > 0:
            item_to_put_back = await temp_queue.get()
            await flows_queue.put(item_to_put_back)
        return flow_name, formatted_text

    async def flow_task(self, text:str):
        try:
            total_flow_steps = flows_queue.qsize()
            descript = ''
            await bot_embeds.send('flow', f'Processing Flow for {self.user_name} with {total_flow_steps} steps', descript, channel=self.channel)

            while flows_queue.qsize() > 0:
                # flow_queue items are removed in init_tags() while running the subtask
                flow_name, text = await self.peek_flow_queue(text)
                remaining_flow_steps = flows_queue.qsize()

                descript = descript.replace("**Processing", ":white_check_mark: **")
                descript += f'**Processing Step {total_flow_steps + 1 - remaining_flow_steps}/{total_flow_steps}**{flow_name}\n'
                await bot_embeds.edit(name='flow', description=descript)

                # CREATE A SUBTASK AND RUN IT
                flows_task: Task = Task('flows', self.ictx, text=text)
                await flows_task.run_subtask('flows')
                await flows_task.stop_typing() # ensure typing tasks stopped
                del flows_task # delete finished flows task

            descript = descript.replace("**Processing", ":white_check_mark: **")
            await bot_embeds.edit('flow', f"Flow completed for {self.user_name}", descript)
        except TaskCensored:
            raise
        except Exception as e:
            log.error(f"An error occurred while processing a Flow: {e}")
            await bot_embeds.edit_or_send('flow', "An error occurred while processing a Flow", e, channel=self.channel)

        flows_event.clear()              # flag that flow is no longer processing
        flows_queue.task_done()          # flow queue task is complete

    async def run_flow_if_any(self, text:str, ictx:CtxInteraction):
        if flows_queue.qsize() > 0:
            self.ictx      = ictx
            self.user_name = get_user_ctx_inter(ictx).display_name
            self.channel   = ictx.channel
            history_char, history_mode = get_char_mode_for_history(self.ictx)
            if is_direct_message(ictx):
                self.local_history = self.local_history if self.local_history else bot_history.get_history_for(self.ictx.channel.id, history_char, history_mode).dont_save()
            else:
                self.local_history = self.local_history if self.local_history else bot_history.get_history_for(self.ictx.channel.id, history_char, history_mode)

            # flows are activated in process_llm_payload_tags(), and is where the flow queue is populated
            await self.flow_task(text)

flows = Flows()

#################################################################
################ IMGGEN SERVER RESTART COMMAND ##################
#################################################################
if imggen_enabled and api.imggen.post_server_restart:
    # Function to attempt restarting the SD WebUI Client in the event it gets stuck
    @client.hybrid_command(description=f"Immediately restarts the main media generation server.")
    @guild_or_owner_only()
    async def restart_sd_client(ctx: commands.Context):
        if api.imggen.is_sdwebui():
            await ctx.send(f"**`/restart_sd_client` __will not work__ unless {api.imggen.name} was launched with flag: `--api-server-stop`**", delete_after=10)
        await api.imggen.post_server_restart.call(retry=0)
        title = f"{ctx.author.display_name} used '/restart_sd_client'. Restarting {api.imggen.name} ..."
        if api.imggen.get_progress:
            await bot_embeds.send('system', title=title, description='Attempting to re-establish connection in 5 seconds (Attempt 1 of 10)', channel=ctx.channel)
            log.info(title)
            response = None
            retry = 1
            while response is None and retry < 11:
                await bot_embeds.edit('system', description=f'Attempting to re-establish connection in 5 seconds (Attempt {retry} of 10)')
                await asyncio.sleep(5)
                response = await api.imggen.get_progress.call(retry=0)
                retry += 1
            if response:
                title = f"{api.imggen.name} restarted successfully."
                await bot_embeds.edit('system', title=title, description=f"Connection re-established after {retry} out of 10 attempts.")
                log.info(title)
            else:
                title = f"{api.imggen.name} server unresponsive after Restarting."
                await bot_embeds.edit('system', title=title, description="Connection was not re-established after 10 attempts.")
                log.error(title)

#################################################################
######################## /IMAGE COMMAND #########################
#################################################################
if imggen_enabled:
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
            size_choices = [app_commands.Choice(name=option['name'], value=option['name'])
                            for option in size_options]
            style_choices = [app_commands.Choice(name=option['name'], value=option['name'])
                             for option in style_options]
            if tgwui_enabled:
                use_llm_choices = [app_commands.Choice(name="No, just img gen from my prompt", value="No"),
                                app_commands.Choice(name="Yes, send my prompt to the LLM", value="Yes"),
                                app_commands.Choice(name="Yes, auto-prefixed: 'Provide a detailed image prompt description for: '", value="YesWithPrefix")]
            else:
                use_llm_choices = [app_commands.Choice(name="**disabled** (LLM not available)", value="disabled")]
            return size_choices, style_choices, use_llm_choices

        except Exception as e:
            log.error(f"An error occurred while building choices for /image: {e}")

    async def get_imgcmd_options():
        try:
            options = load_file(shared_path.cmd_options, {})
            options = dict(options)
            # Get sizes and aspect ratios from 'dict_cmdoptions.yaml'
            sizes = options.get('sizes', {})
            aspect_ratios = [size.get("ratio") for size in sizes.get('ratios', [])]
            # Calculate the average and aspect ratio sizes
            current_avg = bot_database.last_imgmodel_res
            ratio_options = calculate_aspect_ratio_sizes(current_avg, aspect_ratios)
            # Collect any defined static sizes
            static_options = sizes.get('static_sizes', [])
            # Merge dynamic and static sizes
            size_options = (ratio_options or []) + (static_options or [])
            # Get style and controlnet options
            style_options = options.get('styles', {})
            return size_options, style_options

        except Exception as e:
            log.error(f"An error occurred while building options for /image: {e}")
            return None, None

    async def get_cnet_data() -> dict:
        filtered_cnet_data = {}

        async def check_cnet_online():
            try:
                online = await api.imggen.get_controlnet_models.call(retry=0)
                if online:
                    return True
                else: 
                    return False
            except Exception:
                log.warning(f"ControlNet is enabled in config.yaml, but was not responsive from {api.imggen.name} API.")
            return False

        if config.controlnet_enabled():
            try:
                all_control_types:dict = await api.imggen.get_controlnet_control_types.call(retry=0, main=True)
                for key, value in all_control_types.items():
                    if key == "All":
                        continue
                    if key in ["Reference", "Revision", "Shuffle"]:
                        value['name'] = key
                        filtered_cnet_data[key] = value
                    elif value["default_model"] != "None":
                        value['name'] = key
                        filtered_cnet_data[key] = value
            except Exception:
                cnet_online = await check_cnet_online()
                if cnet_online:
                    log.warning("ControlNet is both enabled in config.yaml and detected. However, the '/image' command relies on the '/controlnet/control_types' \
                        API endpoint which is missing. See here: (https://github.com/altoiddealer/ad_discordbot/wiki/troubleshooting).")
        return filtered_cnet_data

    # Get size and style options for /image command
    size_options, style_options = asyncio.run(get_imgcmd_options())
    size_choices, style_choices, use_llm_choices = asyncio.run(get_imgcmd_choices(size_options, style_options))

    # Check if ControlNet enabled in config
    cnet_status = 'Guides image diffusion using an input image or map.' if config.controlnet_enabled() else '**option disabled** (ControlNet not available)'
    # Check if ReActor enabled in config
    reactor_status = 'For best results, attach a square (1:1) cropped image of a face, to swap into the output.' if config.reactor_enabled() else '**option disabled** (ReActor not available)'

    use_llm_status = 'Whether to send your prompt to LLM. Results may vary!' if tgwui_enabled else '**option disabled** (LLM is not integrated)'

    @client.hybrid_command(name="image", description=f"Generate an image{f' using {api.imggen.name}' if api.imggen else ''}")
    @app_commands.describe(use_llm=use_llm_status)
    @app_commands.describe(style='Applies a positive/negative prompt preset')
    @app_commands.describe(img2img='Diffuses from an input image instead of pure latent noise.')
    @app_commands.describe(img2img_mask='Masks the diffusion strength for the img2img input. Requires img2img.')
    @app_commands.describe(face_swap=reactor_status)
    @app_commands.describe(controlnet=reactor_status)
    @app_commands.choices(use_llm=use_llm_choices)
    @app_commands.choices(size=size_choices)
    @app_commands.choices(style=style_choices)
    @configurable_for_dm_if(lambda ctx: 'image' in config.discord_dm_setting('allowed_commands', []))
    async def image(ctx: commands.Context, prompt: str, use_llm: typing.Optional[app_commands.Choice[str]], size: typing.Optional[app_commands.Choice[str]], style: typing.Optional[app_commands.Choice[str]], 
                    neg_prompt: typing.Optional[str], img2img: typing.Optional[discord.Attachment], img2img_mask: typing.Optional[discord.Attachment], 
                    face_swap: typing.Optional[discord.Attachment], controlnet: typing.Optional[discord.Attachment]):
        user_selections = {"prompt": prompt, "use_llm": use_llm.value if use_llm else None, "size": size.value if size else None, "style": style.value if style else {}, "neg_prompt": neg_prompt if neg_prompt else '',
                            "img2img": img2img if img2img else None, "img2img_mask": img2img_mask if img2img_mask else None,
                            "face_swap": face_swap if face_swap else None, "cnet": controlnet if controlnet else None}
        await process_image(ctx, user_selections)

    async def process_image(ctx: commands.Context, selections):
        # CREATE TASK - CHECK IF ONLINE
        image_cmd_task = Task('image_cmd', ctx)
        setattr(image_cmd_task, 'imgcmd_task', True)
        # Do not process if SD WebUI is offline
        if not await api_online(client_type='imggen', ictx=ctx):
            await ctx.reply("Stable Diffusion is not online.", ephemeral=True, delete_after=5)
            return
        # User inputs from /image command
        pos_prompt:str = selections['prompt']
        if pos_prompt.startswith('-# '):
            pos_prompt = pos_prompt[3:]
        use_llm = selections.get('use_llm', None) if tgwui_enabled else None
        size = selections.get('size', None)
        style = selections.get('style', {})
        neg_prompt = selections.get('neg_prompt', '')
        img2img = selections.get('img2img', None)
        img2img_mask = selections.get('img2img_mask', None)
        face_swap = selections.get('face_swap', None) if config.reactor_enabled() else None
        cnet = selections.get('cnet', None) if config.controlnet_enabled() else None

        imgmodel_settings:ImgModel = get_imgmodel_settings(ctx)

        # Defaults
        mode = 'txt2img'
        size_dict = {}
        faceswapimg = None
        img2img_dict = {}
        cnet_dict = {}
        try:
            pos_prompt = await dynamic_prompting(pos_prompt)
            log_msg = ""
            if use_llm and use_llm != 'No':
                log_msg += "\nUse LLM: True (image was generated from LLM reply)"
            if size:
                selected_size = next((option for option in size_options if option['name'] == size), None)
                if selected_size:
                    size_dict['width'] = selected_size.get('width')
                    size_dict['height'] = selected_size.get('height')
                log_msg += f"\nSize: {size}"
            if style:
                selected_style_option = next((option for option in style_options if option['name'] == style), None)
                if selected_style_option:
                    style = selected_style_option
                log_msg += f"\nStyle: {style.get('name', 'Unknown')}"
            if neg_prompt:
                log_msg += f"\nNegative Prompt: {neg_prompt}"
            if img2img:
                async def process_image_img2img(img2img, img2img_dict, mode, log_msg):
                    i2i_image = await imgmodel_settings.handle_image_input(img2img, file_type='image')
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
                    mode = 'img2img' # Change mode to img2img
                    log_msg += f"\nImg2Img with denoise strength: {denoising_strength}"
                    return img2img_dict, mode, log_msg
                try:
                    img2img_dict, mode, log_msg = await process_image_img2img(img2img, img2img_dict, mode, log_msg)
                except Exception as e:
                    log.error(f"An error occurred while configuring Img2Img for /image command: {e}")
            if img2img_mask:
                if img2img:
                    img2img_mask_img = await imgmodel_settings.handle_image_input(img2img_mask, file_type='image')
                    img2img_dict['mask'] = img2img_mask_img
                    log_msg += "\nInpainting Mask Provided"
                else:
                    await ctx.send("Inpainting requires im2img. Not applying img2img_mask mask...", ephemeral=True)
            if face_swap:
                faceswapimg = await imgmodel_settings.handle_image_input(face_swap, file_type='image')
                log_msg += "\nFace Swap Image Provided"
            if cnet:
                # Get filtered ControlNet data
                cnet_data = await get_cnet_data()
                async def process_image_controlnet(cnet, cnet_dict, log_msg):
                    try:
                        cnet_dict['image'] = await imgmodel_settings.handle_image_input(cnet, file_type='image')
                    except Exception as e:
                        log.error(f"Error decoding ControlNet input image for '/image' command: {e}")
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
                        log.error(f"An error occurred while setting ControlNet Control Type in '/image' command: {e}")
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
                            except Exception as e:
                                log.error(f"Error building ControlNet options for '/image' command: {e}")
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
                        log.error(f"An error occurred while configuring initial ControlNet options from '/image' command: {e}")
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
                        log.error(f"An error occurred while configuring secondary ControlNet options from /image command: {e}")
                    cnet_dict.update({'enabled': True, 'save_detected_map': True})
                    log_msg += f"\nControlNet: (Module: {cnet_dict['module']}, Model: {cnet_dict['model']})"
                    return cnet_dict, log_msg
                try:
                    cnet_dict, log_msg = await process_image_controlnet(cnet, cnet_dict, log_msg)
                except Exception as e:
                    log.error(f"An error occurred while configuring ControlNet for /image command: {e}")

            image_cmd_task.prompt = pos_prompt

            if use_llm and use_llm == 'YesWithPrefix':
                pos_prompt = 'Provide a detailed image prompt description (without any preamble or additional text) for: ' + pos_prompt
            image_cmd_task.text = pos_prompt

            # UPDATE TASK WITH PARAMS
            imgcmd_params = {}
            imgcmd_params['size']       = size_dict
            imgcmd_params['neg_prompt'] = neg_prompt
            imgcmd_params['style']      = style
            imgcmd_params['face_swap']  = faceswapimg
            imgcmd_params['controlnet'] = cnet_dict
            imgcmd_params['img2img']    = img2img_dict
            imgcmd_params['message']    = log_msg

            image_cmd_task.params = Params(imgcmd=imgcmd_params, mode=mode)

            await ireply(ctx, 'image') # send a response msg to the user

            # QUEUE TASK
            log.info(f'{ctx.author.display_name} used "/image": "{pos_prompt}"')
            if use_llm and use_llm != 'No':
                # Set more parameters
                params: Params = image_cmd_task.params
                params.should_gen_text = True
                params.should_gen_image = True
                params.skip_create_user_hmsg = True
                params.skip_create_bot_hmsg = True
                params.save_to_history = False
                # CHANGE NAME (will run 'message' then 'img_gen')
                image_cmd_task.name = 'msg_image_cmd'

            await task_manager.queue_task(image_cmd_task, 'gen_queue')

        except Exception as e:
            log.error(f"An error occurred in image(): {e}")
            traceback.print_exc()

#################################################################
#################### DYNAMIC USER COMMANDS ######################
#################################################################
async def load_custom_commands():
    registered_cmd_names = []

    # Map from string type to actual type annotation
    type_map = {"string": str,
                "user": discord.User,
                "int": int,
                "bool": bool,
                "float": float,
                "channel": discord.abc.GuildChannel,
                "role": discord.Role,
                "mentionable": Union[discord.User, discord.Role],
                "attachment": discord.Attachment}

    commands_file_data = load_file(shared_path.custom_commands)
    command_data:list = commands_file_data.get('commands', [])
    for cmd in command_data:
        try:
            name = cmd["command_name"]
            description = cmd.get("description", "No description")
            options = cmd.get("options", [])
            main_steps = cmd.get("steps", [])
            allow_dm = cmd.get("allow_dm", False)
            queue_to = cmd.get("queue_to") or "gen_queue"
            # Do not allow put to Message queue
            if queue_to == "message_queue":
                queue_to = "gen_queue"

            # Signature parameter list (start with interaction)
            parameters = [
                inspect.Parameter("interaction",
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                annotation=discord.Interaction)
            ]

            # Store choice metadata for later injection
            option_metadata = []

            for opt in options:
                opt_name = opt["name"]
                opt_type_str = opt["type"]
                opt_type = type_map[opt_type_str]
                required = opt.get("required", True)
                choices_raw = opt.get("choices")
                steps: Optional[list] = opt.get("steps")
                opt_description = opt.get("description")
                if opt_description and opt_type_str == "attachment":
                    opt_description += " (size limits apply)"

                if choices_raw:
                    # Case: dict {label: value}
                    if isinstance(choices_raw, dict):
                        annotation = app_commands.Choice[str]
                        choices = [app_commands.Choice(name=label, value=value)
                            for label, value in choices_raw.items()]

                    # Case: list (dicts or primitives)
                    elif isinstance(choices_raw, list):
                        if all(isinstance(c, dict) and "name" in c and "value" in c for c in choices_raw):
                            annotation = app_commands.Choice[str]
                            choices = [app_commands.Choice(name=c["name"], value=c["value"])
                                for c in choices_raw]
                        else:
                            value_type = type(choices_raw[0])
                            annotation = app_commands.Choice[value_type]
                            choices = [app_commands.Choice(name=str(c), value=c)
                                for c in choices_raw]
                else:
                    annotation = opt_type
                    choices = None

                default_value = opt.get("default", inspect.Parameter.empty if required else None)
                param = inspect.Parameter(name=opt_name,
                                        kind=inspect.Parameter.KEYWORD_ONLY,
                                        default=default_value,
                                        annotation=annotation)

                parameters.append(param)
                option_metadata.append({"name": opt_name,
                                        "description": opt_description,
                                        "choices": choices,
                                        "steps": steps})

            # Create function signature
            sig = inspect.Signature(parameters)

            def make_callback(command_name:str, option_metadata:dict, queue:str, main_steps:list):
                async def callback_template(*args, **kwargs):
                    interaction = kwargs.pop("interaction", args[0] if args else None)

                    await ireply(interaction, f'/{command_name}') # send a response msg to the user

                    # Convert Choices to their actual value (but not None)
                    raw_kwargs = {k: (v.value if isinstance(v, app_commands.Choice) else v)
                                  for k, v in kwargs.items()
                                  if v is not None}
                    # Create a Task and queue it
                    custom_command_task = Task('custom_command', interaction)
                    custom_cmd_config = {'custom_cmd_name': command_name,
                                         'custom_cmd_selections': raw_kwargs,
                                         'custom_cmd_option_meta': option_metadata,
                                         'custom_cmd_steps': main_steps}
                    setattr(custom_command_task, 'custom_cmd_config', custom_cmd_config)
                    await task_manager.queue_task(custom_command_task, queue_name=queue)
                    user_name = interaction.user.display_name if hasattr(interaction, "user") else interaction.author.display_name
                    log.info(f'{user_name} used user defined command "/{command_name}"')

                return callback_template

            callback_template = make_callback(name, option_metadata, queue_to, main_steps)
            dynamic_callback = types.FunctionType(
                callback_template.__code__,
                globals(),
                name,
                argdefs=(),
                closure=callback_template.__closure__,
            )
            dynamic_callback.__annotations__ = {p.name: p.annotation for p in parameters}
            dynamic_callback.__signature__ = sig

            checked_callback = custom_commands_check_dm(name, allow_dm)(dynamic_callback)

            command = app_commands.Command(name=name,
                                           description=description,
                                           callback=checked_callback)

            # Inject choices (private API workaround)
            for meta in option_metadata:
                param_name = meta["name"]
                if param_name in command._params:
                    param = command._params[param_name]
                    if meta.get("choices"):
                        param.choices = meta["choices"]
                    if hasattr(param, "description"):
                        param.description = meta["description"]
                    else:
                        setattr(param, "description", meta["description"])
                else:
                    print(f"⚠ Warning: Parameter '{param_name}' not found in command._params")

            client.tree.add_command(command)
            registered_cmd_names.append(name)
        except Exception as e:
            log.error(f'[Custom Commands] An error occured while initializing "/{name}": {e}')
    if registered_cmd_names:
        formatted_names = ', '.join(f"'{f'/{name}'}'" for name in registered_cmd_names)
        log.info(f"[Custom Commands] Registered: {formatted_names}")

async def setup_hook():
    try:
        await load_custom_commands()
    except Exception as e:
        log.error(f'[Custom Commands] An error occured while initializing: {e}')
asyncio.run(setup_hook())

#################################################################
######################### MISC COMMANDS #########################
#################################################################
async def is_direct_message_and_not_owner(ictx:CtxInteraction):
    user: Union[discord.User, discord.Member] = get_user_ctx_inter(ictx)
    owner = await client.is_owner(ictx.user)
    return is_direct_message(ictx) and not owner

@client.event
async def on_error(event_name, *args, **kwargs):
    print(traceback.format_exc())
    log.warning(f'Event error in {event_name}')

@client.event
async def on_command_error(ctx:commands.Context, error:Union[HybridCommandError, CommandError, AppCommandError, CommandInvokeError, DiscordException, Exception]):
    while isinstance(error, (HybridCommandError, CommandInvokeError)):
        error = error.original
        
    if isinstance(error, AlertUserError):
        await ctx.reply(embed=discord.Embed(description=str(error), color=discord.Color.gold()), ephemeral=True, delete_after=15)
        return

    # ignore certain errors in console
    if not isinstance(error, (
        discord.app_commands.errors.MissingPermissions,
        commands.errors.MissingPermissions,
    )):
        print(''.join(traceback.format_tb(error.__traceback__)))
        log.warning(f'Command /{ctx.command} failed: Error <{type(error).__name__}>')
        log.warning(error)
        
    await ctx.reply(embed=discord.Embed(description=f'Command `/{ctx.command}` failed\n```\n{error}\n```', color=discord.Color.red()), ephemeral=True, delete_after=15)


@client.listen('on_app_command_error')
async def on_app_command_error(inter:discord.Interaction, error:discord.app_commands.AppCommandError):
    log.debug(f'App command error: {error}')
    await inter.response.send_message(str(error), ephemeral=True, delete_after=15)


if bot_embeds.enabled('system'):
    @client.hybrid_command(description="Display help menu")
    async def helpmenu(ctx):
        await ctx.send(embed = bot_embeds.helpmenu())

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
        await bot_embeds.send('system', "Bot LLM Gen Statistics:", f">>> {formatted_description}", channel=ctx.channel)

@client.hybrid_command(description="Toggle current channel as an announcement channel for the bot (model changes)")
@app_commands.checks.has_permissions(manage_channels=True)
@guild_only()
async def announce(ctx: commands.Context, channel:Optional[discord.TextChannel]=None):
    channel = channel or ctx.channel # type: ignore
    if channel.id in bot_database.announce_channels:
        bot_database.announce_channels.remove(channel.id) # If the channel is already in the announce channels, remove it
        action_message = f'Removed {channel.mention} from announce channels. Use "/announce" again if you want to add it back.'
    else:
        # If the channel is not in the announce channels, add it
        bot_database.announce_channels.append(channel.id)
        action_message = f'Added {channel.mention} to announce channels. Use "/announce" again to remove it.'

    bot_database.save()
    await ctx.reply(action_message, delete_after=15)

@client.hybrid_command(description="Toggle current channel as main channel for bot to auto-reply without needing to be called")
@app_commands.checks.has_permissions(manage_channels=True)
@guild_only()
async def main(ctx: commands.Context, channel:Optional[discord.TextChannel]=None):
    channel = channel or ctx.channel # type: ignore
    
    if channel.id in bot_database.main_channels:
        bot_database.main_channels.remove(channel.id) # If the channel is already in the main channels, remove it
        action_message = f'Removed {channel.mention} from main channels. Use "/main" again if you want to add it back.'
    else:
        # If the channel is not in the main channels, add it
        bot_database.main_channels.append(channel.id)
        action_message = f'Added {channel.mention} to main channels. Use "/main" again to remove it.'

    bot_database.save()
    await ctx.reply(action_message, delete_after=15)


async def update_ttsgen(ictx:CtxInteraction, status:str='enabled', rebuild_cmd_opts=True):
    # Enforce only one TTS method enabled
    enforce_one_tts_method()
    # Process Voice Clients
    vc_guild_ids = [ictx.guild.id] if config.is_per_server() else [guild.id for guild in client.guilds]
    for vc_guild_id in vc_guild_ids:
        await voice_clients.toggle_voice_client(vc_guild_id, status)
    if rebuild_cmd_opts:
        # Build '/speak' options if TTS client was not online during intialization
        if not speak_cmd_options.voice_hash_dict:
            await speak_cmd_options.build_options()

async def announce_api_changes(ictx:CtxInteraction, api_name:str, status:str):
    message = f"{ictx.author.display_name} {status} API"
    log.info(f"{message}: {api_name}")
    if bot_embeds.enabled('change'):
        # Send change embed to interaction channel
        await bot_embeds.send('change', message, f'**{api_name}**', channel=ictx.channel)
        if bot_database.announce_channels:
            # Send embeds to announcement channels
            await bg_task_queue.put(announce_changes(f'{status} API', f'**{api_name}**', ictx))

@client.hybrid_command(description="Toggle available API Clients on/off")
@guild_or_owner_only()
async def toggle_api(ctx: commands.Context):
    # Collect all clients from api.all_clients
    all_apis = api.clients or {}
    if not all_apis:
        await ctx.send('There are no APIs available', ephemeral=True)
        return

    # Create a reverse map from main clients to their role label
    main_roles = {id(api.textgen): "main TextGen client",
                  id(api.imggen): "main ImgGen client",
                  id(api.ttsgen): "main TTSGen client"}

    # Build display name map with status and potential main role
    display_name_to_key = {}
    for key, client in all_apis.items():
        status = "enabled" if client.enabled else "disabled"
        main_suffix = f" **{main_roles.get(id(client))}**" if id(client) in main_roles else ""
        display_name = f"{key} ({status}){main_suffix}"
        display_name_to_key[display_name] = key

    # Use the display names for the menu
    items_for_api_menus = sorted(display_name_to_key.keys())

    apis_view = SelectOptionsView(
        items_for_api_menus,
        custom_id_prefix='apis',
        placeholder_prefix='APIs: ',
        unload_item=None
    )
    view_message = await ctx.send('### Select an API.', view=apis_view, ephemeral=True)
    await apis_view.wait()

    selected_item = apis_view.get_selected()
    await view_message.delete()

    # Lookup the real key and get the APIClient
    original_key = display_name_to_key.get(selected_item)
    selected_api:APIClient = all_apis.get(original_key)
    # Typicallly if command timed out
    if not selected_api:
        return
    
    # Apply Toggle
    new_status_str = await selected_api.toggle()
    if new_status_str is None:
        await ctx.reply(f"Failed to toggle **{original_key}**.", delete_after=5, ephemeral=True)
        return
    await ctx.reply(f"**{original_key}** is now **{new_status_str}**", delete_after=5, ephemeral=True)
    
    # Process TTSGen API changes
    if selected_api == api.ttsgen:
        await update_ttsgen(ctx, status=new_status_str, rebuild_cmd_opts=(new_status_str == 'enabled'))

    # Announce changes
    await announce_api_changes(ctx, api_name=original_key, status=new_status_str)

@client.hybrid_command(description='Change the API client for a main function (imggen, ttsgen, textgen)')
@guild_or_owner_only()
async def change_main_api(ctx: commands.Context):
    all_clients = api.clients or {}
    if not all_clients:
        await ctx.send("There are no APIs available.", ephemeral=True)
        return
    # Map of function name to (current_client, expected_class)
    function_specs = {'imggen': (api.imggen, ImgGenClient),
                      'ttsgen': (api.ttsgen, TTSGenClient),
                      'textgen': (api.textgen, TextGenClient)}
    views_to_send = []

    for func_name, (current_client, expected_cls) in function_specs.items():
        # Filter all clients that match the expected class and are enabled
        eligible_clients = {name: client for name, client in all_clients.items()
                            if isinstance(client, expected_cls) and client.enabled}
        if not eligible_clients:
            continue
        # If current client exists, remove it from selection list
        if current_client in eligible_clients.values():
            eligible_clients = {name: client for name, client in eligible_clients.items()
                                if client is not current_client}
        # If at least one client is selectable, show the view
        if eligible_clients:
            menu_items = sorted(eligible_clients.keys())
            view = SelectOptionsView(menu_items,
                                     custom_id_prefix=f"{func_name}_api",
                                     placeholder_prefix=f"{func_name.upper()} API: ",
                                     unload_item=None)
            views_to_send.append((func_name, view))

    if not views_to_send:
        await ctx.send("No alternative enabled clients available for any main function.", ephemeral=True)
        return

    # Convert to dict for easier lookup
    views_dict = dict(views_to_send)

    # If multiple function types are eligible, show a menu to pick one
    if len(views_dict) > 1:
        class FunctionSelect(discord.Select):
            def __init__(self, view: 'FunctionSelectView'):
                self.view_ref = view
                options = [discord.SelectOption(label=func.upper(), value=func) for func in views_dict]
                super().__init__(placeholder="Select a function to reassign...", options=options)

            async def callback(self, interaction: discord.Interaction):
                self.view_ref.selected_func = self.values[0]
                self.view_ref.interaction = interaction
                self.view_ref.stop()

        class FunctionSelectView(discord.View):
            def __init__(self):
                super().__init__(timeout=60)
                self.selected_func = None
                self.interaction = None
                self.add_item(FunctionSelect(self))

        func_select_view = FunctionSelectView()
        await ctx.send("Choose a main function to reassign its API client:", view=func_select_view, ephemeral=True)
        await func_select_view.wait()

        selected_func = func_select_view.selected_func
        if not selected_func:
            return  # Timeout or cancelled

        interaction = func_select_view.interaction
    else:
        # Only one function type — auto-select it
        selected_func = next(iter(views_dict))
        interaction = ctx  # Original interaction

    # Shared final dispatch
    selected_view = views_dict[selected_func]
    send_func = interaction.response.send_message if hasattr(interaction, "response") else interaction.send
    message = await send_func(f"Select new API client for **{selected_func}**:", view=selected_view, ephemeral=True)
    await selected_view.wait()

    selected_item = selected_view.get_selected()
    await message.delete()

    selected_api:APIClient = all_clients.get(selected_item)
    # Typicallly if command timed out
    if not selected_api:
        return

    setattr(api, selected_func, selected_api)

    await ctx.reply(f"**Main {selected_func}** is now **{selected_item}**", delete_after=5, ephemeral=True)

    # Announce changes
    await announce_api_changes(ctx, api_name=selected_item, status=f'changed main {selected_func}')

    # Process TTSGen API changes
    if selected_func == 'imggen':
        bot_settings.init_imgmodel()
        if config.is_per_server():
            for settings in guild_settings.values():
                settings.init_imgmodel()
        await imgmodel(ctx)

    # Process TTSGen API changes
    elif selected_func == 'ttsgen':
        await update_ttsgen(ctx)

@client.hybrid_command(description="Update dropdown menus without restarting bot script.")
@guild_or_owner_only()
async def sync(ctx: commands.Context):
    await ctx.reply('Syncing client tree. Note: Menus may not update instantly.', ephemeral=True, delete_after=15)
    log.info(f"{ctx.author.display_name} used '/sync' to sync the client.tree (refresh commands).")
    await bg_task_queue.put(client.tree.sync()) # Process this in the background

#################################################################
######################### LLM COMMANDS ##########################
#################################################################
if tgwui_enabled:
    # /reset_conversation command - Resets current character
    @client.hybrid_command(description="Reset the conversation with current character")
    @configurable_for_dm_if(lambda ctx: config.discord_dm_setting('allow_chatting', True))
    async def reset_conversation(ctx: commands.Context):
        await ireply(ctx, 'conversation reset') # send a response msg to the user
        
        last_character = bot_settings.get_last_setting_for("last_character", ctx)
        log.info(f'{ctx.author.display_name} used "/reset_conversation": "{last_character}"')
        # offload to TaskManager() queue
        reset_params = Params(character={'char_name': last_character, 'verb': 'Resetting', 'mode': 'reset'})
        reset_task = Task('reset', ctx, params=reset_params)
        await task_manager.queue_task(reset_task, 'history_queue')

    # /save_conversation command
    @client.hybrid_command(description="Saves the current conversation to a new file in text-generation-webui/logs/")
    @guild_or_owner_only()
    async def save_conversation(ctx: commands.Context):
        history_char, history_mode = get_char_mode_for_history(ctx)
        await bot_history.get_history_for(ctx.channel.id, history_char, history_mode).save(timeout=0, force=True)
        await ctx.reply('Saved current conversation history', ephemeral=True)

    # Context menu command to edit both a discord message and HMessage
    @client.tree.context_menu(name="edit history")
    async def edit_history(inter: discord.Interaction, message: discord.Message):
        if not (message.author == inter.user or message.author == client.user):
            await inter.response.send_message("You can only edit your own or bot's messages.", ephemeral=True, delete_after=5)
            return
        history_char, history_mode = get_char_mode_for_history(inter)
        local_history = bot_history.get_history_for(inter.channel.id, history_char, history_mode)
        if not local_history:
            await inter.response.send_message(f'There is currently no chat history to "edit history" from.', ephemeral=True, delete_after=5)
            return
        matched_hmessage = local_history.search(lambda m: m.id == message.id or message.id in m.related_ids)
        if not matched_hmessage:
            await inter.response.send_message("Message not found in current chat history.", ephemeral=True, delete_after=5)
            return
        
        # offload to TaskManager() queue
        log.info(f'{inter.user.display_name} used "edit history"')
        edit_history_task = Task('edit_history', inter, matched_hmessage=matched_hmessage, target_discord_msg=message)
        await task_manager.queue_task(edit_history_task, 'history_queue')

    # Context menu command to hide a message pair
    @client.tree.context_menu(name="toggle as hidden")
    async def hide_or_reveal_history(inter: discord.Interaction, message: discord.Message):
        if not (message.author == inter.user or message.author == client.user):
            await inter.response.send_message("You can only hide your own or bot's messages.", ephemeral=True, delete_after=5)
            return
        try:
            history_char, history_mode = get_char_mode_for_history(inter)
            local_history = bot_history.get_history_for(inter.channel.id, history_char, history_mode)
            target_hmessage = local_history.search(lambda m: m.id == message.id or message.id in m.related_ids)
            if not target_hmessage:
                await inter.response.send_message("Message not found in current chat history.", ephemeral=True, delete_after=5)
                return
        except Exception as e:
            log.error(f'An error occured while getting history for "Hide History": {e}')

        await ireply(inter, 'toggle as hidden') # send a response msg to the user

        log.info(f'{inter.user.display_name} used "hide or reveal history"')
        hide_or_reveal_history_task = Task('hide_or_reveal_history', inter, local_history=local_history, target_discord_msg=message, target_hmessage=target_hmessage) # custom kwargs
        await task_manager.queue_task(hide_or_reveal_history_task, 'history_queue')

    # Initialize Continue/Regenerate Context commands
    async def process_cont_regen_cmds(inter:discord.Interaction, message:discord.Message, cmd:str, mode:str=None):
        if not (message.author == inter.user or message.author == client.user):
            await inter.response.send_message(f'You can only "{cmd}" from messages written by yourself or from the bot.', ephemeral=True, delete_after=7)
            return
        history_char, history_mode = get_char_mode_for_history(inter)
        local_history = bot_history.get_history_for(inter.channel.id, history_char, history_mode)
        if not local_history:
            await inter.response.send_message(f'There is currently no chat history to "{cmd}" from.', ephemeral=True, delete_after=5)
            return
        target_hmessage = local_history.search(lambda m: m.id == message.id or message.id in m.related_ids)
        if not target_hmessage:
            await inter.response.send_message("Message not found in current chat history.", ephemeral=True, delete_after=5)
            return

        await ireply(inter, cmd) # send a response msg to the user

        # offload to TaskManager() queue
        if cmd == 'Continue':
            log.info(f'{inter.user.display_name} used "{cmd}"')
            continue_task = Task('continue', inter, local_history=local_history, target_discord_msg=message) # custom kwarg
            await task_manager.queue_task(continue_task, 'history_queue')
        else:
            log.info(f'{inter.user.display_name} used "{cmd} ({mode})"')
            regenerate_task = Task('regenerate', inter, local_history=local_history, target_discord_msg=message, target_hmessage=target_hmessage, mode=mode) # custom kwargs
            await task_manager.queue_task(regenerate_task, 'history_queue')

    # Context menu command to Regenerate from selected user message and create new history
    @client.tree.context_menu(name="regenerate create")
    async def regen_create_llm_gen(inter: discord.Interaction, message:discord.Message):
        await process_cont_regen_cmds(inter, message, 'Regenerate', 'create')

    # Context menu command to Regenerate from selected user message and replace the original bot response
    @client.tree.context_menu(name="regenerate replace")
    async def regen_replace_llm_gen(inter: discord.Interaction, message:discord.Message):
        await process_cont_regen_cmds(inter, message, 'Regenerate', 'replace')

    # Context menu command to Continue last reply
    @client.tree.context_menu(name="continue")
    async def continue_llm_gen(inter: discord.Interaction, message:discord.Message):
        await process_cont_regen_cmds(inter, message, 'Continue')

def load_default_character(settings:"Settings", guild_id:int|None=None):
    try:
        # Update stored database / tgwui_shared_module.settings values for character
        bot_settings.set_last_setting_for("last_character", 'default', guild_id=guild_id, save_now=True)
        # Fix any invalid settings while warning user
        settings.fix_settings()
        # save settings
        settings.save()
    except Exception as e:
        log.error(f"Error loading default character. {e}")
        print(traceback.format_exc())

# Collect character information
async def character_loader(char_name, settings:"Settings", guild_id:int|None=None):
    if char_name is None:
        load_default_character(settings, guild_id)
        return
    try:
        # Get data using textgen-webui native character loading function
        _, name, _, greeting, context = custom_load_character(char_name, '', '', try_tgwui=tgwui_enabled)
        missing_keys = [key for key, value in {'name': name, 'greeting': greeting, 'context': context}.items() if not value]
        if any (missing_keys):
            log.warning(f'Note that character "{char_name}" is missing the following info:"{missing_keys}".')
        textgen_data = {'name': name, 'greeting': greeting, 'context': context}
        # Check for extra bot data
        char_data = await load_character_data(char_name, try_tgwui=tgwui_enabled)
        char_instruct = char_data.get('instruction_template_str', None)
        # Merge with basesettings
        char_data = merge_base(char_data, 'llmcontext')
        # Reset warning for character specific TTS
        bot_database.update_was_warned('char_tts')

        # Gather context specific keys from the character data
        char_llmcontext = {}
        use_voice_channels = True
        for key, value in char_data.items():
            if tgwui_enabled and key == 'extensions' and value and isinstance(value, dict):
                if not tts_is_enabled(for_mode='tgwui'):
                    for subkey, _ in value.items():
                        if subkey in tgwui.tts.supported_extensions and char_data[key][subkey].get('activate'):
                            char_data[key][subkey]['activate'] = False
                await tgwui.update_extensions(value)
                char_llmcontext['extensions'] = value
            elif key == 'use_voice_channel':
                use_voice_channels:bool = value
                char_llmcontext['use_voice_channel'] = value
            elif key == 'tags':
                value = base_tags.update_tags(value) # Unpack any tag presets
                char_llmcontext['tags'] = value
        # Connect to voice channels
        if tts_is_enabled(and_online=True):
            if guild_id:
                await voice_clients.voice_channel(guild_id, use_voice_channels)
            else:
                for vc_guild_id in bot_database.voice_channels:
                    await voice_clients.voice_channel(vc_guild_id, use_voice_channels)
        # Merge llmcontext data and extra data
        char_llmcontext.update(textgen_data)
        # Update stored database / tgwui_shared_module.settings values for character
        bot_settings.set_last_setting_for("last_character", char_name, guild_id=guild_id, save_now=True)
        if tgwui_enabled:
            tgwui_shared_module.settings['character'] = char_name

        # Collect behavior data
        char_behavior = char_data.get('behavior', {})
        char_behavior = merge_base(char_behavior, 'behavior')
        # Collect llmstate data
        char_llmstate = char_data.get('state', {})
        char_llmstate = merge_base(char_llmstate, 'llmstate,state')
        char_llmstate['character_menu'] = char_name

        # Commit the character data to settings
        settings.llmcontext.update(dict(char_llmcontext))
        # Update State dict
        state_dict = settings.llmstate.state
        update_dict(state_dict, dict(char_llmstate))
        # Update Behavior
        settings.behavior.update(dict(char_behavior), char_name)
        # Fix any invalid settings while warning user
        settings.fix_settings()
        # save settings
        settings.save()

        # Print mode in cmd
        guild_msg = f' in {settings._guild_name}' if guild_id else ''
        log.info(f'Mode is set to "{state_dict["mode"]}"{guild_msg}.')

        # Check for any char defined or model defined instruct_template
        if tgwui_enabled:
            update_instruct = char_instruct or tgwui.instruction_template_str or None
            if update_instruct:
                state_dict['instruction_template_str'] = update_instruct
    except Exception as e:
        log.error(f"Error loading character. Check spelling and file structure. Use bot cmd '/character' to try again. {e}")
        print(traceback.format_exc())

# Task to manage discord profile updates
delayed_profile_update_tasks:dict[int, asyncio.Task] = {}

async def delayed_profile_update(display_name, avatar_fp, remaining_cooldown, guild:discord.Guild|None=None):
    try:
        await asyncio.sleep(remaining_cooldown)

        guild_id = None
        if guild:
            guild_id = guild.id

        servers = client.guilds
        if config.is_per_character() and guild:
            servers = [guild]
        for server in servers:
            client_member = server.get_member(client.user.id)
            await client_member.edit(nick=display_name)

        if avatar_fp:
            with open(avatar_fp, 'rb') as f:
                avatar = f.read()
                await client.user.edit(avatar=avatar)
                bot_settings.set_last_setting_for("last_avatar", avatar_fp, guild_id=guild_id, save_now=True)

        log.info(f"Updated discord client profile: (display name: {display_name}; Avatar: {'Updated' if avatar_fp else 'Unchanged'}).")
        log.info("Profile can be updated again in 10 minutes.")
        bot_settings.set_last_setting_for("last_change", time.time(), guild_id=guild_id)
    except Exception as e:
        log.error(f"Error while changing character username or avatar: {e}")

def find_avatar_in_paths(base_name: str, paths: list[str]) -> str | None:
    for path in paths:
        for ext in ['png', 'jpg', 'gif']:
            full_path = os.path.join(path, f"{base_name}.{ext}")
            if os.path.exists(full_path):
                return full_path
    return None

def get_avatar_for(char_name: str, ictx: CtxInteraction):
    character_paths: list[str] = get_all_character_paths()
    avatar_fp = None

    if config.is_per_character():
        avatar_fn = config.per_server_settings.get("character_avatar") or char_name
        search_paths = [shared_path.dir_user_images] if avatar_fn else []
        avatar_fp = find_avatar_in_paths(avatar_fn, search_paths)
    else:
        avatar_fp = find_avatar_in_paths(char_name, character_paths)

    last_avatar = bot_settings.get_last_setting_for("last_avatar", guild_id=ictx.guild.id)
    if avatar_fp == last_avatar:
        return None

    return avatar_fp

async def update_client_profile(char_name:str, ictx:CtxInteraction):
    try:
        global delayed_profile_update_tasks
        # Cancel delayed profile update task if one is already pending
        update_task:Optional[asyncio.Task] = delayed_profile_update_tasks.get(ictx.guild.id)
        if update_task and not update_task.done():
            update_task.cancel()
        # Do not update profile if name is same and no update task is scheduled
        elif ictx.guild.get_member(client.user.id).display_name == char_name:
            return
        # get avatar file path
        avatar_fp = get_avatar_for(char_name, ictx)
        # Check for cooldown before allowing profile change
        last_change = bot_settings.get_last_setting_for("last_change", guild_id=ictx.guild.id)
        last_cooldown = last_change + timedelta(minutes=10).seconds
        if time.time() >= last_cooldown:
            # Apply changes immediately if outside 10 minute cooldown
            delayed_profile_update_tasks[ictx.guild.id] = asyncio.create_task(delayed_profile_update(char_name, avatar_fp, 0, ictx.guild))
        else:
            remaining_cooldown = last_cooldown - time.time()
            seconds = int(remaining_cooldown)
            await ictx.channel.send(f'**Due to Discord limitations, character name/avatar will update in {seconds} seconds.**', delete_after=10)
            log.info(f"Due to Discord limitations, character name/avatar will update in {remaining_cooldown} seconds.")
            delayed_profile_update_tasks[ictx.guild.id] = asyncio.create_task(delayed_profile_update(char_name, avatar_fp, seconds, ictx.guild))
    except Exception as e:
        log.error(f"An error occurred while updating Discord profile: {e}")

# Apply character changes
async def change_character(char_name, ictx:CtxInteraction):
    try:
        settings:Settings = get_settings(ictx)
        # Load the character
        await character_loader(char_name, settings, ictx.guild.id)
        # Rebuild idle weights
        bot_status.build_idle_weights()
        # Update discord username / avatar
        await update_client_profile(char_name, ictx)
    except Exception as e:
        await ictx.channel.send(f"An error occurred while changing character: {e}")
        log.error(f"An error occurred while changing character: {e}")

async def process_character(ctx, selected_character_value):
    try:
        if not selected_character_value:
            await ctx.reply('**No character was selected**.', ephemeral=True, delete_after=5)
            return
        char_name = Path(selected_character_value).stem
        await ireply(ctx, 'character change') # send a response msg to the user

        log.info(f'{ctx.author.display_name} used "/character": "{char_name}"')
        # offload to TaskManager() queue
        change_char_params = Params(character={'char_name': char_name, 'verb': 'Changing', 'mode': 'change'})
        change_char_task = Task('change_char', ctx, params=change_char_params)
        await task_manager.queue_task(change_char_task, 'history_queue')

    except Exception as e:
        log.error(f"Error processing selected character from /character command: {e}")

def get_tgwui_character_path() -> str|None:
    if tgwui_enabled:
        # default to old location
        tgwui_chars = os.path.join(shared_path.dir_tgwui, "characters")
        # check for new nested location
        new_tgwui_chars = os.path.join(shared_path.dir_tgwui, "user_data", "characters")
        if os.path.exists(new_tgwui_chars):
            tgwui_chars = new_tgwui_chars
        return tgwui_chars
    return None

def get_all_character_paths() -> list:
    character_paths = [shared_path.dir_user_characters]
    tgwui_char_path = get_tgwui_character_path()
    if tgwui_char_path:
        character_paths.append(tgwui_char_path)
    return character_paths
        
def get_all_characters() -> Tuple[list, list]:
    all_characters = []
    filtered_characters = []
    character_paths = get_all_character_paths()
    try:
        for character_path in character_paths:
            for file in sorted(Path(character_path).glob("*")):
                if file.suffix in [".json", ".yml", ".yaml"]:
                    character = {}
                    character['name'] = file.stem
                    all_characters.append(character)

                    char_data = load_file(file, {})
                    if not char_data:
                        continue

                    if char_data.get('bot_in_character_menu', True):
                        filtered_characters.append(character)

    except Exception as e:
        log.error(f"An error occurred while getting all characters: {e}")
    return all_characters, filtered_characters

# Command to change characters
@client.hybrid_command(description="Choose a character")
@guild_only()
async def character(ctx: commands.Context):
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

#################################################################
####################### IMGMODEL COMMANDS #######################
#################################################################
if imggen_enabled:

    # Register command for helper function to toggle auto-select imgmodel
    @client.hybrid_command(description='Toggles the automatic Img model changing task')
    @guild_or_owner_only()
    async def toggle_auto_change_imgmodels(ctx: commands.Context):
        imgmodel_settings:ImgModel = get_imgmodel_settings(ctx)
        imgmodel_update_task = imgmodel_settings._imgmodel_update_task
        if imgmodel_update_task and not imgmodel_update_task.done():
            imgmodel_update_task.cancel()
            await ctx.send("Auto-change Imgmodels task was cancelled.", ephemeral=True, delete_after=5)
            log.info("[Auto Change Imgmodels] Task was cancelled via '/toggle_auto_change_imgmodels_task'")
        else:
            await bg_task_queue.put(imgmodel_settings.start_auto_change_imgmodels())
            await ctx.send("Auto-change Img models task was started.", ephemeral=True, delete_after=5)


    async def process_imgmodel(ctx: commands.Context, selected_imgmodel:str):
        user_name = get_user_ctx_inter(ctx).display_name or None
        try:
            if not selected_imgmodel:
                await ctx.reply('**No Img model was selected**.', ephemeral=True, delete_after=5)
                return
            await ireply(ctx, 'Img model change') # send a response msg to the user

            log.info(f'{user_name} used "/imgmodel": "{selected_imgmodel}"')
            # offload to TaskManager() queue
            imgmodel_settings:ImgModel = get_imgmodel_settings(ctx)
            imgmodel_params = await imgmodel_settings.get_model_params(selected_imgmodel)
            change_imgmodel_task = Task('change_imgmodel', ctx, params=Params(imgmodel=imgmodel_params))
            await task_manager.queue_task(change_imgmodel_task, 'gen_queue')

        except Exception as e:
            log.error(f"Error processing selected imgmodel from /imgmodel command: {e}")

    @client.hybrid_command(description="Choose an Img Model")
    @guild_or_owner_only()
    async def imgmodel(ctx: commands.Context):
        imgmodel_settings:ImgModel = get_imgmodel_settings(ctx)
        imgmodels = await imgmodel_settings.get_filtered_imgmodels_list(ctx)
        if imgmodels:
            imgmodel_names = imgmodel_settings.collect_model_names(imgmodels)
            warned_too_many_imgmodel = False # TODO use the warned_once feature?
            imgmodels_view = SelectOptionsView(imgmodel_names,
                                               custom_id_prefix='imgmodels',
                                               placeholder_prefix='ImgModels: ',
                                               unload_item=None,
                                               warned=warned_too_many_imgmodel)
            view_message = await ctx.send('### Select an Image Model.', view=imgmodels_view, ephemeral=True)
            await imgmodels_view.wait()
            selected_item = imgmodels_view.get_selected()
            await view_message.delete()
            await process_imgmodel(ctx, selected_item)
        else:
            await ctx.send('There are no Img models available', ephemeral=True)

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

        log.info(f'{ctx.author.display_name} used "/llmmodel": "{selected_llmmodel}"')
        # offload to TaskManager() queue
        change_llmmodel_params = Params(llmmodel={'llmmodel_name': selected_llmmodel, 'verb': 'Changing', 'mode': 'change'})
        change_llmmodel_task = Task('change_llmmodel', ctx, params=change_llmmodel_params)
        await task_manager.queue_task(change_llmmodel_task, 'gen_queue')

    except Exception as e:
        log.error(f"Error processing /llmmodel command: {e}")

if tgwui_enabled:

    @client.hybrid_command(description="Choose an LLM Model")
    @guild_or_owner_only()
    async def llmmodel(ctx: commands.Context):
        all_llmmodels = tgwui_utils_module.get_available_models()
        if all_llmmodels:
            items_for_llm_model = [i for i in all_llmmodels]
            if 'None' in items_for_llm_model:
                items_for_llm_model.remove('None')
            warned_too_many_llm_model = False # TODO use the warned_once feature?
            llmmodels_view = SelectOptionsView(items_for_llm_model,
                                            custom_id_prefix='llmmodels',
                                            placeholder_prefix='LLMModels: ',
                                            unload_item='None',
                                            warned=warned_too_many_llm_model)
            view_message = await ctx.send('### Select an LLM Model.', view=llmmodels_view, ephemeral=True)
            await llmmodels_view.wait()
            selected_item = llmmodels_view.get_selected()
            await view_message.delete()
            await process_llmmodel(ctx, selected_item)
        else:
            await ctx.send('There are no LLM models available', ephemeral=True)

#################################################################
######################### /SPEAK COMMAND ########################
#################################################################
async def process_speak_silero_non_eng(ctx: commands.Context, lang):
    non_eng_speaker = None
    non_eng_model = None
    try:
        with open('extensions/silero_tts/languages.json', 'r') as file:
            languages_data = json.load(file)
        if lang in languages_data:
            default_voice = languages_data[lang].get('default_voice')
            if default_voice: 
                non_eng_speaker = default_voice
            silero_model = languages_data[lang].get('model_id')
            if silero_model: 
                non_eng_model = silero_model
            tts_args = {'silero_tts': {'language': lang, 'speaker': non_eng_speaker, 'model_id': non_eng_model}}
        if not (non_eng_speaker and non_eng_model):
            await ctx.send(f'Could not determine the correct voice and model ID for language "{lang}". Defaulting to English.', ephemeral=True)
            tts_args = {'silero_tts': {'language': 'English', 'speaker': 'en_1'}}
    except Exception as e:
        log.error(f"Error processing non-English voice for silero_tts: {e}")
        await ctx.send(f"Error processing non-English voice for silero_tts: {e}", ephemeral=True)
    return tts_args

async def process_speak_args(ctx: commands.Context, selected_voice=None, lang=None, user_voice=None):
    api_tts_on, tgwui_tts_on = tts_is_enabled(and_online=True, for_mode='both')
    tts_args = {}
    try:
        # API handling
        if api_tts_on:
            api_voice_key = api.ttsgen.post_generate.voice_input_key
            api_lang_key = api.ttsgen.post_generate.language_input_key
            if selected_voice and api_voice_key:
                tts_args[api_voice_key] = selected_voice
            if lang and api_lang_key:
                tts_args[api_lang_key] = lang
        # TGWUI TTS extension handling
        elif tgwui_tts_on:
            if lang:
                if tgwui.tts.extension == 'elevenlabs_tts':
                    if lang != 'English':
                        tts_args.setdefault(tgwui.tts.extension, {}).setdefault('model', 'eleven_multilingual_v1')
                        # Currently no language parameter for elevenlabs_tts
                else:
                    tts_args.setdefault(tgwui.tts.extension, {}).setdefault(tgwui.tts.lang_key, lang)
                    tts_args[tgwui.tts.extension][tgwui.tts.lang_key] = lang
            if selected_voice or user_voice:
                tts_args.setdefault(tgwui.tts.extension, {}).setdefault(tgwui.tts.voice_key, 'temp_voice.wav' if user_voice else selected_voice)
            elif tgwui.tts.extension == 'silero_tts' and lang:
                if lang != 'English':
                    tts_args = await process_speak_silero_non_eng(ctx, lang) # returns complete args for silero_tts
                    if selected_voice: 
                        await ctx.send(f'Currently, non-English languages will use a default voice (not using "{selected_voice}")', ephemeral=True)
            elif tgwui.tts.extension in tgwui.last_extension_params and tgwui.tts.voice_key in tgwui.last_extension_params[tgwui.tts.extension]:
                pass # Default to voice in last_extension_params
            elif f'{tgwui.tts.extension}-{tgwui.tts.voice_key}' in tgwui_shared_module.settings:
                pass # Default to voice in tgwui_shared_module.settings
    except Exception as e:
        log.error(f"Error processing tts options: {e}")
        await ctx.send(f"Error processing tts options: {e}", ephemeral=True)
    return tts_args

async def convert_and_resample_mp3(ctx, mp3_file):
    try:
        audio = AudioSegment.from_mp3(mp3_file)
        if audio.channels == 2:
            audio = audio.set_channels(1)   # should be Mono
        audio = audio.set_frame_rate(22050) # ideal sample rate
        audio = audio.set_sample_width(2)   # 2 bytes for 16 bits
        output_dir = os.path.dirname(mp3_file)
        if not config.path_allowed(output_dir):
            raise RuntimeError(f"Tried saving audio file to a path which is not in config.yaml 'allowed_paths': {output_dir}")
        wav_filename = os.path.splitext(os.path.basename(mp3_file))[0] + '.wav'
        wav_path = f"{output_dir}/{wav_filename}"
        audio.export(wav_path, format="wav")
        log.info(f'User provided file "{mp3_file}" was converted to .wav for "/speak" command')
        return wav_path
    except Exception as e:
        log.error(f"Error converting user's .mp3 to .wav: {e}")
        await ctx.send("An error occurred while processing the voice file.", ephemeral=True)
    finally:
        if mp3_file:
            os.remove(mp3_file)

async def process_user_voice(ctx: commands.Context, voice_input=None):
    api_tts_on, _ = tts_is_enabled(and_online=True, for_mode='both')
    if api_tts_on:
        if voice_input:
            await ctx.send("Sorry, the bot's configured TTS method does not currently support user voice file input.", ephemeral=True)
        return None
    try:
        if not (voice_input and getattr(voice_input, 'content_type', '').startswith("audio/")):
            return ''
        if 'alltalk' not in tgwui.tts.extension and tgwui.tts.extension != 'coqui_tts':
            await ctx.send("Sorry, current tts extension does not allow using a voice attachment (only works for 'alltalk_tts' and 'coqui_tts)", ephemeral=True)
            return ''
        voiceurl = voice_input.url
        voiceurl_without_params = voiceurl.split('?')[0]
        if not voiceurl_without_params.endswith((".wav", ".mp3")):
            await ctx.send("Invalid audio format. Please try again with a WAV or MP3 file.", ephemeral=True)
            return ''
        voice_data_ext = voiceurl_without_params[-4:]
        user_voice = f'extensions/{tgwui.tts.extension}/voices/temp_voice{voice_data_ext}'
        if not config.path_allowed(user_voice):
            await ctx.send(f'The bot is not configured to allow writing the audio file to the required location, "{user_voice}"', ephemeral=True)
            return ''
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
                user_voice = await convert_and_resample_mp3(ctx, user_voice)
            except Exception:
                if user_voice: 
                    os.remove(user_voice)
        return user_voice
    except Exception as e:
        log.error(f"Error processing user provided voice file: {e}")
        await ctx.send("An error occurred while processing the voice file.", ephemeral=True)

async def process_speak(ctx: commands.Context, input_text, selected_voice=None, lang=None, voice_input=None):
    try:
        # Only generate TTS for the server conntected to Voice Channel
        if (is_direct_message(ctx) or not voice_clients.guild_vcs.get(ctx.guild.id)) \
            and int(config.ttsgen.get('play_mode', 0)) == 0:
            await ctx.send('Voice Channel is not enabled on this server', ephemeral=True, delete_after=5)
            return
        user_voice = await process_user_voice(ctx, voice_input)
        tts_args = await process_speak_args(ctx, selected_voice, lang, user_voice)
        await ireply(ctx, 'tts') # send a response msg to the user

        log.info(f'{ctx.author.display_name} used "/speak": "{input_text}"')
        # offload to TaskManager() queue
        speak_params = Params(tts_args=tts_args, user_voice=user_voice)
        speak_task = Task('speak', ctx, text=input_text, params=speak_params)
        await task_manager.queue_task(speak_task, 'gen_queue')

    except Exception as e:
        log.error(f"Error processing tts request: {e}")
        await ctx.send(f"Error processing tts request: {e}", ephemeral=True)

class SpeakCmdOptions:
    def __init__(self):
        self.voice_hash_dict:dict = {}
        self.lang_options = [app_commands.Choice(name='**disabled option', value='disabled')]
        self.lang_options_label:str = 'languages'
        self.voice_options = [app_commands.Choice(name='**disabled option', value='disabled')]
        self.voice_options_label:str = 'voices'
        self.voice_options1 = [app_commands.Choice(name='**disabled option', value='disabled')]
        self.voice_options1_label:str = 'voices1'
        self.voice_options2 = [app_commands.Choice(name='**disabled option', value='disabled')]
        self.voice_options2_label:str = 'voices2'

    def split_options(self, all_voices:list, lang_list:list):
        self.voice_options.clear()
        self.voice_options.extend(app_commands.Choice(name=voice_name.replace('_', ' ').title(), value=str(hash(voice_name))) for voice_name in all_voices[:25])
        self.voice_options_label = f'voices_{self.voice_options[0].name[0]}-{self.voice_options[-1].name[0]}'.lower()
        if len(all_voices) > 25:
            self.voice_options1.clear()
            self.voice_options1.extend(app_commands.Choice(name=voice_name.replace('_', ' ').title(), value=str(hash(voice_name))) for voice_name in all_voices[25:50])
            self.voice_options1_label = f'voices_{self.voice_options1[0].name[0]}-{self.voice_options1[-1].name[0]}'.lower()
            if self.voice_options1_label == self.voice_options_label:
                self.voice_options1_label = f'voices_{self.voice_options1_label}_1'
            if len(all_voices) > 50:
                self.voice_options2.clear()
                self.voice_options2.extend(app_commands.Choice(name=voice_name.replace('_', ' ').title(), value=str(hash(voice_name))) for voice_name in all_voices[50:75])
                self.voice_options2_label = f'voices_{self.voice_options2[0].name[0]}-{self.voice_options2[-1].name[0]}'.lower()
                if self.voice_options2_label == self.voice_options_label or self.voice_options2_label == self.voice_options1_label:
                    self.voice_options2_label = f'voices_{self.voice_options2_label}_2'
                if len(all_voices) > 75:
                    log.warning("'/speak' command only allows up to 75 voices. Some voices were omitted.")
        if lang_list:
            self.lang_options.clear()
            self.lang_options.extend(app_commands.Choice(name=lang, value=lang) for lang in lang_list)
            self.lang_options_label = 'languages'

    async def get_options(self):
        lang_list, all_voices = [], []
        try:
            # TGWUI Extension Mode
            if tgwui_enabled and tgwui.tts.extension and (tgwui.tts.extension in tgwui.tts.supported_extensions):
                lang_list, all_voices = await tgwui.tts.fetch_speak_options()
            # API mode
            elif api.ttsgen and api.ttsgen.enabled:
                lang_list, all_voices = await api.ttsgen.fetch_speak_options()
        except Exception as e:
            log.error(f"Error getting /speak options: {e}")
        return lang_list, all_voices

    async def build_options(self, sync:bool=True):
        lang_list, all_voices = await self.get_options()
        # Rebuild options
        if all_voices:
            self.voice_hash_dict = {str(hash(voice_name)):voice_name for voice_name in all_voices}
            self.split_options(all_voices, lang_list)
        if sync:
            await client.tree.sync()

speak_cmd_options = SpeakCmdOptions()

@client.hybrid_command(name="speak", description='AI will speak your text using a selected voice (pick only one)')
@app_commands.rename(voice_1 = speak_cmd_options.voice_options_label)
@app_commands.describe(voice_1 = speak_cmd_options.voice_options_label.upper())
@app_commands.choices(voice_1 = speak_cmd_options.voice_options)
@app_commands.rename(voice_2 = speak_cmd_options.voice_options1_label)
@app_commands.describe(voice_2 = speak_cmd_options.voice_options1_label.upper())
@app_commands.choices(voice_2 = speak_cmd_options.voice_options1)
@app_commands.rename(voice_3 = speak_cmd_options.voice_options2_label)
@app_commands.describe(voice_3 = speak_cmd_options.voice_options2_label.upper())
@app_commands.choices(voice_3 = speak_cmd_options.voice_options2)
@app_commands.rename(lang = speak_cmd_options.lang_options_label)
@app_commands.choices(lang = speak_cmd_options.lang_options)
@configurable_for_dm_if(lambda ctx: 'speak' in config.discord_dm_setting('allowed_commands', []))
async def speak(ctx: commands.Context, input_text: str, voice_1: typing.Optional[app_commands.Choice[str]], 
                voice_2: typing.Optional[app_commands.Choice[str]], voice_3: typing.Optional[app_commands.Choice[str]], 
                lang: typing.Optional[app_commands.Choice[str]], voice_input: typing.Optional[discord.Attachment]):
    if sum(1 for v in (voice_1, voice_2, voice_3) if v) > 1:
        await ctx.send("A voice was picked from two separate menus. Using the first selection.", ephemeral=True)
    selected_voice = ((voice_1 or voice_2 or voice_3) and (voice_1 or voice_2 or voice_3).value) or ''
    if selected_voice:
        selected_voice = speak_cmd_options.voice_hash_dict[selected_voice]
    voice_input = voice_input if voice_input is not None else ''
    lang = lang.value if (lang is not None and lang != 'disabled') else ''
    await process_speak(ctx, input_text, selected_voice, lang, voice_input)

#################################################################
######################## /PROMPT COMMAND ########################
#################################################################
async def process_prompt(ctx: commands.Context, selections:dict):
    # User inputs from /image command
    prompt:str = selections.get('prompt', '')
    begin_reply_with:Optional[str] = selections.get('begin_reply_with', None)
    mode:Optional[str] = selections.get('mode', None)
    system_message:Optional[str] = selections.get('system_message', None)
    load_history:Optional[str] = selections.get('load_history', None)
    save_to_history:Optional[str] = selections.get('save_to_history', None)
    response_type:Optional[str] = selections.get('response_type', None)

    if not prompt:
        await ctx.reply("A prompt is required for '/prompt' command", ephemeral=True, delete_after=5)
        return

    try:
        await ireply(ctx, 'prompt') # send a response msg to the user

        prompt_params = Params() # Set the tasks params

        log_msg = "\n"
        if begin_reply_with:
            setattr(prompt_params, 'begin_reply_with', begin_reply_with)
            log_msg += f"\n> __Reply continued from__: {begin_reply_with}"
        if mode:
            setattr(prompt_params, 'mode', mode)
            log_msg += f"\n> __Mode__: {mode.title()}"
        if system_message is not None:
            setattr(prompt_params, 'system_message', system_message)
            log_msg += f"\n> __System Message__: {system_message}"
        if load_history is not None and load_history != 'all':
            load_history = int(load_history)
            # Get channel history
            history_char, history_mode = get_char_mode_for_history(ctx)
            local_history = bot_history.get_history_for(ctx.channel.id, history_char, history_mode)
            i_list, v_list = local_history.render_to_tgwui_tuple()
            if load_history <= 0:
                i_list, v_list = [], []
            else:
                num_to_retain = min(load_history, len(i_list))
                i_list, v_list = i_list[-num_to_retain:], v_list[-num_to_retain:]
            setattr(prompt_params, 'prompt_load_history', (i_list, v_list))
            log_msg += f"\n> __Load History__: Limited to {len(i_list)} recent exchanges"
        if save_to_history is not None:
            setattr(prompt_params, 'prompt_save_to_history', True if save_to_history == "yes" else False)
            log_msg += "\n> __Save History__: Interaction will not be saved" if save_to_history == "no" else ""
        if response_type:
            setattr(prompt_params, 'prompt_response_type', response_type)
            log_msg += f"\n> __Response Type__: {response_type}"

        setattr(prompt_params, 'prompt_message', log_msg)

        # Send an embed to channel with user's prompt and params
        title = f"{ctx.author.display_name} used '/prompt':"
        description = f"**{ctx.author.display_name}**: {prompt}{log_msg}"
        await bot_embeds.send('system', title, description, channel=ctx.channel)

        log.info(f'{title} "{prompt}"')
        # offload to TaskManager() queue
        prompt_task = Task('message', ctx, text=prompt, params=prompt_params)
        await task_manager.queue_task(prompt_task, 'history_queue')

    except Exception as e:
        log.error(f"Error processing '/prompt': {e}")
        await ctx.send(f"Error processing '/prompt': {e}", ephemeral=True)

if tgwui_enabled:
    @client.hybrid_command(name="prompt", description=f'Generate text with advanced options')
    @app_commands.describe(prompt='Your prompt to the LLM.')
    @app_commands.describe(begin_reply_with='The LLM will continue their reply from this.')
    @app_commands.describe(mode='"instruct" will omit character context and draw more attention to your prompt.')
    @app_commands.choices(mode=[app_commands.Choice(name="chat", value="chat"), app_commands.Choice(name="instruct", value="instruct")])
    @app_commands.describe(system_message='A non-user instruction to the LLM. May not have any effect in "chat" mode (model dependent).')
    @app_commands.describe(load_history='The number of recent chat history exchanges the LLM sees for this interaction.')
    @app_commands.choices(load_history=[app_commands.Choice(name="All", value="all"),
                                        app_commands.Choice(name="None", value="0"),
                                        *[app_commands.Choice(name=str(i), value=str(i)) for i in range(1, 11)]])
    @app_commands.describe(save_to_history='Whether the LLM should remember this message exchange.')
    @app_commands.choices(save_to_history=[app_commands.Choice(name="Yes", value="yes"), app_commands.Choice(name="No", value="no")])
    @app_commands.describe(response_type="The type of response you want from the LLM. Use '/image' cmd for advanced image requests.")
    @app_commands.choices(response_type=[app_commands.Choice(name="Text response only", value="Text"),
                                     app_commands.Choice(name="Image response only", value="Image"),
                                     app_commands.Choice(name="Text and Image response", value="TextImage")])
    @configurable_for_dm_if(lambda ctx: 'prompt' in config.discord_dm_setting('allowed_commands', []))
    async def prompt(ctx: commands.Context, prompt: str, begin_reply_with: typing.Optional[str], mode: typing.Optional[app_commands.Choice[str]], 
                     system_message: typing.Optional[str], load_history: typing.Optional[app_commands.Choice[str]],
                     save_to_history: typing.Optional[app_commands.Choice[str]], response_type: typing.Optional[app_commands.Choice[str]]):
        user_selections = {"prompt": prompt, "begin_reply_with": begin_reply_with if begin_reply_with else None, "mode": mode.value if mode else None, 
                           "system_message": system_message if system_message else None, "load_history": load_history.value if load_history else None,
                           "save_to_history": save_to_history.value if save_to_history else None, "response_type": response_type.value if response_type else None}
        await process_prompt(ctx, user_selections)

#################################################################
######################### BOT STATUS ############################
#################################################################
# Bot can "go idle" based on 'responsiveness' behavior setting
class BotStatus:
    def __init__(self):
        self.online = True
        self.come_online_time = None
        self.come_online_task = None
        self.responsiveness = 1.0
        self.idle_range = []
        self.idle_weights = []
        self.current_response_delay:Optional[float] = None # If status is idle (not "online"), the next message will set the delay. When online, resets to None.
        self.go_idle_time = None
        self.go_idle_task = None

    async def come_online(self):
        '''Ensures the bot is online immediately.'''
        self.cancel_go_idle_task()
        if not self.online:
            self.online = True
            await client.change_presence(status=discord.Status.online)
        # Reset variables
        self.current_response_delay = None # only reset this when completed without cancel
        self.come_online_time = None
        self.come_online_task = None

    async def come_online_after(self, time_until_online:float):
        log.debug(f"Bot will be online in {time_until_online} seconds.")
        self.come_online_time = time.time() + time_until_online
        try:
            await asyncio.sleep(time_until_online)
            await self.come_online()
        except asyncio.CancelledError:
            log.debug("come_online_after task was cancelled")
            self.come_online_time = None
            self.come_online_task = None

    def cancel_come_online_task(self):
        if self.come_online_task and not self.come_online_task.done():
            self.come_online_task.cancel()

    async def schedule_come_online(self, online_time:float=time.time()):
        '''Ensures the bot is online at given time'''
        self.cancel_come_online_task()
        time_until_online = max(0, online_time - time.time()) # get time difference for sleeping
        if time_until_online > 0:
            self.come_online_task = asyncio.create_task(self.come_online_after(time_until_online))
        else:
            await self.come_online()

    def build_idle_weights(self):
        # Use largest available responsiveness setting
        self.responsiveness = bot_settings.behavior.responsiveness
        if config.is_per_character():
            all_resp_sets = []
            for settings in guild_settings.values():
                settings:"Settings"
                all_resp_sets.append(settings.behavior.responsiveness)
            if all_resp_sets:
                self.responsiveness = max(all_resp_sets)
        responsiveness = max(0.0, min(1.0, self.responsiveness)) # clamped between 0.0 and 1.0
        if responsiveness == 1.0:
            return
        num_values = 10           # arbitrary number of values and weights to generate
        max_time_until_idle = 600 # arbitrary max timeframe in seconds
        # Generate evenly spaced values in range of num_values
        self.idle_range = [round(i * max_time_until_idle / (num_values - 1), 3) for i in range(num_values)]
        self.idle_range[0] = self.idle_range[1] # Never go idle immediately
        # Generate the weights from responsiveness
        self.idle_weights = get_normalized_weights(target = responsiveness, list_len = num_values)

    async def go_idle(self):
        if self.online:
            self.online = False
            await client.change_presence(status=discord.Status.idle)
        log.info("Bot is now idle.")
        self.go_idle_time = None
        self.go_idle_task = None

    async def go_idle_after(self, time_until_idle:float):
        try:
            time_until_idle = int(time_until_idle)
            log.info(f"Bot will be idle in {time_until_idle} seconds.")
            self.go_idle_time = time.time() + time_until_idle
            for i in range(time_until_idle):
                await asyncio.sleep(1)
                #log.debug(f'idle countdown {time_until_idle-i}')
            await self.go_idle()
        except asyncio.CancelledError:
            log.debug("Go idle task was cancelled.")
            self.go_idle_time = None
            self.go_idle_task = None

    def cancel_go_idle_task(self):
        # Cancel any prior go idle task
        if self.go_idle_task and not self.go_idle_task.done():
            self.go_idle_task.cancel()
            self.go_idle_time = None
            self.go_idle_task = None

    async def schedule_go_idle(self):
        # Don't schedule go idle task if message responses are queued
        if not message_manager.send_msg_queue.empty():
            return
        '''Sets the bot status to idle at a randomly selected time'''
        responsiveness = max(0.0, min(1.0, self.responsiveness)) # clamped between 0.0 and 1.0
        if responsiveness == 1.0:
            return # Never go idle
        # cancel previously set go idle task
        self.cancel_go_idle_task()
        # choose a value with weighted probability based on 'responsiveness' bot behavior
        time_until_idle = random.choices(self.idle_range, self.idle_weights)[0]
        self.go_idle_task = asyncio.create_task(self.go_idle_after(time_until_idle))

bot_status = BotStatus()

#################################################################
####################### MESSAGE MANAGER #########################
#################################################################
# Manages "normal" discord message requests (not from commands, "Flows", etc)
class MessageManager():
    def __init__(self):
        self.counter = 0
        self.last_send_time = None
        self.next_send_time = None
        self.send_msg_queue = asyncio.PriorityQueue()
        self.send_msg_event = asyncio.Event()
        self.send_msg_task = None
        # self.last_channel = None

    async def send_message(self):
        _, _, task = await self.send_msg_queue.get()
        task:Task
        self.send_msg_event.set()
        try:
            # Queued chunk message
            if task.name == 'chunk_message':
                await task.send_response_chunk(task.llm_resp)
            # Queued 'on_message' or 'spontaneous_message'
            else:
                await task.message_post_llm_task()
        except Exception as e:
            log.error('An error occurred while sending a delayed message:', e)
        await task.stop_typing()
        del task                                # delete task object
        self.last_send_time = time.time()       # log time
        self.send_msg_event.clear()
        self.send_msg_queue.task_done()         # Accept next task
        await self.schedule_next_message_send() # schedule the next message send

    async def send_message_after(self, time_until_send:float):
        try:
            # Fetches and sends the next item in the queue
            await asyncio.sleep(time_until_send)
            await self.send_message()
        except asyncio.CancelledError:
            self.last_send_time = time.time()
            self.send_msg_event.clear()

    async def schedule_next_message_send(self):
        # reset next send time if no other messages queued
        if self.send_msg_queue.empty():
            self.next_send_time = None
            return
        # Unqueue next message
        num, send_time, task = await self.send_msg_queue.get()
        task:Task
        updated_istyping_time = await task.message.update_timing()
        task.istyping.start(start_time=updated_istyping_time)
        # Ensure at least enough time elapsed to write message
        minimum_send_time = self.last_send_time + task.message.seconds_to_write
        updated_send_time = min(minimum_send_time, send_time)
        # put task back and schedule it
        await self.send_msg_queue.put((num, updated_send_time, task))
        await self.schedule_send_message(updated_send_time)

    async def schedule_send_message(self, send_time):
        # Set the send time
        self.next_send_time = send_time
        # convert time difference to seconds for sleeping
        time_until_send = max(0, send_time - time.time())
        # Create message send task
        self.send_msg_task = asyncio.create_task(self.send_message_after(time_until_send))

    async def cancel_send_message(self):
        if self.send_msg_task and not self.send_msg_task.done():
            self.send_msg_task.cancel()

    # Queue delayed message tasks (self-sorting) which will run after predetermined time
    async def queue_delayed_message(self, task:Task):
        num = task.message.num
        send_time = task.message.send_time
        # Add to self-sorted queue
        await self.send_msg_queue.put((num, send_time, task))
        # Schedule message send
        if self.next_send_time is None:
            await self.schedule_send_message(send_time)

    # Queue discord messages / Spontaneous Messages to TaskManager(). Self sorts by 'num'
    async def queue_message_task(self, task:Task, settings:"Settings"):
        # Update counter
        self.counter += 1
        num = self.counter
        received_time = time.time()
        # Determine any response / read text delays (Only applicable to 'normal discord messages')
        response_delay = settings.behavior.set_response_delay() if task.name == 'on_message' else 0
        read_text_delay = settings.behavior.get_text_delay(task.text) if task.name == 'on_message' else 0
        # Assign Message() and attributes
        task.message = Message(settings, num, received_time, response_delay, read_text_delay)
        # Schedule come online. Typing will be scheduled when message is unqueued.
        await bot_status.schedule_come_online(task.message.come_online_time)
        await task_manager.queue_task(task, 'message_queue', num)

message_manager = MessageManager()

#################################################################
#################### SPONTANEOUS MESSAGING ######################
#################################################################
class SpontaneousMessaging():
    def __init__(self):
        self.tasks = {}

    async def reset_for_channel(self, task_name:str, ictx:CtxInteraction):
        # Only reset from discord message or '/prompt' cmd
        if task_name in ['on_message', 'prompt']:
            current_chan_msg_task = self.tasks.get(ictx.channel.id, (None, 0))
            task, _ = current_chan_msg_task
            if task:
                if not task.done():
                    task.cancel()
                self.tasks.pop(ictx.channel.id)

    async def reset_for_server(self, ictx:CtxInteraction):
        if not hasattr(ictx, 'guild'):
            return
        guild_channel_ids = {channel.id for channel in ictx.guild.channels}
        to_remove = []
        for chan_id, (task, _) in self.tasks.items():
            if chan_id in guild_channel_ids:
                if task and not task.done():
                    task.cancel()
                to_remove.append(chan_id)
        for chan_id in to_remove:
            self.tasks.pop(chan_id, None)

    async def reset_all(self):
        to_remove = []
        for chan_id, (task, _) in self.tasks.items():
            if task and not task.done():
                task.cancel()
            to_remove.append(chan_id)
        for chan_id in to_remove:
            self.tasks.pop(chan_id, None)

    async def run_task(self, ictx:CtxInteraction, prompt:str, wait:int):
        await asyncio.sleep(wait)
        # create message task with the randomly selected prompt
        try:
            log.info(f'Prompting for a spontaneous message: "{prompt}"')
            # offload to TaskManager() queue
            spontaneous_message_task = Task('spontaneous_message', ictx, text=prompt)
            # get settings instance
            settings:Settings = get_settings(ictx)
            await message_manager.queue_message_task(spontaneous_message_task, settings)
        except Exception as e:
            log.error(f"Error while processing a Spontaneous Message: {e}")

    async def init_task(self, settings:"Settings", ictx:CtxInteraction, task:asyncio.Task, tally:int):
        # Randomly select wait duration from start/end range 
        wait = random.uniform(settings.behavior.spontaneous_msg_min_wait, settings.behavior.spontaneous_msg_max_wait)
        wait_secs = round(wait*60)
        # select a random prompt
        random_prompt = random.choice(settings.behavior.spontaneous_msg_prompts)
        if not random_prompt:
            random_prompt = '''[SYSTEM] The conversation has been inactive for {time_since_last_msg}, so you should say something.'''

        prompt = await dynamic_prompting(random_prompt)
        # Cancel any existing task (does not reset tally)
        if task and not task.done():
            task.cancel()
        # Start the new task
        new_task = asyncio.create_task(self.run_task(ictx, prompt, wait_secs))
        # update self variable with new task
        self.tasks[ictx.channel.id] = (new_task, tally)
        log.debug(f"Created a spontaneous msg task (channel: {ictx.channel.id}, delay: {wait_secs}), tally: {tally}.") # Debug because we want surprises from this feature
    
    async def set_for_channel(self, ictx:CtxInteraction):
        # First conditional check
        if ictx:
            # get settings instance
            settings:Settings = get_settings(ictx)
            if (random.random() < settings.behavior.spontaneous_msg_chance):
                # Get any existing message task
                current_chan_msg_task = self.tasks.get(ictx.channel.id, (None, 0))
                task, tally = current_chan_msg_task
                # Second conditional check
                if task is None or settings.behavior.spontaneous_msg_max_consecutive == -1 \
                    or tally + 1 < settings.behavior.spontaneous_msg_max_consecutive:
                    # Initialize the spontaneous message task
                    await self.init_task(settings, ictx, task, tally)

spontaneous_messaging = SpontaneousMessaging()

#################################################################
########################### SETTINGS ############################
#################################################################
guild_settings:dict[int, "Settings"] = {}

# Returns either guild specific or main instance of Settings() 
def get_settings(ictx:CtxInteraction|None=None) -> "Settings":
    if config.is_per_server() and ictx and not is_direct_message(ictx):
        return guild_settings.get(ictx.guild.id, bot_settings)
    return bot_settings

def get_imgmodel_settings(ictx:CtxInteraction|None=None) -> "ImgModel":
    if config.is_per_server_imgmodels():
        settings = get_settings(ictx)
        return settings.imgmodel
    return bot_settings.imgmodel

class SettingsBase:
    def get_vars(self):
        return {k:v for k,v in vars(self).items() if not k.startswith('_')}

    def update(self, new_dict:dict):
        for key, value in new_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_nested_var(self, keys):
        current_level = self.__dict__
        for key in keys:
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            elif hasattr(current_level, key):
                current_level = getattr(current_level, key)
            else:
                return None  # Return None if the path is invalid
        return current_level

# Sub-classes under a main class 'Settings'
class Behavior(SettingsBase):
    def __init__(self):
        self.reply_to_itself = 0.0
        self.chance_to_reply_to_other_bots = 0.5
        self.reply_to_bots_when_addressed = 0.3
        self.only_speak_when_spoken_to = True
        self.ignore_parentheses = True
        self.go_wild_in_channel = True
        self.conversation_recency = 600
        self._user_conversations = {}
        # Chance for bot reply to be sent in chunks to discord chat
        self.chance_to_stream_reply = 0.0
        self.stream_reply_triggers = ['\n', '.']
        # Behaviors to be more like a computer program or humanlike
        self.maximum_typing_speed = -1
        self.responsiveness = 1.0
        self.msg_size_affects_delay = False
        self.max_reply_delay = 30.0
        self._response_delay_values = []   # self.response_delay_values and self.response_delay_weights
        self._response_delay_weights = []  # are calculated from the 3 settings above them via build_response_weights()
        self._text_delay_values = []       # similar to response_delays, except weights need to be made for each message
        # Spontaneous messaging
        self.spontaneous_msg_chance = 0.0
        self.spontaneous_msg_max_consecutive = 1
        self.spontaneous_msg_min_wait = 10.0
        self.spontaneous_msg_max_wait = 60.0
        self.spontaneous_msg_prompts = []

    def update(self, new_behavior:dict, char_name:str):
        for key, value in new_behavior.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.build_response_delay_weights()
        self.build_text_delay_values()
        self.print_behavior_message(char_name)

    def print_behavior_message(self, char_name:str):
        log.info(f"{char_name}'s Behavior:")
        max_responsiveness_msg = '• Processes messages at uncapped speeds, and will never go idle.'
        responsiveness_msg = '• Behaves more humanlike. Delays to respond after going idle.'
        log.info(f'{max_responsiveness_msg if self.responsiveness >= 1.0 else responsiveness_msg} (responsiveness: {self.responsiveness})')
        if self.msg_size_affects_delay:
            log.info(f'• Takes time to read messages. (msg_size_affects_delay: {self.msg_size_affects_delay})')
        if self.maximum_typing_speed > 0:
            log.info(f'• Takes longer to write responses. (maximum_typing_speed: {self.maximum_typing_speed})')

    # Response delays
    def build_response_delay_weights(self):
        responsiveness = max(0.0, min(1.0, self.responsiveness)) # clamped between 0.0 and 1.0
        if responsiveness == 1.0:
            return
        num_values = 10 # arbitrary number of values and weights to generate
        # Generate evenly spaced values from 0 to max_reply_delay
        self._response_delay_values = [round(i * self.max_reply_delay / (num_values - 1), 3) for i in range(num_values)]
        # Generate the weights from responsiveness (inverted)
        inv_responsiveness = (1.0 - responsiveness)
        self._response_delay_weights = get_normalized_weights(target = inv_responsiveness, list_len = num_values)

    def build_text_delay_values(self):
        responsiveness = max(0.0, min(1.0, self.responsiveness)) # clamped between 0.0 and 1.0
        if responsiveness == 1.0 or not self.msg_size_affects_delay:
            return
        # Manipulate the possible text delays to something more reasonable, while still rooted in 'responsiveness'
        self._text_delay_values = copy.deepcopy(self._response_delay_values)
        self._text_delay_values.pop(0) # Remove "0" delay
        # Remove second half of delay values
        # num_values = len(text_values) // 2
        # self.text_delay_values = text_values[:num_values]

    # Currently not used...
    def merge_weights(self, text_weights:list) -> list:
        # Combine text weights with delay weights
        combined_weights = [w1 + w2 for w1, w2 in zip(self._response_delay_weights, text_weights)]
        # Normalize combined weights to sum up to 1.0
        total_combined_weight = sum(combined_weights)
        merged_weights = [weight / total_combined_weight for weight in combined_weights]
        return merged_weights

    def set_response_delay(self) -> float:
        # No delay if bot is online or user config is max responsiveness
        if bot_status.online or self.responsiveness >= 1.0:
            bot_status.current_response_delay = 0
            return 0
        # Choose a delay if none currently set
        if bot_status.current_response_delay is None:
            chosen_delay = random.choices(self._response_delay_values, self._response_delay_weights)[0]
            bot_status.current_response_delay = chosen_delay
        return bot_status.current_response_delay

    def get_text_delay(self, text:str) -> float:
        if not self.msg_size_affects_delay:
            return 0
        # calculate text_weights for message text
        num_values = len(self._text_delay_values)
        text_len = len(text)
        text_factor = min(max(text_len / 450, 0.0), 1.0)  # Normalize text length to [0, 1]
        text_delay_weights = get_normalized_weights(target = text_factor, list_len = num_values, strength=2.0) # use stronger weights for text factor
        # randomly select and return text delay
        chosen_delay = (random.choices(self._text_delay_values, text_delay_weights)[0]) / 2 # Halve it
        log.debug(f"Read Text delay: {chosen_delay}. Chosen from range: {self._text_delay_values}")
        return chosen_delay


    # Active conversations
    def update_user_dict(self, user_id):
        # Update the last conversation time for a user
        self._user_conversations[user_id] = datetime.now()

    def in_active_conversation(self, user_id) -> bool:
        # Check if a user is in an active conversation with the bot
        last_conversation_time = self._user_conversations.get(user_id)
        if last_conversation_time:
            time_since_last_conversation = datetime.now() - last_conversation_time
            return time_since_last_conversation.total_seconds() < self.conversation_recency
        return False


    # Checks if bot should reply to a message
    def bot_should_reply(self, message:discord.Message, text:str, last_character:str) -> bool:
        main_condition = is_direct_message(message) or (message.channel.id in bot_database.main_channels)

        if client.waiting_for.get(message.author.id):
            return False
        if is_direct_message(message) and not config.discord['direct_messages'].get('allow_chatting', True):
            return False
        # Don't reply to @everyone
        if message.mention_everyone:
            return False
        # Only reply to itself if configured to
        if message.author == client.user and not check_probability(self.reply_to_itself):
            return False
        # Whether to reply to other bots
        if message.author.bot and re.search(rf'\b{re.escape(last_character.lower())}\b', text, re.IGNORECASE) and main_condition:
            if 'bye' in text.lower(): # don't reply if another bot is saying goodbye
                return False
            return check_probability(self.reply_to_bots_when_addressed)
        # Whether to reply when text is nested in parentheses
        if self.ignore_parentheses and (message.content.startswith('(') and message.content.endswith(')')) or (message.content.startswith('<:') and message.content.endswith(':>')):
            return False
        # Whether to reply if only speak when spoken to
        if (self.only_speak_when_spoken_to and (client.user.mentioned_in(message) or any(word in message.content.lower() for word in last_character.lower().split()))) \
            or (self.in_active_conversation(message.author.id) and main_condition):
            return True
        reply = False
        # few more conditions
        if message.author.bot and main_condition:
            reply = check_probability(self.chance_to_reply_to_other_bots)
        if self.go_wild_in_channel and main_condition:
            reply = True
        if reply:
            self.update_user_dict(message.author.id)
        return reply

    def probability_to_reply(self, probability) -> bool:
        probability = max(0.0, min(1.0, probability))
        # Determine if the bot should reply based on a probability
        return random.random() < probability

# Base class is intended to handle unknown clients. Payload management will rely on '__overrides__' injection method only.
# Subclasses for known clients will manage payloads and settings more expectedly.
class ImgModel(SettingsBase):
    def __init__(self):
        self._imgmodel_update_task:Optional[asyncio.Task] = None
        self._guild_id = None
        self._guild_name = None
        # Convenience keys
        get_imgmodels_ep = getattr(api.imggen, 'get_imgmodels', None)
        post_options_ep = getattr(api.imggen, 'post_options', None)

        self._name_key:str = getattr(get_imgmodels_ep, 'imgmodel_name_key', '')
        self._value_key:str = getattr(get_imgmodels_ep, 'imgmodel_value_key', '')
        self._filename_key:str = getattr(get_imgmodels_ep, 'imgmodel_filename_key', '')
        self._imgmodel_input_key:Optional[str] = getattr(post_options_ep, 'imgmodel_input_key', None)
        # database-like
        self.last_imgmodel_name:str = ''
        self.last_imgmodel_value:str = ''
        self.last_imgmodel_res:int = 1024
        # Override base values
        self.payload_mods:dict = {}
        self.tags:TAG_LIST = []

    def get_any_imgmodel_key(self, priority:str='name') -> str:
        # Look for the first non-empty value in priority order
        keys = ['name', 'value', 'filename']
        if priority != 'name':
            keys.remove(priority)
            keys.insert(0, priority)
        for k in keys:
            value = getattr(self, f'_{k}_key', '')
            if value:
                return value
        return ''

    def clean_payload(self, payload:dict):
        pass

    def fix_update_values(self, updates:dict):
        pass

    def handle_payload_updates(self, updates:dict, task:"Task") -> dict:
        self.fix_update_values(updates)
        task.vars.update_from_dict(updates)

    # Update vars and __overrides__ dict in user's default payload with any imgmodel payload mods
    def override_payload(self, task:"Task") -> dict:
        return self.handle_payload_updates(self.payload_mods, task)
    
    def apply_final_prompts_for_task(self, task:"Task"):
        active_ep = task.params.get_active_imggen_ep()
        prompt_key, neg_prompt_key = active_ep.get_prompt_keys()
        if prompt_key:
            set_key(data=task.payload, path=prompt_key, value=task.prompt)
        if neg_prompt_key:
            set_key(data=task.payload, path=neg_prompt_key, value=task.neg_prompt)

    def apply_imgcmd_params(self, task:"Task"):
        # Just update vars for unknown clients (expected to be injected via __overrides__)
        task.update_vars_from_imgcmd()

    def apply_payload_param_variances(self, updates:dict, task:"Task"):
        self.fix_update_values(updates)
        summed_updates = sum_update_dict(vars(task.vars), updates, in_place=False, updates_only=True, merge_unmatched=False)
        task.vars.update_from_dict(summed_updates)

    def apply_controlnet(self, controlnet, task:"Task"):
        if not bot_database.was_warned('controlnet_unsupported'):
            log.warning("[TAGS] ControlNet was triggered, but is currently only supported for A1111-like Web UIs (A1111/Forge/ReForge)")
            bot_database.update_was_warned('controlnet_unsupported')
    def apply_forge_couple(self, forge_couple, task:"Task"):
        if not bot_database.was_warned('forge_couple_unsupported'):
            log.warning("[TAGS] Forge Couple was triggered, but is currently only supported for SD Forge")
            bot_database.update_was_warned('forge_couple_unsupported')
    def apply_layerdiffuse(self, layerdiffuse, task:"Task"):
        if not bot_database.was_warned('layerdiffuse_unsupported'):
            log.warning("[TAGS] Layerdiffuse was triggered, but is currently only supported for SD Forge")
            bot_database.update_was_warned('layerdiffuse_unsupported')
    def apply_reactor(self, reactor, task:"Task"):
        if not bot_database.was_warned('reactor_unsupported'):
            log.warning("[TAGS] ReActor was triggered, but is currently only supported for A1111-like Web UIs (A1111/Forge/ReForge)")
            bot_database.update_was_warned('reactor_unsupported')
    
    async def handle_image_bytes(self, image:bytes, file_type: Optional[str] = None, filename: Optional[str] = None) -> str:
        if not api.imggen.post_upload:
            return base64.b64encode(image).decode('utf-8')
        else:
            await api.imggen.post_upload.upload_files(input_data=image, file_name=filename, file_obj_key=file_type)
            return filename

    async def handle_image_input(self, source: Union[discord.Attachment, bytes], file_type: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        If post_upload endpoint is enabled, processes/uploads the file.
        Otherwise, converts to base64.

        Returns:
            str: Either the filename of the uploaded file, or a base64-encoded string if not uploaded.
        """
        if isinstance(source, discord.Attachment):
            file_bytes = await source.read()
            filename = source.filename
        elif isinstance(source, bytes):
            file_bytes = source
            if not filename:
                raise ValueError("Filename must be provided when passing bytes.")
        else:
            raise TypeError("Unsupported image input type. Expected discord.Attachment or bytes.")
        
        return await self.handle_image_bytes(file_bytes, file_type, filename)

    def get_sampler_and_scheduler_mapping(self) -> dict:
        return {}
    
    def normalize_sampler_name(self, name:str) -> str:
        normalized = name.strip().lower().replace("+", "p").replace(" ", "_")
        if normalized.startswith("k_"):
            normalized = normalized[2:]
        return normalized

    def check_sampler_or_scheduler_value(self, value: str) -> str:
        valid_values = [v.lower() for v in api.imggen._sampler_names + api.imggen._schedulers]
        if not valid_values:
            return value
        normalized = self.normalize_sampler_name(value)
        if normalized in valid_values:
            return normalized
        mapping = self.get_sampler_and_scheduler_mapping()
        return mapping.get(normalized, value)

    def parse_lora_matches(self, text: str) -> list[tuple[str, str, str, float, str]]:
        """
        Returns list of (full_tag, original_name, strength_str, strength_float, resolved_name)
        """
        matches = []
        if not api.imggen._lora_names:
            return matches

        lora_matches = patterns.sd_lora_split.findall(text)
        if not lora_matches:
            return matches

        valid_lora_names = api.imggen._lora_names

        for name, strength_str in lora_matches:
            try:
                strength = float(strength_str)
            except ValueError:
                continue
            stripped = name.strip()
            resolved = next((v for v in valid_lora_names if stripped in v), stripped)
            full_tag = f"<lora:{name}:{strength_str}>"
            matches.append((full_tag, name, strength_str, strength, resolved))
        return matches

    def handle_loras(self, text: str, task: "Task") -> str:
        for full_tag, _, _, strength, resolved_name in self.parse_lora_matches(text):
            task.vars.add_lora(resolved_name, strength)
            text = text.replace(full_tag, '')
        return text

    def collect_model_names(self, imgmodels:list):
        if not imgmodels:
            return []
        first = imgmodels[0]
        if isinstance(first, dict):
            display_key = self.get_any_imgmodel_key()
            return [i[display_key] for i in imgmodels]
        elif isinstance(first, str):
            return imgmodels
        else:
            raise ValueError("Unsupported element type in imgmodels list")
    
    async def get_model_params(self, imgmodel:str|dict, mode:str='change', verb:str='Changing') -> dict:
        if isinstance(imgmodel, str):
            params = await self.get_imgmodel_data(imgmodel)
        elif isinstance(imgmodel, dict):
            params = {self._name_key: imgmodel.get(self._name_key, ''),
                      self._value_key: imgmodel.get(self._value_key, '')}
        else:
            raise ValueError(f'Unsupported imgmodel value: {imgmodel}')
        params['mode'] = mode
        params['verb'] = verb
        return params

    def apply_filters(self, items: list[Union[str, dict]], filter_list: list[str], exclude_list: list[str]) -> list:
        def item_matches(text: str) -> bool:
            # Check inclusion
            if filter_list and not any(re.search(re.escape(f), text, re.IGNORECASE) for f in filter_list):
                return False
            # Check exclusion
            if exclude_list and any(re.search(re.escape(e), text, re.IGNORECASE) for e in exclude_list):
                return False
            return True

        def get_text(item: Union[str, dict]) -> str:
            if isinstance(item, str):
                return item
            elif isinstance(item, dict):
                return item.get(self._name_key, '') + item.get(self._value_key, '')
            else:
                raise ValueError(f"Unsupported item type: {type(item)}")

        return [item for item in items if item_matches(get_text(item))]

    # Apply user defined filters to imgmodel list
    async def filter_imgmodels(self, all_imgmodels:list, ictx:CtxInteraction=None) -> list:
        filtered_imgmodels = all_imgmodels
        try:
            imgmodels_data = load_file(shared_path.img_models, {})
            # Apply global filters
            global_filters = imgmodels_data.get('settings', {}).get('filter', [])
            global_excludes = imgmodels_data.get('settings', {}).get('exclude', [])
            filtered_imgmodels = self.apply_filters(all_imgmodels, global_filters, global_excludes)
            # Apply per-server filters
            per_server_filters = imgmodels_data.get('settings', {}).get('per_server_filters', [])
            gid = self._guild_id or (ictx.guild.id if ictx and ictx.guild else None)
            if gid:
                for preset in per_server_filters:
                    preset:dict
                    if gid == preset.get('guild_id'):
                        filtered_imgmodels = self.apply_filters(filtered_imgmodels, preset.get('filter', []), preset.get('exclude', []))
                        break
        except Exception as e:
            log.error(f"Error filtering image model list: {e}")
        return filtered_imgmodels

    # Get and filter a current list of imgmodels from API
    async def get_filtered_imgmodels_list(self, ictx:CtxInteraction=None) -> list:
        all_imgmodels = []
        try:
            all_imgmodels = await api.imggen.fetch_imgmodels()
            return await self.filter_imgmodels(all_imgmodels, ictx)
        except Exception as e:
            log.error(f"Error fetching image models: {e}")
        return all_imgmodels

    # Check filesize/filters with selected imgmodel to assume resolution / tags
    async def guess_model_data(self, imgmodel_data: dict, presets: list[dict]) -> dict | None:
        try:
            imgmodel_name = imgmodel_data.get(self._name_key) or ''
            imgmodel_value = imgmodel_data.get(self._value_key) or ''
            imgmodel = imgmodel_value or imgmodel_name
            match_counts = []

            for preset in presets:
                exact_match = preset.get('exact_match')
                if exact_match and exact_match in [imgmodel_name, imgmodel_value]:
                    log.info(f'Applying exact match imgmodel preset for "{exact_match}".')
                    return preset

                filter_list = [f for f in preset.get('filter', []) if f.strip()]
                exclude_list = [e for e in preset.get('exclude', []) if e.strip()]
                match_count = 0

                for filter_text in filter_list:
                    if re.search(re.escape(filter_text), imgmodel, re.IGNORECASE):
                        match_count += 1

                for exclude_text in exclude_list:
                    if re.search(re.escape(exclude_text), imgmodel, re.IGNORECASE):
                        match_count -= 1

                match_counts.append((preset, match_count))

            match_counts.sort(key=lambda x: x[1], reverse=True)
            matched_preset = match_counts[0][0] if match_counts else None
            return matched_preset
        except Exception as e:
            log.error(f"Error guessing selected imgmodel data: {e}")
            return {}

    def clean_options_before_saving(self, options:dict) -> dict:
        return options

    # Save new Img model data
    async def save_new_imgmodel_options(self, ictx:CtxInteraction, new_imgmodel_options:dict, imgmodel_tags):
        # Update settings
        self.payload_mods = self.clean_options_before_saving(new_imgmodel_options)
        self.tags = imgmodel_tags

        # Check if old/new average resolution is different
        if self.payload_mods.get('width') and self.payload_mods.get('height'):
            new_avg = avg_from_dims(self.payload_mods['width'], self.payload_mods['height'])
            if new_avg != self.last_imgmodel_res or new_avg != bot_database.last_imgmodel_res: # legacy check
                self.last_imgmodel_res = new_avg
                bot_database.last_imgmodel_res = new_avg
                # update /image cmd res options
                await bg_task_queue.put(update_size_options(new_avg))

        settings:Settings = get_settings(ictx)
        # Fix any invalid settings
        settings.fix_settings()
        if ictx:
            # Save file. Don't save from auto-select imgmodels task.
            settings.save()

    # subclass behavior
    def get_extra_settings(self, imgmodel_data:dict) -> dict:
        return {}
    
    def collect_extra_preset_data(self, matched_preset:dict):
        pass

    # Merge selected imgmodel/tag data with base settings
    async def update_imgmodel_options(self, imgmodel_data:dict) -> Tuple[dict, list]:
        imgmodel_options = {}
        imgmodel_tags = []
        try:
            # Get extra model information
            imgmodels_data = load_file(shared_path.img_models, {})
            if imgmodels_data.get('settings', {}).get('auto_change_imgmodels', {}).get('guess_model_params', True):
                imgmodel_presets = copy.deepcopy(imgmodels_data.get('presets', []))
                matched_preset = await self.guess_model_data(imgmodel_data, imgmodel_presets)
                if matched_preset:
                    imgmodel_tags = matched_preset.pop('tags', [])
                    imgmodel_options = matched_preset.get('payload', {})
                    self.collect_extra_preset_data(matched_preset)
            # Unpack any tag presets
            imgmodel_tags = base_tags.update_tags(imgmodel_tags)
        except Exception as e:
            log.error(f"Error merging selected imgmodel data with base imgmodel data: {e}")
        return imgmodel_options, imgmodel_tags

    async def post_options(self, options_payload:dict):
        try:
            await api.imggen.post_options.call(input_data=options_payload, sanitize=True)
        except Exception as e:
            log.error(f"Error posting updated imgmodel settings to API: {e}")
            raise

    async def change_imgmodel(self, imgmodel_data:dict, ictx:CtxInteraction=None, save:bool=True) -> dict:
        # Retain model details
        self.last_imgmodel_name = imgmodel_data.get(self._name_key, '')
        self.last_imgmodel_value = imgmodel_data.get(self._value_key, '')

        # Guess model params, merge with basesettings
        imgmodel_options, imgmodel_tags = await self.update_imgmodel_options(imgmodel_data)

        # Collect options payload
        options_payload = api.imggen.post_options.get_payload() if getattr(api.imggen, 'post_options', None) else {}
        if self._imgmodel_input_key:
            value_key = self.get_any_imgmodel_key(priority='value')
            options_payload[self._imgmodel_input_key] = imgmodel_data[value_key]

        # Factors extras (override_settings, etc) if applicable
        options_payload.update(self.get_extra_settings(imgmodel_options))

        # Post new settings to API (Per-server must be payload-driven)
        if not config.is_per_server_imgmodels():
            await self.post_options(options_payload)

        # Save settings
        if save:
            await self.save_new_imgmodel_options(ictx, imgmodel_options, imgmodel_tags)
            # Restart auto-change imgmodel task if triggered by a user interaction
            if ictx:
                if self._imgmodel_update_task and not self._imgmodel_update_task.done():
                    self._imgmodel_update_task.cancel()
                    await bg_task_queue.put(self.start_auto_change_imgmodels('restarted'))

        return options_payload

    async def get_imgmodel_data(self, imgmodel_value:str, ictx:CtxInteraction=None) -> dict:
        try:
            imgmodel_data = {}
            all_imgmodels = await self.get_filtered_imgmodels_list(ictx)
            for imgmodel in all_imgmodels:
                # check that the value matches a valid checkpoint
                if imgmodel_value == (imgmodel.get(self._name_key) or imgmodel.get(self._value_key)):
                    imgmodel_data = {k: imgmodel[k]
                                    for k in (self._value_key, self._name_key, self._filename_key)
                                    if k and k in imgmodel}
                    break
            if not imgmodel_data:
                log.error(f'Img model not found: {imgmodel_value}')
            return imgmodel_data
        except Exception as e:
            log.error(f"Error getting selected imgmodel data: {e}")
            return {}

    ## Function to automatically change image models
    # Select imgmodel based on mode, while avoid repeating current imgmodel
    async def auto_select_imgmodel(self, mode='random') -> str|dict:
        try:
            all_imgmodels:list = await self.get_filtered_imgmodels_list()
            all_imgmodel_names = self.collect_model_names(all_imgmodels)

            current_index = None
            if self.last_imgmodel_name and self.last_imgmodel_name in all_imgmodel_names:
                current_index = all_imgmodel_names.index(self.last_imgmodel_name)

            if mode == 'random':
                if current_index is not None and len(all_imgmodels) > 1:
                    all_imgmodels.pop(current_index)
                return random.choice(all_imgmodels)

            elif mode == 'cycle':
                if current_index is not None:
                    next_index = (current_index + 1) % len(all_imgmodel_names)  # Cycle to the beginning if at the end
                    return all_imgmodels[next_index]
                else:
                    log.info("[Auto Change Imgmodels] Previous imgmodel name was not matched in list of fetched imgmodels.")
                    log.info("[Auto Change Imgmodels] New imgmodel was selected at random instead of 'cycle'.")
                    return random.choice(all_imgmodels) # If no image model set yet, select randomly
        except Exception as e:
            log.error(f"[Auto Change Imgmodels] Error selecting image model: {e}")

    # Task to auto-select an imgmodel at user defined interval
    async def auto_update_imgmodel_task(self, mode, duration):
        while True:
            try:
                await asyncio.sleep(duration)
                # Select an imgmodel automatically
                imgmodel = await self.auto_select_imgmodel(mode)
                imgmodel_params = await self.get_model_params(imgmodel=imgmodel)
                # CREATE TASK AND QUEUE IT
                change_imgmodel_task = Task('change_imgmodel', ictx=None, params=Params(imgmodel=imgmodel_params))
                await task_manager.queue_task(change_imgmodel_task, 'gen_queue')

            except Exception as e:
                log.error(f"[Auto Change Imgmodels] Error updating image model: {e}")

    # helper function to begin auto-select imgmodel task
    async def start_auto_change_imgmodels(self, status:str='Started'):
        try:
            # load imgmodel settings file and read config
            imgmodels_data:dict = load_file(shared_path.img_models, {})
            auto_change_settings = imgmodels_data.get('settings', {}).get('auto_change_imgmodels', {})
            mode = auto_change_settings.get('mode', 'random')
            frequency = auto_change_settings.get('frequency', 1.0)
            duration = frequency*3600 # 3600 = 1 hour

            self._imgmodel_update_task = client.loop.create_task(self.auto_update_imgmodel_task(mode, duration))

            guild_msg = f", Guild: {self._guild_name}" if self._guild_name else ''

            log.info(f"[Auto Change Imgmodels] {status} (Mode: '{mode}', Frequency: {frequency} hrs{guild_msg}).")
        except Exception as e:
            log.error(f"[Auto Change Imgmodels] Error starting task: {e}")

    # From change_imgmodel_task
    async def change_imgmodel_task(self, task:"Task"):
        print_model_name:str = task.params.imgmodel.get(self.get_any_imgmodel_key(priority='name'))
        imgmodel_value:str = task.params.imgmodel.get(self.get_any_imgmodel_key(priority='value'))

        if not imgmodel_value:
            await task.embeds.send('change', 'Failed to change Img model:', f'Img model not found: {print_model_name}')
            return False

        try:
            # Send embed
            if not task.ictx:
                task.user_name = 'Automatically'
            mode = task.params.imgmodel.pop('mode', 'change')    # default to 'change
            verb = task.params.imgmodel.pop('verb', 'Changing')  # default to 'Changing'
            await task.embeds.send('change', f'{verb} Img model ... ', f'{verb} to {print_model_name}')

            # Swap Image model
            if mode == 'swap' or mode == 'swap_back':
                new_options = await self.change_imgmodel(task.params.imgmodel, task.ictx, save=False)
                await task.embeds.delete('change') # delete embed
                return new_options

            # Change Image model
            await self.change_imgmodel(task.params.imgmodel, task.ictx)

            # Announce change
            await task.embeds.delete('change') # delete any embed
            await task.embeds.send('change', f"{task.user_name} changed Img model:", f'**{print_model_name}**')
            if bot_database.announce_channels and task.embeds.enabled('change'):
                # Send embeds to announcement channels
                await bg_task_queue.put(announce_changes('changed Img model', print_model_name, task.ictx))

            log.info(f"Image model changed to: {print_model_name}")
            if config.discord['post_active_settings'].get('enabled', True):
                settings_keys = ['imgmodel']
                # Auto-change imgmodel task will not have an interaction
                if task.ictx and config.is_per_server():
                    await bg_task_queue.put(post_active_settings(task.ictx.guild.id, settings_keys))
                else:
                    await bg_task_queue.put(post_active_settings_to_all(settings_keys))

        except Exception as e:
            log.error(f"Error changing Img model: {e}")
            await task.embeds.edit_or_send('change', "An error occurred while changing Img model", e)
            traceback.print_exc()
            return False

class ImgModel_Comfy(ImgModel):
    def __init__(self):
        super().__init__()
        # Convenience keys
        self._name_key:str = 'model_name'
        self._value_key:str = 'title'
        self._filename_key:str = ''
        self._imgmodel_input_key = None

        self.delete_nodes:list[str] = []

    def clean_payload(self, payload: dict):
        if self.delete_nodes:
            comfy_delete_and_reroute_nodes(payload, self.delete_nodes)

    async def post_options(self, options_payload:dict):
        pass

    def collect_extra_preset_data(self, matched_preset:dict):
        self.delete_nodes = matched_preset.get('comfy_delete_nodes', [])

    async def get_imgmodel_data(self, imgmodel_value:str, ictx:CtxInteraction=None) -> dict:
        return {self._name_key: imgmodel_value,
                self._value_key: imgmodel_value}

    def apply_final_prompts_for_task(self, task:"Task"):
        task.update_vars()

    def apply_imgcmd_params(self, task:"Task"):
        imgcmd_vars = {}
        imgcmd_params = task.params.imgcmd
        if imgcmd_params.get('size'):
            imgcmd_vars['width'] = imgcmd_params['size']['width']
            imgcmd_vars['height'] = imgcmd_params['size']['height']
        if imgcmd_params.get('img2img'):
            if imgcmd_params['img2img'].get('image'):
                imgcmd_vars['i2i_image'] = imgcmd_params['img2img']['image']
            if imgcmd_params['img2img'].get('mask'):
                imgcmd_vars['i2i_mask'] = imgcmd_params['img2img']['mask']
            if imgcmd_params['img2img'].get('denoising_strength'):
                imgcmd_vars['denoising_strength'] = imgcmd_params['img2img']['denoising_strength']
        if imgcmd_params.get('cnet_dict'):
            # TODO support ControlNet in /image cmd for ComfyUI
            log.warning("ControlNet not yet supported for ComfyUI via /image command")
            # cnet_dict = imgcmd_params['cnet_dict']
            # for key, value in cnet_dict.items():
            #     attr_name = f'cnet_{key}'
            #     imgcmd_vars[attr_name] = value
        if imgcmd_params.get('face_swap'):
            imgcmd_vars['face_swap'] = imgcmd_params['face_swap']

        task.vars.update_from_dict(imgcmd_vars)

    def get_sampler_and_scheduler_mapping(self) -> dict:
        return {"dpmpp_2m_sde_heun": "dpmpp_2m_sde",
                "ddim_cfgpp": "ddim",
                "plms": "lms",
                "unipc": "uni_pc",
                "restart": "res_multistep",
                "sgmuniform": "sgm_uniform",
                "ddim": "ddim_uniform",
                "uniform": "ddim_uniform"}

    def fix_update_values(self, updates: dict):
        for k, v in updates.items():
            if k in ['sampler_name', 'scheduler'] and isinstance(v, str):
                updates[k] = self.check_sampler_or_scheduler_value(v)

    def handle_payload_updates(self, updates:dict, task:"Task") -> dict:
        self.fix_update_values(updates)
        task.vars.update_from_dict(updates)

class ImgModel_Swarm(ImgModel):
    def __init__(self):
        super().__init__()
        # Convenience keys
        self._name_key:str = 'model_name'
        self._value_key:str = 'title'
        self._filename_key:str = ''
        self._imgmodel_input_key:str = 'model'

    def clean_payload(self, payload):
        # resolves duplicate negatives while preserving order
        payload['negativeprompt'] = consolidate_prompt_strings(payload.get('negativeprompt', ''))
        payload['model'] = self.last_imgmodel_value
        payload['donotsaveintermediates'] = True
        payload['images'] = 1

    def handle_loras(self, text: str, task: "Task") -> str:
        for full_tag, _, strength_str, _, resolved_name in self.parse_lora_matches(text):
            # Replace the tag with an updated version where the name is replaced
            updated_tag = f"<lora:{resolved_name}:{strength_str}>"
            text = text.replace(full_tag, updated_tag)
        return text

    async def post_options(self, options_payload:dict):
        payload = {'model': options_payload['model']}
        response = await api.imggen.post_options.call(input_data=payload)
        if response and response.get('error'):
            raise Exception(response['error'])

    async def get_imgmodel_data(self, imgmodel_value:str, ictx:CtxInteraction=None) -> dict:
        return {self._name_key: imgmodel_value,
                self._value_key: imgmodel_value}

    def apply_final_prompts_for_task(self, task:"Task"):
        task.payload['prompt'] = task.prompt
        task.payload['negativeprompt'] = task.neg_prompt

    def apply_imgcmd_params(self, task:"Task"):
        try:
            imgcmd_params = task.params.imgcmd
            size: Optional[dict]       = imgcmd_params['size']
            face_swap :Optional[str]   = imgcmd_params['face_swap']
            controlnet: Optional[dict] = imgcmd_params['controlnet']
            img2img: dict              = imgcmd_params['img2img']
            img2img_mask               = img2img.get('mask', '')

            if img2img:
                task.payload['initimage'] = img2img['image']
                task.payload['initimagecreativity'] = img2img['denoising_strength']
            if img2img_mask:
                task.payload['maskimage'] = img2img_mask
            if size:
                task.payload.update(size)
        except Exception as e:
            log.error(f"Error applying imgcmd params: {e}")

    def get_sampler_and_scheduler_mapping(self) -> dict:
        return {"euler_a": "euler_ancestral",
                "dpmpp_2m_sde_heun": "dpmpp_2m_sde",
                "ddim_cfgpp": "ddim",
                "plms": "lms",
                "unipc": "uni_pc",
                "restart": "res_multistep",
                "sgmuniform": "sgm_uniform",
                "ddim": "ddim_uniform",
                "uniform": "ddim_uniform",
                "polyexponential": "exponential",
                "align_your_steps_gits": "align_your_steps",
                "align_your_steps_11": "align_your_steps",
                "align_your_steps_32": "align_your_steps"}

    async def handle_image_bytes(self, image:bytes, file_type: Optional[str] = None, filename: Optional[str] = None) -> str:
        return base64.b64encode(image).decode('utf-8')
    
    def fix_update_values(self, updates: dict):
        key_map = {'cfg_scale': 'cfgscale',
                   'negative_prompt': 'negativeprompt',
                   'CLIP_stop_at_last_layers': 'clipstopatlayer',
                   'sd_vae': 'vae',
                   'distilled_cfg_scale': 'fluxguidancescale',
                   'denoising_strength': 'initimagecreativity',
                   'sampler_name': 'sampler'}
        for old_key, new_key in key_map.items():
            if old_key in updates:
                updates[new_key] = updates.pop(old_key)
        for key in ['sampler', 'scheduler']:
            if key in updates:
                updates[key] = self.check_sampler_or_scheduler_value(updates[key])

    def handle_payload_updates(self, updates:dict, task:"Task"):
        self.fix_update_values(updates)
        update_dict(task.payload, updates)

    def apply_payload_param_variances(self, updates:dict, task:"Task"):
        self.fix_update_values(updates)
        sum_update_dict(task.payload, updates)


class ImgModel_SDWebUI(ImgModel):
    def __init__(self):
        super().__init__()
        # Convenience keys
        self._name_key:str = 'model_name'
        self._value_key:str = 'title'
        self._filename_key:str = 'filename'
        self._imgmodel_input_key:str = 'sd_model_checkpoint'

    async def handle_image_bytes(self, image:bytes, file_type: Optional[str] = None, filename: Optional[str] = None) -> str:
        return base64.b64encode(image).decode('utf-8')
    
    def fix_update_values(self, updates: dict):
        for k, v in updates.items():
            if k in ['sampler_name', 'scheduler'] and isinstance(v, str):
                updates[k] = self.check_sampler_or_scheduler_value(v)

    def handle_payload_updates(self, updates:dict, task:"Task"):
        self.fix_update_values(updates)
        update_dict(task.payload, updates)

    def apply_payload_param_variances(self, updates:dict, task:"Task"):
        self.fix_update_values(updates)
        sum_update_dict(task.payload, updates)

    def apply_final_prompts_for_task(self, task:"Task"):
        task.payload['prompt'] = task.prompt
        task.payload['negative_prompt'] = task.neg_prompt

    def apply_imgcmd_params(self, task:"Task"):
        try:
            imgcmd_params = task.params.imgcmd
            size: Optional[dict]       = imgcmd_params['size']
            face_swap :Optional[str]   = imgcmd_params['face_swap']
            controlnet: Optional[dict] = imgcmd_params['controlnet']
            img2img: dict              = imgcmd_params['img2img']
            img2img_mask               = img2img.get('mask', '')

            if img2img:
                task.payload['init_images'] = [img2img['image']]
                task.payload['denoising_strength'] = img2img['denoising_strength']
            if img2img_mask:
                task.payload['mask'] = img2img_mask
            if size:
                task.payload.update(size)
            if face_swap or controlnet:
                alwayson_scripts:dict = task.payload.setdefault('alwayson_scripts', {})
                if face_swap:
                    alwayson_scripts.setdefault('reactor', {}).setdefault('args', {})['image'] = face_swap # image in base64 format
                    alwayson_scripts['reactor']['args']['enabled'] = True # Enable
            if controlnet:
                cnet_dict = alwayson_scripts.setdefault('controlnet', {})
                cnet_args = cnet_dict.setdefault('args', [])
                if len(cnet_args) == 0:
                    cnet_args.append({})
                cnet_args[0].update(controlnet)
        except Exception as e:
            log.error(f"Error applying imgcmd params: {e}")

    def clean_payload(self, payload:dict):
        try:
            # resolves duplicate negatives while preserving order
            payload['negative_prompt'] = consolidate_prompt_strings(payload.get('negative_prompt', ''))
            ## Clean up extension keys
            # get alwayson_scripts dict
            alwayson_scripts:dict = payload.get('alwayson_scripts', {})
            # Clean ControlNet
            if alwayson_scripts.get('controlnet'):
                # Delete all 'controlnet' keys if disabled
                if not config.controlnet_enabled():
                    del alwayson_scripts['controlnet']
                else:
                    # Delete all 'controlnet' keys if empty
                    if not alwayson_scripts['controlnet']['args']:
                        del alwayson_scripts['controlnet']
                    # Compatibility fix for 'resize_mode' and 'control_mode'
                    else:
                        for index, cnet_module in enumerate(copy.deepcopy(alwayson_scripts['controlnet']['args'])):
                            cnet_enabled = cnet_module.get('enabled', False)
                            if not cnet_enabled:
                                del alwayson_scripts['controlnet']['args'][index]
                                continue
                            resize_mode = cnet_module.get('resize_mode')
                            if resize_mode is not None and isinstance(resize_mode, int):
                                resize_mode_string = 'Just Resize' if resize_mode == 0 else 'Crop and Resize' if resize_mode == 1 else 'Resize and Fill'
                                alwayson_scripts['controlnet']['args'][index]['resize_mode'] = resize_mode_string
                            control_mode = cnet_module.get('control_mode')
                            if control_mode is not None and isinstance(control_mode, int):
                                cnet_mode_str = 'Balanced' if control_mode == 0 else 'My prompt is more important' if control_mode == 1 else 'ControlNet is more important'
                                alwayson_scripts['controlnet']['args'][index]['control_mode'] = cnet_mode_str
            # Clean Forge Couple
            if alwayson_scripts.get('forge_couple'):
                # Delete all 'forge_couple' keys if disabled by config
                if not config.forgecouple_enabled() or payload.get('init_images'):
                    del alwayson_scripts['forge_couple']
                else:
                    # convert dictionary to list
                    if isinstance(payload['alwayson_scripts']['forge_couple']['args'], dict):
                        payload['alwayson_scripts']['forge_couple']['args'] = list(payload['alwayson_scripts']['forge_couple']['args'].values())
                    # Add the required space between "forge" and "couple" ("forge couple")
                    payload['alwayson_scripts']['forge couple'] = payload['alwayson_scripts'].pop('forge_couple')
            # Clean layerdiffuse
            if alwayson_scripts.get('layerdiffuse'):
                # Delete all 'layerdiffuse' keys if disabled by config
                if not config.layerdiffuse_enabled():
                    del alwayson_scripts['layerdiffuse']
                # convert dictionary to list
                elif isinstance(payload['alwayson_scripts']['layerdiffuse']['args'], dict):
                    payload['alwayson_scripts']['layerdiffuse']['args'] = list(payload['alwayson_scripts']['layerdiffuse']['args'].values())
            # Clean ReActor
            if alwayson_scripts.get('reactor'):
                # Delete all 'reactor' keys if disabled by config
                if not config.reactor_enabled():
                    del alwayson_scripts['reactor']
                # convert dictionary to list
                elif isinstance(payload['alwayson_scripts']['reactor']['args'], dict):
                    payload['alwayson_scripts']['reactor']['args'] = list(payload['alwayson_scripts']['reactor']['args'].values())

        except Exception as e:
            log.error(f"An error occurred when cleaning imggen payload: {e}")

    # Manage override_settings for A1111-like APIs. returns override_settings
    def get_extra_settings(self, imgmodel_data:dict) -> dict:
        override_settings = {}
        # Set defaults
        override_settings:dict = imgmodel_data.setdefault('override_settings', {})
        # For per-server imgmodels, only the image request payload will drive model changes (won't change now via API)
        if config.is_per_server_imgmodels():
            override_settings[self._value_key] = imgmodel_data[self._value_key]
        return override_settings

    def apply_controlnet(self, controlnet, task: "Task"):
        task.payload['alwayson_scripts']['controlnet']['args'] = controlnet

    def apply_reactor(self, reactor, task: "Task"):
        task.payload['alwayson_scripts']['reactor']['args'].update(reactor)
        if reactor.get('mask'):
            task.payload['alwayson_scripts']['reactor']['args']['save_original'] = True

    def get_sampler_and_scheduler_mapping(self) -> dict:
        return {'euler_cfg_pp': 'k_euler',
                'euler_ancestral_cfg_pp': 'k_euler_ancestral',
                'dpm_2_ancestral': 'k_dpm_2_a',
                'dpmpp_2s_ancestral': 'k_dpmpp_2s_a',
                'dpmpp_2s_ancestral_cfg_pp': 'k_dpmpp_2s_a',
                'dpmpp_sde_gpu': 'k_dpmpp_sde',
                'dpmpp_2m_cfg_pp': 'k_dpmpp_2m',
                'dpmpp_2m_sde_gpu': 'k_dpmpp_2m_sde',
                'dpmpp_3m_sde_gpu': 'k_dpmpp_3m_sde',
                'res_multistep': 'restart',
                'res_multistep_cfg_pp': 'restart',
                'res_multistep_ancestral': 'restart',
                'res_multistep_ancestral_cfg_pp': 'restart',
                'uni_pc': 'unipc',
                'uni_pc_bh2': 'unipc'}

    def check_sampler_or_scheduler_value(self, value: str) -> str:
        mapping = self.get_sampler_and_scheduler_mapping()
        return mapping.get(value, value)

    def clean_options_before_saving(self, options:dict) -> dict:
        # Remove options we do not want to retain in Settings
        override_settings:dict = options.get('override_settings', {})
        if override_settings and not config.is_per_server_imgmodels():
            imgmodel_options = list(override_settings.keys())  # list all options
            if imgmodel_options:
                log.info(f"[Change Imgmodel] Applying Options which won't be retained for next bot startup: {imgmodel_options}")
            # Remove override_settings
            options.pop('override_settings')
        return options

class ImgModel_A1111(ImgModel_SDWebUI):
    def __init__(self):
        super().__init__()

class ImgModel_ReForge(ImgModel_SDWebUI):
    def __init__(self):
        super().__init__()

class ImgModel_Forge(ImgModel_SDWebUI):
    def __init__(self):
        super().__init__()

    def apply_forge_couple(self, forge_couple, task: "Task"):
        task.payload['alwayson_scripts']['forge_couple']['args'].update(forge_couple)
        task.payload['alwayson_scripts']['forge_couple']['args']['enable'] = True
        log.info(f"[TAGS] Enabled forge_couple: {forge_couple}")

    def apply_layerdiffuse(self, layerdiffuse, task: "Task"):
        task.payload['alwayson_scripts']['layerdiffuse']['args'].update(layerdiffuse)
        task.payload['alwayson_scripts']['layerdiffuse']['args']['enabled'] = True
        log.info(f"[TAGS] Enabled layerdiffuse: {layerdiffuse}")

    # Manage override_settings for Forge. returns override_settings
    def get_extra_settings(self, imgmodel_data:dict) -> dict:
        override_settings = {}
        # Set defaults
        override_settings:dict = imgmodel_data.setdefault('override_settings', {})
        # For per-server imgmodels, only the image request payload will drive model changes (won't change now via API)
        if config.is_per_server_imgmodels():
            override_settings[self._value_key] = imgmodel_data[self._value_key]
        if not bot_database.was_warned('forge_clip_state_dict'):
            # Forge manages VAE / Text Encoders using "forge_additional_modules" during change model request.
            log.info("Factoring required option for Forge: 'forge_additional_modules'.")
            log.info("If you get a Forge error 'You do not have Clip State Dict!', please double-check your presets in 'dict_imgmodels.yaml'")
            bot_database.update_was_warned('forge_clip_state_dict')
        # Ensure required params for Forge model loading
        if not override_settings.get('forge_additional_modules'):
            forge_additional_modules = []
            if override_settings.get('sd_vae') and override_settings['sd_vae'] != "Automatic":
                vae = override_settings['sd_vae']
                forge_additional_modules.append(vae)
                log.info(f'[Change Imgmodel] VAE "{vae}" was added to Forge options "forge_additional_modules".')
            override_settings['forge_additional_modules'] = forge_additional_modules
        return override_settings


class LLMContext(SettingsBase):
    def __init__(self):
        self.context = 'The following is a conversation with an AI Large Language Model. The AI has been trained to answer questions, provide recommendations, and help with decision making. The AI follows user requests. The AI thinks outside the box.'
        self.extensions = {}
        self.greeting = '' # 'How can I help you today?'
        self.name = 'AI'
        self.use_voice_channel = False
        self.bot_in_character_menu = True
        self.tags = []


class LLMState(SettingsBase):
    def __init__(self):
        self.text = ''
        self.state = {
            # These are defaults for 'Midnight Enigma' preset
            'preset': '',
            'grammar_string': '',
            'add_bos_token': True,
            'auto_max_new_tokens': True,
            'ban_eos_token': False,
            'character_menu': '',
            'chat_generation_attempts': 1,
            'chat_prompt_size': 2048,
            'custom_stopping_strings': [],
            'custom_system_message': '',
            'custom_token_bans': '',
            'do_sample': True,
            'dry_multiplier': 0,
            'dry_base': 1.75,
            'dry_allowed_length': 2,
            'dry_sequence_breakers': '"\\n", ":", "\\"", "*"',
            'dynamic_temperature': False,
            'dynatemp_low': 1,
            'dynatemp_high': 1,
            'dynatemp_exponent': 1,
            'enable_thinking': True,
            'encoder_repetition_penalty': 1,
            'epsilon_cutoff': 0,
            'eta_cutoff': 0,
            'frequency_penalty': 0,
            'greeting': '',
            'guidance_scale': 1,
            'history': {'internal': [], 'visible': []},
            'max_new_tokens': 512,
            'max_tokens_second': 0,
            'max_updates_second': 12,
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
            'reasoning_effort': 'medium',
            'repetition_penalty': 1.18,
            'repetition_penalty_range': 1024,
            'sampler_priority': [],
            'seed': -1.0,
            'show_after': '',
            'skip_special_tokens': True,
            'smoothing_curve': 1,
            'smoothing_factor': 0,
            'static_cache': False,
            'stop_at_newline': False,
            'stopping_strings': [],
            'stream': True,
            'temperature': 0.98,
            'temperature_last': False,
            'tfs': 1,
            'top_a': 0,
            'top_k': 100,
            'top_n_sigma': 0,
            'top_p': 0.37,
            'truncation_length': 2048,
            'turn_template': '',
            'typical_p': 1,
            'user_bio': '',
            'xtc_threshold': 0.1,
            'xtc_probability': 0,
            'chat_template_str': '''{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {%- if message['content'] -%}\n            {{- message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n        {%- if user_bio -%}\n            {{- user_bio + '\\n\\n' -}}\n        {%- endif -%}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{- name1 + ': ' + message['content'] + '\\n'-}}\n        {%- else -%}\n            {{- name2 + ': ' + message['content'] + '\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}''',
            'instruction_template_str': '''{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not ns.found -%}\n    {{- '' + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + '\\n\\n' -}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- '' + message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{-'### Instruction:\\n' + message['content'] + '\\n\\n'-}}\n        {%- else -%}\n            {{-'### Response:\\n' + message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{-'### Response:\\n'-}}\n{%- endif -%}''',
            'chat-instruct_command': '''Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>'''
            }
        self.regenerate = False
        self._continue = False


# Returns the value of default settings as a dictionary
def defaults_to_dict():
    return {'behavior': Behavior().get_vars(),
            'imgmodel': ImgModel().get_vars(),
            'llmcontext': LLMContext().get_vars(),
            'llmstate': LLMState().get_vars()}

# Initializes as settings file (bot_database -> BaseFileMemory)
class Settings(BaseFileMemory):
    def __init__(self, guild:discord.Guild|None=None):
        self._bot_id = 0
        self._guild_id = guild.id if guild else None
        self._guild_name = guild.name if guild else None
        # settings values
        self.behavior: Behavior
        self.imgmodel: ImgModel
        self.llmcontext: LLMContext
        self.llmstate: LLMState
        # Always initializes 'bot_settings' instance. Can initialize per-guild settings instances.
        self._settings_fp = shared_path.active_settings
        if guild:
            self._settings_fp = os.path.join(shared_path.dir_internal_settings, f'{self._guild_id}_settings.yaml')
        # Initializes the settings file -> load() -> updates values -> run_migration()
        super().__init__(self._settings_fp, version=3, missing_okay=True)

    def init_settings(self, data):
        if not isinstance(data, dict):
            raise Exception(f'Failed to import: "{self._fp}" wrong data type, expected dict, got {type(data)}')
        for k, v in data.items():
            if k in ['db_version']:
                continue
            elif k in ['behavior', 'imgmodel', 'llmcontext', 'llmstate']:
                main_key = getattr(self, k, None)
                if isinstance(main_key, (Behavior, ImgModel, LLMContext, LLMState)):
                    main_key_dict = vars(main_key)
                    if isinstance(v, dict) and isinstance(main_key_dict, dict):
                        update_dict(main_key_dict, v, merge_unmatched=False)
            else:
                log.warning(f'Received unexpected key when initializing Settings: "{k}"')
                setattr(self, k, v)

    def print_per_server_msg(self):
        bot_database.update_was_warned('first_server_settings')
        #log.info("[Per Server Settings] Important information about this feature:")
        log.info("[Per Server Settings] Note: 'dict_base_settings.yaml' applies to ALL server settings. Omit settings you do not want shared!")

    def init_imgmodel(self):
        if not api.imggen:
            self.imgmodel = ImgModel()
        else:
            if api.imggen.is_comfy():
                self.imgmodel = ImgModel_Comfy()
            elif api.imggen.is_swarm():
                self.imgmodel = ImgModel_Swarm()
            elif api.imggen.is_reforge():
                self.imgmodel = ImgModel_ReForge()
            elif api.imggen.is_forge():
                self.imgmodel = ImgModel_Forge()
            elif api.imggen.is_sdwebui():
                self.imgmodel = ImgModel_A1111()
            else:
                self.imgmodel = ImgModel()

    def load_defaults(self):
        self.init_imgmodel()
        self.behavior = Behavior()
        self.llmcontext = LLMContext()
        self.llmstate = LLMState()

    # overrides BaseFileMemory method
    def load(self, data=None):
        self.load_defaults()
        # check if any settings were ever retained for guild
        last_guild_settings = None
        if self._guild_id:
            if not bot_database.last_guild_settings and not bot_database.was_warned('first_server_settings'):
                self.print_per_server_msg()
            last_guild_settings = bot_database.last_guild_settings.get(self._guild_id)
        # load file for 'bot_settings' or for existing guild settings
        if (not self._guild_id) or last_guild_settings:
            data = load_file(self._fp, {}, missing_okay=self._missing_okay)
            self.init_settings(data)
        # Initialize new guild settings from current bot_settings
        else:
            log.info(f'[Per Server Settings] Initializing "{self._guild_name}" with copy of your main settings.')
            data = copy.deepcopy(bot_settings.get_vars(public_only=True))
            self.init_settings(data)
            # Skip update_settings() (already applied to bot_settings)
            # file will typically save while loading each character, but may only load one character
            if config.is_per_character:
                self.save()

    # overrides BaseFileMemory method
    def run_migration(self):
        if self._settings_fp == shared_path.active_settings:
            _old_active = os.path.join(shared_path.dir_internal, 'activesettings.yaml')
            self._migrate_from_file(_old_active, load=True)

    # overrides BaseFileMemory method
    def get_vars(self, public_only=False):
        # return dict excluding "_key" keys
        if public_only:
            return {'behavior': self.behavior.get_vars(),
                    'imgmodel': self.imgmodel.get_vars(),
                    'llmcontext': self.llmcontext.get_vars(),
                    'llmstate': self.llmstate.get_vars()}
        # return complete dict
        else:
            return {'behavior': vars(self.behavior),
                    'imgmodel': vars(self.imgmodel),
                    'llmcontext': vars(self.llmcontext),
                    'llmstate': vars(self.llmstate)}
    
    def save(self):
        data = self.get_vars(public_only=True)
        data['db_version'] = self._latest_version
        data = self.save_pre_process(data)
        save_yaml_file(self._fp, data)

    @property
    def name(self):
        return self.llmcontext.name

    def fix_settings(self, save_now=False):
        default_settings:dict = defaults_to_dict()
        self_settings:dict = self.get_vars()
        # Add any missing required settings, while warning for any missing
        warned = bot_database.was_warned('fixed_base_settings')
        _, was_warned = fix_dict(self_settings, default_settings, 'dict_base_settings.yaml', warned)
        bot_database.update_was_warned('fixed_base_settings', was_warned)
        if save_now:
            self.save()


    # Manage get/set database information for guild-specific or not
    def get_last_setting_for(self, key:str, ictx:CtxInteraction|None=None, guild_id:int|None=None):
        value = None
        guild_id = guild_id if guild_id is not None else self._guild_id
        if config.is_per_server() and (ictx or guild_id):
            if ictx and not is_direct_message(ictx):
                guild_id = ictx.guild.id
            value = bot_database.get_last_guild_setting(guild_id, key)
        # Return value, or last setting (not guild specific)
        return value if value else getattr(bot_database, key, None)

    def set_last_setting_for(self, key:str, value:Any, ictx:CtxInteraction|None=None, guild_id:int|None=None, save_now:bool=False):
        guild_id = guild_id if guild_id is not None else self._guild_id
        if config.is_per_server() and (ictx or guild_id):
            if ictx and not is_direct_message(ictx):
                guild_id = ictx.guild.id
            bot_database.set_last_guild_setting(guild_id, key, value, save_now)
        # Update the main value regardless or per-guild settings
        bot_database.set(key, value, save_now)
        return

bot_settings = Settings()

#################################################################
################## CUSTOM HISTORY MANAGEMENT ####################
#################################################################
def get_char_mode_for_history(ictx:CtxInteraction|None=None, settings=None):
    settings:Settings = settings or get_settings(ictx)
    state_dict = settings.llmstate.state
    mode = state_dict['mode']
    character = state_dict["character_menu"] or 'unknown_character'
    return character, mode

@dataclass_json
@dataclass
class CustomHistory(History):
    manager: 'CustomHistoryManager' = field(metadata=cnf(dont_save=True))
    fp_unique_id: Optional[str] = field(default=None)
    fp_character: Optional[str] = field(default=None)
    fp_mode: Optional[str] = field(default=None)
    fp_internal_id: Optional[str] = field(default=None)
    
    _first_save_debug: bool = field(default=True, metadata=cnf(dont_save=True))
    
    
    def __copy__(self):
        new = super().__copy__()
        new.fp_unique_id = self.fp_unique_id
        new.fp_character = self.fp_character
        new.fp_mode = self.fp_mode
        new.fp_internal_id = self.fp_internal_id
        return new
    
    
    def fresh(self):
        new = super().fresh()
        new.fp_unique_id = None # only reset time to create a new file in the same dir.
        return new
    
    
    def set_save_info(self, internal_id, character, mode):
        self.fp_character = character
        self.fp_mode = mode
        self.fp_internal_id = internal_id
        
        has_file_name = self.fp_unique_id
        if not self.fp_unique_id:
            self.fp_unique_id = datetime.now().strftime('%Y%m%d-%H-%M-%S')

        history_dir = self.manager.history_dir_template.format(character=self.fp_character, mode=self.fp_mode, id=self.fp_internal_id)
        self.fp = os.path.join(history_dir, f'{self.fp_unique_id}.json')
        
        if not has_file_name:
            log.info(f'Internal history file will be saved to: {self.fp}')
    
    
    async def save(self, fp=None, timeout=300, force=False, force_tgwui=False):
        try:
            status = await super().save(fp=fp, timeout=timeout, force=force)
            self._save_for_tgwui(status, force=force_tgwui)
            
            if not status: # don't bother saving if nothing changed
                return False
            
            self._first_save_debug = False
            return status

        except Exception as e:
            print(traceback.format_exc())
            log.critical(e)
            
            
    def save_sync(self, fp=None, force=False, force_tgwui=False):
        try:
            status = super().save_sync(fp=fp, force=force)
            if tgwui_enabled:
                self._save_for_tgwui(status, force=force_tgwui)
            
            if not status: # don't bother saving if nothing changed
                return False
            
            self._first_save_debug = False
            return status

        except Exception as e:
            print(traceback.format_exc())
            log.critical(e)
            
            
    def _save_for_tgwui(self, status, force=False):
        if (status and self.manager.export_for_tgwui) or force:
            save_history_func = get_tgwui_functions('save_history')
            save_history_func(self.render_to_tgwui(), f'{self.fp_unique_id}_{self.fp_internal_id}', self.fp_character, self.fp_mode)
            if self._first_save_debug:
                log.debug(f'''TGWUI chat history saved to "/logs/{self.fp_mode}/{self.fp_character}/{self.fp_unique_id}_{self.fp_internal_id}.json"''')
        
    
    def last_exchange(self):
        if not self.empty:
            last_hmessage:HMessage = self[-1]
            previous_hmessage = last_hmessage.reply_to
            return previous_hmessage, last_hmessage
        return None, None
    

@dataclass
class CustomHistoryManager(HistoryManager):
    history_dir_template: str = field(default=os.path.join(shared_path.dir_history, '{id}', '{character}_{mode}'), init=False)
    
    
    def get_history_dir_template(self, id_):
        _, character, mode = self.get_id_parts(id_)
        return self.history_dir_template.format(character=character, mode=mode, id=id_)
        
    
    def search_for_fp(self, id_:ChannelID):
        # Note: this is an internal function part of get_history_for

        # get the first item split by _
        # For this to work, make sure all ids start with ID_... edit the end as you wish.
        internal_id = id_.split('_',1)[0] 
        
        # TODO users should not be digging in the internals, the name of the folders/files shouldn't matter
        # But in the case you do want to add the channel/guild name to the folder 
        # searching for folders could also be implemented.
        
        # TODO I don't really like this because it wont enable per channel characters easily later on.
        # should add a **search_params to pass down to enable
        history_dir = self.get_history_dir_template(internal_id)
        if not os.path.isdir(history_dir):
            return
        
        # get latest valid history file
        for file in reversed(os.listdir(history_dir)):
            return os.path.join(history_dir, file)
            
            
    def get_history_for(self, id_: Optional[ChannelID|int]=None, character=None, mode=None, fp=None, cached_only=False) -> Optional[CustomHistory]:
        '''
        if not autoload_history:
            New files
        
        
        if change_char == keep:
            if channels == single:
                One global history file
                
            if channels == multiple:
                History per channel
                All characters mixed together
                
        if change_char == new:
            if autoload_history:
                Load history on start, change on switch
            
            if channels == single:
                New global file on char switch
                New file when switching back A>B>A
                
            if channels == multiple:
                New files for each channel on character switch
                New file when switching back A>B>A
        '''
        # Should import old logs or not.
        search = self.autoload_history
        
        # TODO if there's a setting about keeping history between characters, maybe duplicating would be better?
        # or just edit the ID here to match both
        
        id_, character, mode = self.get_id_parts(id_, character, mode)
        full_id = f'{id_}_{character}_{mode}'
        history:Optional[CustomHistory] = super().get_history_for(full_id, fp=fp, search=search, cached_only=cached_only) # type: ignore
        if history is not None:
            history.set_save_info(internal_id=id_, character=character, mode=mode)
        return history
    
    
    def new_history_for(self, id_: Optional[ChannelID|int], character=None, mode=None) -> CustomHistory:
        id_, character, mode = self.get_id_parts(id_, character, mode)
        full_id = f'{id_}_{character}_{mode}'
        return super().new_history_for(full_id) # type: ignore
    
    
    def get_id_parts(self, id_: Optional[ChannelID|int], character=None, mode=None):
        state_dict = bot_settings.llmstate.state
        mode = mode or state_dict['mode']
        character = character or state_dict["character_menu"] or 'unknown_character'
        
        if not self.per_channel_history:
            id_ = 'global'
            
        if self.change_char_history_method == 'keep':
            character = 'Mixed'
            mode = 'mixed'
        
        return id_, character, mode

bot_history = CustomHistoryManager(class_builder_history=CustomHistory, **config.textgen.get('chat_history', {}))

async def async_cleanup():
    for guild_id in voice_clients.guild_vcs:
        if voice_clients.is_connected(guild_id):
            await voice_clients.guild_vcs[guild_id].disconnect()

def exit_handler():
    log.info('Running cleanup tasks:')
    bot_history.save_all_sync()
    try:
        asyncio.run(async_cleanup())
    except Exception as e:
        log.error(f"Error during async cleanup: {e}")
    log.info('Done')


def kill_handler(signum, frame):
    log.debug(f"Signal {signum} received, initiating shutdown...")
    exit_handler()
    sys.exit(0)
    
    
def kill_handler_windows(signum, frame):
    log.debug(f"Signal {signum} received, initiating shutdown...")
    sys.exit(0)
    
    
def on_window_close(ctrl_type):
    log.debug(f"Console window is closing, (signal {ctrl_type})")
    exit_handler()
    return False


if sys.platform == "win32":
    import win32api
    win32api.SetConsoleCtrlHandler(on_window_close, True)
    
    signal.signal(signal.SIGINT, kill_handler_windows)
    signal.signal(signal.SIGTERM, kill_handler_windows)
    
else:
    signal.signal(signal.SIGINT, kill_handler)
    signal.signal(signal.SIGTERM, kill_handler)


# Manually start the bot so we can catch keyboard interupts
async def runner():
    async with client:
        try:
            await client.start(TOKEN, reconnect=True)
        except discord.errors.PrivilegedIntentsRequired:
            log.error("The bot requires the privileged intent 'message_content' to be enabled in your discord developer portal.")
            log.error("Please update the intents for the bot and try again.")
            sys.exit(2)

discord.utils.setup_logging(
            handler=log_file_handler,
            formatter=log_file_formatter,
            level=_logging.INFO,
            root=False,
        )
asyncio.run(runner())

