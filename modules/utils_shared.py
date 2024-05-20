from ad_discordbot.modules.logs import import_track, log, get_logger; import_track(__file__, fp=True)
import asyncio
import os
import re
logging = get_logger(__name__)

task_semaphore = asyncio.Semaphore(1)

class SharedPath:
    dir_root = 'ad_discordbot'
    dir_internal = os.path.join(dir_root, 'internal')
    os.makedirs(dir_internal, exist_ok=True)

    # Internal
    active_settings = os.path.join(dir_internal, 'activesettings.yaml')
    starboard = os.path.join(dir_internal, 'starboard_messages.yaml')
    database = os.path.join(dir_internal, 'database.yaml')

    # Configs
    config = os.path.join(dir_root, 'config.yaml')
    base_settings = os.path.join(dir_root, 'dict_base_settings.yaml')
    cmd_options = os.path.join(dir_root, 'dict_cmdoptions.yaml')
    img_models = os.path.join(dir_root, 'dict_imgmodels.yaml')
    tags = os.path.join(dir_root, 'dict_tags.yaml')
    
    dir_wildcards = os.path.join(dir_root, 'wildcards')
    os.makedirs(dir_wildcards, exist_ok=True)

shared_path = SharedPath()

class SharedRegex: # Search for [ (]r['"] in vscode
    braces = re.compile(r'{{([^{}]+?)}}(?=[^\w$:]|$$|$)') # {{this syntax|separate items can be divided|another item}}
    wildcard = re.compile(r'##[\w-]+(?=[^\w-]|$)') # ##this-syntax represents a wildcard .txt file
    audio_src = re.compile(r'audio src="file/(.*?\.(wav|mp3))"', flags=re.IGNORECASE)
    
    sd_lora = re.compile(r'<lora:[^:]+:[^>]+>')
    sd_lora_weight = re.compile(r'(?<=:)\d+(\.\d+)?')
    
    recent_msg_roles = re.compile(r'\{(user|llm|history)_([0-9]+)\}', flags=re.IGNORECASE)
    
    curly_brackets = re.compile(r'\{[^{}]*\}') # Selects all {dicts} {}
    in_curly_brackets = re.compile(r'\{([^{}]+)\}') # Selects {contents} of dicts
    brackets = re.compile(r'\[[^\[\]]*\]') # Selects whole list
    in_brackets = re.compile(r'\[([^\[\]]*)\]') # Selects contents of list

    instant_tags = re.compile(r'\[\[([^\[\]]*?(?:\[\[.*?\]\][^\[\]]*?)*?)\]\]')

patterns = SharedRegex()