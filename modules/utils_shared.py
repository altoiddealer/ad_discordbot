from ad_discordbot.modules.logs import import_track, log, get_logger; import_track(__file__, fp=True)
import asyncio
import os
import re
logging = get_logger(__name__)

task_semaphore = asyncio.Semaphore(1)

class SharedPath:

    def init_shared_path(root, dir, reason) -> str:
        path = os.path.join(root, dir)
        if not os.path.exists(path):
            logging.info(f'Creating "/{dir}/" for {reason}.')
        os.makedirs(path, exist_ok=True)
        return path

    dir_root = 'ad_discordbot'

    # Internal
    dir_internal = init_shared_path(dir_root, 'internal', 'persistent settings not intended to be modified by users')
    active_settings = os.path.join(dir_internal, 'activesettings.yaml')
    starboard = os.path.join(dir_internal, 'starboard_messages.yaml')
    database = os.path.join(dir_internal, 'database.yaml')
    statistics = os.path.join(dir_internal, 'statistics.yaml')

    # Configs
    config = os.path.join(dir_root, 'config.yaml')
    base_settings = os.path.join(dir_root, 'dict_base_settings.yaml')
    cmd_options = os.path.join(dir_root, 'dict_cmdoptions.yaml')
    img_models = os.path.join(dir_root, 'dict_imgmodels.yaml')
    tags = os.path.join(dir_root, 'dict_tags.yaml')

    # Wildcards
    init_shared_path(dir_root, 'wildcards', "wildcard files for Dynamic Prompting feature. Refer to the bot's wiki on GitHub for more information.")

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