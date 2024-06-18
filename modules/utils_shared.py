import asyncio
import os
import re
from shutil import copyfile

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

task_semaphore = asyncio.Semaphore(1)

class SharedPath:

    def init_user_config_files(root, src_dir, file) -> str:
        dest_path = os.path.join(root, file)
        if not os.path.exists(dest_path):
            src_path = os.path.join(src_dir, file)
            if os.path.exists(src_path):
                copyfile(src_path, dest_path)
                log.info(f'Copied default user setting template "/{file}/" to "{root}".')
            else:
                log.error(f'Required settings file "/{file}/" not found in "{root}" or "{src_dir}".')
        return dest_path

    def init_shared_paths(root, dir, reason) -> str:
        path = os.path.join(root, dir)
        if not os.path.exists(path):
            log.info(f'Creating "{path}" for {reason}.')
        os.makedirs(path, exist_ok=True)
        return path

    dir_tgwui = os.path.abspath('.') # because the start file goes up one dir.
    log.debug(f'TGWUI dir: {dir_tgwui}')
    dir_root = 'ad_discordbot'

    # Internal
    dir_internal = init_shared_paths(dir_root, 'internal', 'persistent settings not intended to be modified by users')
    active_settings = os.path.join(dir_internal, 'activesettings.yaml')
    starboard = os.path.join(dir_internal, 'starboard_messages.yaml')
    database = os.path.join(dir_internal, 'database.yaml')
    statistics = os.path.join(dir_internal, 'statistics.yaml')
    
    dir_history = init_shared_paths(dir_internal, 'history', 'storing internal history states')

    # Configs
    templates = os.path.join(dir_root, 'settings_templates')

    config = init_user_config_files(dir_root, templates, 'config.yaml')
    base_settings = init_user_config_files(dir_root, templates, 'dict_base_settings.yaml')
    cmd_options = init_user_config_files(dir_root, templates, 'dict_cmdoptions.yaml')
    img_models = init_user_config_files(dir_root, templates, 'dict_imgmodels.yaml')
    tags = init_user_config_files(dir_root, templates, 'dict_tags.yaml')

    # Wildcards
    dir_wildcards = init_shared_paths(dir_root, 'wildcards', "wildcard files for Dynamic Prompting feature. Refer to the bot's wiki on GitHub for more information.")

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

    history_labels = re.compile(r'^\*\`\(.*?\)\`\*\n')

    mention_prefix = re.compile(r'^<@!?\d+>')

patterns = SharedRegex()

class SharedBotEmojis:
    hidden_emoji = '🙈'
    regen_emoji = '🔃'
    continue_emoji = '⏩'

bot_emojis = SharedBotEmojis()
