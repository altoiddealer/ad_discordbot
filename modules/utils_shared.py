import asyncio
import os
import re
from shutil import copyfile
from modules.utils_files import load_file
from modules.utils_misc import fix_dict

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

task_processing = asyncio.Event()
bg_task_queue = asyncio.Queue()
flows_queue = asyncio.Queue()
flows_event = asyncio.Event()

class SharedPath:

    def init_user_config_files(root, src_dir, file) -> str:
        dest_path = os.path.join(root, file)
        src_path = os.path.join(src_dir, file)
        if not os.path.exists(dest_path):
            if os.path.exists(src_path):
                copyfile(src_path, dest_path)
                log.info(f'Copied default user setting template "/{file}/" to "{root}".')
            else:
                log.error(f'Required settings file "/{file}/" not found in "{root}" or "{src_dir}".')
        return dest_path, src_path

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
    dir_internal_settings = init_shared_paths(dir_internal, 'settings', 'more persistent settings not intended to be modified by users')
    active_settings = os.path.join(dir_internal_settings, 'activesettings.yaml')
    starboard = os.path.join(dir_internal, 'starboard_messages.yaml')
    database = os.path.join(dir_internal, 'database.yaml')
    statistics = os.path.join(dir_internal, 'statistics.yaml')
    
    dir_history = init_shared_paths(dir_internal, 'history', 'storing internal history states')

    # Configs
    templates = os.path.join(dir_root, 'settings_templates')

    config, config_template = init_user_config_files(dir_root, templates, 'config.yaml')
    base_settings, base_settings_template = init_user_config_files(dir_root, templates, 'dict_base_settings.yaml')
    cmd_options, cmd_options_template = init_user_config_files(dir_root, templates, 'dict_cmdoptions.yaml')
    img_models, img_models_template = init_user_config_files(dir_root, templates, 'dict_imgmodels.yaml')
    tags, tags_template = init_user_config_files(dir_root, templates, 'dict_tags.yaml')

    # Wildcards
    dir_wildcards = init_shared_paths(dir_root, 'wildcards', "wildcard files for Dynamic Prompting feature. Refer to the bot's wiki on GitHub for more information.")

    # User images
    dir_user_images = init_shared_paths(dir_root, 'user_images', "Images that the user may use for various bot functions.")

shared_path = SharedPath()

from modules.database import BaseFileMemory

class Config(BaseFileMemory):
    def __init__(self) -> None:
        self.discord: dict
        self.per_server_settings: dict
        self.dynamic_prompting_enabled: bool
        self.textgenwebui: dict
        self.sd: dict
        super().__init__(shared_path.config, version=2, missing_okay=True)
        self.fix_config()

    def fix_config(self):
        config_dict = self.get_vars()
        # Load the template config
        config_template = load_file(shared_path.config_template, {})
        # Update the user config with any missing values from the template
        fix_dict(config_dict, config_template, 'config.yaml')

    def is_per_server(self):
        return self.per_server_settings.get('enabled', False)
    
    def is_per_character(self):
        if self.is_per_server:
            return self.per_server_settings.get('per_server_characters', False)
        return False

    def run_migration(self):
        _old_active = os.path.join(shared_path.dir_root, 'config.py')
        self._migrate_from_file(_old_active, load=True)

    def discord_dm_setting(self, key, default=None):
        return self.get('discord', {}).get('direct_messages', {}).get(key, default)

config = Config()

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

    seed_value = re.compile(r'Seed: (\d+)')

    markdown_patterns = [
        re.compile(r'\*'),        # *
        re.compile(r'\*\*'),      # **
        re.compile(r'\*\*\*'),    # ***
        re.compile(r'__'),        # __
        re.compile(r'`'),         # `
        re.compile(r'```')        # ```
    ]
    
    @classmethod
    def check_markdown_balanced(cls, text):
        for pattern in cls.markdown_patterns:
            occurrences = len(pattern.findall(text))
            if occurrences % 2 != 0:
                return False
        return True

patterns = SharedRegex()

class SharedBotEmojis:
    hidden_emoji = config.discord.get('history_reactions', {}).get('hidden_emoji', '🙈')
    regen_emoji = config.discord.get('history_reactions', {}).get('regen_emoji', '🔃')
    continue_emoji = config.discord.get('history_reactions', {}).get('continue_emoji', '⏩')

    @classmethod
    def get_emojis(cls):
        return [cls.hidden_emoji, cls.regen_emoji, cls.continue_emoji]

bot_emojis = SharedBotEmojis()
