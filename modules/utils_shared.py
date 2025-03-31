import argparse
import asyncio
import os
import re
import sys
from shutil import copyfile, move
from modules.utils_files import load_file
from modules.utils_misc import fix_dict

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

task_event = asyncio.Event()
bg_task_queue = asyncio.Queue()
flows_queue = asyncio.Queue()
flows_event = asyncio.Event()

# Intercept custom bot arguments
def parse_bot_args():
    bot_arg_list = ["--is-tgwui-integrated", "--limit-history", "--token"]
    flag_only_args = {"--is-tgwui-integrated"}
    
    bot_argv = []
    for arg in bot_arg_list:
        try:
            index = sys.argv.index(arg)
        except ValueError:
            continue

        bot_argv.append(sys.argv.pop(index))

        # If the argument requires a value, pop the next item as well
        if arg not in flag_only_args and index < len(sys.argv):
            bot_argv.append(sys.argv.pop(index))

    # Define the argument parser
    parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54))
    parser.add_argument("--token", type=str, help="Discord bot token to use their API.")
    parser.add_argument("--limit-history", type=int, help="When the history gets too large, performance issues can occur. Limit the history to improve performance.")
    parser.add_argument("--is-tgwui-integrated", action="store_true", help="Indicates integration with TGWUI.")

    # Parse the arguments
    bot_args = parser.parse_args(bot_argv)
    return bot_args

bot_args = parse_bot_args()

is_tgwui_integrated = bot_args.is_tgwui_integrated

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

    # Get the parent directory (bot's root directory) relative to this directory (modules folder)
    utils_shared_file_dir = os.path.dirname(os.path.abspath(__file__))
    bot_root_dir = os.path.dirname(utils_shared_file_dir)

    # Define root paths
    dir_root = bot_root_dir if is_tgwui_integrated else os.getcwd()
    dir_tgwui = os.getcwd() if is_tgwui_integrated else ''

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


    # User
    user_dir = init_shared_paths(dir_root, 'user', "for files which may be used for bot functions.")

    dir_user_wildcards = init_shared_paths(user_dir, 'wildcards', "wildcard files for Dynamic Prompting feature. Refer to the bot's wiki on GitHub for more information.")
    old_wildcards = os.path.join(dir_root, 'wildcards')
    if os.path.exists(old_wildcards):
        log.warning(f'Please migrate your existing "/wildcards" directory to: "{dir_user_wildcards}"')

    dir_user_images = init_shared_paths(user_dir, 'images', "Images that the user may use for various bot functions.")
    old_user_images = os.path.join(dir_root, 'user_images')
    if os.path.exists(old_user_images):
        log.info(f'Please migrate your existing "/user_images" to: "{dir_user_images}"')

shared_path = SharedPath()

# SharedPath() must initialize before BaseFileMemory() and Database()
from modules.database import BaseFileMemory, Database

bot_database = Database()

class Config(BaseFileMemory):
    def __init__(self) -> None:
        self.discord: dict
        self.per_server_settings: dict
        self.dynamic_prompting_enabled: bool
        self.textgenwebui: dict
        self.sd: dict
        super().__init__(shared_path.config, version=2, missing_okay=True)
        self.fix_config()

    def load_defaults(self, data: dict):
        self.discord = data.pop('discord', {})
        self.per_server_settings = data.pop('per_server_settings', {})
        self.dynamic_prompting_enabled = data.pop('dynamic_prompting_enabled', True)
        self.textgenwebui = data.pop('textgenwebui', {})
        self.sd = data.pop('sd', {})

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
    
    def is_per_server_imgmodels(self):
        return self.per_server_settings.get('per_server_imgmodel_settings', False)

    def run_migration(self):
        _old_active = os.path.join(shared_path.dir_root, 'config.py')
        self._migrate_from_file(_old_active, load=True)

    def discord_dm_setting(self, key, default=None):
        return self.get('discord', {}).get('direct_messages', {}).get(key, default)

config = Config()

class SharedRegex: # Search for [ (]r['"] in vscode
    braces = re.compile(r'{{([^{}]+?)}}(?=[^\w$:]|$$|$)') # {{this syntax|separate items can be divided|another item}}
    wildcard = re.compile(r'##[\w-]+(?=[^\w-]|$)') # ##this-syntax represents a wildcard .txt file
    audio_src = re.compile(r'src="file/([^"]+\.(wav|mp3))\b', flags=re.IGNORECASE)

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
