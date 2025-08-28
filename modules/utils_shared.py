import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from shutil import copyfile, move, rmtree
from modules.utils_files import load_file
from modules.utils_misc import fix_dict
import discord
from discord.ext import commands

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

task_event = asyncio.Event()
bg_task_queue = asyncio.Queue()
flows_queue = asyncio.Queue()
flows_event = asyncio.Event()

# Intercept custom bot arguments
def parse_bot_args():
    bot_arg_list = ["--is-tgwui-integrated", "--limit-history", "--token", "--lazy-load-llm"]
    flag_only_args = {"--is-tgwui-integrated", "--lazy-load-llm"}
    
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
    parser.add_argument("--lazy-load-llm", action="store_true", help="If true, loads LLM in response to first text gen request (not during script init).")
    parser.add_argument("--is-tgwui-integrated", action="store_true", help="Indicates integration with TGWUI.")

    # Parse the arguments
    bot_args = parser.parse_args(bot_argv)
    return bot_args

bot_args = parse_bot_args()

is_tgwui_integrated = bot_args.is_tgwui_integrated

# Set discord intents
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix=".", intents=intents)
client.is_first_on_ready = True # type: ignore
client.waiting_for = {} # type: ignore

class SharedPath:

    def init_user_config_files(dest_dir, src_dir, file) -> str:
        dest_path = os.path.join(dest_dir, file)
        src_path = os.path.join(src_dir, file)
        if not os.path.exists(dest_path):
            if os.path.exists(src_path):
                copyfile(src_path, dest_path)
                log.info(f'Copied default user setting template "{file}" to "{dest_dir}".')
            else:
                log.error(f'Required settings file "{file}" not found in "{dest_dir}" or "{src_dir}".')
        return dest_path, src_path

    def init_shared_paths(root, dir_name:str, reason:str) -> str:
        path = os.path.join(root, dir_name)
        if not os.path.exists(path) and reason:
            log.info(f'Creating "{path}" for {reason}.')
        os.makedirs(path, exist_ok=True)
        return path
    
    def migrate_old_settings(dir_root, dir_user_settings):
        old_settings_files = ['config.yaml',
                            'dict_base_settings.yaml',
                            'dict_api_settings.yaml',
                            'dict_cmdoptions.yaml',
                            'dict_imgmodels.yaml',
                            'dict_tags.yaml']
        for filename in old_settings_files:
            old_path = os.path.join(dir_root, filename)
            if os.path.exists(old_path):
                new_path = os.path.join(dir_user_settings, filename)
                try:
                    move(old_path, new_path)
                    log.info(f'Migrated "{filename}" from root dir to "/user/settings/"')
                except Exception as e:
                    log.warning(f'Tried to migrate "{filename}" to "/user/settings/". Please migrate manually or try again: {e}')

    # Get the parent directory (bot's root directory) relative to this directory (modules folder)
    utils_shared_file_dir = os.path.dirname(os.path.abspath(__file__))
    bot_root_dir = os.path.dirname(utils_shared_file_dir)

    # Root
    dir_root = bot_root_dir if is_tgwui_integrated else os.getcwd()
    dir_tgwui = os.getcwd() if is_tgwui_integrated else ''

    # Internal
    dir_internal = init_shared_paths(dir_root, 'internal', 'persistent settings not intended to be modified by users')
    # Cache
    dir_internal_cache = os.path.join(dir_internal, 'cache')
    rmtree(dir_internal_cache, ignore_errors=True)
    os.makedirs(dir_internal_cache, exist_ok=True)
    # Settings / Database / Statistics
    dir_internal_settings = init_shared_paths(dir_internal, 'settings', 'more persistent settings not intended to be modified by users')
    active_settings = os.path.join(dir_internal_settings, 'activesettings.yaml')
    starboard = os.path.join(dir_internal, 'starboard_messages.yaml')
    stt_blacklist = os.path.join(dir_internal, 'stt_blacklist.yaml')
    database = os.path.join(dir_internal, 'database.yaml')
    statistics = os.path.join(dir_internal, 'statistics.yaml')
    
    dir_history = init_shared_paths(dir_internal, 'history', 'storing internal history states')

    # Output
    output_dir = init_shared_paths(dir_root, 'output', "for content generated by the bot (except chat logs found in '/internal/history')")

    # Settings Templates
    templates = os.path.join(dir_root, 'settings_templates')

    # User
    dir_user = init_shared_paths(dir_root, 'user', "for files which may be used for bot functions")

    # User Settings
    dir_user_settings = init_shared_paths(dir_user, 'settings', "for bot settings and configurations")

    migrate_old_settings(dir_root, dir_user_settings)
    bot_token, _ = init_user_config_files(dir_user_settings, templates, 'bot_token.yaml')
    config, config_template = init_user_config_files(dir_user_settings, templates, 'config.yaml')
    api_settings, _ = init_user_config_files(dir_user_settings, templates, 'dict_api_settings.yaml')
    base_settings, _ = init_user_config_files(dir_user_settings, templates, 'dict_base_settings.yaml')
    custom_commands, _ = init_user_config_files(dir_user_settings, templates, 'dict_commands.yaml')
    cmd_options, _ = init_user_config_files(dir_user_settings, templates, 'dict_cmdoptions.yaml')
    img_models, _ = init_user_config_files(dir_user_settings, templates, 'dict_imgmodels.yaml')
    tags, _ = init_user_config_files(dir_user_settings, templates, 'dict_tags.yaml')

    # User Characters
    dir_user_characters = init_shared_paths(dir_user, 'characters', f"Character files to be used with the bot{' (merge in with TGWUI characters)' if is_tgwui_integrated else ''}.")
    # User Payloads
    dir_user_payloads = init_shared_paths(dir_user, 'payloads', "Payloads to be used for API calls.")
    # User Images
    dir_user_images = init_shared_paths(dir_user, 'images', "Images to be used for various bot functions.")
    old_user_images = os.path.join(dir_root, 'user_images')
    if os.path.exists(old_user_images):
        log.warning(f'Please migrate your existing "/user_images" contents to: "{dir_user_images}".')
    # User Wildcards
    dir_user_wildcards = init_shared_paths(dir_user, 'wildcards', "wildcard files for Dynamic Prompting feature. Refer to the bot's wiki on GitHub for more information")
    old_wildcards = os.path.join(dir_root, 'wildcards')
    if os.path.exists(old_wildcards):
        log.warning(f'Please migrate your existing "/wildcards" content to: "{dir_user_wildcards}"')

shared_path = SharedPath()

# SharedPath() must initialize before BaseFileMemory() and Database()
from modules.database import BaseFileMemory, Database, STTBlacklist

bot_database = Database()
stt_blacklist = STTBlacklist()

class Config(BaseFileMemory):
    def __init__(self) -> None:
        self.discord: dict
        self.allowed_paths: list
        self.task_queues: dict
        self.per_server_settings: dict
        self.dynamic_prompting_enabled: bool
        self.textgen: dict
        self.ttsgen: dict
        self.imggen: dict
        self.stt: dict
        super().__init__(shared_path.config, version=2, missing_okay=True)
        self._fix_config()
        self._sanitize_paths()

    def load_defaults(self, data: dict):
        self.discord = data.pop('discord', {})
        self.allowed_paths = data.pop('allowed_paths', [])
        self.task_queues = data.pop('task_queues', {})
        self.per_server_settings = data.pop('per_server_settings', {})
        self.dynamic_prompting_enabled = data.pop('dynamic_prompting_enabled', True)
        self.textgen = data.pop('textgen', {})
        self.ttsgen = data.pop('ttsgen', {})
        self.imggen = data.pop('imggen', {})
        self.stt = data.pop('stt', {})

    def _fix_config(self):
        config_dict = self.get_vars()
        # Load the template config
        config_template = load_file(shared_path.config_template, {})
        # Update the user config with any missing values from the template
        fix_dict(config_dict, config_template, 'config.yaml')

    def _sanitize_paths(self):
        raw_paths = self.allowed_paths + [shared_path.output_dir]
        if is_tgwui_integrated:
            raw_paths.append(shared_path.dir_tgwui)
        paths = [Path(p) for p in raw_paths]

        sanitized = []
        for path in paths:
            path:Path
            abs_path = (shared_path.dir_root / path).resolve() if not path.is_absolute() else path.resolve()
            if abs_path.exists():
                sanitized.append(abs_path)
            else:
                log.warning(f"'allowed_path' does not exist and will be ignored from save checks: {abs_path}")
        self.allowed_paths = sanitized

    def path_allowed(self, path: str) -> bool:
        """Check if the input path (relative or absolute) is allowed. Symlinks permitted."""
        input_path = Path(path)
        if not input_path.is_absolute():
            abs_path = (shared_path.dir_root / input_path).absolute()
        else:
            abs_path = input_path.absolute()

        for allowed_base in self.allowed_paths:
            if abs_path.is_relative_to(allowed_base):
                return True
        return False
    
    def should_lazy_load_llm(self) -> bool:
        return bot_args.lazy_load_llm or self.textgen.get('lazy_load_llm', False)

    def is_per_server(self) -> bool:
        return self.per_server_settings.get('enabled', False)
    
    def is_per_character(self) -> bool:
        if self.is_per_server:
            return self.per_server_settings.get('per_server_characters', False)
        return False
    
    def is_per_server_imgmodels(self) -> bool:
        return self.per_server_settings.get('per_server_imgmodel_settings', False)

    def discord_dm_setting(self, key, default=None) -> bool:
        return self.get('discord', {}).get('direct_messages', {}).get(key, default)
    
    def tts_enabled(self) -> bool:
        return self.get('ttsgen', {}).get('enabled')
    
    def controlnet_enabled(self) -> bool:
        return self.imggen.get('extensions', {}).get('controlnet_enabled', False)
    def forgecouple_enabled(self) -> bool:
        return self.imggen.get('extensions', {}).get('forgecouple_enabled', False)
    def layerdiffuse_enabled(self) -> bool:
        return self.imggen.get('extensions', {}).get('layerdiffuse_enabled', False)
    def reactor_enabled(self) -> bool:
        return self.imggen.get('extensions', {}).get('reactor_enabled', False)
    def loractl_enabled(self) -> bool:
        return self.imggen.get('extensions', {}).get('loractl', {}).get('enabled', False)

config = Config()


class BotToken(BaseFileMemory):
    def __init__(self) -> None:
        self.TOKEN: str
        super().__init__(shared_path.bot_token, version=1, missing_okay=True)

    def load_defaults(self, data: dict):
        # Don't save token to file if provided via args
        self.TOKEN = bot_args.token if bot_args.token else None
        if self.TOKEN:
            return

        saved_token = data.pop('TOKEN', None)
        if saved_token:
            self.TOKEN = saved_token
            return
        
        self.TOKEN = config.discord.get('TOKEN') # deprecated
        if self.TOKEN and not saved_token:
            self.save()

        if not self.TOKEN:
            self.prompt_for_token()

    def prompt_for_token(self):
        print(f'\nA Discord bot token is required. You may enter it now or manually in "{shared_path.bot_token}".\n'
              'You may also use "--token {token}" in "CMD_FLAGS.txt".\n'
              'For help regarding Discord bot token, see Install instructions on the project page:\n'
              '(https://github.com/altoiddealer/ad_discordbot)')

        print('\nDiscord bot token (enter "0" to exit):\n')
        token = (input().strip())
        print()
        if token == '0':
            log.error("Discord bot token is required. Exiting.")
            sys.exit(2)
        elif token:
            self.TOKEN = token
            self.save()
            log.info(f"Discord bot token saved to {shared_path.bot_token}")
        else:
            log.error("Discord bot token is required. Exiting.")
            sys.exit(2)

bot_token = BotToken()

TOKEN = bot_token.TOKEN

class SharedRegex: # Search for [ (]r['"] in vscode
    braces = re.compile(r'{{([^{}]+?)}}(?=[^\w$:]|$$|$)') # {{this syntax|separate items can be divided|another item}}
    wildcard = re.compile(r'##[\w-]+(?=[^\w-]|$)') # ##this-syntax represents a wildcard .txt file
    audio_src = re.compile(r'src="file/([^"]+\.(wav|mp3))\b', flags=re.IGNORECASE)

    sd_lora = re.compile(r'<lora:[^:]+:[^>]+>')
    sd_lora_weight = re.compile(r'(?<=:)\d+(\.\d+)?')
    sd_lora_split = re.compile(r'<lora:([^:]+):([^>]+)>')

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

# SharedPath() must initialize before API().
_api = None
async def get_api():
    global _api
    if _api is None:
        from modules.apis import API
        _api = API()
        await _api.init()
    return _api
