from modules.utils_files import load_file, save_yaml_file
from datetime import timedelta
import time

from modules.database_migration_v1_v2 import OldDatabase
from modules.utils_shared import shared_path
from modules.utils_files import make_fp_unique
from modules.utils_aspect_ratios import init_avg_from_dims

import os

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

class BaseFileMemory:
    def __init__(self, fp, version=0, missing_okay=False) -> None:
        self._latest_version = version
        self._fp = fp
        self._did_migration = False
        self._missing_okay = missing_okay

        # Attempts to load, but if file not found, continue to migration
        self.load()

        self.run_migration()

        if self._did_migration:
            self.save()

    def _key_check(self, key):
        if key.startswith('_'):
            raise Exception(f'Memory key cannot start with "_": {key!r}')

    ###########
    # Dict like
    def keys(self):
        return self.get_vars().keys()

    def values(self):
        return self.get_vars().values()

    def items(self):
        return self.get_vars().items()

    def __contains__(self, key):
        self._key_check(key)
        return hasattr(self, key)

    def __iter__(self):
        return iter(self.get_vars())

    def __setitem__(self, key, item):
        self._key_check(key)
        setattr(self, key, item)

    def __getitem__(self, key):
        self._key_check(key)
        return getattr(self, key)

    def __delitem__(self, key):
        self._key_check(key)
        delattr(self, key)

    ########
    # Saving
    def get_vars(self):
        return {k:v for k,v in vars(self).items() if not k.startswith('_')}

    def save(self):
        data = self.get_vars()
        data['db_version'] = self._latest_version
        data = self.save_pre_process(data)
        save_yaml_file(self._fp, data)

    def set(self, key, value, save_now=True):
        setattr(self, key, value)
        if save_now:
            self.save()

    def save_pre_process(self, data):
        return data

    #########
    # Loading
    def load_defaults(self, data: dict):
        pass

    def load(self, data:dict=None):
        if not data:
            data = load_file(self._fp, {}, missing_okay=self._missing_okay)
            if not isinstance(data, dict):
                raise Exception(f'Failed to import: "{self._fp}" wrong data type, expected dict, got {type(data)}')

            # data = self.version_upgrade(data) # TODO implement later

        self.load_defaults(data)

        for k,v in data.items():
            setattr(self, k, v)

    def run_migration(self):
        pass

    def get(self, key, default=None):
        return getattr(self, key, default)

    ###########
    # Migration
    def _migrate_from_file(self, from_fp, load:bool):
        'Migrate an old file where the new file may already exist in correct location from git.'
        if not os.path.isfile(from_fp):
            return

        if os.path.isfile(self._fp):
            log.warning(f'File at "{self._fp}" already exists, renaming.')
            os.rename(self._fp, make_fp_unique(self._fp))

        log.info(f'Migrating file to "{self._fp}"')
        os.rename(from_fp, self._fp)
        self._did_migration = True

        if load:
            self.load()
        return True

    def version_upgrade(self, data):
        version = data.get('db_version', 0)
        if version == self._latest_version:
            return data

        log.debug(f'Upgrading "{self._fp}"')

        for upgrade in range(version, self._latest_version):
            upgrade += 1
            log.debug(f'Upgrading "{self._fp}" to v{upgrade}')

            func = f'_upgrade_to_v{upgrade}'
            if not hasattr(self, func):
                raise Exception(f'Could not upgrade database structure to v{upgrade}, missing function.')

            data = getattr(self, func)(data)

        return data

class Database(BaseFileMemory):
    def __init__(self) -> None:
        self.take_notes_about_users = None # not yet implemented
        self.learn_about_and_use_guild_emojis = None # not yet implemented
        self.read_chatlog = None # not yet implemented

        self.first_run:bool
        self.last_base_tags_modified:float
        self.last_character:str
        self.last_change:float
        self.last_imgmodel_name:str
        self.last_imgmodel_checkpoint:str
        self.last_imgmodel_res: int
        self.last_guild_settings:dict[dict[str, int]]
        self.last_user_msg:dict[str, float]
        self.last_bot_msg:dict[str, float]
        self.announce_channels:list[int]
        self.main_channels:list[int]
        self.voice_channels:dict[str, int]
        self.settings_channels:dict[int, int]
        self.settings_sent:dict[dict[str, int]]
        self.warned_once:dict[str, bool]

        super().__init__(shared_path.database, version=2, missing_okay=True)

    def run_migration(self):
        self._migrate_v1_v2()

        _old_active = os.path.join('bot_database_v2.yaml')
        self._migrate_from_file(_old_active, load=True)

    def _migrate_v1_v2(self):
        old = OldDatabase()
        if not old.migrate:
            return

        self.first_run = old.first_run
        self.last_character = old.last_character or self.last_character
        self.last_change = old.last_change or self.last_change
        self.last_user_msg = old.last_user_msg or self.last_user_msg
        self.main_channels = old.main_channels or self.main_channels
        self.warned_once = old.warned_once or self.warned_once

        self._did_migration = True

    def load_defaults(self, data: dict):
        self.first_run = data.pop('first_run', True)
        self.last_base_tags_modified = data.pop('last_base_tags_modified', None)
        self.last_character = data.pop('last_character', None)
        self.last_change = data.pop('last_change', (time.time() - timedelta(minutes=10).seconds))
        self.last_imgmodel_name = data.pop('last_imgmodel_name', '')
        self.last_imgmodel_checkpoint = data.pop('last_imgmodel_checkpoint', '')
        self.last_imgmodel_res = data.pop('last_imgmodel_res', None)
        if not self.last_imgmodel_res:
            self.last_imgmodel_res = init_avg_from_dims()
        self.last_guild_settings = data.pop('last_guild_settings', {})
        self.last_user_msg = data.pop('last_user_msg', {})
        self.last_bot_msg = data.pop('last_bot_msg', {})
        self.announce_channels = data.pop('announce_channels', [])
        self.main_channels = data.pop('main_channels', [])
        self.voice_channels = data.pop('voice_channels', {})
        self.settings_channels = data.pop('settings_channels', {})
        self.settings_sent = data.pop('settings_sent', {})
        data['warned_once'] = {}

    def save_pre_process(self, data):
        data.pop('warned_once', None)
        return data


    # Per guild last settings logging
    def get_last_guild_setting(self, guild_id:int, key:str):
        last_guild_settings:dict = self.last_guild_settings.get(guild_id, {})
        return last_guild_settings.get(key)

    def set_last_guild_setting(self, guild_id:int, key:str, value, save_now=False):
        self.last_guild_settings.setdefault(guild_id, {})
        self.last_guild_settings[guild_id][key] = value
        if save_now:
            self.save()


    # Last messages logging
    def get_member_dict(self, member:str) -> dict:
        member_attr_str = f"last_{member}_msg"
        member_dict = getattr(self, member_attr_str, None)
        if not isinstance(member_dict, dict):
            member_dict = {}
        return member_dict

    def update_last_msg_for(self, channel_id, member:str, save_now=False):
        member_dict = self.get_member_dict(member)

        if channel_id not in member_dict:
            save_now = True

        member_dict[channel_id] = time.time()
        if save_now:
            self.save()

    def get_last_msg_for(self, channel_id, member:str|None=None):
        if member:
            member_dict = self.get_member_dict(member)
            return member_dict.get(channel_id) # return most recent for requested member

        user_member_dict = self.get_member_dict('user')
        last_user_msg = user_member_dict.get(channel_id, 0)

        bot_member_dict = self.get_member_dict('bot')
        last_bot_msg = bot_member_dict.get(channel_id, 0)

        return max(last_user_msg, last_bot_msg) # Return most recent from any user/bot


    # Warning log management
    def was_warned(self, flag_name):
        return self.warned_once.get(flag_name, False)

    def update_was_warned(self, flag_name, value=True):
        self.warned_once[flag_name] = value


    # Voice channels management
    def update_voice_channels(self, guild_id, channel_id, save_now=True):
        self.voice_channels[guild_id] = channel_id
        if save_now:
            self.save()
    

    # Settings channels management (channel where new/updated settings will be posted)
    def get_settings_channel_id_for(self, guild_id:int) -> int:
        return self.settings_channels.get(guild_id)

    def update_settings_channel(self, guild_id:int, new_channel_id:int, save_now=True):
        self.settings_channels[guild_id] = new_channel_id
        self.settings_sent[guild_id] = {}
        if save_now:
            self.save()


    # Post active settings feature management
    def get_settings_msgs_for(self, guild_id:int, key:str|None=None) -> None|dict|list:
        settings_sent:dict|None = self.settings_sent.get(guild_id)
        if settings_sent is None:
            return None # No channel ever set
        if not key:
            return settings_sent # return entire dict

        # create key if not yet existing
        old_msg_ids_list = self.settings_sent[guild_id].setdefault(key, [])
        return old_msg_ids_list

    def update_settings_key_msgs_for(self, guild_id, key:str, new_msg_ids:list, save_now=True):
        settings_key = self.get_settings_msgs_for(guild_id, key)
        if settings_key is None:
            return
        # Update settings key value with new msg ids list
        self.settings_sent[guild_id][key] = new_msg_ids
        if save_now:
            self.save()


class StarBoard(BaseFileMemory):
    def __init__(self) -> None:
        self.messages:list
        super().__init__(shared_path.starboard, version=2, missing_okay=True)

    def load_defaults(self, data: dict):
        self.messages = data.pop('messages', [])

    def run_migration(self):
        _old_active = os.path.join(shared_path.dir_root, 'starboard_messages.yaml')
        state = self._migrate_from_file(_old_active, load=False) # v1
        if state:
            data = load_file(self._fp, [])      # load old file as list
            self.load(data=dict(messages=data)) # convert list to dict


class _Statistic:
    def __init__(self, db, data) -> None:
        self.db: Statistics = db
        self.data: dict = data

    def set(self, key, value, save_now=False):
        if key not in self.data:
            save_now = True

        self.data[key] = value
        if save_now:
            self.db.save()

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __setitem__(self, key, item):
        self.data[key] = item

    def __getitem__(self, key):
        return self.data[key]


class Statistics(BaseFileMemory):
    def __init__(self) -> None:
        self._llm_gen_time_start_last: float
        self.llm: _Statistic

        super().__init__(shared_path.statistics, version=1, missing_okay=True)

    def load_defaults(self, data: dict):
        self.llm = _Statistic(self, data.pop('llm', {}))

    def save_pre_process(self, data):
        # Replace outgoing data with json serializable
        for k,v in data.items():
            if isinstance(v, _Statistic):
                data[k] = v.data

        return data
