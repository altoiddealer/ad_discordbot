import logging
from ad_discordbot.modules.utils_files import load_file, save_yaml_file
import time

from ad_discordbot.modules.database_migration_v1_v2 import OldDatabase
from ad_discordbot.modules.utils_shared import shared_path

class BaseFileMemory:
    def __init__(self, fp) -> None:
        self._fp = fp
        self._did_migration = False
        
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
        save_yaml_file(self._fp, data)
        
    def set(self, key, value, save_now=True):
        setattr(self, key, value)
        if save_now:
            self.save()
            
    #########
    # Loading
    def load_defaults(self, data: dict):
        pass
    
    def load(self):
        data = load_file(self._fp) or {}
        self.load_defaults(data)
        
        for k,v in data.items():
            setattr(self, k, v)
            
    def run_migration(self):
        pass
    
    def get(self, key, default=None):
        if key in self:
            return getattr(self, key)
        
        return default

class Database(BaseFileMemory):
    def __init__(self, fp='bot_database_v2.yaml') -> None:
        self.take_notes_about_users = None # not yet implemented
        self.learn_about_and_use_guild_emojis = None # not yet implemented
        self.read_chatlog = None # not yet implemented
        
        self.first_run:bool
        self.last_character:str
        self.last_change:float
        self.last_user_msg:float
        self.main_channels:list[int]
        self.warned_once:dict[str, bool]
        
        super().__init__(fp)
        
    def run_migration(self):
        self._migrate_v1_v2()
        
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
        self.last_character = data.pop('last_character', None)
        self.last_change = data.pop('last_change', time.time())
        self.last_user_msg = data.pop('last_user_msg', time.time())
        self.main_channels = data.pop('main_channels', [])
        self.warned_once = data.pop('warned_once', {})
        
        
    def was_warned(self, flag_name):
        return self.warned_once.get(flag_name, False)

    def update_was_warned(self, flag_name, value=True, save_now=True):
        self.warned_once[flag_name] = value
        if save_now:
            self.save()
            
class ActiveSettings(BaseFileMemory):
    def __init__(self) -> None:
        # Set default type hints here
        super().__init__(shared_path.active_settings)
        
    def load_defaults(self, data: dict):
        # Set defaults here
        pass