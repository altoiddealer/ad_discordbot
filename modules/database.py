import logging
from ad_discordbot.modules.utils_files import load_file, save_yaml_file
import time

from ad_discordbot.modules.database_migration_v1_v2 import OldDatabase


class Database:
    def __init__(self):
        self.take_notes_about_users = None # not yet implemented
        self.learn_about_and_use_guild_emojis = None # not yet implemented
        self.read_chatlog = None # not yet implemented
        
        self._did_migration = False
        
        self.first_run:bool
        self.last_character:str
        self.last_change:float
        self.last_user_msg:float
        self.main_channels:list[int]
        self.warned_once:dict[str, bool]
        
        
        self._fp = 'bot_database_v2.yaml'
        self.load()
        
        self.migrate_v1_v2()
        
        if self._did_migration:
            self.save()
            
    def get_vars(self):
        return {k:v for k,v in vars(self).items() if not k.startswith('_')}
        
    def migrate_v1_v2(self):
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
        
    def save(self):
        data = self.get_vars()
        save_yaml_file(self._fp, data)
        
    def load(self):
        data = load_file(self._fp) or {}
        self.first_run = data.pop('first_run', True)
        self.last_character = data.pop('last_character', None)
        self.last_change = data.pop('last_change', time.time())
        self.last_user_msg = data.pop('last_user_msg', time.time())
        self.main_channels = data.pop('main_channels', [])
        self.warned_once = data.pop('warned_once', {})
        
        for k,v in data.items():
            setattr(self, k, v)
        
        
    def set(self, key, value, save_now=True):
        setattr(self, key, value)
        if save_now:
            self.save()
        
    def was_warned(self, flag_name):
        return self.warned_once.get(flag_name, False)

    def update_was_warned(self, flag_name, value=True, save_now=True):
        self.warned_once[flag_name] = value
        if save_now:
            self.save()