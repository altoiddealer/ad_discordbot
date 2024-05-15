import logging
from utils_files import load_file, save_yaml_file
import time

from database_migration_v1_v2 import OldDatabase


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
        self.warned_once:dict[str, int]
        
        
        self._fp = 'bot_database_v2.yaml'
        self.load()
        
        self.migrate_v1_v2()
        
        if self._did_migration:
            self.save()
        
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
        data = {}
        data['first_run'] = self.first_run
        data['last_character'] = self.last_character
        data['last_change'] = self.last_change
        data['last_user_msg'] = self.last_user_msg
        data['main_channels'] = self.main_channels
        data['warned_once'] = self.warned_once
        save_yaml_file(self._fp, data)
        
    def load(self):
        data = load_file(self._fp)
        data = data or {}
        self.first_run = data.get('first_run', True)
        self.last_character = data.get('last_character')
        self.last_change = data.get('last_change', time.time())
        self.last_user_msg = data.get('last_user_msg', time.time())
        self.main_channels = data.get('main_channels', [])
        self.warned_once = data.get('warned_once', {})
        
        
    def set(self, key, value, save_now=True):
        setattr(self, key, value)
        if save_now:
            self.save()
        
    def was_warned(self, flag_name):
        value = self.warned_once.get(flag_name)
        return value # Return None by default, as this is what the previous code did.

    def update_was_warned(self, flag_name, value, save_now=True):
        self.warned_once[flag_name] = value
        if save_now:
            self.save()