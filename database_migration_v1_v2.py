import logging
from datetime import datetime, timedelta
import sqlite3
import os

class OldDatabase:
    def __init__(self):
        self.migrate = False
        if not os.path.isfile('bot.db'):
            return
        
        logging.warning('Old bot.db file exists')
        logging.info('Migrating old database to new bot_database_v2.yaml')
        os.rename('bot.db', 'OUTDATED_bot_v1.db')
        
        self.first_run = self.initialize_first_run()
        self.last_character = self.initialize_last_setting('last_character')
        self.last_change = self.initialize_last_time('last_change')
        self.last_user_msg = self.initialize_last_time('last_user_msg')
        self.main_channels = self.initialize_main_channels()
        self.warned_once = self.initialize_warned_once()
        self.migrate = True
        
        
    def initialize_last_setting(self, location):
        try:
            with sqlite3.connect('OUTDATED_bot_v1.db') as conn:
                c = conn.cursor()
                c.execute(f'''CREATE TABLE IF NOT EXISTS {location} (setting TEXT)''')
                c.execute(f'''SELECT setting FROM {location}''')
                row = c.fetchone()
                if row is None:
                    return
                
                return row[0]
        except Exception as e:
            logging.error(f"Error initializing {location}: {e}")

    def initialize_first_run(self):
        with sqlite3.connect('OUTDATED_bot_v1.db') as conn:
            c = conn.cursor()
            c.execute('''SELECT name FROM sqlite_master WHERE type='table' AND name='first_run' ''')
            is_first_run_table_exists = c.fetchone()
            if not is_first_run_table_exists:
                c.execute('''CREATE TABLE IF NOT EXISTS first_run (is_first_run BOOLEAN)''')
                c.execute('''INSERT INTO first_run (is_first_run) VALUES (1)''')
                conn.commit()
                return True
            c.execute('''SELECT COUNT(*) FROM first_run''')
            is_first_run_exists = c.fetchone()[0]
            return is_first_run_exists == 0

    def initialize_last_time(self, location='last_change'):
        try:
            with sqlite3.connect('OUTDATED_bot_v1.db') as conn:
                c = conn.cursor()
                c.execute(f'''CREATE TABLE IF NOT EXISTS {location} (timestamp TEXT)''')
                c.execute(f'''SELECT timestamp FROM {location}''')
                timestamp = c.fetchone()
                
                ts = timestamp[0] if timestamp else None
                
                if ts is None:
                    now = datetime.now()
                    ts = now.strftime('%Y-%m-%d %H:%M:%S')
                    return ts
        
            ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
            ts = ts.timestamp()
            return ts

        except Exception as e:
            logging.error(f"Error initializing {location}: {e}")


    def initialize_main_channels(self):
        with sqlite3.connect('OUTDATED_bot_v1.db') as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS main_channels (channel_id TEXT UNIQUE)''')
            c.execute('''SELECT channel_id FROM main_channels''')
            result = [int(row[0]) for row in c.fetchall()]
            return result if result else []

    def initialize_warned_once(self):
        with sqlite3.connect('OUTDATED_bot_v1.db') as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS warned_once (flag_name TEXT UNIQUE, value INTEGER)''')
            conn.commit()
        
        data = {}
        for k,_ in [('loractl', 0), ('char_tts', 0), ('no_llmmodel', 0), ('forgecouple', 0), ('layerdiffuse', 0), ('dynaprompt', 0)]:
            v = self.was_warned(k)
            if v:
                data[k] = v
        return data

    def was_warned(self, flag_name):
        with sqlite3.connect('OUTDATED_bot_v1.db') as conn:
            c = conn.cursor()
            c.execute('''SELECT value FROM warned_once WHERE flag_name = ?''', (flag_name,))
            result = c.fetchone()
            if result:
                return result[0]
            else:
                return None