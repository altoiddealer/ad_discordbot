from datetime import datetime
import logging
import sqlite3

class Database:
    def __init__(self):
        self.take_notes_about_users = None # not yet implemented
        self.learn_about_and_use_guild_emojis = None # not yet implemented
        self.read_chatlog = None # not yet implemented
        self.first_run = self.initialize_first_run()
        self.last_character = self.initialize_last_setting('last_character')
        self.last_change = self.initialize_last_time('last_change')
        self.last_user_msg = self.initialize_last_time('last_user_msg')
        self.main_channels = self.initialize_main_channels()
        self.warned_once = self.initialize_warned_once()

    def initialize_first_run(self):
        with sqlite3.connect('bot.db') as conn:
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

    def initialize_last_setting(self, location):
        try:
            with sqlite3.connect('bot.db') as conn:
                c = conn.cursor()
                c.execute(f'''CREATE TABLE IF NOT EXISTS {location} (setting TEXT)''')
        except Exception as e:
            logging.error(f"Error initializing {location}: {e}")

    def update_last_setting(self, value, location):
        try:
            with sqlite3.connect('bot.db') as conn:
                c = conn.cursor()
                c.execute(f'''UPDATE {location} SET setting = ?''', (value,))
                c.execute(f'''INSERT INTO {location} (setting) SELECT (?) WHERE NOT EXISTS (SELECT 1 FROM {location})''', (value,))
                conn.commit()
            setattr(self, location, value)
        except Exception as e:
            logging.error(f"An error occurred while logging '{value}' to '{location}' in bot.db: {e}")

    def initialize_last_time(self, location='last_change'):
        try:
            with sqlite3.connect('bot.db') as conn:
                c = conn.cursor()
                c.execute(f'''CREATE TABLE IF NOT EXISTS {location} (timestamp TEXT)''')
                c.execute(f'''SELECT timestamp FROM {location}''')
                timestamp = c.fetchone()
                if timestamp is None or timestamp[0] is None:
                    now = datetime.now()
                    formatted_now = now.strftime('%Y-%m-%d %H:%M:%S')
                    if timestamp is None:
                        c.execute(f'''INSERT INTO {location} (timestamp) VALUES (?)''', (formatted_now,))
                    else:
                        c.execute(f'''UPDATE {location} SET timestamp = ?''', (formatted_now,))
                    conn.commit()
                    return formatted_now
            return timestamp[0] if timestamp else None
        except Exception as e:
            logging.error(f"Error initializing {location}: {e}")

    def update_last_time(self, location='last_change'):
        try:
            with sqlite3.connect('bot.db') as conn:
                c = conn.cursor()
                now = datetime.now()
                formatted_now = now.strftime('%Y-%m-%d %H:%M:%S')
                c.execute(f'''UPDATE {location} SET timestamp = ?''', (formatted_now,))
                conn.commit()
            setattr(self, location, formatted_now)
        except Exception as e:
            logging.error(f"An error occurred while logging time of profile update to bot.db: {e}")

    def initialize_main_channels(self):
        with sqlite3.connect('bot.db') as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS main_channels (channel_id TEXT UNIQUE)''')
            c.execute('''SELECT channel_id FROM main_channels''')
            result = [int(row[0]) for row in c.fetchall()]
            return result if result else []

    def initialize_warned_once(self):
        with sqlite3.connect('bot.db') as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS warned_once (flag_name TEXT UNIQUE, value INTEGER)''')
            flags_to_insert = [('loractl', 0), ('char_tts', 0), ('no_llmmodel', 0), ('forgecouple', 0), ('layerdiffuse', 0), ('dynaprompt', 0)]
            for flag_name, value in flags_to_insert:
                c.execute('''INSERT OR REPLACE INTO warned_once (flag_name, value) VALUES (?, ?)''', (flag_name, value))
            conn.commit()

    def was_warned(self, flag_name):
        with sqlite3.connect('bot.db') as conn:
            c = conn.cursor()
            c.execute('''SELECT value FROM warned_once WHERE flag_name = ?''', (flag_name,))
            result = c.fetchone()
            if result:
                return result[0]
            else:
                return None

    def update_was_warned(self, flag_name, value):
        with sqlite3.connect('bot.db') as conn:
            c = conn.cursor()
            c.execute('''UPDATE warned_once SET value = ? WHERE flag_name = ?''', (value, flag_name))
            conn.commit()