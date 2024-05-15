from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import random
import logging
import logging.handlers
import json
import re
import glob
import os
import warnings
import discord
from discord.ext import commands
from discord import app_commands, File
from discord.ext.commands.context import Context
import typing
import torch
import io
import base64
import yaml
from PIL import Image, PngImagePlugin
import requests
import sqlite3
import pprint
import aiohttp
import math
import time
from itertools import product
from threading import Lock, Thread
from pydub import AudioSegment
import copy
from shutil import copyfile
import sys
import traceback


# Function to load .json, .yml or .yaml files
def load_file(file_path):
    try:
        file_suffix = Path(file_path).suffix.lower()

        if file_suffix in [".json", ".yml", ".yaml"]:
            with open(file_path, 'r', encoding='utf-8') as file:
                if file_suffix in [".json"]:
                    data = json.load(file)
                else:
                    data = yaml.safe_load(file)
            return data
        else:
            logging.error(f"Unsupported file format: {file_suffix}")
            return None
    except FileNotFoundError:
        logging.error(f"File not found: {file_suffix}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while reading {file_path}: {str(e)}")
        return None
    
def merge_base(newsettings, basekey):
    def deep_update(original, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                deep_update(original[key], value)
            else:
                original[key] = value
    try:
        base_settings = load_file('ad_discordbot/dict_base_settings.yaml')
        keys = basekey.split(',')
        current_dict = base_settings
        for key in keys:
            if key in current_dict:
                current_dict = current_dict[key].copy()
            else:
                return None
        deep_update(current_dict, newsettings) # Recursively update the dictionary
        return current_dict
    except Exception as e:
        logging.error(f"Error loading ad_discordbot/dict_base_settings.yaml ({basekey}): {e}")
        return newsettings
    
def save_yaml_file(file_path, data):
    try:
        with open(file_path, 'w') as file:
            yaml.dump(data, file, encoding='utf-8', default_flow_style=False, width=float("inf"), sort_keys=False)
    except Exception as e:
        logging.error(f"An error occurred while saving {file_path}: {str(e)}")