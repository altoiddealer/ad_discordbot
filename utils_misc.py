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

# Adds missing keys/values
def fix_dict(set, req):
    for k, req_v in req.items():
        if k not in set:
            set[k] = req_v
        elif isinstance(req_v, dict):
            fix_dict(set[k], req_v)
    return set

# Updates matched keys, AND adds missing keys
def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    for k in d.keys() - u.keys():
        u[k] = d[k]
    return u

# Updates matched keys, AND adds missing keys, BUT sums together number values
def sum_update_dict(d, u):
    def get_decimal_places(value):
        # Function to get the number of decimal places in a float.
        if isinstance(value, float):
            return len(str(value).split('.')[1])
        else:
            return 0
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = sum_update_dict(d.get(k, {}), v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            current_value = d.get(k, 0)
            max_decimal_places = max(get_decimal_places(current_value), get_decimal_places(v))
            d[k] = round(current_value + v, max_decimal_places)
        else:
            d[k] = v
    for k in d.keys() - u.keys():
        u[k] = d[k]
    return u

# Updates matched keys, but DOES NOT add missing keys
def update_dict_matched_keys(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d