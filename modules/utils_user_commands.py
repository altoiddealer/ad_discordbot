from modules.utils_shared import client, bg_task_queue, task_event, bot_emojis, config
import discord
from discord.ext import commands
from typing import Optional, Union
from modules.typing import CtxInteraction
from typing import TYPE_CHECKING
import asyncio
from contextlib import asynccontextmanager

import inspect
import types

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

type_map = {
    "string": str,
    "user": discord.User,
    "int": int,
    "bool": bool,
    "float": float,
    "channel": discord.abc.GuildChannel,
    "role": discord.Role,
    "mentionable": Union[discord.User, discord.Role],
    "attachment": discord.Attachment,
}
