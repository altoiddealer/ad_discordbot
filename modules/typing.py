from discord.ext import commands
import discord
from typing import Union, Any
from discord.errors import DiscordException

ChannelID = str
UserID = int
MessageID = int

CtxInteraction = Union[commands.Context, discord.Interaction, discord.Message]

TAG = dict[str, Any]
TAG_LIST = list[TAG]
TAG_LIST_DICT = dict[str, TAG_LIST]

Message = Union[CtxInteraction, str, dict|None, dict|None]


class AlertUserError(DiscordException):
    pass