from discord.ext import commands
import discord
from typing import Union

ChannelID = str
UserID = int
MessageID = int

CtxInteraction = Union[commands.Context, discord.Interaction, discord.Message]

TAG_LIST = list[dict]
TAG_LIST_DICT = dict[str, TAG_LIST]
SORTED_TAGS = dict[str, Union[TAG_LIST, TAG_LIST_DICT]]