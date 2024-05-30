from discord.ext import commands
import discord
from typing import Union

ChannelID = str
UserID = int
MessageID = int

CtxInteraction = Union[commands.Context, discord.Interaction, discord.Message]