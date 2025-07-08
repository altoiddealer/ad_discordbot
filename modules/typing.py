from discord.ext import commands
import discord
from typing import Union, Any, IO, TypedDict
from discord.errors import DiscordException

ChannelID = str
UserID = int
MessageID = int

CtxInteraction = Union[commands.Context, discord.Interaction, discord.Message]

TAG = dict[str, Any]
TAG_LIST = list[TAG]
TAG_LIST_DICT = dict[str, TAG_LIST]

class FileInput(TypedDict):
    """
    Represents a normalized file dict for send_content_to_discord()

    Attributes:
        file_obj (IO[bytes]): The file-like object (e.g., BytesIO or open file).
        filename (str): The name of the file (e.g., "image.png")
        mime_type (str): The MIME type of the file (e.g., "image/png")
        file_size (int): The size of the file in bytes.
        should_close (bool): Whether the file_obj should be closed after use.
    """
    file_obj: IO[bytes]  # file or BytesIO
    filename: str
    mime_type: str
    file_size: int
    should_close: bool

FILE_INPUT = FileInput
FILE_LIST = list[FILE_INPUT]

Message = Union[CtxInteraction, str, dict|None, dict|None]


class AlertUserError(DiscordException):
    pass