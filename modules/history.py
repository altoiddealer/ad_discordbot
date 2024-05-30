# Designed by Artificiangel
# https://github.com/Artificiangel/llm-history-manager.git for future updates


from ad_discordbot.modules.logs import import_track, log, get_logger; import_track(__file__, fp=True)
logging = get_logger(__name__)
import os
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
import time
from typing import Optional
from ad_discordbot.modules.typing import ChannelID, UserID, MessageID, CtxInteraction
from typing import Union
import copy
from uuid import uuid4
import json
from ad_discordbot.modules.utils_discord import get_message_ctx_inter, get_user_ctx_inter
import asyncio
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Optional,
    TypeVar,
    Union,
    overload,
)

#######
# Utils
# Copied from discord.utils.find for use in non discord projects
T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
_Iter = Union[Iterable[T], AsyncIterable[T]]
MaybeAwaitable = Union[T, Awaitable[T]]
Coro = Coroutine[Any, Any, T]

def _find(predicate: Callable[[T], Any], iterable: Iterable[T], /) -> Optional[T]:
    return next((element for element in iterable if predicate(element)), None)

async def _afind(predicate: Callable[[T], Any], iterable: AsyncIterable[T], /) -> Optional[T]:
    async for element in iterable:
        if predicate(element):
            return element

    return None

@overload
def find(predicate: Callable[[T], Any], iterable: AsyncIterable[T], /) -> Coro[Optional[T]]:
    ...

@overload
def find(predicate: Callable[[T], Any], iterable: Iterable[T], /) -> Optional[T]:
    ...

def find(predicate: Callable[[T], Any], iterable: _Iter[T], /) -> Union[Optional[T], Coro[Optional[T]]]:
    return (
        _afind(predicate, iterable)  # type: ignore
        if hasattr(iterable, '__aiter__')  # isinstance(iterable, collections.abc.AsyncIterable) is too slow
        else _find(predicate, iterable)  # type: ignore
    )


#########################
# Config encoder/decoders
def get_uuid_hex():
    return uuid4().hex


def reduce_to_none(i):
    return None


def cls_get_pass(x):
    return x


def cls_get_uuid(x):
    return x.uuid


def cls_get_id(x):
    return x.uuid if x else None


def cls_get_id_list(x):
    out = []
    for i in x:
        if i is not None:
            i = i.uuid
        out.append(i)
    return out


def cls_get_id_dict(x):
    out = {}
    for k,v in x.items():
        if v is not None:
            v = v.uuid
        out[k] = v
    return out


def decoder_dict_int_key(x):
    return {int(k):v for k,v in x.items()}


config_dict_uuid = config(encoder=cls_get_id_dict, decoder=dict)
config_list_uuid = config(encoder=cls_get_id_list, decoder=list)
config_uuid = config(encoder=cls_get_id, decoder=str)
config_null = config(encoder=reduce_to_none, decoder=cls_get_pass)


#############
# dataclasses
@dataclass_json
@dataclass
class HMessage:
    history: 'History' = field(metadata=config_uuid)
    name: str
    text: str
    role: str # Assistant, user, internal
    author_id: UserID

    replies: Optional[list['HMessage']] = field(default_factory=list, metadata=config_list_uuid)
    reply_to: Optional['HMessage'] = field(default=None, metadata=config_uuid)

    text_visible: str = field(default='')
    id: Optional[MessageID] = field(default=None)
    audio_id: Optional[MessageID] = field(default=None)
    typing: bool = field(default=False)
    spoken: bool = field(default=False)
    created: float = field(default_factory=time.time)
    uuid: str = field(default_factory=get_uuid_hex)


    ##################
    # helper functions
    @property
    def channel(self) -> 'History':
        return self.history


    def __repr__(self):
        replies = f', Replies({len(self.replies)})' if self.replies else ''
        return f'<HMessage ({self.role}) by {self.name!r}: {self.text[:20]!r}{replies}>'


    def delta(self) -> float:
        return time.time()-self.created
    
    
    def save_to_history(self):
        self.history.append(self)
        
        
    def update(self, **kw): # TODO could replace all attrs with private internals so users don't accidentally set them.
        for k, v in kw.items():
            setattr(self, k, v)
            
        if kw:
            self.history.event_save.set()
            
        return self


    ###############
    # relationships
    def mark_as_reply_for(self, message: 'HMessage'):
        if not message:
            return self
        
        self.reply_to = message
        message.replies.append(self)
        self.history.event_save.set()
        return self


    def unmark_reply(self):
        message:HMessage = self.reply_to
        if message:
            if self in message.replies:
                message.replies.pop(self)
        self.history.event_save.set()
        return self


    ###########
    # Rendering
    def render_to_prompt(self, each_new_line=True) -> str:
        if not self.text:
            return ''

        if each_new_line:
            output = []
            for text in self.text.split('\n'):
                if not text:
                    continue
                output.append(f'{self.name}: {text}')
            return '\n'.join(output)

        return f'{self.name}: {self.text}'


    ###########
    # Save/load
    def duplicate_history(self) -> 'History':
        return copy.copy(self.history)
        
    
    def new_history_end_here(self, include_self=True) -> Union['History', None]:
        new_history = self.duplicate_history()
        if not self in new_history:
            return None

        index = new_history.index(self)
        if not include_self:
            index -= 1
        new_history._items = new_history._items[:index+1] # TODO write internal methods to set/get items

        return new_history
    
    
    def new_history_start_here(self, include_self=True) -> Union['History', None]:
        new_history = self.duplicate_history()
        if not self in new_history:
            return None

        index = new_history.index(self)
        if not include_self:
            index += 1
        new_history._items = new_history._items[index:] # TODO write internal methods to set/get items

        return new_history


    def from_ctx(self, ictx: CtxInteraction):
        self.author_id = get_user_ctx_inter(ictx).id
        self.id = get_message_ctx_inter(ictx).id
        # self.history.event_save.set() # TODO maybe?


@dataclass
class HistoryPairForTGWUI:
    user: HMessage = field(default=None)
    assistant: HMessage = field(default=None)


    def add_pair_to(self, internal:list, visible:list):
        internal.append([
            self.user.text if self.user else '',
            self.assistant.text if self.assistant else '',
        ])
        visible.append([
            (self.user.text_visible or self.user.text) if self.user else '',
            (self.assistant.text_visible or self.assistant.text) if self.assistant else '',
        ])
        return self


    def clear(self):
        self.user = None
        self.assistant = None
        return self


    def __bool__(self):
        return any([self.user, self.assistant])


@dataclass_json
@dataclass
class History:
    manager: 'HistoryManager' = field(metadata=config_uuid)
    id: ChannelID
    
    fp: Optional[str] = field(default=None)

    _last: dict[UserID, HMessage] = field(default_factory=dict, init=False, metadata=config(encoder=cls_get_id_dict, decoder=decoder_dict_int_key))
    _items: list[HMessage] = field(default_factory=list, init=False, metadata=config(encoder=cls_get_id_list,))
    uuid: str = field(default_factory=get_uuid_hex)
    _save_event: asyncio.Event = field(default_factory=asyncio.Event, init=False, metadata=config_null)
    _last_save: float = field(default_factory=time.time, init=False, metadata=config_null)

    def __copy__(self, x: 'History'):
        new = self.__class__(
            manager=x.manager,
            id=x.id,
            uuid=x.uuid,
            fp=x.fp,
            )

        new._last = x._last
        new._items = x._items
        return new


    ###########
    # Item list
    def __contains__(self, message: HMessage):
        return message in self._items


    def index(self, message: HMessage):
        return self._items.index(message)


    def clear(self):
        self._items.clear()
        self._last.clear()
        self.event_save.set()
        return self


    def append(self, message: HMessage):
        self._items.append(message)
        self._last[message.author_id] = message
        self.event_save.set()
        return self


    def pop(self, index=-1):
        self.event_save.set()
        return self._items.pop(index=index)


    def __iter__(self):
        return iter(self._items)


    def user_items(self):
        return self._last.items()


    def __setitem__(self, slice:slice, message: HMessage):
        self._items[slice] = message
        self.event_save.set()


    def __getitem__(self, slice:slice):
        return self._items[slice]
    
    
    @property
    def empty(self):
        return not self._items


    ##########
    # Messages
    def new_message(self, name, text, role, author_id, save=True, **kw) -> HMessage:
        message = HMessage(self, name, text, role, author_id, **kw)
        if save:
            self.append(message)
        return message

    
    def role_messages(self, role) -> list[HMessage]:
        return [message for message in self if message.role == role]
    
    
    def search(self, predicate):
        return find(predicate, self._items)
    
    
    def _get_sum_text(self, attr='text'):
        return sum(len(getattr(message, attr, '')) for message in self)
    
    
    def truncate(self, target_str_length: int, force=False):
        if self.manager.limit_history or force:
            while self._get_sum_text('text') > target_str_length or self._get_sum_text('text_visible') > target_str_length:
                if self.empty:
                    break
                
                self.pop(0)
                
        return self


    ###########
    # Rendering
    def render_to_tgwui_tuple(self): # TODO create caching by storing event and clearing on render.
        internal = []
        visible = []
        current_pair = HistoryPairForTGWUI()

        for message in self._items:
            if message.role == 'user':
                if current_pair.user:
                    current_pair.add_pair_to(internal, visible).clear()

                current_pair.user = message


            elif message.role == 'assistant':
                if not current_pair.assistant:
                    current_pair.assistant = message

                current_pair.add_pair_to(internal, visible).clear()


        if current_pair:
            current_pair.add_pair_to(internal, visible).clear()
            
        return internal, visible
            
            
    def render_to_tgwui(self):
        internal, visible = self.render_to_tgwui_tuple()
        return dict(internal=internal, visible=visible)
    

    def render_to_prompt(self, each_new_line=True):
        output = []
        for message in self._items:
            output.append(message.render_to_prompt(each_new_line=each_new_line))
        return '\n'.join(output)


    ###########
    # Save/load
    def last_save_delta(self):
        return time.time()-self._last_save
    
    
    @property
    def event_save(self):
        return self._save_event
        
    
    # def get_unique_save_file_name(self):
    #     return get_uuid_hex()
    
    
    # def get_save_file_name(self) -> str:
    #     if not self.file_name:
    #         self.file_name = self.get_unique_save_file_name()
    #     return self.file_name
    
    
    def trigger_save(self, fp):
        self._last_save =  time.time()
        self.event_save.clear()
        
        history = self.to_dict()
        messages = []
        for message in self._items:
            messages.append(message.to_dict())

        with open(fp, 'w', encoding='utf8') as f:
            json_out = json.dumps(dict(history=history, messages=messages), indent=2)
            f.write(json_out)
            
        log.debug(f'Saved file: {fp}')
            
        return self
    
    
    async def save(self, fp=None, modify_fp=False, timeout=10, force=False):
        if fp is not None and modify_fp:
            fp = self.manager.modify_saved_path(fp, self.id)
        fp = fp or self.fp
        
        delta = self.last_save_delta()
        if timeout:
            # wait at least timeout between saves
            if delta < timeout: 
                await asyncio.sleep(timeout-delta)
        
        if self.event_save.is_set() or force:
            self.trigger_save(fp)
            return True
        
        return False
    
    
    # def save_rendered_tgwui(self, fp):
    #     with open(fp, 'w', encoding='utf8') as f:
    #         json_out = json.dumps(self.render_to_tgwui(), indent=2)
    #         f.write(json_out)
    #     return self
    

    @classmethod
    def load_from(cls, hm: 'HistoryManager', fp, id_: ChannelID=None) -> 'History':
        '''
        Pass an id to overwrite channel id.
        '''
        with open(fp, 'r', encoding='utf8') as f:
            json_in = json.loads(f.read())
            history = json_in.get('history', {})
            messages = json_in.get('messages', [])

        # initialize history and overwrite vars
        history:'History' = cls.from_dict(history)
        history.id = id_ or history.id
        history.fp = fp
        history.manager = hm

        # add history to manager
        hm.add_history(history)

        # initialize messages
        local_message_storage: dict[str, HMessage] = {}
        for message in messages:
            message:HMessage = HMessage.from_dict(message)
            message.history = history
            history.append(message)
            local_message_storage[message.uuid] = message

        # resolve replies
        for message in history:
            message.replies = [local_message_storage.get(m) for m in message.replies]
            message.reply_to = local_message_storage.get(message.reply_to)

        return history


@dataclass
class HistoryManager:
    limit_history: bool = field(default=True)
    autosave_history: bool = field(default=False)
    autoload_history: bool = field(default=False)
    change_char_history_method: bool = field(default='new')
    greeting_or_history: bool = field(default='history')
    per_channel_history: bool = field(default=True)

    _histories: dict[ChannelID, History] = field(default_factory=dict)
    uuid: str = field(default_factory=get_uuid_hex, init=False)
    class_builder_history: type = field(default=History)


    def _get_channel_id(self, id_: ChannelID) -> ChannelID:
        if self.per_channel_history and id_ is None:
            raise Exception(f'Channel id is None and multi channel history enabled.')
        
        if not self.per_channel_history:
            return 0
        
        return id_
    
    def modify_saved_path(self, fp, id_: ChannelID=None):
        id_ = self._get_channel_id(id_)
        # if not id_:
        #     return fp
        
        path, ext = fp.rsplit('.',1)
        return f'{path}.{id_}.{ext}'
    
    
    def search_for_fp(self, id_:ChannelID):
        return
        
    
    def clear_all_history(self):
        for history in self._histories.values():
            history.clear()
            
        self._histories.clear()
        return self
    

    def unload_history(self):
        self._histories.clear()
        return self


    def add_history(self, history: History):
        self._histories[history.id] = history
        return history


    def load_history_from_fp(self, fp=None, id_: ChannelID=None) -> History:
        '''
        Pass an id to overwrite channel id.
        '''
        return self.class_builder_history.load_from(self, fp=fp, id_=id_)
    

    def get_history_for(self, id_: ChannelID=None, fp=None, modify_fp=False, search=False) -> History:
        id_ = self._get_channel_id(id_)

        history = self._histories.get(id_)
        if history is None and fp is not None:
            if modify_fp:
                fp = self.modify_saved_path(fp, id_)
            
            logging.debug(f'No channel {id_}, trying to load from file: {fp}')
            history = self.load_history_from_fp(fp=fp, id_=id_)
            
            
        elif history is None and search:
            logging.debug(f'No channel {id_}, Searching for file')
            fp = self.search_for_fp(id_)
            if fp:
                logging.debug(f'Found: {fp}')
                
                history = self.load_history_from_fp(fp=fp, id_=id_)
            

        if history is None:
            logging.debug(f'No history for channel {id_}, creating new')
            history = self.add_history(self.class_builder_history(self, id_))

        return history

