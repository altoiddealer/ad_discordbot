from ad_discordbot.modules.logs import import_track, log, get_logger; import_track(__file__, fp=True)
logging = get_logger(__name__)
import os
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
import time
from typing import Optional
from ad_discordbot.modules.typing import ChannelID, UserID, MessageID, CtxInteraction
from typing import Union
import discord
import copy
from uuid import uuid4
import json
from ad_discordbot.modules.utils_discord import get_message_ctx_inter, get_user_ctx_inter
import asyncio

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


    ###############
    # relationships
    def mark_as_reply_for(self, message: 'HMessage'):
        if not message:
            return self
        
        self.reply_to = message
        message.replies.append(self)
        return self


    def unmark_reply(self):
        message:HMessage = self.reply_to
        if message:
            if self in message.replies:
                message.replies.pop(self)
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
    def new_history_from_here(self) -> Union['History', None]:
        new_history = copy.copy(self.history)
        if not self in new_history:
            return None

        index = new_history.index(self)
        new_history._items = new_history._items[:index+1] # TODO write internal methods to set/get items

        return new_history


    def from_ctx(self, ictx: CtxInteraction):
        self.author_id = get_user_ctx_inter(ictx).id
        self.id = get_message_ctx_inter(ictx).id


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
    
    file_name: Optional[str] = field(default=None)

    _last: dict[UserID, HMessage] = field(default_factory=dict, init=False, metadata=config(encoder=cls_get_id_dict, decoder=decoder_dict_int_key))
    _items: list[HMessage] = field(default_factory=list, init=False, metadata=config(encoder=cls_get_id_list,))
    uuid: str = field(default_factory=get_uuid_hex)
    _save_event: asyncio.Event = field(default_factory=asyncio.Event, init=False, metadata=config_null)

    def __copy__(self, x: 'History'):
        new = self.__class__(
            manager=x.manager,
            id=x.id,
            uuid=x.uuid,
            file_name=x.file_name,
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
        self._save_event.set()
        return self


    def append(self, message: HMessage):
        self._items.append(message)
        self._last[message.author_id] = message
        self._save_event.set()
        return self


    def pop(self, index=-1):
        self._save_event.set()
        return self._items.pop(index=index)


    def __iter__(self):
        return iter(self._items)


    def user_items(self):
        return self._last.items()


    def __setitem__(self, slice:slice, message: HMessage):
        self._items[slice] = message
        self._save_event.set()


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
    def render_to_tgwui(self):
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

        return dict(internal=internal, visible=visible)


    def render_to_prompt(self, each_new_line=True):
        output = []
        for message in self._items:
            output.append(message.render_to_prompt(each_new_line=each_new_line))
        return '\n'.join(output)


    ###########
    # Save/load
    def get_unique_save_file_name(self):
        return get_uuid_hex()
    
    
    def get_save_file_name(self) -> str:
        if not self.file_name:
            self.file_name = self.get_unique_save_file_name()
        return self.file_name
    
    
    def trigger_save(self, fp):
        self._save_event.clear()
        
        history = self.to_dict()
        messages = []
        for message in self._items:
            messages.append(message.to_dict())

        with open(fp, 'w', encoding='utf8') as f:
            json_out = json.dumps(dict(history=history, messages=messages), indent=2)
            f.write(json_out)
            
        return self
    
    
    async def save(self, fp, timeout=30):
        if timeout:
            await asyncio.sleep(timeout)
        
        if self._save_event.is_set():
            self.trigger_save(fp)
            return True
        
        return False
    
    
    # def save_rendered_tgwui(self, fp):
    #     with open(fp, 'w', encoding='utf8') as f:
    #         json_out = json.dumps(self.render_to_tgwui(), indent=2)
    #         f.write(json_out)
    #     return self
    

    @classmethod
    def load_from(cls, fp, hm: 'HistoryManager', id_: ChannelID=None) -> 'History':
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
    per_channel_history_enabled: bool = field(default=True)

    _histories: dict[ChannelID, History] = field(default_factory=dict)
    uuid: str = field(default_factory=get_uuid_hex, init=False)
    class_builder_history: type = field(default=History)


    def _get_channel_id(self, id_: ChannelID) -> ChannelID:
        if self.per_channel_history_enabled and id_ is None:
            raise Exception(f'Channel id is None and multi channel history enabled.')
        
        if not self.per_channel_history_enabled:
            return 0
        
        return id_


    def add_history(self, history: History):
        self._histories[history.id] = history
        return history


    def get_history_for(self, id_: ChannelID=None, fp=None) -> History:
        id_ = self._get_channel_id(id_)

        history = self._histories.get(id_)
        if history is None and fp is not None:
            print('no channel, trying to load from file:')
            history = self.load_history_from(fp, id_)

        if history is None:
            history = self.add_history(self.class_builder_history(self, id_))

        return history


    def load_history_from(self, fp, id_: ChannelID=None) -> History:
        '''
        Pass an id to overwrite channel id.
        '''
        return self.class_builder_history.load_from(fp, self, id_)
    
    

if __name__ == '__main__':
    file_path = os.path.join('ad_discordbot', 'modules', 'TMP_h2.json')
    hm = HistoryManager()
    h = hm.get_history_for(100)

    m = HMessage(h, 'Kat', 'Hey!', 'user', 11111)
    h.append(m)

    m2 = HMessage(h, 'Skye', 'Hello Kat', 'assistant', 22222)
    h.append(m2)
    m2.mark_as_reply_for(m)


    h.append(HMessage(h, 'Kat', 'What is 2+2?', 'user', 11111))
    h.append(HMessage(h, 'Skye', '4', 'assistant', 22222))
    h.append(HMessage(h, 'Skye', 'What else can I assist you with?', 'assistant', 22222))
    h.append(HMessage(h, 'Skye', 'Goodbye', 'assistant', 22222))
    h.append(HMessage(h, 'Kat', 'No wait!', 'user', 11111))

    print(h)

    print(h.render_to_tgwui())
    print(h.render_to_prompt())

    # print(h.to_dict())

    h.save(file_path)

    print('------------------')
    print('After save')

    hm = HistoryManager()
    h = hm.get_history_for(100, file_path)
    print(h)

    print(h.render_to_tgwui())
    print(h.render_to_prompt())


    # print(h.to_dict())

    print(h[0].replies)
    print(h[1].reply_to)
    print(h[1:3])


    # print(json.dumps(h.render_to_tgwui(), indent=2))
    