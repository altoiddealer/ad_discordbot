from modules.utils_shared import task_processing, bot_emojis
import discord
from discord.ext import commands
from typing import Optional, Union
from modules.typing import CtxInteraction
from typing import TYPE_CHECKING
import asyncio

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

if TYPE_CHECKING:
    from modules.history import HistoryManager, History, HMessage  # noqa: F401
    
    
MAX_MESSAGE_LENGTH = 1980
# MAX_MESSAGE_LENGTH = 200 # testing

def guild_only():
    async def predicate(ctx):
        if ctx.guild is None:
            raise commands.CheckFailure("This command can only be used in a server.")
        return True
    return commands.check(predicate)


def configurable_for_dm_if(func):
    async def predicate(ctx):
        if ctx.guild is None:
            try:
                if func(ctx):
                    return True
                
            except Exception as e:
                log.warning(f'Something went wrong with check: {e}')
            
            raise commands.CheckFailure("The bot is not configured to process this command in direct messages")
        
        return True
    
    return commands.check(predicate)

def is_direct_message(ictx:CtxInteraction):
    return ictx and getattr(ictx, 'guild') is None \
        and hasattr(ictx, 'channel') and isinstance(ictx.channel, discord.DMChannel)


def get_hmessage_emojis(hmessage:'HMessage') -> list[str]:
    history_emojis = {'is_continued': bot_emojis.continue_emoji,
                      'regenerated_from': bot_emojis.regen_emoji,
                      'hidden': bot_emojis.hidden_emoji}

    emojis_for_hmessage = []

    for key, value in history_emojis.items():
        if getattr(hmessage, key, False):
            emojis_for_hmessage.append(value)
    
    return emojis_for_hmessage

async def update_message_reactions(client_user:discord.ClientUser, emojis_list:list[str], discord_msg:discord.Message):
    try:
        reactions_to_add = emojis_list
        already_reacted = []
        reactions_to_remove = []

        bot_emojis_list = bot_emojis.get_emojis()

        # check for existing reactions
        for reaction in discord_msg.reactions:
            if reaction.emoji in bot_emojis_list:
                async for user in reaction.users():
                    if user == client_user:
                        if reaction.emoji not in emojis_list:
                            reactions_to_remove.append(reaction.emoji)
                        else:
                            already_reacted.append(reaction.emoji)

        for reaction in reactions_to_remove:
            await discord_msg.remove_reaction(reaction, client_user)

        for reaction in reactions_to_add:
            if reaction not in already_reacted:
                await discord_msg.add_reaction(reaction)

    except Exception as e:
        log.error(f"Error updating reactions for discord message id '{discord_msg.id}': {e}")


# Applies history reactions to a list of discord messages, such as all related messages from 'Continue' function
async def apply_reactions_to_messages(client_user:discord.ClientUser,
                                      ictx:CtxInteraction,
                                      hmsg:Optional['HMessage']=None,
                                      msg_id_list:Optional[list[int]]=None,
                                      ictx_msg:Optional[discord.Message]=None):
    try:
        if hmsg is None:
            return
        if msg_id_list is None:
            msg_id_list = [hmsg.id]

        # Get correct emojis for current message
        emojis_for_msg = get_hmessage_emojis(hmsg)
        
        # Iterate over list of discord message IDs and update reactions for the discord message objects
        for msg_id in msg_id_list:
            # skip fetching ictx message if provided and matched
            if ictx_msg and msg_id == ictx_msg.id:
                discord_msg = ictx_msg
            # fetch message object from id
            else:
                discord_msg = await ictx.channel.fetch_message(msg_id)
            # Update reactions for the message
            await update_message_reactions(client_user, emojis_for_msg, discord_msg)
    except Exception as e:
        log.error(f'Error while processing reactions for messages: {e}')

# Delete discord message without "delete_after" attribute
async def sleep_delete_message(message: discord.Message, wait:int=5):
    try:
        await asyncio.sleep(wait)
        await message.delete()
    except Exception as e:
        log.error(f'Failed to delete stubborn message: {e}')

# Send message response to user's interaction command
async def ireply(ictx: 'CtxInteraction', process):
    try:
        if task_processing.is_set():
            message = f'Your {process} request was added to the task queue'
        else:
            message = f'Processing your {process} request'
        
        if hasattr(ictx, 'reply') and callable(getattr(ictx, 'reply')):
            await ictx.reply(message, ephemeral=True, delete_after=5)
        elif hasattr(ictx, 'response') and callable(getattr(ictx.response, 'send_message')):
            await ictx.response.send_message(message, ephemeral=True, delete_after=5)
        else:
            raise AttributeError("ictx object has neither 'reply' nor 'response.send' methods")

    except Exception as e:
        log.error(f"Error sending message response to user's interaction command: {e}")


async def send_long_message(channel, message_text, bot_hmessage:Optional['HMessage']=None, ref_message:Optional[discord.Message]=None) -> discord.Message:
    """ Splits a longer message into parts while preserving sentence boundaries and code blocks """
    active_lang = ''

    # Helper function to ensure even pairs of code block markdown
    def ensure_even_code_blocks(chunk_text, code_block_inserted):
        nonlocal active_lang  # Declare active_lang as nonlocal to modify the global variable
        code_block_languages = ["asciidoc", "autohotkey", "bash", "coffeescript", "cpp", "cs", "css", "diff", "fix", "glsl", "ini", "json", "md", "ml", "prolog", "ps", "py", "tex", "xl", "xml", "yaml", "html"]
        code_block_count = chunk_text.count("```")
        if code_block_inserted:
            # If a code block was inserted in the previous chunk, add a leading set of "```"
            chunk_text = f"```{active_lang}\n" + chunk_text
            code_block_inserted = False  # Reset the code_block_inserted flag
        code_block_count = chunk_text.count("```")
        if code_block_count % 2 == 1:
            # Check last code block for syntax like "```yaml"
            last_code_block_index = chunk_text.rfind("```")
            last_code_block = chunk_text[last_code_block_index + len("```"):].strip()
            for lang in code_block_languages:
                if (last_code_block.lower()).startswith(lang):
                    active_lang = lang
                    break  # Stop checking if a match is found
            # If there is an odd number of code blocks, add a closing set of "```"
            chunk_text += "```"
            code_block_inserted = True
        return chunk_text, code_block_inserted

    if len(message_text) <= MAX_MESSAGE_LENGTH:
        sent_message = await channel.send(message_text, reference=ref_message)
    else:
        code_block_inserted = False  # Initialize code_block_inserted to False
        first_chunk_sent = False
        while message_text:
            # Find the last occurrence of either a line break or the end of a sentence
            last_line_break = message_text.rfind("\n", 0, MAX_MESSAGE_LENGTH)
            last_sentence_end = message_text.rfind(". ", 0, MAX_MESSAGE_LENGTH)
            # Determine the index to split the string
            if last_line_break >= 0 and last_sentence_end >= 0:
                # If both a line break and a sentence end were found, choose the one that occurred last
                chunk_length = max(last_line_break, last_sentence_end) + 1
            elif last_line_break >= 0:
                # If only a line break was found, use it as the split point
                chunk_length = last_line_break + 1
            elif last_sentence_end >= 0:
                # If only a sentence end was found, use it as the split point
                chunk_length = last_sentence_end + 2  # Include the period and space
            else:
                chunk_length = MAX_MESSAGE_LENGTH # If neither was found, split at the maximum limit of 2000 characters
            chunk_text = message_text[:chunk_length]
            chunk_text, code_block_inserted = ensure_even_code_blocks(chunk_text, code_block_inserted)
            if not first_chunk_sent:
                sent_message = await channel.send(chunk_text, reference=ref_message)
            else:
                sent_message = await channel.send(chunk_text)
            if bot_hmessage:
                bot_hmessage.related_ids.append(sent_message.id)
                
            message_text = message_text[chunk_length:]
            if len(message_text) <= MAX_MESSAGE_LENGTH:
                # Send the remaining text as a single chunk if it's shorter than or equal to 2000 characters
                chunk_text, code_block_inserted = ensure_even_code_blocks(message_text, code_block_inserted)
                sent_message = await channel.send(chunk_text)
                break
            
    if bot_hmessage:
        bot_hmessage.id = sent_message.id

    return sent_message

async def replace_msg_in_history_and_discord(client_user:discord.Client, ictx:CtxInteraction, params, text:str, text_visible:str, apply_reactions:bool=True) -> Optional['HMessage']:
    channel = ictx.channel
    updated_hmessage: Optional[HMessage] = getattr(params, 'user_hmessage_to_update', None) or getattr(params, 'bot_hmessage_to_update', None)
    hmsg_hidden = getattr(params, 'user_hmsg_hidden', None) or getattr(params, 'bot_hmsg_hidden', None) or updated_hmessage.hidden 
    target_discord_msg_id = getattr(params, 'target_discord_msg_id', None)
    ref_message = getattr(params, 'ref_message', None)
    try:
        if not target_discord_msg_id:
            target_discord_msg_id = updated_hmessage.id
        target_discord_msg = await channel.fetch_message(target_discord_msg_id)

        # Only modify discord message(s) if they are responses from the bot
        if target_discord_msg.author == client_user:
            # Collect all messages that are part of the original message
            messages_to_remove = [updated_hmessage.id] + updated_hmessage.related_ids
            # Remove target message from the list
            if target_discord_msg.id in messages_to_remove:
                messages_to_remove.remove(target_discord_msg.id)
            # Delete all other messages from discord
            for message_id in messages_to_remove:
                local_message = await channel.fetch_message(message_id)
                if local_message:
                    await local_message.delete()

            # Update original discord message, or send new one if too long
            if len(text) < MAX_MESSAGE_LENGTH:
                await target_discord_msg.edit(content=text)
            else:
                await target_discord_msg.delete()
                if getattr(params, 'bot_hmessage_to_update', None):
                    await send_long_message(channel, text, bot_hmessage=updated_hmessage, ref_message=ref_message)
                else:
                    await send_long_message(channel, text, bot_hmessage=None, ref_message=ref_message)

        # Clear related IDs attribute
        updated_hmessage.related_ids.clear() # TODO maybe add a 'fresh' method to HMessage? - For Reality
        # Update the HMessage
        updated_hmessage.update(text=text, text_visible=text_visible, hidden=hmsg_hidden)

        # Apply any reactions applicable to message
        if apply_reactions:
            await apply_reactions_to_messages(client_user, ictx, updated_hmessage)

        return updated_hmessage
    except Exception as e:
        log.error(f"An error occurred while replacing message in history and Discord: {e}")
        return None

async def rebuild_chunked_message(ictx:CtxInteraction, msg_id_list:list=None, ictx_msg:discord.Message=None):
    rebuilt_message = ''
    for msg_id in msg_id_list:
        try:
            if msg_id == ictx_msg:
                chunk_message = ictx_msg
            else:
                chunk_message = await ictx.channel.fetch_message(msg_id)
            rebuilt_message += chunk_message.clean_content
            log.debug("rebuilt_message:", rebuilt_message)
        except Exception:
            log.warning(f'Failed to get message content for id {msg_id}.')
            rebuilt_message = ''
            break
    return rebuilt_message


# Modal for editing history
class EditMessageModal(discord.ui.Modal, title="Edit Message in History"):
    def __init__(self, client_user: Optional[discord.ClientUser], ictx:CtxInteraction, matched_hmessage: 'HMessage', target_discord_msg: discord.Message, apply_reactions:bool=True, params=None):
        super().__init__()
        self.target_discord_msg = target_discord_msg
        self.matched_hmessage = matched_hmessage
        self.client_user = client_user
        self.ictx = ictx
        self.apply_reactions = apply_reactions
        self.params = params

        # Add TextInput dynamically with default value
        self.new_content = discord.ui.TextInput(
            label='New Message Content', 
            style=discord.TextStyle.paragraph, 
            min_length=1, 
            default=matched_hmessage.text)

        self.add_item(self.new_content)

    async def on_submit(self, inter: discord.Interaction):
        # Update text in history
        new_text = self.new_content.value
        setattr(self.params, 'user_hmessage_to_update', self.matched_hmessage)
        await replace_msg_in_history_and_discord(self.client_user, self.ictx, params=self.params, text=new_text, text_visible=new_text, apply_reactions=self.apply_reactions)
        if self.target_discord_msg.author != self.client_user:
            await inter.response.send_message("Message history has been edited successfully (Note: the bot cannot update your discord message).", ephemeral=True, delete_after=7)
        else:
            await inter.response.send_message("Message history has been edited successfully.", ephemeral=True, delete_after=5)

class SelectedListItem(discord.ui.Select):
    def __init__(self, options, placeholder, custom_id):
        super().__init__(placeholder=placeholder, min_values=0, max_values=1, options=options, custom_id=custom_id)

    async def callback(self, interaction: discord.Interaction):
        if self.values:
            self.view.selected_item = int(self.values[0])
        await interaction.response.defer()
        # if self.view.num_menus == 1: # Stop the view if there is only one menu item (skip "Submit" button)
        self.view.stop()

class SelectOptionsView(discord.ui.View):
    '''
    Use view.warned to check if too many items message has been logged.
    Pass warned=True to bypass warning.
    '''

    def __init__(self, all_items, max_menus=4, max_items_per_menu=25, custom_id_prefix='items', placeholder_prefix='Items ', unload_item=None, warned=False):
        super().__init__()
        # Get item value for Submit and Unload buttons
        #models_submit_btn = None
        models_unload_btn = None
        for child in self.children:
            if child.custom_id == 'models_unload':
                models_unload_btn = child
            # elif child.custom_id == 'models_submit':
            #     models_submit_btn = child

        # Value for Unload model, if any
        self.unload_item = unload_item
        # Remove "Unload" button if N/A for command
        if not self.unload_item:
            self.remove_item(models_unload_btn)

        self.selected_item = None
        self.warned = warned

        assert max_items_per_menu <= 25
        assert max_menus <= 4

        self.all_items = all_items
        #self.num_menus = 0

        all_choices = [discord.SelectOption(label=name[:100], value=ii) for ii, name in enumerate(self.all_items)]

        for menu_ii in range(max_menus): # 4 max dropdowns
            local_options = all_choices[max_items_per_menu*menu_ii: max_items_per_menu*(menu_ii+1)]
            if not local_options: # end of items
                break

            self.add_item(SelectedListItem(options=local_options,
                                            placeholder=f'{placeholder_prefix}{self.label_formatter(local_options, menu_ii)}',
                                            custom_id=f"{custom_id_prefix}_{menu_ii}_select",
                                            ))
            #self.num_menus += 1 # Count dropdowns. If only one, "Submit" button will be removed

        menu_ii += 1
        local_options = all_choices[max_items_per_menu*menu_ii: max_items_per_menu*(menu_ii+1)]
        if local_options and not self.warned:
            log.warning(f'Too many models, the menu will be truncated to the first {max_items_per_menu*max_menus}.')
            self.warned = True

        # Remove Submit button if only one dropdown
        # if self.num_menus == 1:
        #     self.remove_item(models_submit_btn)

    def label_formatter(self, local_options, menu_ii):
        return f'{local_options[0].label[0]}-{local_options[-1].label[0]}'.upper()

    def get_selected(self, items:list=None):
        if self.selected_item == self.unload_item:
            return self.unload_item
        items = items or self.all_items
        return items[self.selected_item]

    # We may want this in the future...
    # @discord.ui.button(label='Submit', style=discord.ButtonStyle.primary, custom_id="models_submit", row=4)
    # async def submit_button(self, interaction: discord.Interaction, button:discord.ui.Button):
    #     if self.selected_item is None:
    #         await interaction.response.send_message('No Image model selected.', ephemeral=True, delete_after=5)
    #     else:
    #         await interaction.response.defer()
    #         self.stop()

    @discord.ui.button(label='Unload Model', style=discord.ButtonStyle.danger, custom_id="models_unload", row=4)
    async def unload_model_button(self, interaction: discord.Interaction, button:discord.ui.Button):
        self.selected_item = self.unload_item
        await interaction.response.defer()
        self.stop()


def get_user_ctx_inter(ictx: CtxInteraction) -> Union[discord.User, discord.Member]:
    # Found instances of "i" with \((self, )?i[^a-z_\)]
    if isinstance(ictx, discord.Interaction):
        return ictx.user
    return ictx.author


def get_message_ctx_inter(ictx: CtxInteraction) -> discord.Message:
    if isinstance(ictx, discord.Message):
        return ictx
    return ictx.message


class Embeds:
    def __init__(self, config:dict, ictx:CtxInteraction|None=None):
        self.channel:discord.TextChannel|None = ictx.channel if ictx else None
        self.color:int = config['discord'].get('embed_settings', {}).get('color', 0x1e1f22)
        self.enabled_embeds:dict = config['discord'].get('embed_settings', {}).get('show_embeds', {})

        self.root_url:str = 'https://github.com/altoiddealer/ad_discordbot'

        self.embeds:dict = {}
        self.sent_msg_embeds:dict = {}

        self.init_default_embeds()

    def enabled(self, name:str) -> bool:
        return self.enabled_embeds.get(name, True) # all enabled by default

    def init_default_embeds(self):
        if self.enabled('system'):
            self.create("system", "System Notification", " ", url=self.root_url, color=self.color)
        if self.enabled('images'):
            self.create("img_gen", "Processing image generation ...", " ", url=self.root_url, color=self.color)
            self.create("img_send", "User requested an image ...", " ", url=self.root_url, color=self.color)
        if self.enabled('change'):
            self.create("change", "Change Notification", " ", url=self.root_url, color=self.color)
        if self.enabled('flow'):
            self.create("flow", "Flow Notification", " ", url_suffix="/wiki/tags", color=self.color)

    def create(self, name:str, title:str=' ', description:str=' ', url_suffix:str|None=None, url:str|None=None, color:int|None=None) -> discord.Embed:
        if url or url_suffix:
            url = url if url_suffix is None else f'{self.root_url}{url_suffix}'
        self.embeds[name] = discord.Embed(title=title, description=description, url=url, color=color)
        return self.embeds[name]

    def get(self, name:str) -> discord.Embed|None:
        return self.embeds.get(name, None)

    def get_sent_msg(self, name:str) -> discord.Message|None:
        return self.sent_msg_embeds.get(name, None)
    
    async def delete(self, name:str):
        previously_sent_embed:discord.Message = self.sent_msg_embeds.pop(name, None)
        if previously_sent_embed:
            await previously_sent_embed.delete()

    def update(self, name:str, title:str|None=None, description:str|None=None, color:int|None=None, url_suffix:str|None=None, url:str|None=None) -> discord.Embed:
        embed:discord.Embed = self.embeds.get(name)
        if title:
            embed.title = title
        if description:
            embed.description = description
        if color:
            embed.color = color
        if url or url_suffix:
            embed.url = url if url else f'{self.root_url}{url_suffix}'
        return embed
    
    async def edit(self, name:str, title:str|None=None, description:str|None=None, color:int|None=None, url_suffix:str|None=None, url:str|None=None) -> None|discord.Message:
        # Return if not configured
        if not self.enabled(name):
            return
        # Get the previously sent embed
        previously_sent_embed:discord.Message = self.sent_msg_embeds.pop(name, None)
        # Retain the message while editing Embed
        if previously_sent_embed:
            self.sent_msg_embeds[name] = await previously_sent_embed.edit(embed = self.update(name, title, description, color, url_suffix, url))
            return self.sent_msg_embeds[name]
        return None

    async def send(self, name:str, title:str|None=None, description:str|None=None, color:int|None=None, url_suffix:str|None=None, url:str|None=None, channel:discord.TextChannel|None=None, delete_after:int|None=None) -> None|discord.Message:
        send_channel = channel or self.channel or None
        # Return if not configured
        if not self.enabled(name) or (self.channel is None and channel is None):
            return
        # Retain the message while sending Embed
        updated_embed = self.update(name, title, description, color, url_suffix, url)
        self.sent_msg_embeds[name] = await send_channel.send(embed = updated_embed, delete_after=delete_after)
        return self.sent_msg_embeds[name]

    async def edit_or_send(self, name:str, title:str|None=None, description:str|None=None, color:int|None=None, url_suffix:str|None=None, url:str|None=None, channel:discord.TextChannel|None=None) -> None|discord.Embed|discord.Message:
        send_channel = channel or self.channel or None
        # Return if not configured
        if not self.enabled(name):
            return
        # Get the previously sent embed
        previously_sent_embed:discord.Message = self.sent_msg_embeds.pop(name, None)
        # Retain the message while sending/editing Embed
        if previously_sent_embed:
            self.sent_msg_embeds[name] = await previously_sent_embed.edit(embed = self.update(name, title, description, color, url_suffix, url))
            return self.sent_msg_embeds[name]
        elif send_channel is None:
            return None
        else:
            self.sent_msg_embeds[name] = await send_channel.send(embed = self.update(name, title, description, color, url_suffix, url))
            return self.sent_msg_embeds[name]

    def helpmenu(self) -> discord.Embed:
        system_json = {
            "title": "Welcome to ad_discordbot!",
            \
            "description": """
            **/helpmenu** - Display this message
            **/character** - Change character
            **/main** - Toggle if Bot always replies, per channel
            **/image** - prompt an image to be generated (or try "draw <subject>")
            **/speak** - if TTS settings are enabled, the bot can speak your text
            **__Changing settings__** ('.../ad\_discordbot/dict\_.yaml' files)
            **/imgmodel** - Change Img model and any model-specific settings
            """,
            \
            "url": self.root_url,
            "color": self.color
        }
        return discord.Embed().from_dict(system_json)