from modules.logs import import_track, log, get_logger; import_track(__file__, fp=True)
log = get_logger(__name__)
logging = log
from modules.utils_shared import task_semaphore, bot_emojis
import discord
from discord.ext import commands
from typing import Union
from modules.typing import CtxInteraction
from typing import TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from modules.history import HistoryManager, History, HMessage
    
    
MAX_MESSAGE_LENGTH = 1980
# MAX_MESSAGE_LENGTH = 200 # testing

def guild_only():
    async def predicate(ctx):
        if ctx.guild is None:
            raise commands.CheckFailure("This command can only be used in a server.")
        return True
    return commands.check(predicate)

async def react_to_user_message(clientuser: discord.User, channel, user_message:'HMessage'=None):
    try:
        user_message_id = getattr(user_message, 'id', None)
        if user_message_id and getattr(user_message, 'hidden', None) is not None:
            emoji = bot_emojis.hidden_emoji
            has_reacted = False
            discord_message = await channel.fetch_message(user_message_id)
            # check for any existing reaction
            for reaction in discord_message.reactions:
                if str(reaction.emoji) == emoji:
                    async for user in reaction.users():
                        if user == clientuser:
                            has_reacted = True
                            break
            if user_message.hidden == True and has_reacted == False:
                await discord_message.add_reaction(emoji)
            elif user_message.hidden == False and has_reacted == True:
                await discord_message.remove_reaction(emoji, clientuser)
    except Exception as e:
        log.error(f"Error reacting to user message: {e}")

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
        if task_semaphore.locked():  # If a queued item is currently being processed
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


async def send_long_message(channel, message_text, bot_message:'HMessage'=None) -> int:
    """ Splits a longer message into parts while preserving sentence boundaries and code blocks """
    activelang = ''

    # Helper function to ensure even pairs of code block markdown
    def ensure_even_code_blocks(chunk_text, code_block_inserted):
        nonlocal activelang  # Declare activelang as nonlocal to modify the global variable
        code_block_languages = ["asciidoc", "autohotkey", "bash", "coffeescript", "cpp", "cs", "css", "diff", "fix", "glsl", "ini", "json", "md", "ml", "prolog", "ps", "py", "tex", "xl", "xml", "yaml", "html"]
        code_block_count = chunk_text.count("```")
        if code_block_inserted:
            # If a code block was inserted in the previous chunk, add a leading set of "```"
            chunk_text = f"```{activelang}\n" + chunk_text
            code_block_inserted = False  # Reset the code_block_inserted flag
        code_block_count = chunk_text.count("```")
        if code_block_count % 2 == 1:
            # Check last code block for syntax like "```yaml"
            last_code_block_index = chunk_text.rfind("```")
            last_code_block = chunk_text[last_code_block_index + len("```"):].strip()
            for lang in code_block_languages:
                if (last_code_block.lower()).startswith(lang):
                    activelang = lang
                    break  # Stop checking if a match is found
            # If there is an odd number of code blocks, add a closing set of "```"
            chunk_text += "```"
            code_block_inserted = True
        return chunk_text, code_block_inserted

    if len(message_text) <= MAX_MESSAGE_LENGTH:
        sent_message = await channel.send(message_text)
    else:
        code_block_inserted = False  # Initialize code_block_inserted to False
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
            sent_message = await channel.send(chunk_text)
            if bot_message:
                bot_message.related_ids.append(sent_message.id)
                
            message_text = message_text[chunk_length:]
            if len(message_text) <= MAX_MESSAGE_LENGTH:
                # Send the remaining text as a single chunk if it's shorter than or equal to 2000 characters
                chunk_text, code_block_inserted = ensure_even_code_blocks(message_text, code_block_inserted)
                sent_message = await channel.send(chunk_text)
                break
            
    if bot_message:
        bot_message.id = sent_message.id

    return sent_message.id


# Model for editing history
class EditMessageModal(discord.ui.Modal, title="Edit Message in History"):
    def __init__(self, clientuser: discord.User, matched_hmessage: 'HMessage', target_message: discord.Message, local_history:'History'=None):
        super().__init__()
        self.target_message = target_message
        self.matched_hmessage = matched_hmessage
        self.clientuser = clientuser
        
        default_text = target_message.clean_content
        if local_history is not None:
            default_text = local_history.get_labeled_history_text(matched_hmessage, target_message.content, mention_mode='demention', label_mode='delabel')

        # Add TextInput dynamically with default value
        self.new_content = discord.ui.TextInput(
            label='New Message Content', 
            style=discord.TextStyle.paragraph, 
            min_length=1, 
            default=default_text)

        self.add_item(self.new_content)

    async def on_submit(self, inter: discord.Interaction):
        # Update text in history
        edited_message = self.new_content.value
        compound_message = ''
        # Try rebuilding text if target message was a message chunk
        if self.matched_hmessage.related_ids:
            all_original_msg_ids = [self.matched_hmessage.id] + self.matched_hmessage.related_ids
            all_original_msg_ids.sort()
            for orig_msg_id in all_original_msg_ids:
                if self.matched_hmessage.id != orig_msg_id:
                    try:
                        original_chunk_message = await inter.channel.fetch_message(orig_msg_id)
                        compound_message += original_chunk_message.clean_content
                    except:
                        log.warning(f'Failed to get message content for id {orig_msg_id} for "Edit History".')
                        compound_message = ''
                        break                    
                else:
                    compound_message += edited_message
        if compound_message:
            edited_message = compound_message
        # Update the HMessage with the new value
        self.matched_hmessage.update(text=edited_message)
        await inter.response.send_message("Message history has been edited successfully.", ephemeral=True, delete_after=5)

        # Update text in discord message
        if self.clientuser == self.target_message.author:
            await self.target_message.edit(content=edited_message[:2000])
        else:
            await inter.response.send_message("Note: The bot cannot update your message contents in Discord.", ephemeral=True, delete_after=5)
        # Warn if text was truncated
        if len(edited_message) >= 2000:
            await inter.response.send_message("Message exceeded discord text limits and was truncated to 2,000 characters. It was still replaced entirely in history", ephemeral=True, delete_after=10)

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