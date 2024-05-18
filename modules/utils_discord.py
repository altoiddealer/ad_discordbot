from ad_discordbot.modules.logs import import_track, log, get_logger; import_track(__file__, fp=True)
logging = get_logger(__name__)
from ad_discordbot.modules.utils_shared import task_semaphore
import discord


# Send message response to user's interaction command
async def ireply(i, process):
    try:
        if task_semaphore.locked(): # If a queued item is currently being processed
            ireply = await i.reply(f'Your {process} request was added to the task queue', ephemeral=True, delete_after=5)
            # del_time = 5
        else:
            ireply = await i.reply(f'Processing your {process} request', ephemeral=True, delete_after=3)
        #     del_time = 1
    except Exception as e:
        logging.error(f"Error sending message response to user's interaction command: {e}")
        
        
        
async def send_long_message(channel, message_text):
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

    if len(message_text) <= 1980:
        sent_message = await channel.send(message_text)
    else:
        code_block_inserted = False  # Initialize code_block_inserted to False
        while message_text:
            # Find the last occurrence of either a line break or the end of a sentence
            last_line_break = message_text.rfind("\n", 0, 1980)
            last_sentence_end = message_text.rfind(". ", 0, 1980)
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
                chunk_length = 1980 # If neither was found, split at the maximum limit of 2000 characters
            chunk_text = message_text[:chunk_length]
            chunk_text, code_block_inserted = ensure_even_code_blocks(chunk_text, code_block_inserted)
            sent_message = await channel.send(chunk_text)
            message_text = message_text[chunk_length:]
            if len(message_text) <= 1980:
                # Send the remaining text as a single chunk if it's shorter than or equal to 2000 characters
                chunk_text, code_block_inserted = ensure_even_code_blocks(message_text, code_block_inserted)
                sent_message = await channel.send(chunk_text)
                break
            
class SelectedListItem(discord.ui.Select):
    def __init__(self, options, placeholder, custom_id):
        super().__init__(placeholder=placeholder, min_values=0, max_values=1, options=options, custom_id=custom_id)

    async def callback(self, interaction: discord.Interaction):
        if self.values:
            self.view.selected_item = int(self.values[0])
        await interaction.response.defer()
        # Stop the view if there is only one menu item (skip "Submit" button)
        if self.view.num_menus == 1:
            self.view.stop()
        
class SelectOptionsView(discord.ui.View):
    '''
    Use view.warned to check if too many items message has been logged.
    Pass warned=True to bypass warning.
    '''

    def __init__(self, all_items, max_menus=4, max_items_per_menu=25, custom_id_prefix='items', placeholder_prefix='Items ', unload_item=None, warned=False):
        super().__init__()
        # Get item value for Submit and Unload buttons
        models_submit_btn = None
        models_unload_btn = None
        for child in self.children:
            if child.custom_id == 'models_submit':
                models_submit_btn = child
            elif child.custom_id == 'models_unload':
                models_unload_btn = child

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
        self.num_menus = 0

        all_choices = [discord.SelectOption(label=name[:100], value=ii) for ii, name in enumerate(self.all_items)]

        for menu_ii in range(max_menus): # 4 max dropdowns
            local_options = all_choices[max_items_per_menu*menu_ii: max_items_per_menu*(menu_ii+1)]
            if not local_options: # end of items
                break
            
            self.add_item(SelectedListItem(options=local_options,
                                            placeholder=f'{placeholder_prefix}{self.label_formatter(local_options, menu_ii)}', 
                                            custom_id=f"{custom_id_prefix}_{menu_ii}_select",
                                            ))
            
        menu_ii += 1
        self.num_menus += 1 # Count dropdowns. If only one, "Submit" button will be removed
        local_options = all_choices[max_items_per_menu*menu_ii: max_items_per_menu*(menu_ii+1)]
        if local_options and not self.warned:
            logging.warning(f'Too many models, the menu will be truncated to the first {max_items_per_menu*max_menus}.')
            self.warned = True

        # Remove Submit button if only one dropdown
        if self.num_menus == 1:
            self.remove_item(models_submit_btn)
            
    def label_formatter(self, local_options, menu_ii):
        return f'{local_options[0].label[0]}-{local_options[-1].label[0]}'.upper()
    
    def get_selected(self, items:list=None):
        if self.selected_item == self.unload_item:
            return self.unload_item
        items = items or self.all_items
        return items[self.selected_item]

    @discord.ui.button(label='Submit', style=discord.ButtonStyle.primary, custom_id="models_submit", row=4)
    async def submit_button(self, interaction: discord.Interaction, button:discord.ui.Button):
        if self.selected_item is None:
            await interaction.response.send_message('No Image model selected.', ephemeral=True, delete_after=5)
        else:
            await interaction.response.defer()
            self.stop()

    @discord.ui.button(label='Unload Model', style=discord.ButtonStyle.danger, custom_id="models_unload", row=4)
    async def unload_model_button(self, interaction: discord.Interaction, button:discord.ui.Button):
        self.selected_item = self.unload_item
        await interaction.response.defer()
        self.stop()