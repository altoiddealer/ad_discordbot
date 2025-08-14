import asyncio
import copy
import html
import json
import os
import re
import sys
import yaml
import aiohttp
import importlib
import traceback
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple, Callable
from modules.typing import CtxInteraction
from modules.utils_shared import shared_path, config, bot_database, patterns, is_tgwui_integrated
from modules.utils_misc import check_probability

sys.path.append(shared_path.dir_tgwui)

import modules.extensions as tgwui_extensions_module
import modules.shared as tgwui_shared_module
import modules.utils as tgwui_utils_module
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import get_model_metadata, update_model_parameters, get_fallback_settings, infer_loader

# Cache for already-loaded functions
_tgwui_cache: dict[str, Callable] = {}

def get_tgwui_functions(name: str) -> Callable:
    """
    Lazily import and return a TGWUI function by name.

    Returns:
        Callable: The requested function.
    """
    # If we already loaded it, just return it
    if name in _tgwui_cache:
        return _tgwui_cache[name]

    # Import and store only once
    if name == "chat":
        from modules.chat import get_stopping_strings, generate_chat_prompt, generate_search_query, add_message_attachment, add_web_search_attachments, update_message_metadata, get_current_timestamp, add_message_version
        from modules.text_generation import generate_reply
        funcs = {"get_stopping_strings": get_stopping_strings,
                 "generate_chat_prompt": generate_chat_prompt,
                 "generate_reply": generate_reply,
                 "generate_search_query": generate_search_query,
                 "add_message_attachment": add_message_attachment,
                 "add_web_search_attachments": add_web_search_attachments,
                 "update_message_metadata": update_message_metadata,
                 "get_current_timestamp": get_current_timestamp,
                 "add_message_version": add_message_version}
        _tgwui_cache.update(funcs)
        return funcs  # Returns dict if "chat" is requested

    elif name == "load_character":
        from modules.chat import load_character
        _tgwui_cache[name] = load_character

    elif name == "chatbot_wrapper":
        from modules.chat import chatbot_wrapper
        _tgwui_cache[name] = chatbot_wrapper

    elif name == "save_history":
        from modules.chat import save_history
        _tgwui_cache[name] = save_history

    elif name == "count_tokens":
        from modules.prompts import count_tokens
        _tgwui_cache[name] = count_tokens

    elif name == "load_model":
        from modules.models import load_model
        _tgwui_cache[name] = load_model

    elif name == "unload_model":
        from modules.models import unload_model
        _tgwui_cache[name] = unload_model

    else:
        raise ValueError(f"Unknown TGWUI function name: {name}")

    return _tgwui_cache[name]

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

class TTS:
    def __init__(self):
        # TGWUI Extension method
        self.enabled:bool = False

        self.settings:dict = config.ttsgen
        self.supported_extensions = ['alltalk_tts', 'coqui_tts', 'silero_tts', 'elevenlabs_tts', 'edge_tts', 'vits_api_tts']
        self.extension:Optional[str] = config.ttsgen.get('tgwui_extension')
        self.voice_key:Optional[str] = None
        self.lang_key:Optional[str] = None

    # Toggles TTS on/off
    async def toggle_tts_extension(self, settings, toggle:str='on') -> bool:
        try:
            #settings:"Settings" = get_settings(ictx)
            llmcontext_dict = vars(settings.llmcontext)
            extensions:dict = copy.deepcopy(llmcontext_dict.get('extensions', {}))
            if toggle == 'off' and extensions.get(self.extension, {}).get('activate'):
                extensions[self.extension]['activate'] = False
                await tgwui.update_extensions(extensions)
                # Return True if subsequent toggle_tts_extension() should enable TTS
                return True
            elif toggle == 'on':
                extensions[self.extension]['activate'] = True
                await tgwui.update_extensions(extensions)
        except Exception as e:
            log.error(f'[{self.tts.extension}] An error occurred while toggling the TTS on/off: {e}')
        return False
    
    async def apply_toggle_tts(self, settings) -> str:
        if self.enabled:
            await self.toggle_tts_extension(settings, toggle='off')
            self.enabled = False
            return 'disabled'
        else:
            await self.toggle_tts_extension(settings, toggle='on')
            self.enabled = True
            return 'enabled'

    async def fetch_speak_options(self) -> Tuple[list, list]:
        try:
            lang_list = []
            all_voices = []

            if self.extension == 'coqui_tts' or 'alltalk' in self.extension:
                lang_list = ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French', 'German', 'Hungarian', 'Italian', 'Japanese', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Spanish', 'Turkish']
                if self.extension == 'coqui_tts':
                    from extensions.coqui_tts.script import get_available_voices
                    all_voices = get_available_voices()
                else:
                    from extensions.alltalk_tts.script import get_available_voices
                    all_voices = get_available_voices()

            elif self.extension == 'silero_tts':
                lang_list = ['English', 'Spanish', 'French', 'German', 'Russian', 'Tatar', 'Ukranian', 'Uzbek', 'English (India)', 'Avar', 'Bashkir', 'Bulgarian', 'Chechen', 'Chuvash', 'Kalmyk', 'Karachay-Balkar', 'Kazakh', 'Khakas', 'Komi-Ziryan', 'Mari', 'Nogai', 'Ossetic', 'Tuvinian', 'Udmurt', 'Yakut']
                log.warning('''There's too many Voice/language permutations to make them all selectable in "/speak" command. Loading a bunch of English options. Non-English languages will automatically play using respective default speaker.''')
                all_voices = [f"en_{index}" for index in range(1, 76)] # will just include English voices in select menus. Other languages will use defaults.

            elif self.extension == 'elevenlabs_tts':
                lang_list = ['English', 'German', 'Polish', 'Spanish', 'Italian', 'French', 'Portuegese', 'Hindi', 'Arabic']
                log.info('''Getting list of available voices for elevenlabs_tts for "/speak" command...''')
                from extensions.elevenlabs_tts.script import refresh_voices, update_api_key # type: ignore
                api_key = '' # If you are using 'elevenlabs_tts' extension, feel free to hardcode the API key here!
                if api_key:
                    update_api_key(api_key)
                all_voices = refresh_voices()

            elif self.extension == 'edge_tts':
                lang_list = ['English']
                from extensions.edge_tts.script import edge_tts # type: ignore
                voices = await edge_tts.list_voices()
                all_voices = [voice['ShortName'] for voice in voices if 'ShortName' in voice and voice['ShortName'].startswith('en-')]

            elif self.extension == 'vits_api_tts':
                lang_list = ['English']
                log.info("Collecting voices for the '/speak' command. If this fails, ensure 'vits_api_tts' is running on default URL 'http://localhost:23456/'.")
                from extensions.vits_api_tts.script import refresh_voices # type: ignore
                all_voices = refresh_voices()

            all_voices.sort() # Sort alphabetically
            return lang_list, all_voices
        except Exception as e:
            log.error(f"Error building options for '/speak' command: {e}")
            return None, None

# Majority of this code section is sourced from 'modules/server.py'
class TGWUI():
    def __init__(self):
        self.enabled:bool = config.textgen.get('enabled', True)
        self.instruction_template_str:str = None
        self.last_extension_params = {}
        self.is_multimodal = False

        self.tts = TTS()

        if self.enabled:
            self.init_settings()

            # monkey patch load_extensions behavior from pre-commit b3fc2cd
            tgwui_extensions_module.load_extensions = self.load_extensions
            self.init_extensions()     # build TGWUI extensions
            self.init_tts_extensions() # build TTS extensions
            self.activate_extensions() # Activate the extensions

            self.init_llmmodels() # Get model from cmd args, or present model list in cmd window
            asyncio.run(self.load_llm_model())

            tgwui_shared_module.generation_lock = Lock()

    def init_settings(self):
        tgwui_shared_module.settings['character'] = bot_database.last_character
        # Loading custom settings
        settings_file = None

        # Paths to check
        tgwui_user_data_dir = os.path.join(shared_path.dir_tgwui, "user_data")
        tgwui_user_data_settings_json = os.path.join(tgwui_user_data_dir, "settings.json")
        tgwui_user_data_settings_yaml = os.path.join(tgwui_user_data_dir, "settings.yaml")
        tgwui_settings_json = os.path.join(shared_path.dir_tgwui, "settings.json")
        tgwui_settings_yaml = os.path.join(shared_path.dir_tgwui, "settings.yaml")

        # Check if a settings file is provided and exists
        if tgwui_shared_module.args.settings is not None and Path(tgwui_shared_module.args.settings).exists():
            settings_file = Path(tgwui_shared_module.args.settings)
        # Check if settings exist in user_data directory
        elif Path(tgwui_user_data_settings_json).exists():
            settings_file = Path(tgwui_user_data_settings_json)
        elif Path(tgwui_user_data_settings_yaml).exists():
            settings_file = Path(tgwui_user_data_settings_yaml)
        # Fall back to the original location
        elif Path(tgwui_settings_json).exists():
            settings_file = Path(tgwui_settings_json)
        elif Path(tgwui_settings_yaml).exists():
            settings_file = Path(tgwui_settings_yaml)

        # Load the settings
        if settings_file is not None:
            log.info(f"Loading text-generation-webui settings from {settings_file}...")
            file_contents = open(settings_file, 'r', encoding='utf-8').read()
            new_settings = json.loads(file_contents) if settings_file.suffix == ".json" else yaml.safe_load(file_contents)
            tgwui_shared_module.settings.update(new_settings)

        # Fallback settings for models
        tgwui_shared_module.model_config['.*'] = get_fallback_settings()
        tgwui_shared_module.model_config.move_to_end('.*', last=False)  # Move to the beginning

    # legacy version of load_extensions() which allows extension params to be updated during runtime
    def load_extensions(self, available_extensions):
        tgwui_extensions_module.state = {}
        for i, name in enumerate(tgwui_shared_module.args.extensions):
            if name not in available_extensions:
                continue
            if name.endswith('_tts') and self.tts.extension is None:
                log.warning(f'A TTS extension "{name}" attempted to load which was not set in config.yaml TTS Settings. Errors are likely to occur.')
            if name != 'api':
                if not bot_database.was_warned(name):
                    bot_database.update_was_warned(name)
                    log.info(f'Loading {"your configured TTS extension" if name == self.tts.extension else "the extension"} "{name}"')
            try:
                try:
                    # Prefer user extension, fall back to system extension
                    user_script_path = Path(f'user_data/extensions/{name}/script.py')
                    if user_script_path.exists():
                        extension = importlib.import_module(f"user_data.extensions.{name}.script")
                    else:
                        extension = importlib.import_module(f"extensions.{name}.script")
                except ModuleNotFoundError:
                    extension_location = Path('user_data/extensions') / name if user_script_path.exists() else Path('extensions') / name
                    log.error(f"[Text Generation WebUI Extension Error for '{name}']\n\n"
                              f"Could not import the requirements for '{name}'. Make sure to install the requirements for the extension.\n\n"
                              f"* To install requirements for all available extensions, launch the\n  update_wizard script for your OS and choose the B option.\n\n"
                              f"* To install the requirements for this extension alone, launch the\n  cmd script for your OS and paste the following command in the\n  terminal window that appears:\n\n"
                              f"Linux / Mac:\n\npip install -r {extension_location}/requirements.txt --upgrade\n\n"
                              f"Windows:\n\npip install -r {extension_location}\\requirements.txt --upgrade\n")
                    raise

                tgwui_extensions_module.apply_settings(extension, name)
                setup_name = f"{name}_setup"
                if hasattr(extension, "setup") and not bot_database.was_warned(setup_name):
                    bot_database.update_was_warned(setup_name)
                    log.warning(f'Extension "{name}" has "setup" attribute. Trying to load...')
                    try:
                        extension.setup()
                    except Exception as e:
                        log.error(f'Setup failed for extension {name}:', e)
                tgwui_extensions_module.state[name] = [True, i, extension]
            except Exception:
                if name == self.tts.extension:
                    self.tts.enabled = False
                    self.tts.extension = None
                tgwui_shared_module.args.extensions.remove(name)
                log.error(f'Failed to load the extension "{name}".')
                traceback.print_exc()

    def init_extensions(self):
        tgwui_shared_module.args.extensions = []
        tgwui_extensions_module.available_extensions = tgwui_utils_module.get_available_extensions()

        # Initialize shared args extensions
        if tgwui_shared_module.settings.get('default_extensions'):
            for extension in tgwui_shared_module.settings['default_extensions']:
                tgwui_shared_module.args.extensions = tgwui_shared_module.args.extensions or []
                if extension not in tgwui_shared_module.args.extensions:
                    tgwui_shared_module.args.extensions.append(extension)

    def init_tts_extensions(self):
        # If any TTS extension defined in config.yaml, set tts bot vars and add extension to tgwui_shared_module.args.extensions
        if self.tts.extension:
            self.tts.enabled = True
            if 'alltalk' in self.tts.extension:
                log.warning(f'[{self.tts.extension}] If using AllTalk v2, extension params may fail to apply (changing voices, etc).')
                self.tts.voice_key = 'voice'
                self.tts.lang_key = 'language'
                # All TTS extensions with "alltalk" in the name are supported
                if self.tts.extension not in self.tts.supported_extensions:
                    self.tts.supported_extensions.append(self.tts.extension)
            elif self.tts.extension == 'coqui_tts':
                self.tts.voice_key = 'voice'
                self.tts.lang_key = 'language'
            elif self.tts.extension in ['vits_api_tts', 'elevenlabs_tts']:
                self.tts.voice_key = 'selected_voice'
                self.tts.lang_key = ''
            elif self.tts.extension in ['silero_tts', 'edge_tts']:
                self.tts.voice_key = 'speaker'
                self.tts.lang_key = 'language'

            if self.tts.extension not in tgwui_shared_module.args.extensions:
                tgwui_shared_module.args.extensions.append(self.tts.extension)
            if self.tts.extension not in self.tts.supported_extensions:
                log.warning(f'[{self.tts.extension}] The "/speak" command will not be registered.')
                log.warning(f'[{self.tts.extension}] Supported TTS extensions: {self.tts.supported_extensions}')

            # Ensure only one TTS extension is running
            excess_tts_clients = []
            for extension in tgwui_shared_module.args.extensions:
                extension:str
                if extension.endswith('_tts') and extension != self.tts.extension:
                    log.warning(f'[{self.tts.extension}] An undefined TTS extension "{extension}" attempted to load. Skipping...')
                    excess_tts_clients.append(extension)
            if excess_tts_clients:
                log.warning(f'[{self.tts.extension}] Skipping: {excess_tts_clients}')
                for extension in excess_tts_clients:
                    tgwui_shared_module.args.extensions.pop(extension)

    def activate_extensions(self):
        if tgwui_shared_module.args.extensions is not None and len(tgwui_shared_module.args.extensions) > 0:
            tgwui_extensions_module.load_extensions(tgwui_extensions_module.available_extensions)

    def init_llmmodels(self):
        all_llmmodels = tgwui_utils_module.get_available_models()

        # Model defined through --model
        if tgwui_shared_module.args.model is not None:
            tgwui_shared_module.model_name = tgwui_shared_module.args.model

        # Only one model is available
        elif len(all_llmmodels) == 1:
            tgwui_shared_module.model_name = all_llmmodels[0]

        # Select the model from a command-line menu
        else:
            if len(all_llmmodels) == 0:
                log.error("No LLM models are available! Please download at least one.")
                sys.exit(0)
            else:
                print('The following LLM models are available:\n')
                for index, model in enumerate(all_llmmodels):
                    print(f'{index+1}. {model}')

                print(f'\nWhich one do you want to load? 1-{len(all_llmmodels)}\n')
                i = int(input()) - 1
                print()

            tgwui_shared_module.model_name = all_llmmodels[i]
            print(f'Loading {tgwui_shared_module.model_name}.\nTo skip model selection, use "--model" in "CMD_FLAGS.txt".')

    # Check user settings (models/config-user.yaml) to determine loader
    def get_llm_model_loader(self, model:str) -> str:
        loader = None
        user_model_settings = {}
        settings = tgwui_shared_module.user_config
        for pat in settings:
            if re.match(pat.lower(), Path(model).name.lower()):
                for k in settings[pat]:
                    user_model_settings[k] = settings[pat][k]
        if 'loader' in user_model_settings:
            loader = user_model_settings['loader']
            return loader
        else:
            loader = infer_loader(model, user_model_settings)
        return loader
    
    def check_if_multimodal(self):
        if hasattr(tgwui_shared_module, 'is_multimodal'):
            return tgwui_shared_module.is_multimodal
        return False

    async def load_llm_model(self, loader=None):
        try:
            # If any model has been selected, load it
            if tgwui_shared_module.model_name != 'None':
                p = Path(tgwui_shared_module.model_name)
                if p.exists():
                    model_name = p.parts[-1]
                    tgwui_shared_module.model_name = model_name
                else:
                    model_name = tgwui_shared_module.model_name

                model_settings = get_model_metadata(model_name)

                self.instruction_template_str = model_settings.get('instruction_template_str', '')

                update_model_parameters(model_settings, initial=True)  # hijack the command-line arguments
                # Load the model
                loop = asyncio.get_event_loop()
                tgwui_shared_module.model, tgwui_shared_module.tokenizer = await loop.run_in_executor(None, load_model, model_name, loader)
                # Check if modal is multimodal
                self.check_if_multimodal()
                # Load any LORA
                if tgwui_shared_module.args.lora:
                    add_lora_to_model(tgwui_shared_module.args.lora)
        except Exception as e:
            log.error(f"An error occurred while loading LLM Model: {e}")

    async def update_extensions(self, params):
        try:
            if self.last_extension_params or params:
                if self.last_extension_params == params:
                    return # Nothing needs updating
                self.last_extension_params = params # Update self dict
            # Update extension settings
            if self.last_extension_params:
                last_extensions = list(self.last_extension_params.keys())
                # Update tgwui_shared_module.settings
                for param in last_extensions:
                    listed_param = self.last_extension_params[param]
                    tgwui_shared_module.settings.update({'{}-{}'.format(param, key): value for key, value in listed_param.items()})
            else:
                log.warning('** No extension params for this character. Reloading extensions with initial values. **')
            tgwui_extensions_module.load_extensions(tgwui_extensions_module.available_extensions)  # Load Extensions (again)
        except Exception as e:
            log.error(f"An error occurred while updating character extension settings: {e}")

tgwui = TGWUI()

def custom_chatbot_wrapper(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False, stream_tts=False):
    """
    Custom version of the native TGWUI chatbot_wrapper()

    Enables streaming output from TTS extensions, instead of once at the end of generation.
    """
    # Lazy import native TGWUI funtions
    chat_funcs = get_tgwui_functions("chat")
    get_stopping_strings = chat_funcs['get_stopping_strings']
    generate_chat_prompt = chat_funcs['generate_chat_prompt']
    generate_reply = chat_funcs['generate_reply']
    add_message_attachment = chat_funcs['add_message_attachment']
    generate_search_query = chat_funcs['generate_search_query']
    add_web_search_attachments = chat_funcs['add_web_search_attachments']
    update_message_metadata = chat_funcs['update_message_metadata']
    get_current_timestamp = chat_funcs['get_current_timestamp']
    add_message_version = chat_funcs['add_message_version']

    # Handle dict format with text and files
    files = []
    if isinstance(text, dict):
        files = text.get('files', [])
        text = text.get('text', '')

    history = state['history']
    output = copy.deepcopy(history)
    output = tgwui_extensions_module.apply_extensions('history', output)
    state = tgwui_extensions_module.apply_extensions('state', state)

    # Handle GPT-OSS as a special case
    if '<|channel|>final<|message|>' in state['instruction_template_str']:
        state['skip_special_tokens'] = False

    # Let the jinja2 template handle the BOS token
    if state['mode'] in ['instruct', 'chat-instruct']:
        state['add_bos_token'] = False

    # Initialize metadata if not present
    if 'metadata' not in output:
        output['metadata'] = {}

    visible_text = None
    stopping_strings = get_stopping_strings(state)
    state['stream'] = state['stream'] if stream_tts == False else True # CUSTOM FOR BOT. FORCES STREAMING.
    is_stream = state['stream']

    # Prepare the input
    if not (regenerate or _continue):
        visible_text = html.escape(text)

        # Process file attachments and store in metadata
        row_idx = len(output['internal'])

        # Add attachments to metadata only, not modifying the message text
        for file_path in files:
            add_message_attachment(output, row_idx, file_path, is_user=True)

        # Add web search results as attachments if enabled
        if state.get('enable_web_search', False):
            search_query = generate_search_query(text, state)
            add_web_search_attachments(output, row_idx, text, search_query, state)

        # Apply extensions
        text, visible_text = tgwui_extensions_module.apply_extensions('chat_input', text, visible_text, state)
        text = tgwui_extensions_module.apply_extensions('input', text, state, is_chat=True)

        # Current row index
        output['internal'].append([text, ''])
        output['visible'].append([visible_text, ''])
        # Add metadata with timestamp
        update_message_metadata(output['metadata'], "user", row_idx, timestamp=get_current_timestamp())

        # *Is typing...*
        if loading_message:
            yield {
                'visible': output['visible'][:-1] + [[output['visible'][-1][0], tgwui_shared_module.processing_message]],
                'internal': output['internal'],
                'metadata': output['metadata']
            }
    else:
        text, visible_text = output['internal'][-1][0], output['visible'][-1][0]
        if regenerate:
            row_idx = len(output['internal']) - 1

            # Store the old response as a version before regenerating
            if not output['metadata'].get(f"assistant_{row_idx}", {}).get('versions'):
                add_message_version(output, "assistant", row_idx, is_current=False)

            # Add new empty version (will be filled during streaming)
            key = f"assistant_{row_idx}"
            output['metadata'][key]["versions"].append({
                "content": "",
                "visible_content": "",
                "timestamp": get_current_timestamp()
            })
            output['metadata'][key]["current_version_index"] = len(output['metadata'][key]["versions"]) - 1

            if loading_message:
                yield {
                    'visible': output['visible'][:-1] + [[visible_text, tgwui_shared_module.processing_message]],
                    'internal': output['internal'][:-1] + [[text, '']],
                    'metadata': output['metadata']
                }
        elif _continue:
            last_reply = [output['internal'][-1][1], output['visible'][-1][1]]
            if loading_message:
                yield {
                    'visible': output['visible'][:-1] + [[visible_text, last_reply[1] + '...']],
                    'internal': output['internal'],
                    'metadata': output['metadata']
                }

    # Generate the prompt
    kwargs = {
        '_continue': _continue,
        'history': output if _continue else {
            k: (v[:-1] if k in ['internal', 'visible'] else v)
            for k, v in output.items()
        }
    }

    prompt = tgwui_extensions_module.apply_extensions('custom_generate_chat_prompt', text, state, **kwargs)
    if prompt is None:
        prompt = generate_chat_prompt(text, state, **kwargs)

    # Add timestamp for assistant's response at the start of generation
    row_idx = len(output['internal']) - 1
    update_message_metadata(output['metadata'], "assistant", row_idx, timestamp=get_current_timestamp(), model_name=tgwui_shared_module.model_name)

    # Generate
    reply = None
    for j, reply in enumerate(generate_reply(prompt, state, stopping_strings=stopping_strings, is_chat=True, for_ui=for_ui)):

        # Extract the reply
        if state['mode'] in ['chat', 'chat-instruct']:
            visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply)
        else:
            visible_reply = reply

        visible_reply = html.escape(visible_reply)

        if tgwui_shared_module.stop_everything:
            # CUSTOM FOR BOT - Caller handles final apply_extensions
            # output['visible'][-1][1] = tgwui_extensions_module.apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
            yield output
            return

        if _continue:
            output['internal'][-1] = [text, last_reply[0] + reply]
            output['visible'][-1] = [visible_text, last_reply[1] + visible_reply]
        elif not (j == 0 and visible_reply.strip() == ''):
            output['internal'][-1] = [text, reply.lstrip(' ')]
            output['visible'][-1] = [visible_text, visible_reply.lstrip(' ')]

        # Keep version metadata in sync during streaming (for regeneration)
        if regenerate:
            row_idx = len(output['internal']) - 1
            key = f"assistant_{row_idx}"
            current_idx = output['metadata'][key]['current_version_index']
            output['metadata'][key]['versions'][current_idx].update({
                'content': output['internal'][row_idx][1],
                'visible_content': output['visible'][row_idx][1]
            })

        if is_stream:
            yield output

    if _continue:
        # Reprocess the entire internal text for extensions (like translation)
        full_internal = output['internal'][-1][1]
        if state['mode'] in ['chat', 'chat-instruct']:
            full_visible = re.sub("(<USER>|<user>|{{user}})", state['name1'], full_internal)
        else:
            full_visible = full_internal

        full_visible = html.escape(full_visible)
        # CUSTOM FOR BOT
        output['visible'][-1][1] = full_visible
    #     output['visible'][-1][1] = tgwui_extensions_module.apply_extensions('output', full_visible, state, is_chat=True)
    # else:
    #     output['visible'][-1][1] = tgwui_extensions_module.apply_extensions('output', output['visible'][-1][1], state, is_chat=True)

    # Final sync for version metadata (in case streaming was disabled)
    if regenerate:
        row_idx = len(output['internal']) - 1
        key = f"assistant_{row_idx}"
        current_idx = output['metadata'][key]['current_version_index']
        output['metadata'][key]['versions'][current_idx].update({
            'content': output['internal'][row_idx][1],
            'visible_content': output['visible'][row_idx][1]
        })

    yield output

