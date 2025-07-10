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
from typing import Optional, Tuple
from modules.typing import CtxInteraction
from modules.utils_shared import shared_path, config, bot_database, patterns, is_tgwui_integrated
from modules.utils_misc import check_probability

sys.path.append(shared_path.dir_tgwui)

import modules.extensions as extensions_module
from modules import shared
from modules import utils
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import get_model_metadata, update_model_parameters, get_fallback_settings, infer_loader
from modules.prompts import count_tokens

# Import chat functions dynamically to avoid circular imports
def get_chat_functions():
    from modules.chat import chatbot_wrapper, load_character, save_history, get_stopping_strings, generate_chat_prompt
    from modules.text_generation import generate_reply
    return chatbot_wrapper, load_character, save_history, get_stopping_strings, generate_chat_prompt, generate_reply

def load_character(character, name1, name2):
    """Wrapper function that dynamically imports load_character to avoid circular imports"""
    chatbot_wrapper, load_character_func, save_history, get_stopping_strings, generate_chat_prompt, generate_reply = get_chat_functions()
    return load_character_func(character, name1, name2)

def chatbot_wrapper(*args, **kwargs):
    """Wrapper function that dynamically imports chatbot_wrapper to avoid circular imports"""
    chatbot_wrapper_func, load_character, save_history, get_stopping_strings, generate_chat_prompt, generate_reply = get_chat_functions()
    return chatbot_wrapper_func(*args, **kwargs)

def save_history(*args, **kwargs):
    """Wrapper function that dynamically imports save_history to avoid circular imports"""
    chatbot_wrapper, load_character, save_history_func, get_stopping_strings, generate_chat_prompt, generate_reply = get_chat_functions()
    return save_history_func(*args, **kwargs)

def count_tokens(*args, **kwargs):
    """Wrapper function that dynamically imports count_tokens to avoid circular imports"""
    from modules.prompts import count_tokens as count_tokens_func
    return count_tokens_func(*args, **kwargs)

def unload_model(*args, **kwargs):
    """Wrapper function that dynamically imports unload_model to avoid circular imports"""
    from modules.models import unload_model as unload_model_func
    return unload_model_func(*args, **kwargs)

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

        self.tts = TTS()

        if self.enabled:
            self.init_settings()

            # monkey patch load_extensions behavior from pre-commit b3fc2cd
            extensions_module.load_extensions = self.load_extensions
            self.init_extensions()     # build TGWUI extensions
            self.init_tts_extensions() # build TTS extensions
            self.activate_extensions() # Activate the extensions

            self.init_llmmodels() # Get model from cmd args, or present model list in cmd window
            asyncio.run(self.load_llm_model())

            shared.generation_lock = Lock()

    def init_settings(self):
        shared.settings['character'] = bot_database.last_character
        # Loading custom settings
        settings_file = None

        # Paths to check
        tgwui_user_data_dir = os.path.join(shared_path.dir_tgwui, "user_data")
        tgwui_user_data_settings_json = os.path.join(tgwui_user_data_dir, "settings.json")
        tgwui_user_data_settings_yaml = os.path.join(tgwui_user_data_dir, "settings.yaml")
        tgwui_settings_json = os.path.join(shared_path.dir_tgwui, "settings.json")
        tgwui_settings_yaml = os.path.join(shared_path.dir_tgwui, "settings.yaml")

        # Check if a settings file is provided and exists
        if shared.args.settings is not None and Path(shared.args.settings).exists():
            settings_file = Path(shared.args.settings)
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
            shared.settings.update(new_settings)

        # Fallback settings for models
        shared.model_config['.*'] = get_fallback_settings()
        shared.model_config.move_to_end('.*', last=False)  # Move to the beginning

    # legacy version of load_extensions() which allows extension params to be updated during runtime
    def load_extensions(self, available_extensions):
        extensions_module.state = {}
        for i, name in enumerate(shared.args.extensions):
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

                extensions_module.apply_settings(extension, name)
                setup_name = f"{name}_setup"
                if hasattr(extension, "setup") and not bot_database.was_warned(setup_name):
                    bot_database.update_was_warned(setup_name)
                    log.warning(f'Extension "{name}" has "setup" attribute. Trying to load...')
                    try:
                        extension.setup()
                    except Exception as e:
                        log.error(f'Setup failed for extension {name}:', e)
                extensions_module.state[name] = [True, i, extension]
            except Exception:
                if name == self.tts.extension:
                    self.tts.enabled = False
                    self.tts.extension = None
                shared.args.extensions.remove(name)
                log.error(f'Failed to load the extension "{name}".')
                traceback.print_exc()

    def init_extensions(self):
        shared.args.extensions = []
        extensions_module.available_extensions = utils.get_available_extensions()

        # Initialize shared args extensions
        if shared.settings.get('default_extensions'):
            for extension in shared.settings['default_extensions']:
                shared.args.extensions = shared.args.extensions or []
                if extension not in shared.args.extensions:
                    shared.args.extensions.append(extension)

    def init_tts_extensions(self):
        # If any TTS extension defined in config.yaml, set tts bot vars and add extension to shared.args.extensions
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

            if self.tts.extension not in shared.args.extensions:
                shared.args.extensions.append(self.tts.extension)
            if self.tts.extension not in self.tts.supported_extensions:
                log.warning(f'[{self.tts.extension}] The "/speak" command will not be registered.')
                log.warning(f'[{self.tts.extension}] Supported TTS extensions: {self.tts.supported_extensions}')

            # Ensure only one TTS extension is running
            excess_tts_clients = []
            for extension in shared.args.extensions:
                extension:str
                if extension.endswith('_tts') and extension != self.tts.extension:
                    log.warning(f'[{self.tts.extension}] An undefined TTS extension "{extension}" attempted to load. Skipping...')
                    excess_tts_clients.append(extension)
            if excess_tts_clients:
                log.warning(f'[{self.tts.extension}] Skipping: {excess_tts_clients}')
                for extension in excess_tts_clients:
                    shared.args.extensions.pop(extension)

    def activate_extensions(self):
        if shared.args.extensions is not None and len(shared.args.extensions) > 0:
            extensions_module.load_extensions(extensions_module.available_extensions)

    def init_llmmodels(self):
        all_llmmodels = utils.get_available_models()

        # Model defined through --model
        if shared.args.model is not None:
            shared.model_name = shared.args.model

        # Only one model is available
        elif len(all_llmmodels) == 1:
            shared.model_name = all_llmmodels[0]

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

            shared.model_name = all_llmmodels[i]
            print(f'Loading {shared.model_name}.\nTo skip model selection, use "--model" in "CMD_FLAGS.txt".')

    # Check user settings (models/config-user.yaml) to determine loader
    def get_llm_model_loader(self, model:str) -> str:
        loader = None
        user_model_settings = {}
        settings = shared.user_config
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

    async def load_llm_model(self, loader=None):
        try:
            # If any model has been selected, load it
            if shared.model_name != 'None':
                p = Path(shared.model_name)
                if p.exists():
                    model_name = p.parts[-1]
                    shared.model_name = model_name
                else:
                    model_name = shared.model_name

                model_settings = get_model_metadata(model_name)

                self.instruction_template_str = model_settings.get('instruction_template_str', '')

                update_model_parameters(model_settings, initial=True)  # hijack the command-line arguments
                # Load the model
                loop = asyncio.get_event_loop()
                shared.model, shared.tokenizer = await loop.run_in_executor(None, load_model, model_name, loader)
                # Load any LORA
                if shared.args.lora:
                    add_lora_to_model(shared.args.lora)
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
                # Update shared.settings
                for param in last_extensions:
                    listed_param = self.last_extension_params[param]
                    shared.settings.update({'{}-{}'.format(param, key): value for key, value in listed_param.items()})
            else:
                log.warning('** No extension params for this character. Reloading extensions with initial values. **')
            extensions_module.load_extensions(extensions_module.available_extensions)  # Load Extensions (again)
        except Exception as e:
            log.error(f"An error occurred while updating character extension settings: {e}")

tgwui = TGWUI()

def custom_chatbot_wrapper(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False, stream_tts=False):
    # 'stream_tts' CUSTOM FOR BOT
    _, _, _, get_stopping_strings, generate_chat_prompt, generate_reply = get_chat_functions()
    
    history = state['history']
    output = copy.deepcopy(history)
    output = extensions_module.apply_extensions('history', output)
    state = extensions_module.apply_extensions('state', state)

    visible_text = None
    stopping_strings = get_stopping_strings(state)
    state['stream'] = state['stream'] if stream_tts == False else True # CUSTOM FOR BOT. FORCES STREAMING.
    is_stream = state['stream']

    # Prepare the input
    if not (regenerate or _continue):
        visible_text = html.escape(text)

        # Apply extensions
        text, visible_text = extensions_module.apply_extensions('chat_input', text, visible_text, state)
        text = extensions_module.apply_extensions('input', text, state, is_chat=True)

        output['internal'].append([text, ''])
        output['visible'].append([visible_text, ''])

        # *Is typing...*
        if loading_message:
            yield {
                'visible': output['visible'][:-1] + [[output['visible'][-1][0], shared.processing_message]],
                'internal': output['internal']
            }
    else:
        text, visible_text = output['internal'][-1][0], output['visible'][-1][0]
        if regenerate:
            if loading_message:
                yield {
                    'visible': output['visible'][:-1] + [[visible_text, shared.processing_message]],
                    'internal': output['internal'][:-1] + [[text, '']]
                }
        elif _continue:
            last_reply = [output['internal'][-1][1], output['visible'][-1][1]]
            if loading_message:
                yield {
                    'visible': output['visible'][:-1] + [[visible_text, last_reply[1] + '...']],
                    'internal': output['internal']
                }

    # Generate the prompt
    kwargs = {
        '_continue': _continue,
        'history': output if _continue else {k: v[:-1] for k, v in output.items()}
    }
    prompt = extensions_module.apply_extensions('custom_generate_chat_prompt', text, state, **kwargs)
    if prompt is None:
        prompt = generate_chat_prompt(text, state, **kwargs)

    # Generate
    reply = None
    for j, reply in enumerate(generate_reply(prompt, state, stopping_strings=stopping_strings, is_chat=True, for_ui=for_ui)):

        # Extract the reply
        visible_reply = reply
        if state['mode'] in ['chat', 'chat-instruct']:
            visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply)

        visible_reply = html.escape(visible_reply)

        if shared.stop_everything:
            output['visible'][-1][1] = extensions_module.apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
            yield output
            return

        if _continue:
            output['internal'][-1] = [text, last_reply[0] + reply]
            output['visible'][-1] = [visible_text, last_reply[1] + visible_reply]
            if is_stream:
                yield output
        elif not (j == 0 and visible_reply.strip() == ''):
            output['internal'][-1] = [text, reply.lstrip(' ')]
            output['visible'][-1] = [visible_text, visible_reply.lstrip(' ')]
            if is_stream:
                yield output

    # CUSTOM FOR BOT
    #output['visible'][-1][1] = extensions_module.apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
    yield output
