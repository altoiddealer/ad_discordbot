import asyncio
import copy
import html
import json
import os
import re
import sys
import yaml
from pathlib import Path
from threading import Lock
from modules.typing import CtxInteraction
from modules.utils_shared import shared_path, config, bot_database, patterns
from modules.utils_misc import check_probability

sys.path.append(shared_path.dir_tgwui)

import modules.extensions as extensions_module
from modules.chat import chatbot_wrapper, load_character, save_history, get_stopping_strings, generate_chat_prompt, generate_reply
from modules import shared
from modules import utils
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import get_model_metadata, update_model_parameters, get_fallback_settings, infer_loader
from modules.prompts import count_tokens

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

class TTS:
    def __init__(self):
        self.enabled:bool = False
        self.settings:dict = config.textgenwebui['tts_settings']
        self.supported_clients = ['alltalk_tts', 'coqui_tts', 'silero_tts', 'elevenlabs_tts', 'edge_tts']
        self.client:str = self.settings.get('extension', '')
        self.api_key:str = ''
        self.voice_key:str = ''
        self.lang_key:str = ''
    
    # runs from TGWUI() class
    def init_tts_extensions(self):
        # Get any supported TTS client found in TGWUI CMD_FLAGS
        fallback_client = None
        for extension in shared.args.extensions:
            extension:str
            if extension in self.supported_clients:
                self.client = extension
                break
            elif extension.endswith('_tts'):
                fallback_client = extension
        if fallback_client and not self.client:
            log.warning(f'tts client "{fallback_client}" was included in launch params, but is not yet confirmed to work.')
            log.warning(f'List of supported tts_clients: {self.supported_clients}')
            log.warning(f'Enabling "{fallback_client}", but there could be issues.')
            self.client = fallback_client

        # If any TTS extension defined in config.yaml, set tts bot vars and add extension to shared.args.extensions
        if self.client:
            if self.client not in self.supported_clients:
                log.warning(f'The "/speak" command will not be registered for "{self.client}".')
            self.enabled = True
            self.api_key = self.settings.get('api_key', None)
            if self.client == 'alltalk_tts':
                self.voice_key = 'voice'
                self.lang_key = 'language'
            elif self.client == 'coqui_tts':
                self.voice_key = 'voice'
                self.lang_key = 'language'
            elif self.client == 'elevenlabs_tts':
                self.voice_key = 'selected_voice'
                self.lang_key = ''
            elif self.client in ['silero_tts', 'edge_tts']:
                self.voice_key = 'speaker'
                self.lang_key = 'language'

            if self.client not in shared.args.extensions:
                shared.args.extensions.append(self.client)

    # Toggles TTS on/off
    async def apply_toggle_tts(self, settings, toggle:str='on', tts_sw:bool=False):
        try:
            #settings:"Settings" = get_settings(ictx)
            llmcontext_dict = vars(settings.llmcontext)
            extensions:dict = copy.deepcopy(llmcontext_dict.get('extensions', {}))
            if toggle == 'off' and extensions.get(self.client, {}).get('activate'):
                extensions[self.client]['activate'] = False
                await tgwui.update_extensions(extensions)
                # Return True if subsequent apply_toggle_tts() should enable TTS
                return True
            if tts_sw:
                extensions[self.client]['activate'] = True
                await tgwui.update_extensions(extensions)
        except Exception as e:
            log.error(f'An error occurred while toggling the TTS on/off: {e}')
        return False

tts = TTS()

# Majority of this code section is sourced from 'modules/server.py'
class TGWUI():
    def __init__(self):
        self.enabled:bool = config.textgenwebui.get('enabled', True)

        self.instruction_template_str:str = None

        self.last_extension_params = {}

        if self.enabled:
            self.init_settings()

            # monkey patch load_extensions behavior from pre-commit b3fc2cd
            extensions_module.load_extensions = self.load_extensions
            self.init_tgwui_extensions()  # build TGWUI extensions
            tts.init_tts_extensions()   # build TTS extensions in TTS()
            self.activate_extensions() # Activate the extensions

            self.init_llmmodels() # Get model from cmd args, or present model list in cmd window
            asyncio.run(self.load_llm_model())

            shared.generation_lock = Lock()

    def init_settings(self):
        shared.settings['character'] = bot_database.last_character
        # Loading custom settings
        settings_file = None
        tgwui_settings_json = os.path.join(shared_path.dir_tgwui, "settings.json")
        tgwui_settings_yaml = os.path.join(shared_path.dir_tgwui, "settings.yaml")
        # Check if a settings file is provided and exists
        if shared.args.settings is not None and Path(shared.args.settings).exists():
            settings_file = Path(shared.args.settings)
        # Check if settings file exists
        elif Path(tgwui_settings_json).exists():
            settings_file = Path(tgwui_settings_json)
        elif Path(tgwui_settings_yaml).exists():
            settings_file = Path(tgwui_settings_yaml)
        if settings_file is not None:
            log.info(f"Loading text-generation-webui settings from {settings_file}...")
            file_contents = open(settings_file, 'r', encoding='utf-8').read()
            new_settings = json.loads(file_contents) if settings_file.suffix == "json" else yaml.safe_load(file_contents)
            shared.settings.update(new_settings)

        # Fallback settings for models
        shared.model_config['.*'] = get_fallback_settings()
        shared.model_config.move_to_end('.*', last=False)  # Move to the beginning

    # legacy version of load_extensions() which allows extension params to be updated during runtime
    def load_extensions(self, extensions, available_extensions):
        extensions_module.state = {}
        for index, name in enumerate(shared.args.extensions):
            if name in available_extensions:
                if name != 'api':
                    if not bot_database.was_warned(name):
                        bot_database.update_was_warned(name)
                        log.info(f'Loading the extension "{name}"')
                try:
                    try:
                        exec(f"import extensions.{name}.script")
                    except ModuleNotFoundError:
                        log.error(f"Could not import the requirements for '{name}'. Make sure to install the requirements for the extension.\n\n \
                                  Linux / Mac:\n\npip install -r extensions/{name}/requirements.txt --upgrade\n\nWindows:\n\npip install -r extensions\\{name}\\requirements.txt --upgrade\n\n \
                                  If you used the one-click installer, paste the command above in the terminal window opened after launching the cmd script for your OS.")
                        raise
                    extension = getattr(extensions, name).script
                    extensions_module.apply_settings(extension, name)
                    setup_name = f"{name}_setup"
                    if hasattr(extension, "setup") and not bot_database.was_warned(setup_name):
                        bot_database.update_was_warned(setup_name)
                        log.warning(f'Extension "{name}" has "setup" attribute. Trying to load...')
                        try:
                            extension.setup()
                        except Exception as e:
                            log.error(f'Setup failed for extension {name}:', e)
                    extensions_module.state[name] = [True, index]
                except Exception:
                    log.error(f'Failed to load the extension "{name}".')

    def init_tgwui_extensions(self):
        shared.args.extensions = []
        extensions_module.available_extensions = utils.get_available_extensions()

        # Initialize shared args extensions
        for extension in shared.settings['default_extensions']:
            shared.args.extensions = shared.args.extensions or []
            if extension not in shared.args.extensions:
                shared.args.extensions.append(extension)

    def activate_extensions(self):
        if shared.args.extensions and len(shared.args.extensions) > 0:
            extensions_module.load_extensions(extensions_module.extensions, extensions_module.available_extensions)

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
            if re.match(pat.lower(), model.lower()):
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
            # Add tts API key if one is provided in config.yaml
            if tts.api_key:
                if tts.client not in self.last_extension_params:
                    self.last_extension_params[tts.client] = {'api_key': tts.api_key}
                else:
                    self.last_extension_params[tts.client].update({'api_key': tts.api_key})
            # Update extension settings
            if self.last_extension_params:
                last_extensions = list(self.last_extension_params.keys())
                # Update shared.settings
                for param in last_extensions:
                    listed_param = self.last_extension_params[param]
                    shared.settings.update({'{}-{}'.format(param, key): value for key, value in listed_param.items()})
            else:
                log.warning('** No extension params for this character. Reloading extensions with initial values. **')
            extensions_module.load_extensions(extensions_module.extensions, extensions_module.available_extensions)  # Load Extensions (again)
        except Exception as e:
            log.error(f"An error occurred while updating character extension settings: {e}")

tgwui = TGWUI()

def custom_chatbot_wrapper(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False, stream_tts=False):
    history = state['history']
    output = copy.deepcopy(history)
    output = extensions_module.apply_extensions('history', output)
    state = extensions_module.apply_extensions('state', state)

    visible_text = None
    stopping_strings = get_stopping_strings(state)
    state['stream'] = state['stream'] if stream_tts == False else True # Custom
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

    output['visible'][-1][1] = extensions_module.apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
    yield output


# def custom_chatbot_wrapper(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False, include_continued_text=False, can_chunk=False, stream_tts=False, settings=None):

#     last_checked = ''

#     def check_should_chunk(partial_resp):
#         nonlocal last_checked
#         chance_to_chunk = settings.behavior.chance_to_stream_reply
#         chunk_syntax = settings.behavior.stream_reply_triggers # ['\n', '.']
#         check_resp:str = partial_resp[len(last_checked):]
#         for syntax in chunk_syntax:
#             if check_resp.endswith(syntax):
#                 # update for next iteration
#                 last_checked += check_resp
#                 # Ensure markdown syntax is not cut off
#                 if not patterns.check_markdown_balanced(last_checked):
#                     return False
#                 # Special handling if syntax is '.' (sentence completion)
#                 elif syntax == '.':
#                     if len(check_resp) > 1 and check_resp[-2].isdigit(): # avoid chunking on numerical list
#                         return False
#                     elif len(check_resp) > 2 and ('\n' in check_resp[-3:-1]): # avoid chunking on other lists
#                         return False
#                     chance_to_chunk = chance_to_chunk * 0.5
#                 return check_probability(chance_to_chunk)

#         return False

#     history = state['history']
#     output = copy.deepcopy(history)
#     output = extensions_module.apply_extensions('history', output)
#     state = extensions_module.apply_extensions('state', state)

#     visible_text = None
#     stopping_strings = get_stopping_strings(state)
#     state['stream'] = state['stream'] if not stream_tts else True
#     is_stream = state['stream']

#     # Prepare the input
#     if not (regenerate or _continue):
#         visible_text = html.escape(text)
#         text, visible_text = extensions_module.apply_extensions('chat_input', text, visible_text, state)
#         text = extensions_module.apply_extensions('input', text, state, is_chat=True)

#         output['internal'].append([text, ''])
#         output['visible'].append([visible_text, ''])

#         if loading_message:
#             yield {
#                 'visible': output['visible'][:-1] + [[output['visible'][-1][0], shared.processing_message]],
#                 'internal': output['internal']
#             }
#     else:
#         text, visible_text = output['internal'][-1][0], output['visible'][-1][0]
#         if regenerate:
#             if loading_message:
#                 yield {
#                     'visible': output['visible'][:-1] + [[visible_text, shared.processing_message]],
#                     'internal': output['internal'][:-1] + [[text, '']]
#                 }
#         elif _continue:
#             last_reply = [output['internal'][-1][1], output['visible'][-1][1]]
#             if loading_message:
#                 yield {
#                     'visible': output['visible'][:-1] + [[visible_text, last_reply[1] + '...']],
#                     'internal': output['internal']
#                 }

#     kwargs = {
#         '_continue': _continue,
#         'history': output if _continue else {k: v[:-1] for k, v in output.items()}
#     }
#     prompt = extensions_module.apply_extensions('custom_generate_chat_prompt', text, state, **kwargs)
#     if prompt is None:
#         prompt = generate_chat_prompt(text, state, **kwargs)

#     reply = None
#     already_chunked = ""
#     continued_from = output['internal'][-1][-1] if _continue else ''
#     include_continued_text = getattr(state, "include_continued_text", False)

#     for j, reply in enumerate(generate_reply(prompt, state, stopping_strings=stopping_strings, is_chat=True, for_ui=for_ui)):

#         visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply)
#         visible_reply = html.escape(visible_reply)

#         if shared.stop_everything:
#             output['visible'][-1][1] = extensions_module.apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
#             yield output
#             return

#         if _continue:
#             output['internal'][-1] = [text, output['internal'][-1][1] + reply]
#             output['visible'][-1] = [visible_text, output['visible'][-1][1] + visible_reply]
#         elif not (j == 0 and visible_reply.strip() == ''):
#             output['internal'][-1] = [text, reply.lstrip(' ')]
#             output['visible'][-1] = [visible_text, visible_reply.lstrip(' ')]

#         base_resp = output['internal'][-1][1]
#         if _continue and not include_continued_text and len(base_resp) > len(continued_from):
#             base_resp = base_resp[len(continued_from):]
        
#         partial_response = base_resp[len(already_chunked):]
#         should_chunk = check_should_chunk(partial_response)

#         if should_chunk:
#             already_chunked += partial_response
#             if is_stream:
#                 yield {
#                     'visible': output['visible'],
#                     'internal': output['internal']
#                 }

#     output['visible'][-1][1] = extensions_module.apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
#     yield output
