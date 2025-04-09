import asyncio
import copy
import html
import json
import os
import re
import sys
import yaml
import aiohttp
from pathlib import Path
from threading import Lock
from typing import Optional
from modules.typing import CtxInteraction
from modules.utils_shared import shared_path, config, bot_database, patterns, is_tgwui_integrated
from modules.utils_misc import check_probability
from modules.tts import tts

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

# Majority of this code section is sourced from 'modules/server.py'
class TGWUI():
    def __init__(self):
        self.enabled:bool = config.textgen.get('enabled', True)
        self.instruction_template_str:str = None
        self.last_extension_params = {}
        
        self.supported_tts_extensions = ['alltalk_tts', 'coqui_tts', 'silero_tts', 'elevenlabs_tts', 'edge_tts', 'vits_api_tts']
        self.tts_extension:Optional[str] = config.ttsgen.get('tgwui_extension')
        self.tts_voice_key:Optional[str] = None
        self.tts_lang_key:Optional[str] = None

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
                if name.endswith('_tts') and self.tts_extension is None:
                    log.warning(f'A TTS extension "{name}" attempted to load which was not set in config.yaml TTS Settings. Errors are likely to occur.')
                if name != 'api':
                    if not bot_database.was_warned(name):
                        bot_database.update_was_warned(name)
                        log.info(f'Loading {"your configured TTS extension" if name == self.tts_extension else "the extension"} "{name}"')
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

    def init_extensions(self):
        shared.args.extensions = []
        extensions_module.available_extensions = utils.get_available_extensions()

        # Initialize shared args extensions
        for extension in shared.settings['default_extensions']:
            shared.args.extensions = shared.args.extensions or []
            if extension not in shared.args.extensions:
                shared.args.extensions.append(extension)

    def init_tts_extensions(self):
        # If any TTS extension defined in config.yaml, set tts bot vars and add extension to shared.args.extensions
        if self.tts_extension:
            self.enabled = True
            if 'alltalk' in self.tts_extension:
                log.warning('[TTS] If using AllTalk v2, extension params may fail to apply (changing voices, etc). Full support is coming soon.')
                self.voice_key = 'voice'
                self.lang_key = 'language'
                # All TTS extensions with "alltalk" in the name are supported
                if self.tts_extension not in self.supported_tts_extensions:
                    self.supported_tts_extensions.append(self.tts_extension)
            elif self.tts_extension == 'coqui_tts':
                self.voice_key = 'voice'
                self.lang_key = 'language'
            elif self.tts_extension in ['vits_api_tts', 'elevenlabs_tts']:
                self.voice_key = 'selected_voice'
                self.lang_key = ''
            elif self.tts_extension in ['silero_tts', 'edge_tts']:
                self.voice_key = 'speaker'
                self.lang_key = 'language'

            if self.tts_extension not in shared.args.extensions:
                shared.args.extensions.append(self.tts_extension)
            if self.tts_extension not in self.supported_tts_extensions:
                log.warning(f'[TTS] The "/speak" command will not be registered for "{self.tts_extension}".')
                log.warning(f'[TTS] List of supported tts_clients: {self.supported_tts_extensions}')

            # Ensure only one TTS extension is running
            excess_tts_clients = []
            for extension in shared.args.extensions:
                extension:str
                if extension.endswith('_tts') and extension != self.tts_extension:
                    log.warning(f'[TTS] Your configured TTS client is "{self.tts_extension}", but another TTS extension "{extension}" attempted to load. Skipping "{extension}".')
                    excess_tts_clients.append(extension)
            if excess_tts_clients:
                log.warning(f'[TTS] Skipping: {excess_tts_clients}')
                for extension in excess_tts_clients:
                    shared.args.extensions.pop(extension)

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
        if tts.api_mode == False:
            return
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
            extensions_module.load_extensions(extensions_module.extensions, extensions_module.available_extensions)  # Load Extensions (again)
        except Exception as e:
            log.error(f"An error occurred while updating character extension settings: {e}")

tgwui = TGWUI()

def custom_chatbot_wrapper(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False, stream_tts=False):
    # 'stream_tts' CUSTOM FOR BOT
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
