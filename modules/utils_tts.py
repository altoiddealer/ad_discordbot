import copy
from modules.utils_shared import is_tgwui_integrated, config

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

# TODO Isolate independant TTS handling from TGWUI extension handling

class TTS:
    def __init__(self):
        self.enabled:bool = False
        self.settings:dict = config.textgenwebui['tts_settings']
        self.supported_clients = ['alltalk_tts', 'coqui_tts', 'silero_tts', 'elevenlabs_tts', 'edge_tts', 'vits_api_tts']
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
            elif self.client in ['vits_api_tts', 'elevenlabs_tts']:
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
