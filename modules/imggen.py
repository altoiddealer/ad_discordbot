import asyncio
from typing import Optional
from modules.utils_shared import is_tgwui_integrated, config, bot_embeds
if is_tgwui_integrated:
    from modules.utils_tgwui import tgwui

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

class SD:
    def __init__(self):
        self.enabled:bool = config.sd.get('enabled', True)
        self.url:str = config.sd.get('SD_URL', 'http://127.0.0.1:7860')
        self.client:str = None
        self.session_id:str = None
        self.last_img_payload = {}

        if self.enabled:
            if asyncio.run(self.online()):
                asyncio.run(self.init_sdclient())

    async def online(self, ictx:CtxInteraction|None=None):
        channel = ictx.channel if ictx else None
        e_title = f"Stable Diffusion is not running at: {self.url}"
        e_description = f"Launch your SD WebUI client with `--api --listen` command line arguments\n\
            Read more [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f'{self.url}/') as response:
                    if response.status == 200:
                        log.debug(f'Request status to SD: {response.status}')
                        return True
                    else:
                        log.warning(f'Non-200 status code received: {response.status}')
                        await bot_embeds.send('system', e_title, e_description, channel=channel, delete_after=10)
                        return False
            except aiohttp.ClientError as exc:
                # never successfully connected
                if self.client is None:
                    log.warning(e_title)
                    log.warning("Launch your SD WebUI client with `--api --listen` command line arguments")
                    log.warning("Image commands/features will function when client is active and accessible via API.'")
                # was previously connected
                else:
                    log.warning(exc)
                await bot_embeds.send('system', e_title, e_description, channel=channel, delete_after=10)
                return False

    async def api(self, endpoint:str, method='get', json=None, retry=True, warn=True) -> dict:
        headers = {'Content-Type': 'application/json'}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method.lower(), url=f'{self.url}{endpoint}', json=json or {}, headers=headers) as response:
                    response_text = await response.text()
                    if response.status == 200:
                        r = await response.json()
                        if self.client is None and endpoint not in ['/sdapi/v1/cmd-flags', '/API/GetNewSession']:
                            await self.init_sdclient()
                            if self.client and not self.client == 'SwarmUI':
                                bot_settings.imgmodel.refresh_enabled_extensions(print=True)
                                for settings in guild_settings.values():
                                    settings:Settings
                                    settings.imgmodel.refresh_enabled_extensions()
                        return r
                    # Try resolving certain issues and retrying
                    elif response.status in [422, 500]:
                        error_json = await response.json()
                        try_resolve = False
                        # Check if it's related to an invalid override script
                        if 'Script' in error_json.get('detail', ''):
                            script_name = error_json['detail'].split("'")[1]  # Extract the script name
                            if json and 'alwayson_scripts' in json:
                                # Remove the problematic script
                                if script_name in json['alwayson_scripts']:
                                    log.info(f"Removing invalid script: {script_name}")
                                    json['alwayson_scripts'].pop(script_name, None)
                                    try_resolve = True
                        elif 'KeyError' in error_json.get('error', ''):
                            # Extract the key name from the error message
                            key_error_msg = error_json.get('errors', '')
                            key_name = key_error_msg.split("'")[1]  # Extract the key inside single quotes
                            log.info(f"Removing invalid key: {key_name}")
                            # Remove the problematic key from the payload
                            if json and key_name in json:
                                json.pop(key_name, None)
                                try_resolve = True
                        if try_resolve:
                            return await self.api(endpoint, method, json, retry=False, warn=warn)
                    # Handle internal server error (status 500)
                    elif response.status == 500:
                        error_json = await response.json()
                        # Check if it's related to the KeyError
                        if 'KeyError' in error_json.get('error', '') and "'forge_inference_memory'" in error_json.get('errors', ''):
                            log.info("Removing problematic key: 'forge_inference_memory'")
                            # Remove 'forge_inference_memory' from the payload if it exists
                            if json and 'forge_inference_memory' in json:
                                json.pop('forge_inference_memory', None)
                            
                            # Retry the request with the modified payload
                            return await self.api(endpoint, method, json, retry=False, warn=warn)

                    # Log the error if the request failed
                    if warn:
                        log.error(f'{self.url}{endpoint} response: {response.status} "{response.reason}"')
                        log.error(f'Response content: {response_text}')
                    
                    # Retry on specific status codes (408, 500)
                    if retry and response.status in [408, 500]:
                        log.info("Retrying the request in 3 seconds...")
                        await asyncio.sleep(3)
                        return await self.api(endpoint, method, json, retry=False)

        except aiohttp.client.ClientConnectionError:
            log.warning(f'Failed to connect to: "{self.url}{endpoint}", offline?')

        except Exception as e:
            if endpoint == '/sdapi/v1/server-restart' or endpoint == '/sdapi/v1/progress':
                return None
            else:
                log.error(f'Error getting data from "{self.url}{endpoint}": {e}')
                traceback.print_exc()

    def determine_client_type(self, r):
        ui_settings_file = r.get("ui_settings_file", "").lower()
        if "reforge" in ui_settings_file:
            self.client = 'SD WebUI ReForge'
        elif "forge" in ui_settings_file:
            self.client = 'SD WebUI Forge'
        elif "webui" in ui_settings_file:
            self.client = 'A1111 SD WebUI'
        else:
            self.client = 'SD WebUI'

    async def try_sdwebuis(self):
        try:
            log.info("Checking if SD Client is A1111, Forge, ReForge, or other.")
            r = await self.api(endpoint='/sdapi/v1/cmd-flags')
            if not r:
                raise ConnectionError(f'Failed to connect to SD API, ensure it is running or disable the API in your config.')
            self.determine_client_type(r)
        except ConnectionError as e:
            log.error(f"Connection error: {e}")
        except Exception as e:
            log.error(f"Unexpected error when checking SD WebUI clients: {e}")
            traceback.print_exc()

    async def try_swarmui(self):
        try:
            log.info("Checking if SD Client is SwarmUI.")
            r = await self.api(endpoint='/API/GetNewSession', method='post', warn=False)
            if r is None:
                return False  # Early return if the response is None or the API call failed

            self.session_id = r.get('session_id', None)
            if self.session_id:
                self.client = 'SwarmUI'
                return True
        except aiohttp.ClientError as e:
            log.error(f"Error getting SwarmUI session: {e}")
        return False

    async def init_sdclient(self):
        if await self.try_swarmui():
            return

        if not self.session_id:
            await self.try_sdwebuis()

sd = SD()
