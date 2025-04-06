import aiohttp
import asyncio
from typing import Any, Dict, Optional, Union
from modules.utils_shared import shared_path, load_file, is_tgwui_integrated, config

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

class API:
    def __init__(self):
        self.clients:dict = {}
        self.imggen_client_name:Optional[str] = None
        self.textgen_client_name:Optional[str] = None
        self.tts_clent_name:Optional[str] = None
        self.init()

    def assign_functions(self, api_config:dict):
        function = api_config.get('function')
        if function is not None and function in ['imggen', 'textgen', 'tts']:
            function_key = function + '_client_name'
            self_function_key = getattr(self, function_key)
            if self_function_key is None:
                self_function_key = api_config['name']

    def api_config_validated(self, api_config) -> bool:
        if not isinstance(api_config, dict):
            log.warning('[APIs] An API definition was not formatted as a dictionary. Ignoring.')
            return False
        name = api_config.get('name')
        if name is None:
            log.warning('[APIs] Encountered an API definition without a "name" key value. Ignoring.')
            return False
        url = api_config.get('base_url')
        if url is None:
            log.warning('[APIs] Encountered an API definition without a "base_url" key value. Ignoring.')
            return False
        return True

    def init(self):
        apis = load_file(shared_path.api_settings)
        for api_config in apis:
            if not self.api_config_validated(api_config):
                continue
            self.assign_functions(api_config)
            # Collect all valid user APIs
            name = api_config['name']
            self.clients[name] = APIClient(
                api_url=api_config["base_url"],
                default_headers=api_config.get("headers"),
                timeout=api_config.get("timeout", 10)
            )

class APIClient:
    def __init__(self, api_url: str, default_headers: Optional[Dict[str, str]] = None, timeout: int = 10):
        self.api_url = api_url.rstrip("/")
        self.default_headers = default_headers or {}
        self.timeout = timeout

    async def request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Union[Dict[str, Any], str, bytes]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[aiohttp.BasicAuth] = None,
        retry: int = 3,
        return_text: bool = False,
        return_raw: bool = False,
        timeout: Optional[int] = None,
    ) -> Union[Dict[str, Any], str, bytes, None]:
        
        url = f"{self.api_url}{endpoint}" if endpoint.startswith("/") else f"{self.api_url}/{endpoint}"
        headers = {**self.default_headers, **(headers or {})}
        timeout = timeout or self.timeout
        
        for attempt in range(retry + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method.upper(), url, params=params, data=data, json=json, headers=headers, auth=auth, timeout=timeout
                    ) as response:
                        
                        response_text = await response.text()
                        
                        if return_raw:
                            return await response.read()
                        
                        if return_text:
                            return response_text
                        
                        if response.status == 200:
                            try:
                                return await response.json()
                            except aiohttp.ContentTypeError:
                                log.warning(f"Non-JSON response received from {url}")
                                return response_text
                        else:
                            log.error(f"HTTP {response.status} Error: {response_text}")
                            response.raise_for_status()
            
            except aiohttp.ClientConnectionError:
                log.warning(f"Connection error to {url}, attempt {attempt + 1}/{retry + 1}")
            except aiohttp.ClientResponseError as e:
                log.error(f"HTTP Error {e.status} on {url}: {e.message}")
            except asyncio.TimeoutError:
                log.error(f"Request to {url} timed out.")
            except Exception as e:
                log.error(f"Unexpected error with {url}: {e}")
            
            if attempt < retry:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None
