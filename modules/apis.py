import aiohttp
import aiofiles
import json
import uuid
import asyncio
import os
import jsonschema
import jsonref
from PIL import Image, PngImagePlugin
import io
import base64
import copy
from typing import Any, Dict, Tuple, List, Optional, Union
from modules.utils_shared import shared_path, load_file, get_api
import modules.utils_processing

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

class APISettings():
    def __init__(self):
        self.main_settings:dict = {}
        self.response_handling_presets:dict = {}
        self.workflows:dict = {}
    def get_config_for(self, section: str, default=None) -> dict:
        return self.main_settings.get(section, default or {})
    def collect_presets(self, presets_list):
        for preset in presets_list:
            if not isinstance(preset, dict):
                log.warning("Encountered a non-dict response handling preset. Skipping.")
                continue
            name = preset.get('name')
            if name:
                self.response_handling_presets[name] = preset
    def apply_preset(self, rh_config: dict) -> dict:
        """
        Merge a config dictionary with its referenced preset (if any),
        giving priority to explicitly defined values in config.
        """
        preset_name = rh_config.get("preset")
        if not preset_name:
            return rh_config

        preset = copy.deepcopy(self.get_preset(preset_name))
        if not preset:
            log.warning(f"Response handling preset '{preset_name}' not found.")
            return rh_config

        # Merge preset with overrides (config wins)
        merged = {**preset, **rh_config}
        merged.pop("preset", None)
        return merged
    def get_preset(self, preset_name: str, default=None) -> dict:
        return self.response_handling_presets.get(preset_name, default or {})
    def collect_workflows(self, workflows_list):
        for workflow in workflows_list:
            if not isinstance(workflow, dict):
                log.warning("Encountered a non-dict Workflow. Skipping.")
                continue
            name = workflow.get('name')
            if name:
                self.workflows[name] = workflow
                # TODO Fix Presets for Workflow Steps
                workflow_steps = workflow.get('steps')
                if workflow_steps:
                    expanded_steps = []
                    for step in workflow_steps:
                        if not isinstance(step, dict):
                            log.error(f"Workflow {name} has improper structure for a 'step' (steps should be lists of dictionaries)")
                        expanded = apisettings.apply_preset(step)
                        expanded_steps.append(expanded)
                    self.workflows[name]['steps'] = expanded_steps

    def get_workflow(self, workflow_name: str, default=None) -> dict:
        return self.workflows.get(workflow_name, default or {})

apisettings = APISettings()


class API:
    def __init__(self):
        # ALL API clients
        self.clients:dict[str, APIClient] = {}
        # Main API clients
        self.imggen:Union[ImgGenClient, DummyClient] = DummyClient(ImgGenClient)
        self.textgen:Union[TextGenClient, DummyClient] = DummyClient(TextGenClient)
        self.ttsgen:Union[TTSGenClient, DummyClient] = DummyClient(TTSGenClient)
        # Collect setup tasks
        self.setup_tasks:list = []

    async def init(self):
        # Load API Settings yaml
        data = load_file(shared_path.api_settings)

        # Collect Main APIs
        apisettings.main_settings = data.get('bot_api_functions', {})
        # Collect Response Handling Presets
        apisettings.collect_presets(data.get('response_handling_presets', {}))
        # Collect Workflows
        apisettings.collect_workflows(data.get('workflows', {}))

        # Reverse lookup for matching API names to their function type
        main_api_name_map = {v.get("api_name"): k for k, v in apisettings.main_settings.items()
                             if isinstance(v, dict) and v.get("api_name")}
        # Map function type to specialized client class
        client_type_map = {"imggen": ImgGenClient,
                           "textgen": TextGenClient,
                           "ttsgen": TTSGenClient}

        # Iterate over all APIs data
        apis:dict = data.get('all_apis', {})
        for api_config in apis:
            if not isinstance(api_config, dict):
                log.warning('An API definition was not formatted as a dictionary. Ignoring.')
                continue
            name = api_config.get("name")
            if not name:
                log.warning("API config missing required 'name'. Skipping.")
                continue
            enabled = api_config.get("enabled", True)
            if not enabled:
                continue

            # Determine if this API is a "main" one
            api_func_type = main_api_name_map.get(name)
            is_main = api_func_type is not None
            # Determine which client class to use
            ClientClass = client_type_map.get(api_func_type, APIClient)

            # Collect all valid user APIs
            try:
                api_client:APIClient = ClientClass(
                    name=name,
                    url=api_config['url'],
                    enabled=enabled,
                    transport=api_config.get('transport', 'http'),
                    websocket_url=api_config.get('websocket_url'),
                    default_headers=api_config.get('default_headers'),
                    default_timeout=api_config.get('default_timeout', 60),
                    auth=api_config.get('auth'),
                    endpoints_config=api_config.get('endpoints', [])
                )
                # Capture all clients
                self.clients[name] = api_client
                # Capture main clients
                if is_main:
                    setattr(self, api_func_type, api_client)
                    log.info(f"Registered main {api_func_type} client: {name}")
                # Collect setup tasks
                if hasattr(api_client, 'setup'):
                    self.setup_tasks.append(api_client.setup())
            except KeyError as e:
                log.warning(f"Skipping API Client due to missing key: {e}")
            except TypeError as e:
                log.warning(f"Failed to create API client '{name}': {e}")

    async def setup_clients(self):
        if not self.setup_tasks:
            return
        await asyncio.gather(*self.setup_tasks)
        self.setup_tasks = []  # Clear after use

    def is_api_object(self, client_name:str|None=None, ep_name:str|None=None, log_missing:bool=False) -> bool:
        client = getattr(self, client_name, None)
        if client is None:
            client = self.clients.get(client_name, None)

        if not isinstance(client, APIClient):
            if log_missing:
                log.warning(f"API client '{client_name}' not found or invalid.")
            return False

        if ep_name is None:
            return True

        endpoint = getattr(client, ep_name, None)
        if endpoint is None:
            endpoint = client.endpoints.get(ep_name, None)

        if not isinstance(endpoint, Endpoint):
            if log_missing:
                log.warning(f"Endpoint '{ep_name}' not found or invalid in client '{client_name}'.")
            return False

        return True


class APIClient:
    def __init__(self,
                 name: str,
                 enabled: bool,
                 url: str,
                 transport: str = 'http',
                 websocket_url: Optional[str] = None,
                 default_headers: Optional[Dict[str, str]] = None,
                 default_timeout: int = 60,
                 auth: Optional[dict] = None,
                 endpoints_config=None):

        self.name = name
        self.session:Optional[aiohttp.ClientSession] = None
        self.enabled = enabled
        self.url = url.rstrip("/")
        self.transport = transport.lower()
        self.websocket_url = websocket_url
        self.default_headers = default_headers or {}
        self.default_timeout = default_timeout
        self.auth = auth
        self.endpoints: dict[str, Endpoint] = {}
        self.openapi_schema = None
        self._endpoint_fetch_payloads = []
        # WebSocket connection
        self.ws = None
        self.client_id = None
        # set auth
        if auth:
            try:
                self.auth = aiohttp.BasicAuth(auth["username"], auth["password"])
            except KeyError:
                log.warning(f"[APIClient:{self.name}] Invalid auth dict: 'username' or 'password' missing.")
                self.auth = None
        # collect endpoints
        if endpoints_config:
            self._collect_endpoints(endpoints_config)

    async def setup(self):
        self.session = aiohttp.ClientSession()
        if self.transport == 'websocket':
            await self._connect_websocket()
        else:
            await self._fetch_openapi_schema()
            self._assign_endpoint_schemas()
            await self._resolve_deferred_payloads()

    async def toggle(self):
        if self.enabled:
            await self.go_offline()
            return 'offline'
        else:
            await self.come_online()
            return 'online'

    async def go_offline(self):
        if not self.enabled:
            return
        log.info(f"API Client '{self.name}' disabled. Use '/toggle_api' to try enabling it when available.")
        self.enabled = False
        if self.session is not None:
            await self.close()

    async def come_online(self):
        if self.enabled:
            return
        await self.setup()
        self.enabled = True
        log.info(f"API Client {self.name} enabled. Use '/toggle_api' to disable.")

    async def _connect_websocket(self):
        if not self.session:
            log.warning(f"[APIClient:{self.name}] Cannot connect websocket — session is not initialized.")
            return

        self.client_id = str(uuid.uuid4())
        url = self.websocket_url or self.url.replace("http", "ws") + "/ws"
        if '?' not in url:
            url = f"{url}?clientId={self.client_id}"
        try:
            self.ws = await self.session.ws_connect(url)
            log.info(f"[APIClient:{self.name}] WebSocket connection established.")
        except Exception as e:
            log.error(f"[APIClient:{self.name}] Failed to connect to WebSocket: {e}")
            self.ws = None

    async def close(self):
        if self.ws:
            try:
                await self.ws.close()
                log.info(f"[APIClient:{self.name}] WebSocket connection closed.")
            except Exception as e:
                log.warning(f"[APIClient:{self.name}] Error closing WebSocket: {e}")
            finally:
                self.ws = None  # ✅ Prevent future use of closed connection

        if self.session:
            await self.session.close()
            self.session = None


    def _create_endpoint(self, EPClass:"Endpoint", ep_dict:dict):
        return EPClass(name=ep_dict["name"],
                        path=ep_dict["path"],
                        method=ep_dict.get("method", "GET"),
                        response_type=ep_dict.get("response_type", "json"),
                        payload_config=ep_dict.get("payload"),
                        rh_config=ep_dict.get("response_handling"),
                        headers=ep_dict.get("headers", self.default_headers),
                        timeout=ep_dict.get("timeout", self.default_timeout),
                        retry=ep_dict.get("retry", 0))
    
    def _get_self_ep_class(self):
        return Endpoint

    def _collect_endpoints(self, endpoints_config:list[dict]):
        for ep_dict in endpoints_config:
            try:
                ep_class:Endpoint = self._get_self_ep_class()
                endpoint:Endpoint = self._create_endpoint(ep_class, ep_dict)
                # link APIClient to Endpoint
                endpoint.client = self
                # get deferred payloads after collecting all endpoints
                if hasattr(endpoint, "_deferred_payload_source"):
                    self._endpoint_fetch_payloads.append(endpoint)
                self.endpoints[endpoint.name] = endpoint
            except KeyError as e:
                log.warning(f"[APIClient:{self.name}] Skipping endpoint due to missing key: {e}")

    def _bind_main_ep_values(self, config_entry: dict):
        if not type(self) in [TextGenClient, TTSGenClient, ImgGenClient]:
            return

        endpoint_name = config_entry.get("endpoint_name")
        endpoint = self.endpoints.get(endpoint_name)
        if not endpoint:
            return

        custom_keys = []
        for key, value in config_entry.items():
            if key == "endpoint_name":
                continue
            if hasattr(endpoint, key):
                setattr(endpoint, key, value)
            else:
                # Allow setting, but track as custom
                setattr(endpoint, key, value)
                custom_keys.append(key)
        if custom_keys:
            log.info(f"[{self.name}] Endpoint '{endpoint_name}' received custom user-defined config keys: {custom_keys}")

    def _bind_main_endpoints(self, config: dict):
        missing = []
        # match key names against self attributes
        for key, config_entry in config.items():
            if not hasattr(self, key):
                continue

            if not isinstance(config_entry, dict):
                missing.append((key, "not a dict"))
                continue

            endpoint_name = config_entry.get("endpoint_name")
            if not endpoint_name or endpoint_name not in self.endpoints:
                missing.append((key, f"endpoint '{endpoint_name}' not found"))
                continue

            self._bind_main_ep_values(config_entry)
            setattr(self, key, self.endpoints[endpoint_name])
        return missing

    async def _fetch_openapi_schema(self):
        try:
            def dereference_schema(schema: dict) -> dict:
                return jsonref.JsonRef.replace_refs(schema)
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.url}/openapi.json") as response:
                    if 200 <= response.status < 300:
                        self.openapi_schema = dereference_schema(await response.json())
                        log.debug(f"Loaded OpenAPI schema for {self.name}")
                    else:
                        log.debug(f"No OpenAPI schema available at {self.name}")
        except Exception as e:
            log.warning(f"Failed to load OpenAPI schema from {self.url} for {self.name}: {e}")
            await self.go_offline()

    def _assign_endpoint_schemas(self):
        if not self.openapi_schema:
            return

        for endpoint in self.endpoints.values():
            endpoint.set_schema_from_openapi(self.openapi_schema)
            if endpoint.schema:
                log.debug(f"[{self.name}] Schema assigned to endpoint '{endpoint.name}'")
            else:
                log.debug(f"[{self.name}] No schema found for endpoint '{endpoint.name}'")

    async def _resolve_deferred_payloads(self):
        for ep in self._endpoint_fetch_payloads:
            ep:Endpoint
            ref = ep._deferred_payload_source
            ref_ep = self.endpoints.get(ref)

            if ref_ep is None:
                log.warning(f"[APIClient:{self.name}] Endpoint '{ep.name}' references unknown payload source: '{ref}'")
                continue

            log.debug(f"[APIClient:{self.name}] Fetching payload for '{ep.name}' using endpoint '{ref}'")

            try:
                data = await ref_ep.call()
                if isinstance(data, dict):
                    ep.payload = data
                else:
                    log.warning(f"[APIClient:{self.name}] Endpoint '{ref}' returned non-dict data for '{ep.name}'")

            except Exception as e:
                log.error(f"[APIClient:{self.name}] Failed to fetch payload from '{ref}' for '{ep.name}': {e}")

    def validate_payload(self, method:str, endpoint:str, json:str|None=None, data:str|None=None):
        if self.openapi_schema:
            try:
                if json:
                    jsonschema.validate(instance=json, schema=self.openapi_schema)
                elif data and isinstance(data, dict):
                    jsonschema.validate(instance=data, schema=self.openapi_schema)
            except jsonschema.ValidationError as e:
                log.error(f"Schema validation failed for {method} {endpoint}: {e.message}")
                raise

    async def is_online(self) -> Tuple[bool, str]:
        try:
            response = await self.request(endpoint='', method='GET', retry=0, return_raw=True, timeout=5)
            if response:
                return True, ''
            else:
                return False, ''
           
        except aiohttp.ClientError as e:
            emsg = f"[APIClient:{self.name}] is offline at url {self.url}: {e}"
            await self.go_offline()
            return False, emsg

    async def request(
        self,
        endpoint: str,
        method: str = "GET",
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Union[Dict[str, Any], str, bytes]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, Any]] = None,
        auth: Optional[aiohttp.BasicAuth] = None,
        retry: int = 0,
        return_text: bool = False,  # still here if used manually
        return_raw: bool = False,
        timeout: Optional[int] = None,
        session: Optional[aiohttp.ClientSession] = None,
        response_type: Optional[str] = None,
    ) -> Union[Dict[str, Any], str, bytes, None]:

        url = f"{self.url}{endpoint}" if endpoint.startswith("/") else f"{self.url}/{endpoint}"
        headers = {**self.default_headers, **(headers or {})}
        timeout = timeout or self.default_timeout
        auth = auth or self.auth

        # Validate payload
        self.validate_payload(method, endpoint, json, data)

        # Optional OpenAPI schema validation
        if self.openapi_schema:
            try:
                if json:
                    jsonschema.validate(instance=json, schema=self.openapi_schema)
                elif data and isinstance(data, dict):
                    jsonschema.validate(instance=data, schema=self.openapi_schema)
            except jsonschema.ValidationError as e:
                log.error(f"Schema validation failed for {method} {endpoint}: {e.message}")
                raise

        for attempt in range(retry + 1):
            try:
                active_session = session or self.session
                if active_session is None:
                    async with aiohttp.ClientSession() as temp_session:
                        return await self._make_request(
                            session=temp_session, method=method, url=url, params=params, data=data,
                            json=json, files=files, headers=headers, auth=auth, timeout=timeout,
                            return_text=return_text, return_raw=return_raw, response_type=response_type
                        )
                else:
                    return await self._make_request(
                        session=active_session, method=method, url=url, params=params, data=data,
                        json=json, files=files, headers=headers, auth=auth, timeout=timeout,
                        return_text=return_text, return_raw=return_raw, response_type=response_type
                    )

            except aiohttp.ClientConnectionError:
                log.warning(f"Connection error to {url}, attempt {attempt + 1}/{retry + 1}")
            except aiohttp.ClientResponseError as e:
                log.error(f"HTTP Error {e.status} on {url}: {e.message}")
            except asyncio.TimeoutError:
                log.error(f"Request to {url} timed out.")
            except Exception as e:
                log.error(f"Unexpected error with {url}: {e}")

            if attempt < retry:
                await asyncio.sleep(2 ** attempt)

        return None

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        method: str,
        json: Optional[Dict[str, Any]],
        data: Optional[Union[Dict[str, Any], str, bytes]],
        params: Optional[Dict[str, Any]],
        files: Optional[Dict[str, Any]],
        headers: Dict[str, str],
        auth: Optional[aiohttp.BasicAuth],
        timeout: int,
        return_text: bool,
        return_raw: bool,
        response_type: Optional[str],
    ):
        request_kwargs = {
            "method": method.upper(),
            "url": url,
            "params": params,
            "headers": headers,
            "auth": auth,
            "timeout": aiohttp.ClientTimeout(total=timeout),
        }

        if files:
            form_data = aiohttp.FormData()
            for key, val in files.items():
                form_data.add_field(key, val, filename=getattr(val, "name", key))
            request_kwargs["data"] = form_data
        else:
            if data is not None:
                request_kwargs["data"] = data
            if json is not None:
                request_kwargs["json"] = json

        async with session.request(**request_kwargs) as response:

            if return_raw:
                return await response.read()
            if return_text:
                return await response.text()

            if 200 <= response.status < 300:
                if response_type == "bytes":
                    return await response.read()
                elif response_type == "text":
                    return await response.text()
                else:  # default to json
                    try:
                        return await response.json()
                    except aiohttp.ContentTypeError:
                        log.warning(f"Non-JSON response received from {url}")
                        return await response.text()
            else:
                response_text = await response.text()
                log.error(f"HTTP {response.status} Error: {response_text}")
                response.raise_for_status()

class DummyClient:
    def __init__(self, target_cls: type):
        annotations = getattr(target_cls, '__annotations__', {})
        for attr_name in annotations:
            setattr(self, attr_name, None)
    def __bool__(self):
        return False  # Makes instances evaluate False

class ImgGenClient(APIClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        imggen_config:dict = apisettings.get_config_for("imggen")
        self.last_img_payload = {}

        self.post_txt2img: Optional[ImgGenEndpoint] = None
        self.post_img2img: Optional[ImgGenEndpoint] = None
        self.get_progress: Optional[ImgGenEndpoint] = None
        self.post_pnginfo: Optional[ImgGenEndpoint] = None
        self.post_options: Optional[ImgGenEndpoint] = None
        self.get_imgmodels: Optional[ImgGenEndpoint] = None
        self.get_imgmodels: Optional[ImgGenEndpoint] = None
        self.get_controlnet_models: Optional[ImgGenEndpoint] = None
        self.get_controlnet_control_types: Optional[ImgGenEndpoint] = None
        self.post_server_restart: Optional[ImgGenEndpoint] = None
        
        self._bind_main_endpoints(imggen_config) 

    # class override to subclass Endpoint()
    def _get_self_ep_class(self):
        return ImgGenEndpoint

    async def get_imggen_progress(self, session: Optional[aiohttp.ClientSession] = None) -> Optional[Dict[str, Any]]:
        if self.get_progress:
            try:
                return await self.get_progress.call(session=session)
            except Exception as e:
                log.warning(f"Progress fetch failed from {self.name}: {e}")
        return None

    async def get_image_data(
            self,
            image_payload: Dict[str, Any],
            mode: str = "txt2img",
            session: Optional[aiohttp.ClientSession] = None
        ) -> Tuple[List[Image.Image], Optional[PngImagePlugin.PngInfo]]:

        images = []
        pnginfo = None
        try:
            ep_for_mode:ImgGenEndpoint = getattr(self, f'post_{mode}')
            response = await ep_for_mode.call(input_data=image_payload, session=session)

            if not isinstance(response, dict):
                return [], response

            for i, img_data in enumerate(response.get('images', [])):
                raw_data = base64.b64decode(img_data.split(",", 1)[0])
                image = Image.open(io.BytesIO(raw_data))
                
                # Get PNG info
                if self.post_pnginfo:
                    png_payload = {"image": "data:image/png;base64," + img_data}
                    r2 = await self.post_pnginfo.call(input_data=png_payload, session=session)
                    if not isinstance(r2, dict):
                        return [], r2
                    png_info_data = r2.get("info")

                    if i == 0 and png_info_data:
                        pnginfo = PngImagePlugin.PngInfo()
                        pnginfo.add_text("parameters", png_info_data)

                image.save(f"{shared_path.dir_temp_images}/temp_img_{i}.png", pnginfo=pnginfo)

                images.append(image)

        except Exception as e:
            log.error(f"Error retrieving or processing image data: {e}")
            return [], e

        return images, pnginfo
    
    def is_sdwebui(self) -> bool:
        return any(substring in self.name.lower() for substring in ['stable', 'a1111', 'sdwebui'])
    
    def is_reforge(self) -> bool:
        return 'reforge' in self.name.lower()
    
    def is_forge(self) -> bool:
        return ('forge' in self.name.lower() and not self.is_reforge())
    
    def supports_loractrl(self) -> bool:
        return (self.is_sdwebui() or self.is_reforge()) and not self.is_forge()
    
    # async def try_swarmui(self):
    #     try:
    #         log.info("Checking if SD Client is SwarmUI.")
    #         r = await self.api(endpoint='/API/GetNewSession', method='post', warn=False)
    #         if r is None:
    #             return False  # Early return if the response is None or the API call failed

    #         self.session_id = r.get('session_id', None)
    #         if self.session_id:
    #             self.client = 'SwarmUI'
    #             return True
    #     except aiohttp.ClientError as e:
    #         log.error(f"Error getting SwarmUI session: {e}")
    #     return False


class TextGenClient(APIClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        textgen_config:dict = apisettings.get_config_for("textgen")
        # TODO Main TextGen API support

        # Collect endpoints used for main TextGen functions
        self._bind_main_endpoints(textgen_config)

    # class override to subclass Endpoint()
    def _get_self_ep_class(self):
        return TextGenEndpoint


class TTSGenClient(APIClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ttsgen_config:dict = apisettings.get_config_for("ttsgen")

        self.get_voices: Optional[TTSGenEndpoint] = None
        self.get_languages: Optional[TTSGenEndpoint] = None
        self.post_generate: Optional[TTSGenEndpoint] = None

        self._bind_main_endpoints(ttsgen_config)

    # class override to subclass Endpoint()
    def _get_self_ep_class(self):
        return TTSGenEndpoint


    async def fetch_speak_options(self):
        lang_list, all_voices = [], []
        try:
            if self.get_languages:
                lang_list = await self.get_languages.call(retry=0, extract_keys="get_languages_key")
            if self.get_voices:
                all_voices = await self.get_voices.call(retry=0, extract_keys="get_voices_key")
            return lang_list, all_voices
        except Exception as e:
            log.error(f'Error fetching options for "/speak" command via API: {e}')
            return None, None


class Endpoint:
    def __init__(self,
                 name: str,
                 path: str,
                 method: str = "GET",
                 response_type: str = "json",
                 payload_config: Optional[str|dict] = None,
                 rh_config: Optional[dict[str, Any]] = None,
                 headers: Optional[Dict[str, str]] = None,
                 timeout: int = 10,
                 retry: int = 0):
        self.client: Optional["APIClient"] = None
        self.name = name
        self.path = path
        self.method = method.upper()
        self.response_type = response_type
        self.response_handling = None
        self.payload = {}
        self.schema: Optional[dict] = None
        self.headers = headers or {}
        self.timeout = timeout
        self.retry = retry
        if rh_config:
            self.init_response_handling(rh_config)
        if payload_config:
            self.init_payload(payload_config)

    def init_response_handling(self, rh_config: dict):
        # Expand top-level response_handling preset
        final_rh = apisettings.apply_preset(rh_config)

        # Expand each post_process step if defined
        post_process_steps = final_rh.get("post_process", [])
        expanded_steps = []
        for step in post_process_steps:
            expanded = apisettings.apply_preset(step)
            expanded_steps.append(expanded)
        final_rh["post_process"] = expanded_steps

        self.response_handling = final_rh

    def init_payload(self, payload_config):
        # dictionary value
        if isinstance(payload_config, dict):
            self.payload = payload_config
        # string value
        elif isinstance(payload_config, str):
            payload_fp = os.path.join(shared_path.dir_user_payloads, payload_config)
            # string is file path
            if os.path.exists(payload_fp):
                self.payload = load_file(payload_fp)
            # any other string should be a 'get' endpoint. Need to get payload after client init.
            else:
                setattr(self, '_deferred_payload_source', payload_config)

    def get_payload(self):
        return copy.deepcopy(self.payload)

    def get_preferred_content_type(self) -> Optional[str]:
        content_type = None
        if isinstance(self.headers, dict):
            content_type = self.headers.get("Content-Type")
        elif isinstance(self.headers, str):
            content_type = self.headers

        if content_type:
            return content_type.split(";")[0].strip().lower()  # Strip params like charset=utf-8

        return None

    def set_schema_from_openapi(self, openapi_schema: dict, force: bool = False):
        if not openapi_schema or (self.schema and not force):
            return

        preferred_content_type = self.get_preferred_content_type()
        self.schema = self.get_schema(openapi_schema, preferred_content_type=preferred_content_type)


    def generate_payload_from_schema(self) -> dict:
        if not self.schema:
            return {}

        def resolve_schema(schema: dict) -> Any:
            # Only use fields with a "default"
            if "default" in schema:
                return schema["default"]

            schema_type = schema.get("type")

            if schema_type == "object":
                result = {}
                for k, v in schema.get("properties", {}).items():
                    value = resolve_schema(v)
                    if value is not None:
                        result[k] = value
                return result if result else None

            if schema_type == "array":
                item_schema = schema.get("items", {})
                value = resolve_schema(item_schema)
                return [value] if value is not None else None

            # Skip anything without a default
            return None

        payload = resolve_schema(self.schema)
        return payload if isinstance(payload, dict) else {}

    def _match_path_key(self, paths: dict) -> Optional[str]:
        """
        Attempt to find the OpenAPI path key that matches this endpoint's path.
        Handles paths with parameters like /item/{id}
        """
        if self.path in paths:
            return self.path  # Direct match

        for defined_path in paths:
            # Convert OpenAPI-style path to regex
            path_regex = re.sub(r"\{[^/]+\}", "[^/]+", defined_path)
            if re.fullmatch(path_regex, self.path):
                return defined_path

        return None

    def get_schema(self, openapi_schema: dict, preferred_content_type: Optional[str] = None) -> Optional[dict]:
        if not openapi_schema:
            return None

        path_key = self._match_path_key(openapi_schema.get("paths", {}))
        if not path_key:
            return None

        method_spec = openapi_schema["paths"][path_key].get(self.method.lower())
        if not method_spec:
            return None

        request_body = method_spec.get("requestBody", {})
        content = request_body.get("content", {})

        # If a preferred type was given or derived, check that first
        if preferred_content_type and preferred_content_type in content:
            return content[preferred_content_type].get("schema")

        # Fall back to common content types
        for content_type in ["application/json", "application/x-www-form-urlencoded", "multipart/form-data"]:
            if content_type in content:
                return content[content_type].get("schema")

        return None

    def sanitize_payload(self, payload: Dict[str, Any], openapi_schema: dict) -> Dict[str, Any]:
        """
        Recursively sanitizes the payload using the OpenAPI schema by removing unknown keys.
        """
        schema = self.schema or self.get_schema(openapi_schema, )
        if not schema:
            log.debug(f"No schema found for {self.method} {self.path} — skipping sanitization")
            return payload

        def _sanitize(data: dict, schema_props: dict, required_fields: list = None) -> dict:
            cleaned = {}
            required_fields = required_fields or []
            missing_fields = []

            for k, v in data.items():
                if k not in schema_props:
                    log.debug(f"Sanitize: removed unknown key '{k}'")
                    continue

                prop_schema = schema_props[k]

                if prop_schema.get("type") == "array" and isinstance(v, list):
                    item_schema = prop_schema.get("items", {})
                    if "properties" in item_schema:
                        cleaned[k] = [_sanitize(item, item_schema["properties"], item_schema.get("required", [])) if isinstance(item, dict) else item for item in v]
                    else:
                        cleaned[k] = v
                elif isinstance(v, dict) and "properties" in prop_schema:
                    cleaned[k] = _sanitize(v, prop_schema["properties"], prop_schema.get("required", []))
                else:
                    cleaned[k] = v

            # After processing all items, check for missing required fields
            for field in required_fields:
                if field not in cleaned:
                    missing_fields.append(field)

            if missing_fields:
                raise ValueError(f"Missing required fields during sanitization for endpoint {self.name}: {missing_fields}")

            return cleaned

        schema_props = schema.get("properties", {})
        required_fields = schema.get("required", [])
        final_cleaned = _sanitize(payload, schema_props, required_fields)

        if not final_cleaned:
            raise ValueError(f"All keys in payload were removed during sanitization for endpoint {self.name}")

        return final_cleaned

    def resolve_input_data(self, input_data, payload_type, payload_map):
        json_payload = None
        data_payload = None
        params_payload = None
        files_payload = None
        input_data = input_data or {}
        explicit_type = payload_type in ["json", "form", "multipart", "query"]
        preferred_content = self.get_preferred_content_type()
        # Use explicit payload map if given
        if payload_map:
            json_payload = {k: input_data[k] for k in payload_map.get("json", []) if k in input_data}
            data_payload = {k: input_data[k] for k in payload_map.get("data", []) if k in input_data}
            params_payload = {k: input_data[k] for k in payload_map.get("params", []) if k in input_data}
            files_payload = {k: input_data[k] for k in payload_map.get("files", []) if k in input_data}
        else:
            if explicit_type:
                if payload_type == "json":
                    json_payload = input_data
                elif payload_type == "form":
                    data_payload = input_data
                elif payload_type == "multipart":
                    files_payload = {}
                    data_payload = {}
                    for k, v in input_data.items():
                        if hasattr(v, 'read'):
                            files_payload[k] = v
                        else:
                            data_payload[k] = v
                elif payload_type == "query":
                    params_payload = input_data
            else:
                if preferred_content == "application/json":
                    json_payload = input_data
                elif preferred_content == "application/x-www-form-urlencoded":
                    data_payload = input_data
                elif preferred_content and preferred_content.startswith("multipart/form-data"):
                    files_payload = {}
                    data_payload = {}
                    for k, v in input_data.items():
                        if hasattr(v, 'read'):
                            files_payload[k] = v
                        else:
                            data_payload[k] = v
                elif preferred_content == "application/octet-stream":
                    data_payload = input_data
                elif preferred_content == "text/plain":
                    data_payload = str(input_data)
                else:
                    log.warning(f"Cannot infer payload type from Content-Type: {preferred_content}. Defaulting to json.")
                    json_payload = input_data
        return json_payload, data_payload, params_payload, files_payload

    async def call(self,
                   input_data: dict = None,
                   payload_type: str = "any",
                   payload_map: dict = None,
                   sanitize:bool=False,
                   extract_keys:str|List[str]|None=None,
                   **kwargs
                   ):
        if self.client is None:
            raise ValueError("Endpoint not bound to an APIClient")
        if not self.client.enabled:
            raise RuntimeError(f"Endpoint {self.name} was called, but API Client '{self.client.name}' is currently disabled. Use '/toggle_api' to enable the client when available.")

        headers = kwargs.pop('headers', self.headers)
        timeout = kwargs.pop('timeout', self.timeout)
        retry = kwargs.pop('retry', self.retry)
        response_type = kwargs.pop('response_type', self.response_type)

        json_payload, data_payload, params_payload, files_payload = self.resolve_input_data(input_data, payload_type, payload_map)

        if sanitize:
            if json_payload and isinstance(json_payload, dict):
                json_payload = self.sanitize_payload(json_payload, self.client.openapi_schema)
            if data_payload and isinstance(data_payload, dict):
                data_payload = self.sanitize_payload(data_payload, self.client.openapi_schema)

        response = await self.client.request(
            endpoint=self.path,
            method=self.method,
            json=json_payload,
            data=data_payload,
            params=params_payload,
            files=files_payload,
            headers=headers,
            timeout=timeout,
            retry=retry,
            response_type=response_type,
            **kwargs
        )
        if not extract_keys:
            return response
        
        return self.extract_main_keys(response, extract_keys)

    # Extracts the key values from the API response, for the Endpoint's key names defined in user API settings
    def extract_main_keys(self, response, ep_keys: str|List[str] = None):
        if not ep_keys or not isinstance(response, dict):
            return response
        
        if isinstance(ep_keys, str):
            key_paths = getattr(self, ep_keys, None)
            return try_paths(response, key_paths)

        elif isinstance(ep_keys, list):
            results = []
            for key_attr in ep_keys:
                key_paths = getattr(self, key_attr, None)
                value = try_paths(response, key_paths)
                results.append(value)
            return tuple(results)

        return response

# Utility function to get a key value like 'data.voices'
def deep_get(d: dict, path: str) -> Any:
    """Safely navigate dot-separated path in a dict."""
    keys = path.split(".")
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return None
    return d

def try_paths(response: dict, paths: Union[str, List[str]]) -> Any:
    """Try one or more deep key paths and return the first match."""
    if isinstance(paths, str):
        return deep_get(response, paths)

    if isinstance(paths, list):
        for path in paths:
            val = deep_get(response, path)
            if val is not None:
                return val
    return None


    # def handle_response(self, response):
    #     rh = self.response_handling
    #     if rh["type"] == "media_file":
    #         return utils_processing.save_from_path(response)
    #     elif rh["type"] == "base64":
    #         return utils_processing.save_base64(response)
    #     elif rh["type"] == "dict":
    #         return {k: response.get(k) for k in rh.get("extract_keys", [])}
    #     return response  # fallback

class TextGenEndpoint(Endpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Defaults

class TTSGenEndpoint(Endpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Defaults
        self.get_voices_key = 'speaker'
        self.get_languages_key = 'languages'
        self.text_input_key = 'text_input'
        self.output_file_path_key = 'output_file_path_key'
        self.language_input_key = 'language'
        self.speaker_input_key = 'character_voice_gen'

class ImgGenEndpoint(Endpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Defaults
        self.control_types_key = 'control_types'
        self.prompt_key = 'prompt'
        self.neg_prompt_key = 'negative_prompt'
        self.seed_key = 'seed'


class WorkflowExecutor:
    def __init__(self, workflow_name: str):
        self.workflow_def:dict = apisettings.get_workflow(workflow_name)
        self.api:API = get_api()
        self.context: Dict[str, Any] = {}  # Stores save_as values
        self.results: Dict[str, Any] = {}  # Final output for inspection or chaining

    async def run(self):
        steps = self.workflow_def.get("steps", [])

        for step in steps:
            if "group" in step:  # parallel group
                await self._run_group_steps(step["group"])
            else:
                await self._run_step(step)

        return self.results

    async def _run_group_steps(self, step_group: List[dict]):
        step_names = []
        for step in step_group:
            step_names.append(step.get('name'))
        log.info(f'Processing API workflow group steps: {step_names}')
        await asyncio.gather(*[self._run_step(step) for step in step_group])

    async def _run_step(self, step: dict):
        name = step.get("name", "<unnamed>")
        log.info(f'Processing API Workflow step: {name}')
        api_name = step.get("api_name")
        endpoint_name = step.get("endpoint_name")

        # Get APIClient and Endpoint
        api_client:APIClient = self.api.clients.get(api_name)
        if not api_client:
            raise ValueError(f"Unknown API client: {api_name}")
        endpoint = api_client.endpoints.get(endpoint_name)
        if not endpoint:
            raise ValueError(f"Unknown endpoint: {endpoint_name} for client {api_name}")

        # Process input
        input_data = await self._resolve_inputs(step.get("input", {}))
        payload_type = step.get("payload_type", "json")
        payload_map = step.get("payload_map")

        response = await endpoint.call(
            input_data=input_data,
            payload_type=payload_type,
            payload_map=payload_map
        )

        # Handle response
        response_handling = step.get("response_handling", {})
        output_key = response_handling.get("output_key")
        result = response.get(output_key) if (isinstance(response, dict) and output_key) else response

        # Store in context if needed
        save_as = step.get("save_as")
        if save_as:
            self.context[save_as] = result
            self.results[save_as] = result


    async def _resolve_inputs(self, inputs: Any) -> Any:
        if isinstance(inputs, str):
            if inputs.startswith("{") and inputs.endswith("}"):
                context_key = inputs[1:-1]
                return self.context.get(context_key)
            return inputs

        elif isinstance(inputs, dict):
            if "path" in inputs and "as" in inputs:
                return await self._process_file_input(inputs["path"], inputs["as"])

            resolved = {}
            for k, v in inputs.items():
                resolved[k] = await self._resolve_inputs(v)
            return resolved

        elif isinstance(inputs, list):
            return [await self._resolve_inputs(item) for item in inputs]

        else:
            return inputs


    async def _process_file_input(self, path: str, input_type: str):
        if input_type == "text":
            async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
                return await f.read()

        elif input_type == "base64":
            async with aiofiles.open(path, mode='rb') as f:
                raw = await f.read()
                return base64.b64encode(raw).decode('utf-8')

        elif input_type == "file":
            # Special behavior — this must be handled outside JSON.
            raise ValueError("File upload input type 'file' should be used with multipart/form-data.")

        elif input_type == "raw":
            async with aiofiles.open(path, mode='rb') as f:
                return await f.read()

        elif input_type == "url":
            return path  # Just return the path (expected to be a full URL)

        else:
            raise ValueError(f"Unknown input type: {input_type}")
