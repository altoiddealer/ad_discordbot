import aiohttp
from json import loads as json_loads
import asyncio
import os
import jsonschema
import jsonref
import time
import re
from PIL import Image, PngImagePlugin
import io
import base64
import copy
from modules.typing import CtxInteraction, FILE_INPUT, APIRequestCancelled
from typing import get_type_hints, get_type_hints, get_origin, get_args, Any, Tuple, Optional, Union, Callable, AsyncGenerator
from modules.utils_shared import client, shared_path, bot_database, load_file
from modules.utils_misc import progress_bar, extract_key, deep_merge, split_at_first_comma, detect_audio_format, remove_meta_keys
import modules.utils_processing as processing
from discord.ui import View, Button
import discord
from modules.utils_discord import Embeds, get_user_ctx_inter

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

class APISettings():
    def __init__(self):
        self.main_settings:dict = {}
        self.misc_settings:dict = {}
        self.presets:dict = {}
        self.workflows:dict = {}

    def get_client_type_map(self):
        return {"imggen": ImgGenClient,
                "textgen": TextGenClient,
                "ttsgen": TTSGenClient}

    def get_config_for(self, section: str, default=None) -> dict:
        return self.main_settings.get(section, default or {})

    def get_workflow_steps_for(self, workflow_name:str) -> dict:
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise RuntimeError(f'[API Workflows] Workflow not found "{workflow_name}"')
        if not isinstance(workflow, dict):
            raise ValueError(f'[API Workflows] Invalid format for workflow "{workflow_name}" (must be dict)')
        workflow_steps = workflow.get('steps')
        if not isinstance(workflow_steps, list):
            raise ValueError(f'[API Workflows] Invalid structure for workflow "{workflow_name}" (include a "step" key)')
        return workflow_steps

    def collect_presets(self, presets_list):
        for preset in presets_list:
            if not isinstance(preset, dict):
                log.warning("Encountered a non-dict response handling preset. Skipping.")
                continue
            name = preset.get('name')
            if name:
                self.presets[name] = preset

    def is_simple_step_preset(self, preset: dict) -> bool:
        """True if preset only contains 'name' and 'steps'."""
        return (isinstance(preset, dict) and
                set(preset.keys()) <= {"name", "steps"} and
                isinstance(preset.get("steps"), list))

    def get_preset(self, preset_name: str, default=None) -> dict:
        return self.presets.get(preset_name, default or {})

    def apply_preset(self, config: dict) -> dict:
        preset_name = config.get("preset")
        if not preset_name:
            return config
        preset = self.get_preset(preset_name)
        if not preset:
            log.warning(f"Preset '{preset_name}' not found.")
            return config
        # Merge preset with overrides (config wins)
        merged = deep_merge(preset, config)
        merged.pop("preset", None)
        return merged

    def apply_presets(self, config):
        """
        Recursively applies presets to all dicts found in the config (which can be a dict or list).
        Returns a new config structure with all presets applied.
        """
        if isinstance(config, dict):
            # Apply preset and get merged result
            config = self.apply_preset(config)
            # Recurse into each value and update it
            for key, value in config.items():
                config[key] = self.apply_presets(value)
            return config
        elif isinstance(config, list):
            new_list = []
            for item in config:
                # list item is a dict containing only 'preset'
                if (isinstance(item, dict) and set(item.keys()) == {"preset"}):
                    preset_name = item["preset"]
                    preset = self.get_preset(preset_name)
                    # insert the preset list items
                    if self.is_simple_step_preset(preset):
                        preset_steps = preset.get("steps", [])
                        new_list.extend(self.apply_presets(pstep) for pstep in preset_steps)
                        continue  # Skip adding the original item
                # Normal recursive case
                new_list.append(self.apply_presets(item))
            return new_list
        else:
            # Base case: return scalar values as-is
            return config

    def collect_workflows(self, workflows_list):
        for workflow in workflows_list:
            if not isinstance(workflow, dict):
                log.warning("Encountered a non-dict Workflow. Skipping.")
                continue
            name = workflow.get('name')
            if not name:
                log.warning("Encountered a Workflow with a 'name'. Skipping.")
                continue
            steps = workflow.get('steps')
            if not steps:
                log.warning("Encountered a Workflow without any 'steps'. Skipping.")
                continue
            self.workflows[name] = workflow
            self.workflows[name]['steps'] = self.apply_presets(steps)

    def collect_settings(self, data:dict):
        # Collect Main APIs
        self.main_settings = data.get('bot_api_functions', {})
        # Collect Misc API function assignments
        self.misc_settings = data.get('misc_api_functions', {})
        # Collect Response Handling Presets
        self.collect_presets(data.get('presets', {}))
        # Collect Workflows
        self.collect_workflows(data.get('workflows', {}))

apisettings = APISettings()

def validate_misc_function(func_key:str, client:"APIClient"):
    config_block = apisettings.get_config_for('misc_api_functions', {})
    func_cfg = config_block.get(func_key, {})
    for task_key, task_cfg in func_cfg.items():
        if not isinstance(task_cfg, dict):
            continue
        ep_name = task_cfg.get("endpoint_name")
        if ep_name:
            endpoint = client.endpoints.get(ep_name)
            if not endpoint:
                log.warning(f"Endpoint '{ep_name}' not found for {func_key}.{task_key}")
                continue
            for k, v in task_cfg.items():
                if k != "endpoint_name":
                    setattr(endpoint, f"_{k}", v)

# Mapping keywords to ImgGenClient subclasses
def get_imggen_client_map():
    return [(["comfy"], ImgGenClient_Comfy, "ComfyUI"),
            (["swarm"], ImgGenClient_Swarm, "SwarmUI"),
            (["reforge"], ImgGenClient_SDWebUI, "ReForge"),
            (["forge"], ImgGenClient_SDWebUI, "Forge"),
            (["stable", "a1111", "sdwebui"], ImgGenClient_SDWebUI, "A1111")]

# Mapping keywords to TTSGenClient subclasses
def get_ttsgen_client_map():
    return [(["alltalk"], TTSGenClient_AllTalk, "Alltalk")]

MAP_FUNCTIONS = {"imggen": get_imggen_client_map,
                 "ttsgen": get_ttsgen_client_map}

def _assign_from_map(name: str, client_type:str, default_class: type|None = None) -> type:
    name_lower = name.lower()

    # Dynamically get the mapping function
    map_func = MAP_FUNCTIONS.get(client_type)
    if not callable(map_func):
        log.error(f"No client map function defined for client type '{client_type}'")
        return APIClient

    client_map = map_func()
    for keywords, client_class, label in client_map:
        if any(k in name_lower for k in keywords):
            log.info(f"{name} recognized as {label}.")
            return client_class
    if default_class:
        log.info(f"{name} is an unknown '{client_type}' client. Main bot functions will rely heavily on user configuration.")
        return default_class
    return None

def assign_client_classes(name: str, func_type: str|None = None) -> type["APIClient"]:
    log.info(f"Checking if client '{name}' is a known API")
    if func_type == "imggen":
        return _assign_from_map(name, 'imggen', ImgGenClient)
    elif func_type == "ttsgen":
        return _assign_from_map(name, 'ttsgen', TTSGenClient)

    # Try both without defaulting to base classes unless necessary
    for client_type in ['imggen', 'ttsgen']:
        ClientClass = _assign_from_map(name, client_type)
        if ClientClass:
            return ClientClass
    return APIClient

class API:
    def __init__(self):
        # ALL API clients
        self.clients:dict[str, APIClient] = {}
        # Main API clients
        self.imggen:Union[ImgGenClient, DummyClient] = DummyClient(ImgGenClient)
        self.textgen:Union[TextGenClient, DummyClient] = DummyClient(TextGenClient)
        self.ttsgen:Union[TTSGenClient, DummyClient] = DummyClient(TTSGenClient)
        # Misc API functions:
        self.upload_large_files:Union[APIClient, DummyClient] = DummyClient(APIClient)
        # Collect setup tasks
        self.setup_tasks:list = []

    async def init(self):
        # Load API Settings yaml
        data = load_file(shared_path.api_settings)

        # Main APIs / Presets / Workflows
        apisettings.collect_settings(data)

        # Reverse lookups for matching API names to their function types
        main_api_name_map = {v.get("api_name"): k for k, v in apisettings.main_settings.items()
                             if isinstance(v, dict) and v.get("api_name")}
        misc_api_name_map = {v.get("api_name"): k for k, v in apisettings.misc_settings.items()
                            if isinstance(v, dict) and v.get("api_name")}
        
        check_clients_online = []

        # Iterate over all APIs data
        apis:dict = data.get('all_apis', {})
        # Expand any presets in the data
        apis = apisettings.apply_presets(apis)
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
            ClientClass = assign_client_classes(name, api_func_type)

            # Collect all valid user APIs
            try:
                api_client: APIClient = ClientClass(
                    name=name,
                    url=api_config['url'],
                    enabled=enabled,
                    websocket_config=api_config.get('websocket'),
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
                # Check if this API client corresponds to a misc function
                misc_func_key = misc_api_name_map.get(name)
                if misc_func_key:
                    setattr(self, misc_func_key, api_client)
                    log.info(f"Registered misc API function '{misc_func_key}': {name}")
                    validate_misc_function(misc_func_key, api_client)
                if hasattr(api_client, 'is_online'):
                    check_clients_online.append(api_client.is_online())
                # Collect setup tasks
                if hasattr(api_client, 'setup'):
                    self.setup_tasks.append(api_client.setup())
            except KeyError as e:
                log.warning(f"Skipping API Client due to missing key: {e}")
            except TypeError as e:
                log.warning(f"Failed to create API client '{name}': {e}")

        await asyncio.gather(*check_clients_online)

    async def setup_all_clients(self):
        if not self.setup_tasks:
            return
        try:
            await asyncio.gather(*self.setup_tasks)
            self.setup_tasks = []  # Clear after use
        except Exception as e:
            pass
    
    def get_client(self, client_type:str|None=None, client_name:str|None=None, strict=False):
        api_client = None
        main_client = getattr(self, client_type) if client_type else None
        if main_client:
            api_client = main_client
        else:
            api_client = self.clients.get(client_name)

        msg_prefix = f'API Client "{client_name}"' if client_type is None else f'Main {client_type.upper()} API Client "{client_name}"'

        if not api_client:
            if strict:
                raise ValueError(f"{msg_prefix} not found or invalid.")
            else:
                log.warning(f"{msg_prefix} not found or invalid.")
            return None

        elif not api_client.enabled:
            if strict:
                raise RuntimeError(f'{msg_prefix} is currently disabled.')
            else:
                log.warning(f'{msg_prefix} is currently disabled.')
            return None

        return api_client

    def get_client_and_endpoint(self, endpoint_name:str, client_type:str|None=None, client_name:str|None=None, strict=False) -> Tuple["APIClient", "Endpoint"]:
        api_client = self.get_client(client_type=client_type, client_name=client_name, strict=strict)
        if not api_client:
            return None, None
        endpoint = api_client.get_endpoint(endpoint_name=endpoint_name, strict=strict)
        return api_client, endpoint

    def get_endpoint_from_config(self, config: dict) -> Tuple[Optional["Endpoint"], dict]:
        client_type = str(config.pop('client_type', '') or '').lower()
        client_name = str(config.pop('client_name', '') or '').lower()
        endpoint_type = str(config.pop('endpoint_type', '') or '').lower()
        endpoint_name = str(config.pop('endpoint_name', '') or '').lower()
        # Main client types
        type_to_client = {'imggen': self.imggen,
                          'textgen': self.textgen,
                          'ttsgen': self.ttsgen}
        # Step 1: Try using client_type (specialized client)
        if client_type in type_to_client:
            client = type_to_client[client_type]
            # 1a: Exact match by endpoint_name
            if endpoint_name:
                ep = getattr(client, endpoint_name, None)
                if isinstance(ep, Endpoint):
                    return ep, config
            # 1b: Fallback: match by endpoint_type
            if endpoint_type:
                for attr in dir(client):
                    ep = getattr(client, attr, None)
                    if isinstance(ep, Endpoint) and endpoint_type in attr.lower():
                        return ep, config
        # Step 2: Try using client_name
        if client_name and client_name in self.clients:
            client = self.clients[client_name]
            if endpoint_name and endpoint_name in client.endpoints:
                return client.endpoints[endpoint_name], config
            if endpoint_type:
                for ep in client.endpoints.values():
                    if endpoint_type in ep.name.lower():
                        return ep, config
        # Step 3: Search all clients if only endpoint_name or endpoint_type is provided
        for client in self.clients.values():
            # Match exact name
            if endpoint_name and endpoint_name in client.endpoints:
                return client.endpoints[endpoint_name], config
            # Match by type
            if endpoint_type:
                for ep in client.endpoints.values():
                    if endpoint_type in ep.name.lower():
                        return ep, config
        # Not found
        return None, config

    def get_misc_function_endpoint(self, func_key: str, task_key: str) -> Union["Endpoint", None]:
        """Safely fetch an endpoint for a misc function"""
        client: APIClient = getattr(self, func_key, None)
        if not isinstance(client, APIClient):
            return

        misc_settings = apisettings.misc_settings
        task_cfg = misc_settings.get(func_key, {}).get(task_key, {})
        ep_name = task_cfg.get("endpoint_name")

        if ep_name:
            return client.endpoints.get(ep_name)


class WebSocketConnectionConfig:
    def __init__(self, **kwargs):
        self.url: Optional[str] = kwargs.get("url")
        self.query_params: dict = kwargs.get("query_params", {})

        # ID / session / channel support
        self.client_id_required: bool = kwargs.get("client_id_required", False)
        self.client_id_format: str = kwargs.get("client_id_format", "uuid")
        self.token_required: bool = kwargs.get("token_required", False)
        self.token_name: str = kwargs.get("token_name", "token")
        self.auth_token: Optional[str] = kwargs.get("auth_token")

        self.session_id_required: bool = kwargs.get("session_id_required", False)
        self.session_id_name: str = kwargs.get("session_id_name", "session_id")
        self.session_id: Optional[str] = kwargs.get("session_id")

        self.channel_required: bool = kwargs.get("channel_required", False)
        self.channel_name: str = kwargs.get("channel_name", "channel")
        self.channel: Optional[str] = kwargs.get("channel")

        self.version_required: bool = kwargs.get("version_required", False)
        self.version_name: str = kwargs.get("version_name", "version")
        self.version: Optional[str] = kwargs.get("version")

        self.headers: dict = kwargs.get("headers", {})

        self.client_id: Optional[str] = None

    def get_context(self) -> dict[str, str]:
        return {"token": self.auth_token or "",
                "client_id": self.client_id or "",
                "session_id": self.session_id or "",
                "channel": self.channel or "",
                "version": self.version or ""}

    def build_headers(self) -> dict[str, str]:
        context = self.get_context()
        return processing.resolve_placeholders(self.headers, context, log_prefix='[Websocket]', log_suffix='for headers')

    def generate_client_id(self, format: str) -> str:
        import secrets
        if format == "uuid":
            import uuid
            return str(uuid.uuid4())
        elif format == "short":
            import string
            return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))
        elif format == "timestamp":
            import time
            return str(int(time.time()))
        elif format == "hex":
            return secrets.token_hex(8)
        elif format == "opaque":
            return secrets.token_urlsafe(16)
        elif format == "int":
            return str(secrets.randbelow(10**8))
        elif format == "machine":
            import socket
            return socket.gethostname()
        elif format == "env_user":
            return os.getenv("USER", "unknown_user")
        else:
            return "client"  # fallback/default

    def build_url(self, fallback_http_url: str) -> str:
        from urllib.parse import urlencode, urlparse, urlunparse, parse_qs
        # Generate client_id if needed
        if self.client_id_required and not self.client_id:
            self.client_id = self.generate_client_id(self.client_id_format)
        # Start with configured URL or fallback from HTTP
        if self.url:
            parsed = urlparse(self.url)
        else:
            scheme = "wss" if fallback_http_url.startswith("https") else "ws"
            base = fallback_http_url.replace("http://", "").replace("https://", "").rstrip("/")
            parsed = urlparse(f"{scheme}://{base}/ws")
        # Build query dictionary from:
        # - Explicit query_params with placeholder replacements
        # - Additional required params like client_id, token, etc.
        final_params = dict(self.query_params)

        if self.client_id_required:
            final_params.setdefault("clientId", self.client_id)
        if self.token_required and self.auth_token:
            final_params[self.token_name] = self.auth_token
        if self.session_id_required and self.session_id:
            final_params[self.session_id_name] = self.session_id
        if self.channel_required and self.channel:
            final_params[self.channel_name] = self.channel
        if self.version_required and self.version:
            final_params[self.version_name] = self.version

        # Replace placeholders (e.g., {client_id}) in query_params
        context = self.get_context()
        resolved_query = processing.resolve_placeholders(final_params, context, log_prefix='[Websocket]', log_suffix='for query parameters')
        original_params = parse_qs(parsed.query)
        for k, v in resolved_query.items():
            original_params[k] = [v]

        final_query = urlencode({k: v[0] for k, v in original_params.items()})
        rebuilt = parsed._replace(query=final_query)

        return urlunparse(rebuilt)


class APIClient:
    _endpoint_class_map: dict[str, type] = {}

    def __init__(self,
                 name: str,
                 enabled: bool,
                 url: str,
                 websocket_config = None,
                 default_headers: Optional[dict[str, str]] = None,
                 default_timeout: int = 60,
                 auth: Optional[dict] = None,
                 endpoints_config=None):

        self.name = name
        self.session:Optional[aiohttp.ClientSession] = None
        self.enabled = enabled
        self.url = url.rstrip("/")
        self.default_headers = default_headers or {}
        self.default_timeout = default_timeout
        self.auth = auth
        self.endpoints: dict[str, Endpoint] = {}
        self.openapi_schema = None
        self.fetching_progress = False
        self.cancel_event = asyncio.Event()
        self._endpoint_fetch_payloads = []
        # WebSocket connection
        self.ws = None
        self.ws_config: Optional[WebSocketConnectionConfig] = None
        if websocket_config:
            self.ws_config = WebSocketConnectionConfig(**websocket_config)
        # set auth
        if auth:
            try:
                self.auth = aiohttp.BasicAuth(auth["username"], auth["password"])
            except KeyError:
                log.warning(f"[{self.name}] Invalid auth dict: 'username' or 'password' missing.")
                self.auth = None
        # collect endpoints
        if endpoints_config:
            self._collect_endpoints(endpoints_config)

    async def setup(self):
        if not self.enabled:
            return
        # Create and retain reusable session per API
        self.session = aiohttp.ClientSession()
        if self.ws_config:
            await self.connect_websocket()
        await self._fetch_openapi_schema()
        self._assign_endpoint_schemas()
        await self._resolve_deferred_payloads()
        await self.main_setup_tasks()

    async def main_setup_tasks(self):
        pass

    def add_required_values_to_payload(self, payload:dict):
        pass

    def get_endpoint(self, endpoint_name:str, strict=False) -> "Endpoint":
        endpoint = self.endpoints.get(endpoint_name)
        if not endpoint:
            if strict:
                raise ValueError(f'[{self.name}] Endpoint "{endpoint_name}" not found or invalid')
            else:
                log.warning(f'[{self.name}] Endpoint "{endpoint_name}" not found or invalid')
            return None
        return endpoint

    async def toggle(self) -> str | None:
        try:
            if self.enabled:
                await self.go_offline()
                return 'disabled'
            else:
                await self.come_online()
                return 'enabled'
        except Exception as e:
            log.error(f"[{self.name}] toggle() failed: {e}")
            return None

    async def go_offline(self) -> bool:
        if not self.enabled:
            return
        log.warning(f"[{self.name}] disabled. Use '/toggle_api' to try enabling it when available.")
        self.enabled = False
        if self.session is not None:
            await self.close() # also closes websocket

    async def come_online(self) -> bool:
        if self.enabled:
            return
        self.enabled = True
        await self.setup()
        log.info(f"[{self.name}] enabled. Use '/toggle_api' to disable.")

    async def connect_websocket(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        url = self.ws_config.build_url(self.url)
        headers = self.ws_config.build_headers()
        try:
            self.ws = await self.session.ws_connect(url, headers=headers)
            log.info(
                f"[{self.name}] WebSocket connection established ({url})")
        except Exception as e:
            log.error(f"[{self.name}] failed to connect to WebSocket: {e}")
            self.ws = None
            raise

    async def close(self):
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                log.warning(f"[{self.name}] Error closing WebSocket: {e}")
            finally:
                self.ws = None

        if self.session:
            await self.session.close()
            self.session = None

    def _create_endpoint(self, EPClass: type, ep_dict: dict):
        return EPClass(name=ep_dict["name"],
                        path=ep_dict.get("path", ""),
                        method=ep_dict.get("method", "GET"),
                        response_type=ep_dict.get("response_type", "json"),
                        payload_type=ep_dict.get("payload_type", "any"),
                        payload_config=ep_dict.get("payload_base", ep_dict.get("payload")),
                        response_handling=ep_dict.get("response_handling"),
                        headers=ep_dict.get("headers", self.default_headers),
                        stream=ep_dict.get("stream", False),
                        timeout=ep_dict.get("timeout", self.default_timeout),
                        retry=ep_dict.get("retry", 0),
                        concurrency_limit=ep_dict.get("concurrency_limit", None))

    def default_endpoint_class(self):
        return Endpoint

    def get_endpoint_class_map(self) -> dict[str, type]:
        return {}

    def _collect_endpoints(self, endpoints_config: list[dict]):
        # Determine applicable settings and endpoint class mappings
        client_type_map = apisettings.get_client_type_map()
        ep_class_map = self.get_endpoint_class_map()
        default = self.default_endpoint_class()

        # Determine client key (e.g., 'imggen', 'textgen', etc.)
        client_key = next((k for k, cls in client_type_map.items() if isinstance(self, cls)), None)
        endpoint_config_block = apisettings.main_settings.get(client_key, {}) if client_key else {}

        # Map from endpoint_name → config key (e.g., "Prompt" → "post_txt2img")
        name_to_config_key = {cfg["endpoint_name"]: key
                              for key, cfg in endpoint_config_block.items()
                              if isinstance(cfg, dict) and "endpoint_name" in cfg}

        for ep_dict in endpoints_config:
            try:
                ep_name = ep_dict["name"]
                config_key = name_to_config_key.get(ep_name)
                ep_class = ep_class_map.get(config_key, default)

                endpoint = self._create_endpoint(ep_class, ep_dict)
                endpoint.client = self

                if hasattr(endpoint, "_deferred_payload_source"):
                    self._endpoint_fetch_payloads.append(endpoint)

                self.endpoints[endpoint.name] = endpoint
            except KeyError as e:
                log.warning(f"[{self.name}] Skipping endpoint due to missing key: {e}")

    def _bind_main_ep_values(self, config_entry: dict):
        if not isinstance(self, (TextGenClient, TTSGenClient, ImgGenClient)):
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
        if self.openapi_schema:
            return
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.url}/openapi.json") as response:
                    if 200 <= response.status < 300:
                        raw_schema = await response.json()
                        self.openapi_schema = jsonref.JsonRef.replace_refs(raw_schema)
                        log.debug(f"[{self.name}] Loaded OpenAPI schema.")
                    else:
                        log.debug(f"[{self.name}] No OpenAPI schema available.")
        except Exception as e:
            log.warning(f"[{self.name}] Failed to load OpenAPI schema from {self.url}: {e}")
            await self.go_offline()
            raise

    def _assign_endpoint_schemas(self):
        if not self.openapi_schema:
            return

        for endpoint in self.endpoints.values():
            endpoint.set_openapi_schema(self.openapi_schema)
            if endpoint.schema:
                log.debug(f"[{self.name}] Schema assigned to endpoint '{endpoint.name}'")
            else:
                log.debug(f"[{self.name}] No schema found for endpoint '{endpoint.name}'")

    async def _resolve_deferred_payloads(self):
        resolved_endpoints = []
        for ep in self._endpoint_fetch_payloads:
            ep:Endpoint
            if ep.payload:
                continue
            ref = ep._deferred_payload_source
            ref_ep = self.endpoints.get(ref)

            if ref_ep is None:
                log.warning(f"[{self.name}] Endpoint '{ep.name}' references unknown payload source: '{ref}'")
                continue

            log.debug(f"[{self.name}] Fetching payload for '{ep.name}' using endpoint '{ref}'")
            try:
                data = await ref_ep.call()
                if isinstance(data, dict):
                    ep.payload = data
                    resolved_endpoints.append(ep)
                else:
                    log.warning(f"[{self.name}] Endpoint '{ref}' returned non-dict data for '{ep.name}'")
            except Exception as e:
                log.error(f"[{self.name}] Failed to fetch payload from '{ref}' for '{ep.name}': {e}")
        # Remove resolved endpoints from the fetch list
        self._endpoint_fetch_payloads = [ep for ep in self._endpoint_fetch_payloads if ep not in resolved_endpoints]

    def validate_payload(self, method:str, endpoint:str, json:str|None=None, data:str|None=None):
        if self.openapi_schema:
            try:
                if json:
                    jsonschema.validate(instance=json, schema=self.openapi_schema)
                elif data and isinstance(data, dict):
                    jsonschema.validate(instance=data, schema=self.openapi_schema)
            except jsonschema.ValidationError as e:
                log.error(f"[{self.name}] Schema validation failed for {method} {endpoint}: {e.message}")
                raise

    async def is_online(self) -> Tuple[bool, str]:
        if not self.enabled:
            return False
        try:
            response = await self.request(endpoint='', method='GET', retry=0, timeout=5)
            if response:
                return True, ''
            else:
                return False, ''
        except aiohttp.ClientError as e:
            emsg = f"[{self.name}] is enabled but unresponsive at url {self.url}: {e}"
            await self.go_offline()
            return False, emsg

    async def _stream_response(self, response: aiohttp.ClientResponse, response_type: str, url: str):
        try:
            if response_type == "text" or response_type == "json":
                async for line in response.content:
                    if not line:
                        continue
                    try:
                        decoded = line.decode("utf-8").strip()
                        if response_type == "json":
                            yield json_loads(decoded)
                        else:
                            yield decoded
                    except Exception as e:
                        log.warning(f"Streaming parse error on line from {url}: {e}")
                        continue
            else:
                log.warning(f"Streaming mode not supported for response_type '{response_type}' — returning raw bytes")
                async for chunk in response.content.iter_chunked(1024):
                    yield chunk
        except Exception as e:
            log.exception(f"Error while streaming from {url}: {e}")

    async def _process_response(self, response: aiohttp.ClientResponse, response_type: Optional[str], url: str):
        try:
            # Default to JSON if not specified
            if response_type == "json" or not response_type:
                return await response.json()

            elif response_type == "text":
                return await response.text()

            elif response_type in ("bytes", "binary"):
                return await response.read()

            elif response_type == "image":
                # Returns raw image bytes — caller can process further
                return await response.read()

            elif response_type == "base64":
                # Base64-encoded plain text body
                encoded = await response.text()
                return base64.b64decode(encoded)

            elif response_type == "base64_json":
                # Common format: {"data": "<base64string>"}
                data = await response.json()
                b64_str = data.get("data")
                if not b64_str:
                    raise ValueError("Missing 'data' field for base64_json response")
                return base64.b64decode(b64_str)

            elif response_type == "data_url":
                # Handle data URL format: data:image/png;base64,....
                data_url = await response.text()
                if "base64," in data_url:
                    base64_part = data_url.split("base64,", 1)[1]
                    return base64.b64decode(base64_part)
                raise ValueError("Invalid data URL format")

            elif response_type == "csv":
                import csv
                text = await response.text()
                return list(csv.reader(io.StringIO(text)))

            elif response_type == "pdf":
                return await response.read()  # treat as bytes

            elif response_type == "yaml":
                import yaml
                text = await response.text()
                return yaml.safe_load(text)

            elif response_type == "html" or response_type == "html_fragment":
                return await response.text()

            elif response_type == "markdown":
                return await response.text()

            elif response_type == "javascript":
                return await response.text()

            elif response_type == "none":
                return None

            else:
                log.warning(f"[{self.name}] Unknown response_type '{response_type}' for {url}, falling back to raw bytes")
                return await response.read()

        except Exception as e:
            log.exception(f"Error processing response from {url} with response_type={response_type}: {e}")
            try:
                return await response.text()
            except Exception:
                return await response.read()

    async def guess_response_type(self, response: aiohttp.ClientResponse):
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        try:
            if content_type == "application/json":
                return await response.json()
            elif content_type.startswith("text/") or content_type in ("application/xml", "text/xml", "text/csv"):
                return await response.text()
            elif content_type.startswith(("audio/", "video/", "image/", "font/")) or content_type in (
                "application/octet-stream",
                "application/pdf",
                "application/zip",
                "application/x-gzip"
            ):
                return await response.read()
            else:
                # Unknown content type — try JSON, then text, then bytes
                try:
                    return await response.json()
                except aiohttp.ContentTypeError:
                    try:
                        return await response.text()
                    except Exception:
                        return await response.read()
        except Exception as e:
            log.warning(f"Failed to decode response (Content-Type: {content_type}): {e}")
            return await response.read()

    def format_endpoint(self, endpoint: str, path_vars: Optional[Union[str, tuple, list, dict]]) -> str:
        if not path_vars:
            return endpoint
        try:
            if isinstance(path_vars, dict):
                return endpoint.format(**path_vars)
            elif isinstance(path_vars, (tuple, list)):
                return endpoint.format(*path_vars)
            else:
                return endpoint.format(path_vars)
        except (IndexError, KeyError, ValueError) as e:
            raise ValueError(f"Failed to format endpoint '{endpoint}' with path_vars {path_vars}: {e}")

    async def request(
        self,
        endpoint: str,
        method: str = "GET",
        json: Optional[dict[str, Any]] = None,
        data: Optional[Union[dict[str, Any], str, bytes]] = None,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        files: Optional[dict[str, Any]] = None,
        auth: Optional[aiohttp.BasicAuth] = None,
        retry: int = 0,
        timeout: Optional[int] = None,
        response_type: Optional[str] = None,
        stream: bool = False,
        path_vars: Optional[Union[str, tuple, list, dict]] = None,
    ) -> Union[dict[str, Any], str, bytes, None]:
        
        # Format endpoint with path variables
        if path_vars:
            endpoint = self.format_endpoint(endpoint, path_vars)

        url = f"{self.url}{endpoint}" if endpoint.startswith("/") else f"{self.url}/{endpoint}"
        headers = {**self.default_headers, **(headers or {})}
        timeout = timeout or self.default_timeout

        # Validate payload
        self.validate_payload(method, endpoint, json, data)
        
        # Finalize request
        request_kwargs = {"method": method.upper(),
                          "url": url,
                          "params": params,
                          "headers": headers,
                          "auth": auth or self.auth,
                          "timeout": aiohttp.ClientTimeout(total=timeout)}

        if data is not None:
            data = remove_meta_keys(data)
            request_kwargs["data"] = data
            # Content-Type will be set in Form automatically
            if isinstance(data, aiohttp.FormData) and headers:
                request_kwargs['headers'].pop("Content-Type", None)
        if json is not None:
            json = remove_meta_keys(json)
            request_kwargs["json"] = json

        # Ensure session exists
        if not self.session:
            self.session = aiohttp.ClientSession()

        for attempt in range(retry + 1):
            try:
                async with self.session.request(**request_kwargs) as response:

                    if 200 <= response.status < 300:
                        content_type = response.headers.get("Content-Type", "")
                        response_headers = dict(response.headers)

                        if stream:
                            return self._stream_response(response, response_type, url)
                        else:
                            if response_type == "bytes":
                                raw = await response.read()
                                return APIResponse(body=raw, headers=response_headers, status=response.status, content_type=content_type, raw=raw)
                            elif response_type == "text":
                                text = await response.text()
                                return APIResponse(body=text, headers=response_headers, status=response.status, content_type=content_type)
                            elif response_type == "json":
                                json_body = await response.json()
                                return APIResponse(body=json_body, headers=response_headers, status=response.status, content_type=content_type)
                            else:
                                guessed = await self.guess_response_type(response)
                                return APIResponse(body=guessed, headers=response_headers, status=response.status, content_type=content_type)
                    else:
                        response_text = await response.text()
                        log.error(f"HTTP {response.status} Error: {response_text}")
                        response.raise_for_status()

            except (aiohttp.ClientConnectionError, aiohttp.ClientResponseError, asyncio.TimeoutError, Exception) as e:
                if isinstance(e, aiohttp.ClientConnectionError):
                    log.warning(f"[{self.name}] Connection error to {url}, attempt {attempt + 1}/{retry + 1}")
                elif isinstance(e, aiohttp.ClientResponseError):
                    log.error(f"[{self.name}] HTTP Error {e.status} on {url}: {e.message}")
                elif isinstance(e, asyncio.TimeoutError):
                    log.error(f"[{self.name}] Request to {url} timed out.")
                else:
                    log.exception(f"[{self.name}] Unexpected error with {url}")
                raise

            if attempt < retry:
                await asyncio.sleep(2 ** attempt)

        return None

    async def _send_ws_message(
        self,
        json: Optional[dict[str, Any]],
        timeout: Optional[int],
        expect_response: bool,
        response_type: Optional[str],
    ) -> Union[dict[str, Any], str, bytes, None]:
        if not json:
            raise ValueError("WebSocket messages must be JSON serializable.")

        timeout = timeout or self.default_timeout

        if not self.ws or self.ws.closed:
            await self.connect_websocket()

        try:
            await self.ws.send_json(json)

            if expect_response:
                msg = await self.ws.receive(timeout=timeout)

                if msg.type == aiohttp.WSMsgType.TEXT:
                    if response_type == "text":
                        return msg.data
                    try:
                        return json.loads(msg.data)
                    except Exception:
                        log.warning("Failed to parse WebSocket text as JSON.")
                        return msg.data
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    return msg.data if response_type == "bytes" else msg.data.decode("utf-8")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise RuntimeError(f"WebSocket error: {msg}")
        except Exception as e:
            log.exception(f"[{self.name}] WebSocket message failed: {e}")
            raise

    async def poll_ws(
        self,
        return_values: dict,
        interval: float = 1.0,
        duration: int = -1,
        num_yields: int = -1,
        timeout: int = 60,
        type_filter: Optional[list[str]] = None,
        data_filter: Optional[dict] = None,
        completion_condition: Optional[Callable[[dict], bool]] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream messages from WebSocket and yield structured data at most once per yield_interval seconds.
        
        - return_values: Dict of key -> dot/bracket path to extract
        - type_filter: List of message types to process
        - data_filter: Dict of required key-values in data field
        - duration: Time in seconds before polling stops
        - timeout: Timeout per WS receive
        - interval: Minimum time in seconds between yields
        """
        start_time = time.monotonic()
        last_successful_yield = time.monotonic()
        yield_count = 0
        last_yield_time = 0.0
        buffered_result = None
        MIN_RECEIVE_TIMEOUT = 1.0

        while True:
            now = time.monotonic()
            elapsed = now - start_time
            time_since_last_yield = now - last_successful_yield

            # Stop polling after duration
            if duration > 0 and elapsed >= duration:
                if buffered_result:
                    yield buffered_result
                log.info(f"[{self.name}] WebSocket polling stopped after duration {duration}s")
                break

            # Stop polling after timeout of no yields
            if timeout > 0 and time_since_last_yield >= timeout:
                if buffered_result:
                    yield buffered_result
                log.info(f"[{self.name}] WebSocket polling stopped after inactivity timeout {timeout}s")
                break

            try:
                if not self.ws or self.ws.closed:
                    await self.connect_websocket()

                # Wait for message OR cancel event OR timeout
                receive_task = asyncio.create_task(self.ws.receive())
                wait_tasks = [receive_task]
                cancel_task = asyncio.create_task(self.cancel_event.wait())
                wait_tasks.append(cancel_task)

                done, pending = await asyncio.wait(
                    wait_tasks,
                    timeout=(interval if interval >= MIN_RECEIVE_TIMEOUT else timeout),
                    return_when=asyncio.FIRST_COMPLETED
                )

                if cancel_task in done:
                    receive_task.cancel()
                    raise APIRequestCancelled(f"[{self.name}] Generation was cancelled by user.", cancel_event=self.cancel_event)

                if receive_task in done:
                    msg = receive_task.result()
                else:
                    receive_task.cancel()
                    # Timeout occurred
                    if buffered_result and (interval == 0.0 or (time.monotonic() - last_yield_time) >= interval):
                        yield buffered_result
                        yield_count += 1
                        last_successful_yield = time.monotonic()
                        last_yield_time = last_successful_yield
                        buffered_result = None
                        if num_yields > 0 and yield_count >= num_yields:
                            log.info(f"[{self.name}] WebSocket polling stopped after num_yields {num_yields}")
                            break
                    continue

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    payload = json_loads(msg.data)
                except Exception:
                    log.warning(f"[{self.name}] Invalid JSON in WebSocket message")
                    continue
                
                # Check for completion condition
                if completion_condition and completion_condition(payload):
                    log.info(f"[{self.name}] Completion condition matched.")
                    yield payload
                    break

                msg_type = payload.get("type")
                data = payload.get("data", {})

                if type_filter and msg_type not in type_filter:
                    continue
                if data_filter and any(data.get(k) != v for k, v in data_filter.items()):
                    continue

                result = {}
                for key, path in return_values.items():
                    try:
                        result[key] = extract_key(payload, path)
                    except Exception as e:
                        log.debug(f"[{self.name}] Extraction failed for {key}: {e}")
                        continue

                if not result:
                    continue

                now = time.monotonic()
                if interval > 0 and (now - last_yield_time) < interval:
                    buffered_result = result  # Buffer until interval is met
                    continue

                yield result
                yield_count += 1
                last_successful_yield = now
                last_yield_time = now
                buffered_result = None

                if num_yields > 0 and yield_count >= num_yields:
                    log.info(f"[{self.name}] WebSocket polling stopped after num_yields {num_yields}")
                    break

            except Exception as e:
                log.exception(f"[{self.name}] WebSocket polling failed: {e}")

    class CancelView(View):
        def __init__(self, cancel_callback: callable, user: discord.User, timeout: float = 300):
            super().__init__(timeout=timeout)
            self.cancel_callback = cancel_callback
            self.author = user

        @discord.ui.button(label="Cancel Generation", style=discord.ButtonStyle.secondary, custom_id="cancel_button")
        async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
            if interaction.user.id != self.author.id and not client.is_owner(interaction.user):
                await interaction.response.send_message("Only the initial user may cancel this task!",
                                                        ephemeral=True)
                return

            await interaction.response.defer()
            await self.cancel_callback()
            self.stop()

    async def track_progress(self,
                             endpoint: Optional["Endpoint"] = None,
                             use_ws=False,
                             interval: float = 1.0,
                             duration: int = -1,
                             num_yields: int = -1,
                             progress_key: str = "progress",
                             max_key: str|None = None,
                             eta_key: str|None = None,
                             message: str = 'Generating',
                             ictx: CtxInteraction|None = None,
                             task = None,
                             type_filter: list[str] = ["progress", "executed"], # websocket
                             data_filter: dict|None = None, # websocket
                             completion_condition: Callable[[dict], bool]|None = None,
                             **kwargs) -> list[dict]:
        """
        Polls an endpoint while sending a progress Embed to discord.
        Pops and manages 'return_values' to ensure polling method only returns progress data
        If `max` is specified, progress is interpreted as a step count and normalized as (progress / max).
        Otherwise, progress is assumed to be a float between 0.0 and 1.0.
        """
        embeds = task.embeds if task else Embeds(ictx)
        ictx = task.ictx if task else ictx

        # Resolve endpoint / websocket
        if not endpoint and not use_ws:
            endpoint = getattr(self, "get_progress", None)
            if not endpoint:
                log.warning(f'[{self.name}] "track_progress" has no configured endpoint. Defaulting to assume websocket method.')
                use_ws = True

        cancel_ep = getattr(self, "post_cancel", None)
        async def cancel_callback():
            payload = cancel_ep.get_payload()
            await cancel_ep.call(input_data=payload)
            self.cancel_event.set()

        # Resolve progress_values
        return_values:dict = kwargs.pop("return_values", {})

        progress_values = {}
        progress_values['progress'] = return_values.get('progress') or progress_key or "progress"
        max_value_key = return_values.get('max') or max_key or None
        if max_value_key:
            progress_values['max'] = max_value_key
        eta_value_key = return_values.get('eta') or return_values.get('eta_relative') or eta_key or None
        if eta_value_key:
            progress_values['eta'] = eta_value_key

        STALL_THRESHOLD = 5.0
        last_progress = 0.0
        stall_time = 0.0

        updates = []

        # Prevent multiple progress tasks on same endpoint from running in tandem
        while self.fetching_progress == True:
            await asyncio.sleep(1.0)
        self.fetching_progress = True

        try:
            title = f'Waiting for {self.name} ...'
            description = f'{progress_bar(0)}'
            eta_message = ''
            embed_msg = await embeds.send('img_gen', title, description)
            # Attach button if cancel endpoint exists
            if cancel_ep and embed_msg and ictx:
                ictx_user = get_user_ctx_inter(ictx)
                view = self.CancelView(cancel_callback, user=ictx_user)
                await embed_msg.edit(view=view)

            poller = self.poll_ws(return_values=progress_values,
                                  interval=interval,
                                  duration=duration,
                                  num_yields=num_yields,
                                  type_filter=type_filter,
                                  data_filter=data_filter,
                                  completion_condition=completion_condition) \
                     if use_ws else \
                     endpoint.poll(return_values=progress_values,
                                   interval=interval,
                                   duration=duration,
                                   num_yields=num_yields,
                                   completion_condition=completion_condition,
                                   **kwargs)

            async for update in poller:
                try:
                    if self.cancel_event.is_set():
                        raise APIRequestCancelled(f"[{self.name}] Generation was cancelled by user.", cancel_event=self.cancel_event)
                    # Collect updates
                    updates.append(update)

                    # Read progress safely
                    raw_progress = update.get("progress", 0.0)
                    raw_max = update.get("max", 1.0)  # default to 1.0

                    try:
                        progress = float(raw_progress)
                        max_value = float(raw_max)
                        progress = progress / max_value
                    except (TypeError, ValueError, ZeroDivisionError):
                        progress = 0.0

                    progress = max(0.0, min(progress, 1.0))  # Clamp between 0.0 and 1.0

                    eta = update.get('eta')

                    # Completion check
                    if completion_condition and completion_condition(update):
                        break
                    elif not completion_condition and progress >= 1.0:
                        break

                    # Check for stalled condition
                    if progress == last_progress:
                        stall_time += interval
                    else:
                        stall_time = 0.0

                    # Edit the Discord Embed
                    if progress > 0.01:
                        comment = " (Stalled)" if stall_time >= STALL_THRESHOLD else ""
                        title = f"{message}: {progress * 100:.0f}%{comment}"
                        if isinstance(eta, float) or isinstance(eta, int):
                            eta_message = f'\n**ETA**: {round(eta, 2)} seconds'
                        description = f"{progress_bar(progress)}{eta_message}"
                        await embeds.edit("img_gen", title, description)

                    last_progress = progress

                except Exception as e:
                    await embeds.edit_or_send('img_gen', f'[{self.name}] An error occurred while {message}', e)
                    break

        finally:
            self.fetching_progress = False
            if self.cancel_event.is_set():
                await embed_msg.edit(view=View())
            else:
                await embeds.delete('img_gen')
        return updates

# Dummy main objects to allow graceful evaluation
def _unwrap_optional(type_hint):
    origin = get_origin(type_hint)
    if origin is Union:
        # Remove NoneType from Union
        non_none = [t for t in get_args(type_hint) if t is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return type_hint

class DummyEndpoint:
    def __init__(self, endpoint_cls: type):
        self.name = "dummy"
    def __getattr__(self, item):
        return None
    def __bool__(self):
        return False

class DummyClient:
    def __init__(self, target_cls: type):
        annotations = get_type_hints(target_cls)
        for attr_name, attr_type in annotations.items():
            real_type = _unwrap_optional(attr_type)
            if isinstance(real_type, type) and issubclass(real_type, Endpoint):
                setattr(self, attr_name, DummyEndpoint(real_type))
            else:
                setattr(self, attr_name, None)
    def __getattr__(self, name):
        return None
    def __bool__(self):
        return False


class ImgGenClient(APIClient):
    post_txt2img: Optional["ImgGenEndpoint_PostTxt2Img"] = None
    post_img2img: Optional["ImgGenEndpoint_PostImg2Img"] = None
    post_cancel: Optional["ImgGenEndpoint_PostCancel"] = None
    get_progress: Optional["ImgGenEndpoint_GetProgress"] = None
    post_pnginfo: Optional["ImgGenEndpoint_PostPNGInfo"] = None
    post_options: Optional["ImgGenEndpoint_PostOptions"] = None
    get_imgmodels: Optional["ImgGenEndpoint_GetImgModels"] = None
    get_controlnet_models: Optional["ImgGenEndpoint_GetControlNetModels"] = None
    post_server_restart: Optional["ImgGenEndpoint_PostServerRestart"] = None
    get_controlnet_control_types: Optional["ImgGenEndpoint_GetControlNetControlTypes"] = None
    get_history: Optional["ImgGenEndpoint_GetHistory"] = None
    get_view: Optional["ImgGenEndpoint_GetView"] = None
    post_upload: Optional["ImgGenEndpoint_PostUpload"] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        imggen_config:dict = apisettings.get_config_for("imggen")
        self._bind_main_endpoints(imggen_config)
        self._sampler_names:list[str] = []
        self._schedulers:list[str] = []
        self._lora_names:list[str] = []

    def get_endpoint_class_map(self) -> dict[str, type]:
        return {"post_txt2img": ImgGenEndpoint_PostTxt2Img,
                "post_img2img": ImgGenEndpoint_PostImg2Img,
                "post_cancel": ImgGenEndpoint_PostCancel,
                "get_progress": ImgGenEndpoint_GetProgress,
                "post_pnginfo": ImgGenEndpoint_PostPNGInfo,
                "post_options": ImgGenEndpoint_PostOptions,
                "get_imgmodels": ImgGenEndpoint_GetImgModels,
                "get_controlnet_models": ImgGenEndpoint_GetControlNetModels,
                "post_server_restart": ImgGenEndpoint_PostServerRestart,
                "get_controlnet_control_types": ImgGenEndpoint_GetControlNetControlTypes,
                "post_upload": ImgGenEndpoint_PostUpload,
                "get_history": ImgGenEndpoint_GetHistory,
                "get_view": ImgGenEndpoint_GetView}

    def default_endpoint_class(self):
        return ImgGenEndpoint
    
    async def fetch_imgmodels(self):
        return await self.get_imgmodels.call()
    
    def decode_and_save_for_index(self,
                                  i: int,
                                  data: str|bytes|list,
                                  pnginfo = None) -> FILE_INPUT:
        if isinstance(data, str):
            try:
                decoded = base64.b64decode(data)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 string: {e}")
        elif isinstance(data, list):
            try:
                decoded = bytes(data)
            except Exception as e:
                raise ValueError(f"Failed to convert list of ints to bytes: {e}")
        elif isinstance(data, bytes):
            decoded = data
        else:
            raise TypeError(f"Expected str, bytes, or list of ints, got {type(data)}")

        # Decode and open the image
        image = Image.open(io.BytesIO(decoded))

        # Save with metadata to BytesIO
        output_bytes = io.BytesIO()
        image.save(output_bytes, format="PNG", pnginfo=pnginfo)
        output_bytes.seek(0)
        output_bytes.name = f"image_{i}.png"
        file_size = output_bytes.getbuffer().nbytes

        return {"file_obj": output_bytes,
                "filename": output_bytes.name,
                "mime_type": "image/png",
                "file_size": file_size,
                "should_close": False}

    async def _post_image_for_pnginfo_data(self, image_data:str):
        # Build payload
        pnginfo_payload = self.post_pnginfo.get_payload()
        info_key = self.post_pnginfo.pnginfo_image_key
        if info_key:
            pnginfo_payload[info_key] = image_data
        # post for image gen data
        return await self.post_pnginfo.call(input_data=pnginfo_payload, main=True)

    async def extract_pnginfo(self, data: str|bytes|list, images_list:list) -> PngImagePlugin.PngInfo | None:
        if not self.post_pnginfo:
            return None
        if isinstance(data, str):
            pnginfo_data = await self._post_image_for_pnginfo_data(data)
        elif isinstance(data, bytes):
            b64str = base64.b64encode(data).decode()
            pnginfo_data = await self._post_image_for_pnginfo_data(b64str)
        else:
            log.warning(f"Unsupported data type for pnginfo: {type(data)}")
            pnginfo_data = None
        if pnginfo_data:
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("parameters", pnginfo_data)
            return pnginfo
        return None

    async def resolve_image_data(self, item, index: int) -> str|bytes|list:
        if isinstance(item, str):
            return split_at_first_comma(item)
        return item

    async def unpack_image_results(self, images:Any) -> str|bytes|list:
        return images

    async def call_track_progress(self, task):
        await self.track_progress(endpoint=self.get_progress,
                                  progress_key=self.get_progress.progress_key,
                                  eta_key=self.get_progress.eta_key,
                                  max_key=self.get_progress.max_key,
                                  message="Generating image",
                                  task=task)

    async def _track_t2i_i2i_progress(self, task):
        try:
            if not self.get_progress:
                await task.embeds.send('img_gen', f'Generating an image with {self.name} ...', '')
                if self.ws:
                    if not bot_database.was_warned("imggen_websocket_progress"):
                        log.warning(f"[{self.name}] If websocket supports tracking progress, and you want to use it for 'main txt2img/img2img', "
                                    "you'll have to omit/null the 'images_result_key' and instead use the 'response_handling' (advanced). "
                                    "Refer to the wiki for more info (https://github.com/altoiddealer/ad_discordbot/wiki).")
                        bot_database.update_was_warned("imggen_websocket_progress")
            else:
                await self.call_track_progress(task)
        except Exception as e:
            log.error(f'Error tracking {self.name} image generation progress: {e}')

    async def call_imggen_endpoint(self, img_payload:dict, mode:str="txt2img"):
        ep_for_mode:Union[ImgGenEndpoint_PostTxt2Img, ImgGenEndpoint_PostImg2Img] = getattr(self, f'post_{mode}')
        return await ep_for_mode.call(input_data=img_payload, main=True)

    async def post_for_images(self, task, img_payload:dict, mode:str="txt2img") -> list[str]:
        # Start progress task and generation task concurrently
        images_task = asyncio.create_task(self.call_imggen_endpoint(img_payload, mode))
        progress_task = asyncio.create_task(self._track_t2i_i2i_progress(task))
        # Wait for images_task to complete
        images_results = await images_task
        # Once images_task is done, cancel progress_task
        if progress_task and not progress_task.done():
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
        return images_results

    async def _main_imggen(self, task) -> Tuple[list[FILE_INPUT], Optional[PngImagePlugin.PngInfo]]:
        img_file_list = []
        pnginfo = None
        try:
            img_payload:dict = task.payload
            mode:str = task.params.mode
            # Get the results
            images_results = await self.post_for_images(task, img_payload, mode)
            if self.cancel_event.is_set():
                raise APIRequestCancelled(f"[{self.name}] Generation was cancelled by user.", cancel_event=self.cancel_event)
            # Ensure the results are a list of base64 or bytes
            images_list = await self.unpack_image_results(images_results)
            # Optionally add png info to first result > save > returns PIL images and pnginfo
            for i, item in enumerate(images_list):
                data = await self.resolve_image_data(item, i)
                if i == 0:
                    pnginfo = await self.extract_pnginfo(data, images_list)
                img_file:FILE_INPUT = self.decode_and_save_for_index(i, data, pnginfo)
                img_file_list.append(img_file)
            # Delete embed on success
            await task.embeds.delete('img_gen')
        except Exception as e:
            e_prefix = f'[{self.name}] Error processing images'
            log.error(f'{e_prefix}: {e}')
            restart_msg = f'\nIf {self.name} remains unresponsive, consider trying "/restart_sd_client" command.' if self.post_server_restart else ''
            await task.embeds.edit_or_send('img_send', e_prefix, f'{e}{restart_msg}')
        return img_file_list, pnginfo

    def is_comfy(self) -> bool:
        return isinstance(self, ImgGenClient_Comfy)

    def is_swarm(self) -> bool:
        return isinstance(self, ImgGenClient_Swarm)

    def is_sdwebui_variant(self) -> bool:
        return isinstance(self, ImgGenClient_SDWebUI)

    def is_sdwebui(self) -> bool:
        return any(substring in self.name.lower() for substring in ['stable', 'a1111', 'sdwebui'])
    
    def is_reforge(self) -> bool:
        return 'reforge' in self.name.lower()
    
    def is_forge(self) -> bool:
        return ('forge' in self.name.lower() and not self.is_reforge())
    
    def supports_loractrl(self) -> bool:
        return (self.is_sdwebui() or self.is_reforge()) and not self.is_forge()

class ImgGenClient_SDWebUI(ImgGenClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def call_track_progress(self, task):
        await self.track_progress(endpoint=self.get_progress,
                                  progress_key='progress',
                                  eta_key='eta_relative',
                                  max_key=None,
                                  message="Generating image",
                                  task=task)

class ImgGenClient_Swarm(ImgGenClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = None

    async def call_track_progress(self, task) -> list[dict]:
        completion_condition = processing.build_completion_condition({'image': "*"})
        return await self.track_progress(endpoint=None,
                                         use_ws=True,
                                         task=task,
                                         message="Generating image",
                                         type_filter=None,
                                         data_filter=None,
                                         progress_key='gen_progress.overall_percent',
                                         completion_condition=completion_condition)

    async def connect_websocket(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        ws_url = self.url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/API/GenerateText2ImageWS"
        try:
            self.ws = await self.session.ws_connect(ws_url)
            log.info(
                f"[{self.name}] WebSocket connection established ({ws_url})")
        except Exception as e:
            log.error(f"[{self.name}] failed to connect to WebSocket: {e}")
            self.ws = None
            raise
    
    async def extract_pnginfo(self, data, images_list) -> None:
        return None

    async def resolve_image_data(self, item:dict, index: int) -> bytes:
        image:str = item['image']
        if image.startswith('View/'):
            # Is a path that needs to get bytes from server
            response = await self.request(endpoint=image, method='GET', retry=0, timeout=10)
            return response.body
        else:
            # Is base64 string
            data = split_at_first_comma(image)
            return base64.b64decode(data)

    async def unpack_image_results(self, results:list) -> list[dict]:
        # Return the last dict value returned by poll_ws()
        last_dict:dict = results.pop()
        return [last_dict]

    async def post_for_images(self, task, img_payload:dict, mode:str="txt2img") -> list[str]:
        if not self.ws or self.ws.closed:
            await self.connect_websocket()
        try:
            self.add_required_values_to_payload(img_payload)
            await self.ws.send_json(img_payload)
            return await self.call_track_progress(task)
        finally:
            await self.ws.close()

    def add_required_values_to_payload(self, payload:dict):
        payload['session_id'] = self.session_id

    def _get_settings_list_for(self, response, key:str):
        if not isinstance(response, APIResponse):
            return []
        return response.body.get(key, [])

    async def main_setup_tasks(self):
        try:
            response:APIResponse = await self.request(endpoint='/API/GetNewSession', json={}, method='POST', retry=0, timeout=5)
            self.session_id = response.body['session_id']
        except Exception as e:
            log.error(f'[{self.name}] Error getting session_id (required for this client): {e}')
            return await self.go_offline()
        try:
            # Collect valid samplers and schedulers
            params_resp:APIResponse = await self.request(endpoint='/API/ListT2IParams', json={'session_id': self.session_id}, method='POST', retry=0, timeout=5)
            params_list = self._get_settings_list_for(params_resp, 'list')
            for settings_dict in params_list:
                if settings_dict.get('id') == 'sampler':
                    self._sampler_names = settings_dict.get('values', [])
                elif settings_dict.get('id') == 'scheduler':
                    self._schedulers = settings_dict.get('values', [])
            # Collect valid LoRAs
            loras_resp:APIResponse = await self.request(endpoint='/API/ListModels', json={'session_id': self.session_id, 'path': '', 'subtype': 'LoRA', 'depth': 10}, method='POST', retry=0, timeout=5)
            loras_list = self._get_settings_list_for(loras_resp, 'files')
            for lora_dict in loras_list:
                if lora_dict.get('name'):
                    self._lora_names.append(lora_dict['name'])
        except Exception as e:
            log.error(f"Error: {e}")
            pass

    async def fetch_imgmodels(self) -> list:
        all_imgmodels = []
        response:APIResponse = await self.request(endpoint='/API/ListModels', json={'session_id': self.session_id, 'path': '', 'depth': 10}, method='POST', retry=0, timeout=5)
        model_list = self._get_settings_list_for(response, 'files')
        for model_dict in model_list:
            if model_dict.get('name'):
                all_imgmodels.append(model_dict['name'])
        return all_imgmodels

class ImgGenClient_Comfy(ImgGenClient):
    get_history: Optional["ImgGenEndpoint"] = None
    get_view: Optional["ImgGenEndpoint"] = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def get_comfy_settings_values_for(self, class_type:str, labels:str | list[str]) -> list|dict:
        '''Returns dict if list of labels. Returns list for just one label string'''
        def extract_values_for_label(req:dict, label:str):
            values = req.get(label)
            if isinstance(values, list) and isinstance(values[0], list):
                return values[0]
            return []
        response = await self.request(endpoint=f'/object_info/{class_type}', method='GET', retry=0, timeout=5)
        if response and isinstance(response, APIResponse):
            req = response.body.get(class_type, {}).get('input', {}).get('required', {})
            if isinstance(labels, str):
                return extract_values_for_label(req, labels)
            elif isinstance(labels, list):
                values = {}
                for label in labels:
                    values[label] = extract_values_for_label(req, label)
                return values
            return []

    async def fetch_imgmodels(self) -> list:
        all_imgmodels = []
        all_imgmodels.extend(await self.get_comfy_settings_values_for(class_type='CheckpointLoaderSimple', labels='ckpt_name'))
        all_imgmodels.extend(await self.get_comfy_settings_values_for(class_type='UnetLoaderGGUF', labels='unet_name'))
        all_imgmodels.extend(await self.get_comfy_settings_values_for(class_type='UNETLoader', labels='unet_name'))
        return all_imgmodels

    async def main_setup_tasks(self):
        try:
            data = await self.get_comfy_settings_values_for(class_type='KSampler', labels=['sampler_name', 'scheduler'])
            self._sampler_names = data.get('sampler_name', [])
            self._schedulers = data.get('scheduler', [])
            self._lora_names = await self.get_comfy_settings_values_for(class_type='LoraLoader', labels='lora_name')
        except Exception as e:
            log.error(f"Error: {e}")
            pass

    async def _free_memory(self, unload_models:bool = True, free_memory:bool = True):
        payload = {'unload_models': unload_models, 'free_memory': free_memory}
        await self.request(endpoint='/free', method='POST', json=payload)

    async def extract_pnginfo(self, data, images_list) -> None:
        return None

    async def _resolve_output_data(self, item:dict) -> bytes:
        if self.get_view:
            return await self.get_view.call(input_data=item)
        else:
            response:APIResponse = await self.request(endpoint=f'/view', params=item, method='GET', response_type='bytes')
            return response.body
    
    async def resolve_image_data(self, item:dict, index:int) -> bytes:
        return await self._resolve_output_data(item)
    
    async def _fetch_prompt_results(self, prompt_id:str, returns:list[str]=['images'], node_ids:list[int]=[]) -> list[dict]:
        if self.get_history:
            history = await self.get_history.call(path_vars=prompt_id)
        else:
            response:APIResponse = await self.request(endpoint=f'/history/{prompt_id}', method='GET')
            history = response.body
        outputs:dict = history.get(prompt_id, {}).get('outputs', {})
        results = []
        for node_id_str, node_output in outputs.items():
            # filter nodes if any provided in list
            node_id = int(node_id_str)
            if node_ids and node_id not in node_ids:
                continue
            # collect outputs
            for output_type in returns:
                if output_type in node_output:
                    results.extend(node_output[output_type])
        if not results:
            log.warning(f"[{self.name}] No outputs were found for criteria (types: {returns}; node_ids: {node_ids if node_ids else 'ANY'})")
        return results

    async def unpack_image_results(self, results: list[dict]) -> list[dict]:
        prompt_id = results[0]["prompt_id"]
        return await self._fetch_prompt_results(prompt_id)
    
    def _build_completion_condition(self, prompt_id:str, completed_node_id:int|None=None) -> Callable[[dict], bool]:
        if completed_node_id is None:
            completion_config = {'type': 'execution_success',
                                 'data': {'prompt_id': prompt_id}}
        else:
            completion_config = {'type': 'executed',
                                 'data': {'prompt_id': prompt_id,
                                          'node': str(completed_node_id)}}
        return processing.build_completion_condition(completion_config, None)

    def _nest_payload_in_prompt(self, payload:dict):
        prompt_value = payload.get('prompt')
        if prompt_value is None:
            prompt_value = copy.deepcopy(payload)
        payload.clear()
        payload['prompt'] = prompt_value

    async def _post_prompt(self,
                           img_payload:dict,
                           mode:str = "txt2img",
                           task = None,
                           ictx:CtxInteraction|None = None,
                           message:str = "Generating image",
                           endpoint:Union["Endpoint", None] = None,
                           completed_node_id:int|None = None) -> str:
        # Ensure meta keys removed
        img_payload = remove_meta_keys(img_payload)
        # Resolve malformatted payload
        self._nest_payload_in_prompt(img_payload)
        # Add Client ID to payload
        img_payload['client_id'] = self.ws_config.client_id
        # Resolve calling method
        if mode or endpoint:
            if mode:
                endpoint: ImgGenEndpoint = getattr(self, f'post_{mode}')
            queued = await endpoint.call(input_data=img_payload, task=task)
        else:
            response:APIResponse = await self.request(endpoint=f'/prompt', json=img_payload, method='POST')
            queued = response.body
        prompt_id = queued['prompt_id']
        # Create a callable completion condition for progress tracking
        completion_condition = self._build_completion_condition(prompt_id, completed_node_id)
        # Track progress
        await self.track_progress(endpoint=None,
                                  use_ws=True,
                                  task=task,
                                  ictx=ictx,
                                  message=message,
                                  return_values={'progress': 'data.value', 'max': 'data.max'},
                                  type_filter=["progress", "executed"],
                                  data_filter={'prompt_id': prompt_id},
                                  completion_condition=completion_condition)
        # Return the prompt ID to fetch results
        return prompt_id

    async def post_for_images(self, task, img_payload:dict, mode:str="txt2img") -> list[str]:
        prompt_id = await self._post_prompt(img_payload, mode, task)
        return [{"prompt_id": prompt_id}]

    async def _execute_prompt(self,
                              payload:dict,
                              endpoint:Union["Endpoint", None] = None,
                              ictx:CtxInteraction|None = None,
                              task = None,
                              message:str = "Generating",
                              outputs:list[str] = ['images'],
                              output_node_ids:list[int] = [],
                              completed_node_id:int|None = None,
                              file_path:str = '',
                              returns:str = 'file_path'):
        # Queue prompt > track progress
        prompt_id = await self._post_prompt(payload, mode=None, task=task, ictx=ictx, message=message, endpoint=endpoint, completed_node_id=completed_node_id)
        # Fetch results (list of bytes)
        results_list = await self._fetch_prompt_results(prompt_id, returns=outputs, node_ids=output_node_ids)
        # Collect results
        save_file_results = []
        for item in results_list:
            bytes = await self._resolve_output_data(item)
            save_dict = await processing.save_any_file(bytes, file_path=file_path, msg_prefix='[StepExecutor] ')            
            save_file_results.append(save_dict[returns] if returns else save_dict)
        return save_file_results


class TextGenClient(APIClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        textgen_config:dict = apisettings.get_config_for("textgen")
        # TODO Main TextGen API support

        # Collect endpoints used for main TextGen functions
        self._bind_main_endpoints(textgen_config)

    def get_endpoint_class_map(self) -> dict[str, type]:
        return {}

    def default_endpoint_class(self):
        return TextGenEndpoint


class TTSGenClient(APIClient):
    get_voices: Optional["TTSGenEndpoint_GetVoices"] = None
    get_languages: Optional["TTSGenEndpoint_GetLanguages"] = None
    post_generate: Optional["TTSGenEndpoint_PostGenerate"] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ttsgen_config:dict = apisettings.get_config_for("ttsgen")
        self._bind_main_endpoints(ttsgen_config)

    def get_endpoint_class_map(self) -> dict[str, type]:
        return {"get_voices": TTSGenEndpoint_GetVoices,
                "get_languages": TTSGenEndpoint_GetLanguages,
                "post_generate": TTSGenEndpoint_PostGenerate}

    def default_endpoint_class(self):
        return TTSGenEndpoint

    async def fetch_speak_options(self):
        lang_list, all_voices = [], []
        try:
            if self.get_languages:
                lang_list = await self.get_languages.call(retry=0, main=True)
            if self.get_voices:
                all_voices = await self.get_voices.call(retry=0, main=True)
            return lang_list, all_voices
        except Exception as e:
            log.error(f'Error fetching options for "/speak" command via API: {e}')
            return None, None

    def is_alltalk(self) -> bool:
        return isinstance(self, TTSGenClient_AllTalk)

class TTSGenClient_AllTalk(TTSGenClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Endpoint:
    def __init__(self,
                 name: str,
                 path: str,
                 method: Optional[str] = "GET",
                 response_type: str = "json",
                 payload_type: str = "any",
                 payload_config: Optional[str|dict] = None,
                 response_handling: Optional[list] = None,
                 headers: Optional[dict[str, str]] = None,
                 stream: bool = False,
                 timeout: int = 10,
                 retry: int = 0,
                 concurrency_limit: Optional[int] = None):

        self.client: Optional["APIClient"] = None
        self.name = name
        self.path = path
        self.method = method.upper() if method is not None else None
        self.response_type = response_type
        self.response_handling = response_handling
        self.payload_type = payload_type
        self.payload = {}
        self.schema: Optional[dict] = None
        self.headers = headers or {}
        self.stream = stream
        self.timeout = timeout
        self.retry = retry
        self._semaphore = asyncio.Semaphore(concurrency_limit) if concurrency_limit else None
        if payload_config:
            self.init_payload(payload_config)

    def pop_payload_comment(self):
        if isinstance(self.payload, dict):
            self.payload.pop('_comment', None)

    def init_payload(self, payload_config):
        # dictionary value
        if isinstance(payload_config, dict):
            self.payload = payload_config
            self.pop_payload_comment()
        # string value
        elif isinstance(payload_config, str):
            payload_fp = os.path.join(shared_path.dir_user_payloads, payload_config)
            # string is file path
            if os.path.exists(payload_fp):
                self.payload = load_file(payload_fp)
                self.pop_payload_comment()
            # any other string should be a 'get' endpoint. Need to get payload after client init.
            else:
                setattr(self, '_deferred_payload_source', payload_config)

    def get_payload(self):
        return copy.deepcopy(self.payload)

    def get_extract_keys(self):
        return None

    def get_preferred_content_type(self) -> Optional[str]:
        content_type = None
        if isinstance(self.headers, dict):
            content_type = self.headers.get("Content-Type")
        elif isinstance(self.headers, str):
            content_type = self.headers

        if content_type:
            return content_type.split(";")[0].strip().lower()  # Strip params like charset=utf-8

        return None

    def set_openapi_schema(self, openapi_schema: dict, force: bool = False):
        if not openapi_schema or (self.schema and not force):
            return

        preferred_content_type = self.get_preferred_content_type()
        if self.method is not None:
            self.schema = self.get_schema(preferred_content_type)


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
            import re
            # Convert OpenAPI-style path to regex
            path_regex = re.sub(r"\{[^/]+\}", "[^/]+", defined_path)
            if re.fullmatch(path_regex, self.path):
                return defined_path

        return None

    def get_schema(self, preferred_content_type: Optional[str] = None) -> Optional[dict]:
        openapi_schema = self.client.openapi_schema
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

    def sanitize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively sanitizes the payload using the OpenAPI schema by removing unknown keys.
        """
        ep_schema = self.schema or self.get_schema()
        if not ep_schema or not ep_schema.get("properties"):
            log.debug(f"No schema or schema properties found for {self.method} {self.path} — skipping sanitization")
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

        schema_props = ep_schema.get("properties", {})
        required_fields = ep_schema.get("required", [])
        final_cleaned = _sanitize(payload, schema_props, required_fields)

        if not final_cleaned:
            raise ValueError(f"All keys in payload were removed during sanitization for endpoint {self.name}")

        return final_cleaned

    async def handle_main_response(self, results: Any, response: Union["APIResponse", None]):
        if response:
            # Automatically handle responses from known APIs
            expected_response_data = await self.get_expected_response_data(response)
            if expected_response_data:
                return expected_response_data

        # Optional key extraction (bypasses StepExecutor)
        extract_keys:Optional[str|list[str]] = self.get_extract_keys()
        if extract_keys is not None:
            if isinstance(extract_keys, str) or \
                (isinstance(extract_keys, list) and all(key is not None for key in extract_keys)):
                return self.extract_main_keys(results, extract_keys)

    def clean_payload(self, payload):
        if isinstance(payload, dict):
            self.client.add_required_values_to_payload(payload)
        return payload

    def prepare_aiohttp_formdata(self, data_payload: dict = None, files_payload: dict = None) -> aiohttp.FormData:
        form = aiohttp.FormData()

        # Add standard form fields
        if data_payload:
            for key, value in data_payload.items():
                form.add_field(name=key, value=str(value))

        # Add file fields
        if files_payload:
            for field_name, file_info in files_payload.items():
                if isinstance(file_info, dict):
                    # Single file for this field
                    fileobj = file_info["file"]
                    filename = file_info.get("filename", "file")
                    content_type = file_info.get("content_type", "application/octet-stream")
                    form.add_field(name=field_name, value=fileobj, filename=filename, content_type=content_type)

                elif isinstance(file_info, list):
                    # Multiple files for this field
                    for i, file_entry in enumerate(file_info):
                        fileobj = file_entry["file"]
                        filename = file_entry.get("filename", f"file_{i}")
                        content_type = file_entry.get("content_type", "application/octet-stream")
                        form.add_field(name=field_name, value=fileobj, filename=filename, content_type=content_type)

                else:
                    raise ValueError(f"Unsupported file payload type for field '{field_name}': {type(file_info)}")

        return form

    def resolve_input_data(self, input_data, payload_type, payload_map):
        json_payload = None
        data_payload = None
        params_payload = None
        files_payload = None
        input_data = input_data or {}
        explicit_type = payload_type in ["json", "form", "multipart", "query"]
        preferred_content = self.get_preferred_content_type()
        if (payload_type == "multipart" or preferred_content.startswith("multipart/form-data")) \
            and input_data and not payload_map:
            if input_data.get('files'):
                payload_map = input_data
            else:
                raise ValueError(f"[{self.name}] has 'payload_type: multipart' but payload does not include 'files' key")
        # Use explicit payload map if given
        if payload_map:
            # Fully structured override
            json_payload = payload_map.get("json")
            data_payload = payload_map.get("data")
            params_payload = payload_map.get("params")
            files_payload = payload_map.get("files")
            if files_payload:
                data_payload = self.prepare_aiohttp_formdata(data_payload, files_payload)
                json_payload = None
                files_payload = None
        else:
            if payload_type == "multipart" or preferred_content.startswith("multipart/form-data"):
                raise ValueError(f"[{self.name}] 'payload_map=' is required for multipart/form-data payloads (cannot be accomplished with 'input_data=')'")
            if explicit_type:
                if payload_type == "json":
                    json_payload = input_data
                elif payload_type == "form":
                    data_payload = input_data
                elif payload_type == "query":
                    params_payload = input_data
            else:
                if preferred_content == "application/json":
                    json_payload = input_data
                elif preferred_content == "application/x-www-form-urlencoded":
                    data_payload = input_data
                elif preferred_content == "application/octet-stream":
                    data_payload = input_data
                elif preferred_content == "text/plain":
                    data_payload = str(input_data)
                else:
                    log.warning(f"Cannot infer payload type from Content-Type: {preferred_content}. Defaulting to json.")
                    json_payload = input_data
        return json_payload, data_payload, params_payload, files_payload

    async def call(self,
                   input_data: dict|None = None,
                   payload_map: dict|None = None,
                   sanitize: bool = False,
                   main: bool = False,
                   task = None,
                   ictx: CtxInteraction|None = None,
                   **kwargs
                   ) -> Any:
        if self.client is None:
            raise ValueError(f"[{self.name}] Endpoint not bound to an APIClient")
        if not self.client.enabled:
            raise RuntimeError(f"[{self.name}] API Client '{self.client.name}' is currently disabled. Use '/toggle_api' to enable the client when available.")

        input_data = self.clean_payload(input_data)

        headers = kwargs.pop('headers', self.headers)
        timeout = kwargs.pop('timeout', self.timeout)
        retry = kwargs.pop('retry', self.retry)
        payload_type = kwargs.pop('payload_type', self.payload_type)
        response_type = kwargs.pop('response_type', self.response_type)

        json_payload, data_payload, params_payload, files_payload = self.resolve_input_data(input_data, payload_type, payload_map)

        response:Optional[APIResponse] = None
        results = {}

        if self.method is None:
            log.info(f"[{self.name}] has 'null' method. The input data will be returned as response data.")
            results = json_payload or data_payload
        else:            
            if sanitize:
                if json_payload and isinstance(json_payload, dict):
                    json_payload = self.sanitize_payload(json_payload)
                if data_payload and isinstance(data_payload, dict):
                    data_payload = self.sanitize_payload(data_payload)

            request_kwargs = {"endpoint": self.path,
                              "method": self.method,
                              "json": json_payload,
                              "data": data_payload,
                              "params": params_payload,
                              "files": files_payload,
                              "headers": headers,
                              "timeout": timeout,
                              "retry": retry,
                              "response_type": response_type,
                              **kwargs}

            if self._semaphore:
                async with self._semaphore:  # Waits if limit is reached
                    response = await self.client.request(**request_kwargs)
            else:
                response = await self.client.request(**request_kwargs)

            if not isinstance(response, APIResponse):
                return response

            results = response.body

        if main:
            # Try handling response from "main functions"
            try:
                expected_main_result = await self.handle_main_response(results, response)
                if expected_main_result:
                    return expected_main_result

            except Exception as e:
                if not bot_database.was_warned(f'{self.name}_main_error'):
                    log.error(f'[{self.name}] Error while handling an expected API response tpye/value: {e}'
                              f'[{self.name}] Falling back to "response_handling". (Only logging this error once)')
                    bot_database.update_was_warned(f'{self.name}_main_error')
    
        # ws_response = await self.process_ws_request(json_payload, data_payload, input_data, **kwargs)
        # Hand off full response to StepExecutor
        if isinstance(self.response_handling, list):
            log.info(f'[{self.name}] Executing "response_handling" ({len(self.response_handling)} processing steps)')
            from modules.stepexecutor import StepExecutor
            handler = StepExecutor(steps=self.response_handling, input_data=response or results, task=task, ictx=ictx, endpoint=self)
            results = await handler.run()

        return results

    async def upload_files(self, 
                           input_data: Any = None,
                           file_name:str = 'file.bin',
                           file_obj_key:str = 'file',
                           normalized_inputs: list[FILE_INPUT] | None = None,
                           **kwargs) -> list:

        normalized_files:list[FILE_INPUT] = normalized_inputs or processing.normalize_file_inputs(input_data, file_name)
        results_list = []

        for file in normalized_files:
            file_obj = file['file_obj']
            filename = file['filename']
            mime_type = file['mime_type']
            should_close = file['should_close']

            file_dict = {"file": file_obj,
                         "filename": filename,
                         "content_type": mime_type}

            mime_category = mime_type.split("/")[0]
            field_name = getattr(self, '_file_key', None) or file_obj_key or mime_category

            payload = self.get_payload()
            payload['files'] = {field_name: file_dict}

            path_vars = mime_category if '{}' in self.path else None
            result = await self.call(payload_map=payload, path_vars=path_vars, **kwargs)

            if isinstance(result, APIResponse):
                results_list.append(result.body)
            else:
                results_list.append(result)

            if should_close:
                file_obj.close()

        return results_list


    async def poll(self,
                   return_values: dict,
                   interval: float = 1.0,
                   duration: int = -1,
                   num_yields: int = -1,
                   completion_condition: Optional[Callable[[dict], bool]] = None,
                   **kwargs) -> AsyncGenerator[dict, None]:
        """
        Poll an API repeatedly, extracting specified values from each response.

        :param return_values: A dict where keys are output keys and values are paths (str or dict) for extract_key().
        :param interval: Time in seconds between polls.
        :param duration: Max duration to poll (in seconds). -1 means no limit.
        :param num_yields: Max number of yields to return. -1 means no limit.
        :yield: Dict with extracted values based on return_values.
        """
        yield_count = 0
        start_time = time.monotonic()

        while True:
            if self.client.cancel_event.is_set():
                raise APIRequestCancelled(f"[{self.client.name}] Generation was cancelled by user.", cancel_event=self.client.cancel_event)
            # Check stop conditions
            if duration > 0:
                elapsed = time.monotonic() - start_time
                if elapsed >= duration:
                    log.info(f"[{self.name}] Polling stopped after duration limit ({duration}s).")
                    break

            try:
                response_data = await self.call(**kwargs)
            except Exception as e:
                log.warning(f"[{self.name}] Progress fetcher failed: {e}")
                raise

            if not response_data:
                try:
                    await asyncio.wait_for(self.client.cancel_event.wait(), timeout=interval)
                    # Generation was cancelled
                    raise APIRequestCancelled(f"[{self.client.name}] Generation was cancelled by user.", cancel_event=self.client.cancel_event)
                except asyncio.TimeoutError:
                    continue # normal timeout happened

            # Check for completion condition
            if completion_condition and completion_condition(response_data):
                log.info(f"[{self.name}] Completion condition matched.")
                yield response_data
                break
            
            # Process response
            if not return_values:
                result = response_data
            else:
                result = {}
                for key, config in return_values.items():
                    try: 
                        result[key] = extract_key(response_data, config)
                    except ValueError as e:
                        if not bot_database.was_warned(f'poll_api_fail_{key}'):
                            log.warning(f"[{self.name}] Failed to extract key '{key}' (only warning once for this)")
                            bot_database.update_was_warned(f'poll_api_fail_{key}')

            yield result

            yield_count += 1
            if num_yields > 0 and yield_count >= num_yields:
                log.info(f"[{self.name}] Polling stopped after num_yields limit ({num_yields}).")
                break
            await asyncio.sleep(interval)

    async def process_ws_request(self, json_payload, data_payload, **kwargs):
        # Compose WebSocket message
        message = {}

        timeout = kwargs.pop('timeout', self.timeout)
        response_type = kwargs.pop('response_type', self.response_type)
        expect_response = kwargs.pop('expect_response', False)

        if isinstance(self.payload, dict):
            message.update(self.payload)

        # Merge input_data or processed payloads
        if json_payload:
            message.update(json_payload)
        if data_payload:
            message.update(data_payload)

        # Optionally include client ID
        if self.client.ws_config.client_id:
            message["client_id"] = self.client.ws_config.client_id

        return await self.client._send_ws_message(
            json=message,
            timeout=timeout,
            expect_response=expect_response,
            response_type=response_type,
            **kwargs
        )

    async def get_expected_response_data(self, response:"APIResponse"):
        return None

    # Extracts the key values from the API response, for the Endpoint's key names defined in user API settings
    def extract_main_keys(self, response, ep_keys:str | list[str]):
        if not isinstance(response, dict):
            log.warning(f'[{self.client.name}] tried to extract value(s) for "{ep_keys}" from the response, but response was non-dict format.')
            return response
        # Try to extract and return one key value        
        if isinstance(ep_keys, str):
            return extract_key(response, ep_keys)
        # Try to extract and return multiple key values as a tuple
        elif isinstance(ep_keys, list):
            results = []
            for key_path in ep_keys:
                value = extract_key(response, key_path)
                results.append(value)
            if results:
                return tuple(results)
        # Key not matched, return original dict
        return response


class TextGenEndpoint(Endpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Defaults


# TTSGen Endpoint Subclasses
class TTSGenEndpoint(Endpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class TTSGenEndpoint_GetVoices(TTSGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_voices_key: Optional[str] = None

    def get_extract_keys(self) -> str|None:
        return self.get_voices_key

class TTSGenEndpoint_GetLanguages(TTSGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_languages_key: Optional[str] = None

    def get_extract_keys(self) -> str|None:
        return self.get_languages_key

class TTSGenEndpoint_PostGenerate(TTSGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_file_path_key: Optional[str] = None
        self.text_input_key: Optional[str] = None
        self.language_input_key: Optional[str] = None
        self.voice_input_key: Optional[str] = None

    def get_extract_keys(self) -> str|None:
        return self.output_file_path_key

    async def get_expected_response_data(self, response:"APIResponse"):
        if isinstance(response.body, dict):
            output_file_path = response.body.get(self.output_file_path_key)
            if isinstance(output_file_path, str):
                if os.path.exists(output_file_path) and output_file_path.lower().endswith(("mp3", "wav")):
                    return output_file_path
                if isinstance(self.client, TTSGenClient_AllTalk) \
                    and response.body.get('output_file_url') \
                    and self.client.endpoints.get('Get Audio'):
                    output_url = response.body['output_file_url']
                    get_audio_ep = self.client.endpoints['Get Audio']
                    response = await get_audio_ep.call(response_type="bytes", path_vars=output_url)

        result = response if not isinstance(response, APIResponse) else response.body
        if isinstance(result, bytes):
            file_format = detect_audio_format(result)
            if file_format in ["mp3", "wav"]:
                file_data = await processing.save_any_file(result, file_format=file_format, file_path='tts')
                return file_data['file_path']

        return None

# ImgGen Endpoint Subclasses
class ImgGenEndpoint(Endpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ImgGenEndpoint_PostTxt2Img(ImgGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_key: Optional[str] = None
        self.neg_prompt_key: Optional[str] = None
        self.seed_key: Optional[str] = None
        self.images_result_key: Optional[str] = None

    def get_extract_keys(self) -> str|None:
        return self.images_result_key

    def get_prompt_keys(self):
        if isinstance(self.client, ImgGenClient_Comfy):
            return None, None
        return self.prompt_key, self.neg_prompt_key

class ImgGenEndpoint_PostImg2Img(ImgGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_key: Optional[str] = None
        self.neg_prompt_key: Optional[str] = None
        self.seed_key: Optional[str] = None
        self.images_result_key: Optional[str] = None

    def get_extract_keys(self) -> str|None:
        return self.images_result_key

    def get_prompt_keys(self):
        if isinstance(self.client, ImgGenClient_Comfy):
            return None, None
        return self.prompt_key, self.neg_prompt_key

class ImgGenEndpoint_PostCancel(ImgGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ImgGenEndpoint_GetProgress(ImgGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_key: Optional[str] = None
        self.eta_key: Optional[str] = None
        self.max_key: Optional[str] = None

class ImgGenEndpoint_PostPNGInfo(ImgGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pnginfo_image_key: Optional[str] = None
        self.pnginfo_result_key: Optional[str] = None

    def get_extract_keys(self) -> str|None:
        return self.pnginfo_result_key

class ImgGenEndpoint_PostOptions(ImgGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imgmodel_input_key: Optional[str] = None

class ImgGenEndpoint_GetImgModels(ImgGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imgmodel_name_key: Optional[str] = None
        self.imgmodel_value_key: Optional[str] = None
        self.imgmodel_filename_key: Optional[str] = None

class ImgGenEndpoint_GetControlNetModels(ImgGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ImgGenEndpoint_PostServerRestart(ImgGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ImgGenEndpoint_GetControlNetControlTypes(ImgGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_types_key: Optional[str] = None

    def get_extract_keys(self) -> str|None:
        return self.control_types_key

class ImgGenEndpoint_GetHistory(ImgGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ImgGenEndpoint_GetView(ImgGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ImgGenEndpoint_PostUpload(ImgGenEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class APIResponse:
    def __init__(
        self,
        body: Union[str, bytes, dict, list],
        headers: dict[str, str],
        status: int,
        content_type: str = None,
        raw: bytes = None,
    ):
        self.body = body
        self.headers = headers
        self.status = status
        self.content_type = content_type or headers.get("Content-Type", "")
        self.raw = raw  # Optional raw bytes for full fidelity saving

    def json(self) -> dict:
        if isinstance(self.body, dict):
            return self.body
        raise TypeError("Response body is not a JSON object")

    def text(self) -> str:
        if isinstance(self.body, str):
            return self.body
        elif isinstance(self.body, bytes):
            return self.body.decode()
        raise TypeError("Response body is not text or bytes")

    def bytes(self) -> bytes:
        if isinstance(self.body, bytes):
            return self.body
        elif isinstance(self.body, str):
            return self.body.encode()
        raise TypeError("Response body is not text or bytes")
