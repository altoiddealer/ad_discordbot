import aiohttp
import aiofiles
import json
import asyncio
import os
import jsonschema
import jsonref
import yaml
from datetime import datetime
from pathlib import Path
import re
from PIL import Image, PngImagePlugin
import io
import base64
import copy
from typing import Any, Dict, Tuple, List, Optional, Union, Type
from modules.utils_shared import shared_path, patterns, load_file, get_api
from modules.utils_misc import valueparser, deep_merge, is_base64, guess_format_from_headers, guess_format_from_data, detect_audio_format, image_bytes_to_data_uri
import modules.utils_processing as processing

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

class APISettings():
    def __init__(self):
        self.main_settings:dict = {}
        self.presets:dict = {}
        self.workflows:dict = {}

    def get_config_for(self, section: str, default=None) -> dict:
        return self.main_settings.get(section, default or {})

    def get_workflow(self, workflow_name: str, default=None) -> dict:
        return self.workflows.get(workflow_name, default or {})

    def collect_presets(self, presets_list):
        for preset in presets_list:
            if not isinstance(preset, dict):
                log.warning("Encountered a non-dict response handling preset. Skipping.")
                continue
            name = preset.get('name')
            if name:
                self.presets[name] = preset

    def apply_preset(self, config: dict) -> dict:
        """
        Merge a config dictionary with its referenced preset (if any),
        giving priority to explicitly defined values in config.
        """
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

    def get_preset(self, preset_name: str, default=None) -> dict:
        return self.presets.get(preset_name, default or {})

    def expand_steps_with_presets(self, steps: list) -> list:
        expanded_steps = []
        for step in steps:
            if not isinstance(step, dict):
                log.error("Encountered a non-dict step. Steps should be dictionaries.")
                continue
            expanded = self.apply_preset(step)
            expanded_steps.append(expanded)
        return expanded_steps

    def collect_workflows(self, workflows_list):
        for workflow in workflows_list:
            if not isinstance(workflow, dict):
                log.warning("Encountered a non-dict Workflow. Skipping.")
                continue
            name = workflow.get('name')
            if name:
                self.workflows[name] = workflow
                workflow_steps = workflow.get('steps')
                if workflow_steps:
                    expanded_steps = self.expand_steps_with_presets(workflow_steps)
                    self.workflows[name]['steps'] = expanded_steps

    def collect_settings(self, data:dict):
        # Collect Main APIs
        self.main_settings = data.get('bot_api_functions', {})
        # Collect Response Handling Presets
        self.collect_presets(data.get('presets', {}))
        # Collect Workflows
        self.collect_workflows(data.get('workflows', {}))

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

        # Main APIs / Presets / Workflows
        apisettings.collect_settings(data)

        # Reverse lookup for matching API names to their function type
        main_api_name_map = {v.get("api_name"): k for k, v in apisettings.main_settings.items()
                             if isinstance(v, dict) and v.get("api_name")}
        # Map function type to specialized client class
        client_type_map = {"imggen": ImgGenClient,
                           "textgen": TextGenClient,
                           "ttsgen": TTSGenClient}
        
        check_clients_online = []

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

    def resolve_placeholders(self, data: dict[str, str]) -> dict[str, str]:
        context = {
            "token": self.auth_token or "",
            "client_id": self.client_id or "",
            "session_id": self.session_id or "",
            "channel": self.channel or "",
            "version": self.version or "",
        }
        return {key: (value.format(**context) if isinstance(value, str) else value)
                for key, value in data.items()}

    def build_headers(self) -> dict[str, str]:
        return self.resolve_placeholders(self.headers)

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
        resolved_query = self.resolve_placeholders(final_params)

        original_params = parse_qs(parsed.query)
        for k, v in resolved_query.items():
            original_params[k] = [v]

        final_query = urlencode({k: v[0] for k, v in original_params.items()})
        rebuilt = parsed._replace(query=final_query)

        return urlunparse(rebuilt)


class APIClient:
    def __init__(self,
                 name: str,
                 enabled: bool,
                 url: str,
                 websocket_config = None,
                 default_headers: Optional[Dict[str, str]] = None,
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
        self._endpoint_fetch_payloads = []
        # WebSocket connection
        self.ws = None
        self.ws_config: Optional[WebSocketConnectionConfig] = None
        if websocket_config:
            self.ws_config = WebSocketConnectionConfig(websocket_config)
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
            await self._connect_websocket()
        await self._fetch_openapi_schema()
        self._assign_endpoint_schemas()
        await self._resolve_deferred_payloads()
        await self.main_setup_tasks()

    async def main_setup_tasks(self):
        pass

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

    async def _connect_websocket(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        url = self.ws_config.build_url(self.url)
        headers = self.ws_config.build_headers()
        try:
            self.ws = await self.session.ws_connect(url, headers=headers)
            log.info(f"[{self.name}] WebSocket connection established.")
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


    def _create_endpoint(self, EPClass: Type[Union["Endpoint", "TextGenEndpoint", "ImgGenEndpoint", "TTSGenEndpoint"]], ep_dict: dict):
        return EPClass(name=ep_dict["name"],
                        path=ep_dict.get("path", ""),
                        method=ep_dict.get("method", "GET"),
                        response_type=ep_dict.get("response_type", "json"),
                        payload_config=ep_dict.get("payload_base"),
                        response_handling=ep_dict.get("response_handling"),
                        headers=ep_dict.get("headers", self.default_headers),
                        stream=ep_dict.get("stream", False),
                        timeout=ep_dict.get("timeout", self.default_timeout),
                        retry=ep_dict.get("retry", 0))
    
    def _get_self_ep_class(self):
        return Endpoint

    def _collect_endpoints(self, endpoints_config:list[dict]):
        for ep_dict in endpoints_config:
            # Expand any presets
            ep_dict = apisettings.apply_preset(ep_dict)
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
                log.warning(f"[{self.name}] Skipping endpoint due to missing key: {e}")

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
                            yield json.loads(decoded)
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
            elif content_type.startswith("image/") or content_type in (
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
        timeout: Optional[int] = None,
        response_type: Optional[str] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], str, bytes, None]:

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
        json: Optional[Dict[str, Any]],
        timeout: Optional[int],
        expect_response: bool,
        response_type: Optional[str],
    ) -> Union[Dict[str, Any], str, bytes, None]:
        if not json:
            raise ValueError("WebSocket messages must be JSON serializable.")

        timeout = timeout or self.default_timeout

        if not self.ws or self.ws.closed:
            await self._connect_websocket()

        try:
            await self.ws.send_json(json)

            if expect_response:
                msg = await self.ws.receive(timeout=timeout)

                if msg.type == aiohttp.WSMsgType.TEXT:
                    if response_type == "text":
                        return msg.data
                    try:
                        return jsonlib.loads(msg.data)
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

    
    def progress_bar(self, value, length=15):
        try:
            filled_length = int(length * value)
            bar = ':black_square_button:' * filled_length + ':black_large_square:' * (length - filled_length)
            return f'{bar}'
        except Exception:
            return 0

    async def track_progress(self, discord_embeds):
        if self.get_progress:
            try:
                eta_message = 'Not yet available'
                await discord_embeds.send('img_gen', f'Waiting for {self.name} ...', f'{self.progress_bar(0)}\n**ETA**: {eta_message}')
                await asyncio.sleep(1)

                retry_count = 0
                while retry_count < 5:
                    progress_data = await self.get_progress.call()
                    if progress_data and progress_data.get('progress', 0) > 0:
                        break
                    log.warning(f'Waiting for progress response from {self.name}, retrying in 1 second (attempt {retry_count + 1}/5)')
                    await asyncio.sleep(1)
                    retry_count += 1
                else:
                    log.error('Reached maximum retry limit')
                    await discord_embeds.edit_or_send('img_gen', f'Error getting progress response from {self.name}.', 'Image generation will continue, but progress will not be tracked.')
                    return

                last_progress = 0
                stall_count = 0
                progress = progress_data.get('progress', 0)
                percent_complete = progress * 100

                while percent_complete < 100 and retry_count < 5:
                    progress_data = await self.get_progress.call()
                    if progress_data:
                        progress = progress_data.get('progress', 0)
                        percent_complete = progress * 100
                        eta = progress_data.get('eta_relative', 0)

                        if eta == 0:
                            progress = 1.0
                            percent_complete = 100

                        if progress <= 0.01:
                            title = f'Preparing to generate image ...'
                        else:
                            eta_message = f'{round(eta, 2)} seconds'
                            comment = ''
                            if progress == last_progress:
                                stall_count += 1
                                if stall_count > 2:
                                    comment = ' (Stalled)'
                            else:
                                stall_count = 0
                            title = f'Generating image: {percent_complete:.0f}%{comment}'

                        description = f"{self.progress_bar(progress)}\n**ETA**: {eta_message}"
                        await discord_embeds.edit('img_gen', title, description)

                        last_progress = progress
                        await asyncio.sleep(1)
                    else:
                        log.warning(f'Connection closed with {self.name}, retrying in 1 second (attempt {retry_count + 1}/5)')
                        await asyncio.sleep(1)
                        retry_count += 1

                await discord_embeds.delete('img_gen')

            except Exception as e:
                log.error(f'Error tracking {self.name} image generation progress: {e}')

    async def save_images_and_return(self, img_payload:dict, mode:str="txt2img") -> Tuple[List[Image.Image], Optional[PngImagePlugin.PngInfo]]:
        images = []
        pnginfo = None
        try:
            ep_for_mode:ImgGenEndpoint = getattr(self, f'post_{mode}')
            response = await ep_for_mode.call(input_data=img_payload)

            if not isinstance(response, dict):
                return [], response

            for i, img_data in enumerate(response.get('images', [])):
                if "," in img_data:
                    img_data = img_data.split(",", 1)[1]
                raw_data = base64.b64decode(img_data)

                # Attempt to detect image format
                image = Image.open(io.BytesIO(raw_data))
                mime_type = Image.MIME.get(image.format, "image/png")
                # Construct proper data URI
                data_uri = image_bytes_to_data_uri(raw_data, mime_type)

                # Call PNGInfo endpoint if applicable
                if self.post_pnginfo:
                    png_payload = {"image": data_uri}

                    r2 = await self.post_pnginfo.call(input_data=png_payload)
                    if not isinstance(r2, dict):
                        return [], r2
                    png_info_data = r2.get("info")

                    if i == 0 and png_info_data:
                        # Retain seed
                        seed_match = patterns.seed_value.search(str(png_info_data))
                        if seed_match:
                            self.last_img_payload['seed'] = int(seed_match.group(1))
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
                 response_handling: Optional[list] = None,
                 headers: Optional[Dict[str, str]] = None,
                 stream: bool = False,
                 timeout: int = 10,
                 retry: int = 0):
        self.client: Optional["APIClient"] = None
        self.name = name
        self.path = path
        self.method = method.upper()
        self.response_type = response_type
        self.response_handling = response_handling
        self.payload = {}
        self.schema: Optional[dict] = None
        self.headers = headers or {}
        self.stream = stream
        self.timeout = timeout
        self.retry = retry
        if payload_config:
            self.init_payload(payload_config)

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

    def set_openapi_schema(self, openapi_schema: dict, force: bool = False):
        if not openapi_schema or (self.schema and not force):
            return

        preferred_content_type = self.get_preferred_content_type()
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

    def sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively sanitizes the payload using the OpenAPI schema by removing unknown keys.
        """
        ep_schema = self.schema or self.get_schema()
        if not ep_schema:
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

        schema_props = ep_schema.get("properties", {})
        required_fields = ep_schema.get("required", [])
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
                   ) -> Any:
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
                json_payload = self.sanitize_payload(json_payload)
            if data_payload and isinstance(data_payload, dict):
                data_payload = self.sanitize_payload(data_payload)

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

        if not isinstance(response, APIResponse):
            return response

        # Legacy compatibility step (uses only the response body)
        main_ep_response = await self.return_main_data(response.body)
        if main_ep_response:
            return main_ep_response
        
        results = response.body

        # Optional key extraction for now (could migrate this into StepExecutor)
        if extract_keys:
            results = self.extract_main_keys(response.body, extract_keys)

        # ws_response = await self.process_ws_request(json_payload, data_payload, input_data, **kwargs)

        # Hand off full response to StepExecutor
        # if isinstance(self.response_handling, list):
        #     handler = StepExecutor(steps=self.response_handling, input=response)
        #     results = await handler.run()

        return results
    
    async def return_main_data(self, response):
        pass


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

    # Extracts the key values from the API response, for the Endpoint's key names defined in user API settings
    def extract_main_keys(self, response, ep_keys: str|List[str] = None):
        # No keys to check or invalid response type for this operation
        if not ep_keys:
            return response
        if not isinstance(response, dict):
            log.warning(f'[{self.client.name}] tried to extract value(s) for "{ep_keys}" from the response, but response was non-dict format.')
            return response
        # Try to extract and return one key value        
        if isinstance(ep_keys, str):
            key_paths = getattr(self, ep_keys, None)
            if key_paths:
                return try_paths(response, key_paths)
        # Try to extract and return multiple key values as a tuple
        elif isinstance(ep_keys, list):
            results = []
            for key_attr in ep_keys:
                key_paths = getattr(self, key_attr, None)
                if key_paths:
                    value = try_paths(response, key_paths)
                    results.append(value)
            if results:
                return tuple(results)
        # Key not matched, return original dict
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


class TextGenEndpoint(Endpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Defaults

    async def return_main_data(self, response):
        pass

class TTSGenEndpoint(Endpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Defaults
        self.get_voices_key:Optional[str] = None
        self.get_languages_key:Optional[str] = None
        self.text_input_key:str = 'text_input'
        self.language_input_key:str = 'language'
        self.speaker_input_key:str = 'speaker'
        self.output_file_path_key:Optional[str] = None

    async def return_main_data(self, response):
        if self == self.client.post_generate:
            if isinstance(response, bytes):
                resp_format = self.response_handling.get('type', 'unknown')
                if resp_format == 'unknown':
                    resp_format = processing.detect_audio_format(response)
                    if resp_format == 'unknown':
                        log.error(f'[{self.name}] Expected response to be mp3 or wav (bytes), but received an unexpected format.')
                        return None
                output_dir = os.path.join(shared_path.output_dir, self.response_handling.get('save_dir', ''))
                save_prefix = os.path.join(shared_path.output_dir, self.response_handling.get('save_prefix', ''))
                save_format = self.response_handling.get('save_format', resp_format)         
                audio_fp:str = processing.save_audio_bytes(response, output_dir, input_format=resp_format, file_prefix=save_prefix, output_format=save_format)
                return audio_fp
        return None

class ImgGenEndpoint(Endpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Defaults
        self.prompt_key:str = 'prompt'
        self.neg_prompt_key:str = 'negative_prompt'
        self.seed_key:str = 'seed'
        self.control_types_key:Optional[str] = None

    async def return_main_data(self, response):
        pass


class APIResponse:
    def __init__(
        self,
        body: Union[str, bytes, dict, list],
        headers: Dict[str, str],
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


class StepExecutor:
    def __init__(self, steps: List[dict], input: Any = None):
        """
        Executes a sequence of data transformation steps with optional context storage.

        Steps are defined as a list of single-key dictionaries, where each key is the step type
        (e.g., "extract_values") and the value is the configuration for that step.

        Each step can optionally include a `save_as` key to store intermediate results in context
        without affecting the main result passed to the next step.
        """
        self.steps = steps
        self.context: Dict[str, Any] = {}
        self.original_input = input
        self.response = input if isinstance(input, APIResponse) else None

        # Store full response in context for access by steps (e.g. headers)
        # if self.response:
        #     self.context["_response"] = self.response
        #     self.context["_headers"] = self.response.headers
        #     self.context["_status"] = self.response.status
        #     self.context["_content_type"] = self.response.content_type

    async def run(self, input_data: Any|None = None) -> Any:
        result = input_data or self._initial_data()

        for step in self.steps:
            if not isinstance(step, dict):
                raise ValueError(f"Invalid step: {step}")

            step = step.copy()
            save_as = step.pop("save_as", None)

            if len(step) != 1:
                raise ValueError(f"Invalid step format: {step}")

            step_name, config = next(iter(step.items()))
            method = getattr(self, f"_step_{step_name}", None)
            if not method:
                raise NotImplementedError(f"Unsupported step: {step_name}")

            # Resolve any placeholders using the current context
            config = self._resolve_context_placeholders(config)

            # Run the step and determine where to store the result
            step_result = await method(result, config) if asyncio.iscoroutinefunction(method) else method(result, config)
            # Apply returns logic
            step_result = self._apply_returns(step_result, result, config)

            if save_as:
                self.context[save_as] = step_result
            else:
                result = step_result  # Only update result if not storing to context

        return result

    def _initial_data(self):
        if isinstance(self.response, APIResponse):
            return self.response.body
        return self.original_input
    
    def _apply_returns(self, result, original_input, config:dict, allowed: List[str], default="data"):
        returns = config.get("returns", default)

        if returns not in allowed:
            log.warning(
                f"Ignoring invalid 'returns' value '{returns}'. "
                f"Allowed: {allowed}. Falling back to default: '{default}'."
            )
            returns = default

        if returns == "data":
            return result
        if returns == "input":
            return original_input
        if returns == "context":
            return self.context
        if returns == "dict":
            if isinstance(result, dict):
                return result
            log.warning(f"'returns': 'dict' requested but result is of type {type(result).__name__}. Falling back to 'data'.")
            return result
        if returns == "path":
            if isinstance(result, str):
                return result
            log.warning(f"'returns': 'path' requested but result is of type {type(result).__name__}. Falling back to 'data'.")
            return result

        log.warning(f"Unhandled 'returns' value: {returns}. Falling back to 'data'.")
        return result

    def step_returns(*allowed: str, default: str = "data"):
        def decorator(func):
            async def async_wrapper(self:"StepExecutor", data, config):
                raw_result = await func(self, data, config)
                return self._apply_returns(raw_result, data, config, allowed, default)

            def sync_wrapper(self:"StepExecutor", data, config):
                raw_result = func(self, data, config)
                return self._apply_returns(raw_result, data, config, allowed, default)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    def _resolve_context_placeholders(self, config: Any) -> Any:
        if not self.context:
            return config
        if isinstance(config, str):
            return config.format(**self.context)
        elif isinstance(config, dict):
            return {k: self._resolve_context_placeholders(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_context_placeholders(i) for i in config]
        return config
    
    def _substitute(self, value):
        if isinstance(value, str) and "{" in value:
            return value.format(**self.context)
        return value

    @step_returns("data", "input", default="data")
    async def _step_for_each(self, data: Any, config: dict) -> list:
        """
        Executes the same steps for each Inits a StepExecutor for each item in a Context list.

        Returns:
            list: A list of results, one per item processed.
        """
        source = config.get("in")
        alias = config.get("as", "item")
        steps = config.get("steps")

        if not steps or not isinstance(steps, list):
            raise ValueError("for_each step requires a 'steps' list.")

        # Determine iterable: from context (by string) or directly as list
        items = self.context.get(source) if isinstance(source, str) else source

        if not isinstance(items, list):
            raise TypeError(f"'for_each' expected a list but got {type(items).__name__}")

        results = []

        for index, item in enumerate(items):
            sub_executor = StepExecutor(steps)
            sub_executor.response = self.response
            sub_executor.context = {
                **self.context,
                alias: item,
                f"{alias}_index": index,
            }
            result = await sub_executor.run(item)
            results.append(result)

        return results

    @step_returns("data", "input", default="data")
    async def _step_call_api(self, data: Any, config: Union[str, dict]) -> Any:
        api:API = get_api()
        client_name = config.get("client_name")
        endpoint_name = config.get("client_name")
        input_data = config.get("input_data")
        api_client:APIClient = api.clients.get(client_name)
        if not api_client:
            raise ValueError(f'Call API step requires API Client, but could not find "{client_name}"')
        if not api_client.enabled:
            raise RuntimeError(f'Call API step failed because "{client_name}" is not enabled.')
        endpoint:Endpoint = api_client.endpoints.get(endpoint_name)
        response = await endpoint.call(input_data=input_data)
        if not isinstance(response, APIResponse):
            return response
        return response.body

    @step_returns("data", "input", default="data")
    def _step_extract_key(self, data: Any, config: Union[str, dict]) -> Any:
        if isinstance(config, dict):
            path = config.get("path")
            default = config.get("default", None)
        else:
            path = config
            default = None

        if not isinstance(path, str):
            raise ValueError("Path must be a string.")

        try:
            parts = re.findall(r'[^.\[\]]+|\[\d+\]', path)
            for part in parts:
                if re.fullmatch(r'\[\d+\]', part):  # list index
                    idx = int(part[1:-1])
                    if isinstance(data, list):
                        data = data[idx]
                    else:
                        raise TypeError(f"Expected list for index access but got {type(data).__name__}")
                else:  # dict key
                    if isinstance(data, dict):
                        data = data[part]
                    else:
                        raise TypeError(f"Expected dict for key '{part}' but got {type(data).__name__}")
            return data
        except (KeyError, IndexError, TypeError) as e:
            if default is not None:
                return default
            raise ValueError(f"Failed to extract path '{path}': {e}")

    @step_returns("data", "input", default="data")
    def _step_extract_values(self, data: Any, config: Dict[str, Union[str, dict]]) -> Dict[str, Any]:
        result = {}
        for key, path_config in config.items():
            result[key] = self._step_extract_key(data, path_config)
        return result

    @step_returns("data", "input", default="data")
    def _step_decode_base64(self, data, config):
        if isinstance(data, str):
            return base64.b64decode(data)
        raise TypeError("Expected base64 string for decode_base64 step")

    @step_returns("data", "input", default="data")
    def _step_type(self, data, to_type):
        type_map = {"int": int, "float": float, "str": str, "bool": bool}
        return type_map[to_type](data)
    
    @step_returns("data", "input", default="data")
    def _step_cast(self, data, config: dict):
        type_map = {
            "int": int,
            "float": float,
            "str": str,
            "bool": lambda x: str(x).lower() in ("true", "1", "yes")
        }

        if not isinstance(data, dict):
            raise TypeError(f"'cast' step requires a dict input, got {type(data).__name__}")

        result = data.copy()
        for key, type_name in config.items():
            if key not in result:
                log.warning(f"'cast' step: key '{key}' not found in data")
                continue
            if type_name not in type_map:
                raise ValueError(f"'cast' step: unsupported type '{type_name}'")
            try:
                result[key] = type_map[type_name](result[key])
            except Exception as e:
                log.warning(f"Failed to cast '{key}' to {type_name}: {e}")

        return result

    @step_returns("data", "input", default="data")
    def _step_evaluate(self, data, value: str) -> Any:
        if not isinstance(value, str):
            raise ValueError("The evaluate step requires a string input.")
        return valueparser.parse_value(value)

    @step_returns("data", "input", default="data")
    def _step_regex(self, data, pattern):
        match = re.search(pattern, data)
        if not match:
            raise ValueError("No regex match found")
        return match.group(1) if match.lastindex else match.group(0)

    @step_returns("data", "input", default="data")
    def _step_format(self, data, fmt):
        return fmt.format(data=data)

    @step_returns("data", "input", default="data")
    def _step_eval(self, data, expression):
        return eval(expression, {"data": data})

    @step_returns("path", "dict", "data", default="path")
    async def _step_save(self, data: Any, config: dict):
        """
        Save input data to a file and return either path, original data, or metadata.

        Config options:
        - file_format: Explicit format (e.g. 'json', 'jpg').
        - file_name: Optional file name (without extension).
        - file_path: Relative directory inside output_dir.
        - returns: 'path' (default), 'data', or 'dict'.
        """

        # 1. Setup file path & naming
        file_name = config.get("file_name", f"data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        file_path = Path(config.get("file_path", ""))
        output_path = shared_path.output_dir / file_path
        output_path.mkdir(parents=True, exist_ok=True)

        # 2. Guess format: config > headers > data
        file_format = config.get("file_format")
        if not file_format:
            if isinstance(self.response, APIResponse):
                file_format = guess_format_from_headers(self.response.headers)
            if not file_format:
                file_format = guess_format_from_data(data)
            if file_format:
                log.info(f'Guessed output file format for "save" step by analyzing headers/data: "{file_format}"')

        full_path = output_path / f"{file_name}.{file_format}"
        binary_formats = {"jpg", "jpeg", "png", "webp", "gif", "mp3", "wav", "mp4", "webm", "bin"}

        # 3. Base64 decoding if applicable
        if isinstance(data, str) and is_base64(data):
            try:
                data = base64.b64decode(data)
                log.info("Detected base64 input; decoded to binary.")
            except Exception as e:
                log.error(f"Failed to decode base64 string: {e}")
                raise

        # 4. Select write mode
        mode = "wb" if file_format in binary_formats else "w"

        # 5. Save logic
        try:
            async with aiofiles.open(full_path, mode) as f:
                if file_format == "json":
                    if isinstance(data, (dict, list)):
                        await f.write(json.dumps(data, indent=2))
                    else:
                        raise TypeError("JSON format requires dict or list.")
                elif file_format == "yaml":
                    if isinstance(data, (dict, list)):
                        await f.write(yaml.dump(data))
                    else:
                        raise TypeError("YAML format requires dict or list.")
                elif file_format == "csv":
                    if isinstance(data, list) and all(isinstance(row, (list, tuple)) for row in data):
                        csv_content = "\n".join([",".join(map(str, row)) for row in data])
                        await f.write(csv_content)
                    else:
                        raise TypeError("CSV format requires list of lists/tuples.")
                elif mode == "w":
                    if not isinstance(data, (str, int, float)):
                        raise TypeError(f"Text format requires str/number, got {type(data).__name__}")
                    await f.write(str(data))
                elif mode == "wb":
                    if isinstance(data, bytes):
                        await f.write(data)
                    elif isinstance(data, str):
                        await f.write(data.encode())
                    else:
                        raise TypeError(f"Binary format requires bytes or str, got {type(data).__name__}")

        except Exception as e:
            log.error(f"Failed to save data as {file_format}: {e}")
            raise

        log.info(f"Saved data to {full_path}")

        return {"path": str(full_path),
                "format": file_format,
                "name": file_name,
                "data": data}


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
