import aiohttp
import asyncio
import os
import jsonschema
import jsonref
from PIL import Image, PngImagePlugin
import io
import base64
from typing import Any, Dict, Tuple, List, Optional, Union
from modules.utils_shared import shared_path, load_file

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

class APISettings():
    def __init__(self):
        self.main_settings:dict = {}
    def get(self, section: str, default=None) -> dict:
        return self.main_settings.get(section, default or {})

apisettings = APISettings()

class API:
    def __init__(self):
        # ALL API clients
        self.clients:dict[str, APIClient] = {}
        # Main API clients
        self.imggen:Optional[ImgGenClient] = None
        self.textgen:Optional[TextGenClient] = None
        self.ttsgen:Optional[TTSGenClient] = None

    async def init(self):
        # Load API Settings yaml
        data = load_file(shared_path.api_settings)

        # Main APIs
        apisettings.main_settings = data.get('bot_api_functions', {})
        # Reverse lookup for matching API names to their function type
        main_api_name_map = {v.get("api_name"): k for k, v in apisettings.main_settings.items()
                             if isinstance(v, dict) and v.get("api_name")}
        # Map function type to specialized client class
        client_type_map = {"imggen": ImgGenClient,
                           "textgen": TextGenClient,
                           "ttsgen": TTSGenClient}

        # Iterate over all APIs data
        apis:dict = data.get('all_apis', {})
        setup_tasks = []
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
                api_client = ClientClass(name=api_config['name'],
                                         url=api_config['url'],
                                         default_headers=api_config.get('default_headers'),
                                         default_timeout=api_config.get('default_timeout', 10),
                                         auth=api_config.get('auth'),
                                         endpoints_config=api_config.get('endpoints', []))
                # Capture all clients
                self.clients[name] = api_client
                # Capture main clients
                if is_main:
                    setattr(self, api_func_type, api_client)
                    log.info(f"Registered main {api_func_type} client: {name}")
                if hasattr(api_client, 'setup'):
                    setup_tasks.append(api_client.setup())
            except KeyError as e:
                log.warning(f"Skipping API Client due to missing key: {e}")
            except TypeError as e:
                log.warning(f"Failed to create API client '{name}': {e}")

        await asyncio.gather(*setup_tasks)


class APIClient:
    def __init__(self,
                 name: str,
                 url: str,
                 default_headers: Optional[Dict[str,str]] = None,
                 default_timeout: int = 120,
                 auth: Optional[dict] = None,
                 endpoints_config=None):

        self.name = name
        self.url = url.rstrip("/")
        self.default_headers = default_headers or {}
        self.default_timeout = default_timeout
        self.auth = auth
        self.endpoints: dict[str, Endpoint] = {}
        self.openapi_schema = None
        self._endpoint_fetch_payloads = [] # to execute after endpoints are collected
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
        await self._fetch_openapi_schema()
        self._assign_endpoint_schemas()
        await self._resolve_deferred_payloads()

    def _collect_endpoints(self, endpoints_config:list[dict]):
        for ep in endpoints_config:
            try:
                endpoint = Endpoint(name=ep["name"],
                                    path=ep["path"],
                                    method=ep.get("method", "GET"),
                                    response_type=ep.get("response_type", "json"),
                                    payload_config=ep.get("payload"),
                                    headers=ep.get("headers", self.default_headers),
                                    timeout=ep.get("timeout", self.default_timeout))
                # get deferred payloads after collecting all endpoints
                if hasattr(endpoint, "_deferred_payload_source"):
                    self._endpoint_fetch_payloads.append(endpoint)

                self.endpoints[endpoint.name] = endpoint
            except KeyError as e:
                log.warning(f"[APIClient:{self.name}] Skipping endpoint due to missing key: {e}")

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
            log.error(f"Failed to load OpenAPI schema from {self.url} for {self.name}: {e}")

    def _assign_endpoint_schemas(self):
        if not self.openapi_schema:
            return

        for endpoint in self.endpoints.values():
            endpoint.set_schema_from_openapi(self.openapi_schema)

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
                data = await ref_ep.call(client=self)
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

    async def online(self) -> bool:
        try:
            response = await self.request(
                endpoint='',
                method='GET',
                retry=0,
                return_text=True,
                timeout=5
            )
            return True if response is not None else False
        except Exception as e:
            log.debug(f"[{self.name}] API offline check failed: {e}")
            return False

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
                if session is not None:
                    # Reuse session if passed
                    return await self._make_request(
                        session=session, method=method, url=url, params=params, data=data,
                        json=json, headers=headers, auth=auth, timeout=timeout,
                        return_text=return_text, return_raw=return_raw, response_type=response_type
                    )
                else:
                    # Create a session for standalone request
                    async with aiohttp.ClientSession() as temp_session:
                        return await self._make_request(
                            session=temp_session, method=method, url=url, params=params, data=data,
                            json=json, headers=headers, auth=auth, timeout=timeout,
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
        method: str,
        url: str,
        params: Optional[Dict[str, Any]],
        data: Optional[Union[Dict[str, Any], str, bytes]],
        json: Optional[Dict[str, Any]],
        headers: Dict[str, str],
        auth: Optional[aiohttp.BasicAuth],
        timeout: int,
        return_text: bool,
        return_raw: bool,
        response_type: Optional[str],
    ):
        async with session.request(
            method.upper(), url, params=params, data=data, json=json,
            headers=headers, auth=auth, timeout=timeout
        ) as response:

            if return_raw:
                return await response.read()
            if return_text:
                return await response.text()

            # Decide response type based on endpoint config
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


class ImgGenClient(APIClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        imggen_config:dict = apisettings.get("imggen")

        self.post_txt2img: Optional[Endpoint] = None
        self.post_img2img: Optional[Endpoint] = None
        self.get_imgmodels: Optional[Endpoint] = None
        self.get_progress: Optional[Endpoint] = None
        self.post_pnginfo: Optional[Endpoint] = None
        self.get_imgmodels: Optional[Endpoint] = None
        self.get_controlnet_models: Optional[Endpoint] = None
        self.get_controlnet_control_types: Optional[Endpoint] = None
        
        # Collect endpoints used for main ImgGen functions
        endpoint_keys = {'post_txt2img_endpoint_name': 'post_txt2img',
                         'post_img2img_endpoint_name': 'post_img2img',
                         'get_progress_endpoint_name': 'get_progress',
                         'post_pnginfo_endpoint_name': 'post_pnginfo',
                         'post_options_endpoint_name': 'post_options',
                         'get_imgmodels_endpoint_name': 'get_imgmodels',
                         'get_controlnet_models_endpoint_name': 'get_controlnet_models',
                         'get_controlnet_control_types_endpoint_name': 'get_controlnet_control_types'}
        for config_key, attr_name in endpoint_keys.items():
            ep_name = imggen_config.get(config_key)
            setattr(self, attr_name, self.endpoints.get(ep_name) if ep_name else None)       

    async def get_imggen_progress(self, session: Optional[aiohttp.ClientSession] = None) -> Optional[Dict[str, Any]]:
        if self.get_progress:
            try:
                return await self.get_progress.call(client=self, session=session)
            except Exception as e:
                log.warning(f"Progress fetch failed from {self.name}: {e}")
        return None

    async def get_image_data(
            self,
            image_payload: Dict[str, Any],
            mode: str = "txt2img",
            session: Optional[aiohttp.ClientSession] = None) -> Tuple[List[Image.Image], Optional[PngImagePlugin.PngInfo]]:

        images = []
        pnginfo = None
        try:
            ep_for_mode:Endpoint = getattr(self, f'post_{mode}')
            response = await ep_for_mode.call(client=self, json=image_payload, session=session)

            if not isinstance(response, dict):
                return [], response

            for i, img_data in enumerate(response.get('images', [])):
                raw_data = base64.b64decode(img_data.split(",", 1)[0])
                image = Image.open(io.BytesIO(raw_data))
                
                # Get PNG info
                if self.post_pnginfo:
                    png_payload = {"image": "data:image/png;base64," + img_data}
                    r2 = await self.post_pnginfo.call(client=self, json=png_payload, session=session)
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


class TextGenClient(APIClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO Main TextGen API support


class TTSGenClient(APIClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ttsgen_config:dict = apisettings.get("ttsgen")

        self.get_voices: Optional[Endpoint] = None
        self.post_generate: Optional[Endpoint] = None

        # Collect endpoints used for main TTSGen functions
        endpoint_keys = {'get_voices_endpoint_name': 'get_voices',
                         'post_generate_endpoint_name': 'post_generate'}
        for config_key, attr_name in endpoint_keys.items():
            ep_name = ttsgen_config.get(config_key)
            setattr(self, attr_name, self.endpoints.get(ep_name) if ep_name else None) 


class Endpoint:
    def __init__(self,
                 name: str,
                 path: str,
                 method: str = "GET",
                 response_type: str = "json",
                 payload_config: Optional[str|dict] = None,
                 headers: Optional[Dict[str, str]] = None,
                 timeout: int = 10):
        self.name = name
        self.path = path
        self.method = method.upper()
        self.response_type = response_type
        self.payload = {}
        self.schema: Optional[dict] = None
        self.headers = headers or {}
        self.timeout = timeout
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

    def get_preferred_content_type(self):
        if isinstance(self.headers, dict):
            return self.headers.get("Content-Type")
        if isinstance(self.headers, str):
            return self.headers.strip().lower()
        return None

    def set_schema_from_openapi(self, openapi_schema: dict):
        if not openapi_schema:
            return

        paths = openapi_schema.get("paths", {})
        endpoint_spec = paths.get(self.path)
        if not endpoint_spec:
            return

        method_spec = endpoint_spec.get(self.method.lower())
        if not method_spec:
            return

        request_body = method_spec.get("requestBody", {})
        content = request_body.get("content", {})
        preferred_content_type = self.get_preferred_content_type()

        if preferred_content_type and preferred_content_type in content:
            schema = content[preferred_content_type].get("schema")
            if schema:
                self.schema = schema
                return

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

    def get_schema(self, openapi_schema: dict) -> Optional[dict]:
        if not openapi_schema:
            return None

        paths = openapi_schema.get("paths", {})
        endpoint_spec = paths.get(self.path)
        if not endpoint_spec:
            return None

        method_spec = endpoint_spec.get(self.method.lower())
        if not method_spec:
            return None

        request_body = method_spec.get("requestBody", {})
        content = request_body.get("content", {})
        app_json = content.get("application/json", {})
        return app_json.get("schema")

    def sanitize_payload(self, payload: Dict[str, Any], openapi_schema: dict, strict: bool=False) -> Dict[str, Any]:
        """
        Recursively sanitizes the payload using the OpenAPI schema by removing unknown keys.
        """
        schema = self.schema or self.get_schema(openapi_schema)
        if not schema:
            log.debug(f"No schema found for {self.method} {self.path} — skipping sanitization")
            return payload

        def _sanitize(data: dict, schema_props: dict) -> dict:
            cleaned = {}
            for k, v in data.items():
                if k not in schema_props:
                    log.debug(f"Sanitize: removed unknown key '{k}'")
                    continue

                prop_schema = schema_props[k]
                if isinstance(v, dict) and "properties" in prop_schema:
                    cleaned[k] = _sanitize(v, prop_schema["properties"])
                else:
                    cleaned[k] = v

            return cleaned

        schema_props = schema.get("properties", {})
        final_cleaned = _sanitize(payload, schema_props)
        if strict and not final_cleaned:
            raise ValueError(f"All keys in payload were removed during sanitization for endpoint {self.name}")

        return final_cleaned

    async def call(self, client:"APIClient", sanitize:bool=False, strict: bool = False, **kwargs):
        """
        Convenience wrapper to call this endpoint directly.
        Assumes `client.request()` exists.

        :param sanitize: If True, attempt to sanitize payload against the schema
        :param strict: If True and sanitize=True, raise an error if payload is fully removed
        """
        json_payload = kwargs.pop('json', None)
        data_payload = kwargs.pop('data', None)

        if sanitize:
            if isinstance(json_payload, dict):
                json_payload = self.sanitize_payload(json_payload, client.openapi_schema, strict=strict)
            if isinstance(data_payload, dict):
                data_payload = self.sanitize_payload(data_payload, client.openapi_schema, strict=strict)

        return await client.request(
            endpoint=self.path,
            method=self.method,
            json=json_payload,
            data=data_payload,
            headers=kwargs.pop('headers', self.headers),
            timeout=kwargs.pop('timeout', self.timeout),
            response_type=kwargs.pop('response_type', self.response_type),
            **kwargs
        )
    
    def __repr__(self):
        return f"<Endpoint {self.method} {self.path}>"


# # Accessing an endpoint
# ep = client.endpoints.get("Post txt2img")
# payload = {
#     "prompt": "a beautiful mountain landscape",
#     "steps": 30,
#     "cfg_scale": 7.5,
# }

# # Validate before sending
# ep.validate_payload(payload, client.openapi_schema)

# # Or just call directly
# response = await ep.call(client, json=payload)