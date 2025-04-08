import aiohttp
import asyncio
import jsonschema
from typing import Any, Dict, Optional, Union
from modules.utils_shared import shared_path, load_file, is_tgwui_integrated, config

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log


class API:
    def __init__(self):
        self.clients:dict[str, APIClient] = {}

        self.imggen_client:Optional[APIClient] = None
        self.textgen_client:Optional[APIClient] = None
        self.tts_client:Optional[APIClient] = None
        self.init()

    def assign_functions(self, api_client:"APIClient"):
        if api_client.function is not None and api_client.function in ['imggen', 'textgen', 'ttsgen']:
            function_key = api_client.function + '_client'
            self_function_key = getattr(self, function_key)
            # Only accept first instance
            if self_function_key is None:
                self_function_key = api_client.name
                log.info(f'[APIs] Assigned "{api_client.name}" as a default client ({api_client.function}).')

    def init(self):
        apis = load_file(shared_path.api_settings)
        for api_config in apis:
            if not isinstance(api_config, dict):
                log.warning('[API] An API definition was not formatted as a dictionary. Ignoring.')
                continue
            # Collect all valid user APIs
            try:
                api_client = APIClient(name=api_config['name'],
                                       url=api_config['url'],
                                       headers=api_config.get('default_headers'),
                                       timeout=api_config.get('default_timeout', 10),
                                       auth=api_config.get('auth'),
                                       endpoints_config=api_config.get('endpoints', []))
                self.clients[api_config['name']] = api_client
            except KeyError as e:
                log.warning(f"[API] Skipping API Client due to missing key: {e}")
            self.assign_functions(api_client)


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
        # fetch schema
        asyncio.create_task(self._fetch_openapi_schema())

    def _collect_endpoints(self, endpoints_config:list[dict]):
        for ep in endpoints_config:
            try:
                endpoint = Endpoint(name=ep["name"],
                                    path=ep["path"],
                                    method=ep.get("method", "GET"),
                                    response_type=ep.get("response_type", "json"),
                                    headers=ep.get("headers", self.default_headers),
                                    timeout=ep.get("timeout", self.default_timeout))
                self.endpoints[endpoint.name] = endpoint
            except KeyError as e:
                log.warning(f"[APIClient] Skipping endpoint due to missing key: {e}")

    async def _fetch_openapi_schema(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.url}/openapi.json") as response:
                    if response.status == 200:
                        self.openapi_schema = await response.json()
                        log.debug(f"Loaded OpenAPI schema for {self.name}")
                    else:
                        log.debug(f"No OpenAPI schema available at {self.name}")
        except Exception as e:
            log.error(f"Failed to load OpenAPI schema from {self.url} for {self.name}: {e}")

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
        return_text: bool = False,
        return_raw: bool = False,
        timeout: Optional[int] = None,
    ) -> Union[Dict[str, Any], str, bytes, None]:
        
        url = f"{self.url}{endpoint}" if endpoint.startswith("/") else f"{self.url}/{endpoint}"
        headers = {**self.default_headers, **(headers or {})}
        timeout = timeout or self.default_timeout
        auth=auth or self.auth
        
        # Validate payload
        self.validate_payload(method, endpoint, json, data)

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


class Endpoint:
    def __init__(self,
                 name: str,
                 path: str,
                 method: str = "GET",
                 response_type: str = "json",
                 headers: Optional[Dict[str, str]] = None,
                 timeout: int = 10):
        self.name = name
        self.path = path
        self.method = method.upper()
        self.response_type = response_type
        self.headers = headers or {}
        self.timeout = timeout

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
        schema = self.get_schema(openapi_schema)
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
        json_payload = kwargs.get('json')
        data_payload = kwargs.get('data')

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
            headers=kwargs.get("headers", self.headers),
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