import asyncio
import re
from PIL import Image, PngImagePlugin
import io
import base64
import filetype
from modules.typing import CtxInteraction
from typing import Any, Optional, Union
from modules.utils_shared import client, shared_path, load_file, get_api
from modules.utils_misc import valueparser, set_key, extract_key
import modules.utils_processing as processing
from modules.apis import apisettings, APIResponse, Endpoint, API, APIClient, ImgGenClient_Comfy

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

async def call_stepexecutor(name:str=None, steps:list=None, input_data:Any=None, task=None, context:dict=None, prefix='Running '):
    if name:
        processing_steps = apisettings.get_workflow_steps_for(name)
        prefix = f'Running Workflow "{name}" with '
    elif steps:
        processing_steps = steps
    else:
        raise RuntimeError('[StepExecutor] Received a config without any workflow name or steps provided.')

    num_steps = len(processing_steps)
    log.info(f'[StepExecutor] {prefix}{num_steps} processing steps)')
    handler = StepExecutor(steps=processing_steps, input_data=input_data, task=task, context=context)
    return await handler.run()

class StepExecutor:
    def __init__(self, steps:list[dict], input_data:Any=None, response=None, task=None, ictx=None, endpoint=None, context=None):
        """
        Executes a sequence of data transformation steps with optional context storage.

        Steps are defined as a list of single-key dictionaries, where each key is the step type
        (e.g., "extract_values") and the value is the configuration for that step.

        Each step can optionally include a `save_as` key to store intermediate results in context
        without affecting the main result passed to the next step.
        """
        self.steps = steps
        self.context: dict[str, Any] = context if context else {}
        self.original_input_data = input_data
        self.response:Optional[APIResponse] = input_data if isinstance(input_data, APIResponse) else response
        self.endpoint:Optional[Endpoint] = endpoint
        self.task = task
        self.ictx:Optional[CtxInteraction] = ictx
        if task:
            self.ictx = task.ictx

    def _split_step(self, step: dict) -> tuple[str, dict, dict]:
        step = step.copy()
        metadata_keys = {"save_as", "on_error", "log", "returns"} #, "skip_if"}  # extensible
        metadata = {k: step.pop(k) for k in metadata_keys if k in step}
        if len(step) != 1:
            raise ValueError(f"[StepExecutor] Invalid step format: {step}")
        step_name, config = next(iter(step.items()))
        return step_name, config, metadata

    def _initial_data(self):
        if isinstance(self.response, APIResponse):
            return self.response.body
        return self.original_input_data

    async def run(self, input_data: Any|None = None) -> Any:
        result = input_data or self._initial_data()

        for step in self.steps:
            if not isinstance(step, dict):
                raise ValueError(f"[StepExecutor] Invalid step: {step}")

            step = step.copy()

            step_name, config, meta = self._split_step(step)

            method = getattr(self, f"_step_{step_name}", None)
            if not method:
                raise NotImplementedError(f"[StepExecutor] Unsupported step: {step_name}")
            
            try:
                if step_name == "for_each":
                    # _step_for_each will resolve placeholders per-item
                    step_result = await method(result, config)
                else:
                    # Resolve any placeholders using the current context
                    config = self._resolve_context_placeholders(result, config)
                    # Run the step and determine where to store the result
                    step_result = await method(result, config) if asyncio.iscoroutinefunction(method) else method(result, config)

                result = self._process_step_result(meta, result, step_result, step_name)

            except Exception as e:
                on_error = meta.get("on_error", "raise")
                if on_error == "skip":
                    log.error(f"[StepExecutor] Step '{step_name}' failed and was skipped: {e}")
                    return result
                elif on_error == "default":
                    # Use default value from config or None
                    default_value = config.get("default", None)
                    log.error(f"[StepExecutor] Step '{step_name}' failed, using default: {default_value} ({e})")
                    result = default_value
                    self._apply_meta_save_as(meta, result, step_name)
                else:  # Default behavior: raise
                    log.error(f"[StepExecutor] An error occurred while processing step '{step_name}': {e}")
                    raise

        # print("final result:", result)
        return result

    ### Meta Handling
    def _apply_meta_returns(self, meta:dict, original_input:Any, step_result:Any, step_name:str) -> Any:
        returns = meta.get("returns", "data")
        if returns and not isinstance(returns, str):
            log.warning(f"[StepExecutor] 'returns' value must be a string, not {type(returns).__name__}. Falling back to 'data'.")
            returns = "data"

        if returns == "data":
            return step_result
        if returns == "input":
            return original_input
        if returns == "context":
            return self.context
        if isinstance(step_result, dict):
            if returns in step_result:
                return step_result[returns]

        log.warning(f"[StepExecutor] 'returns': '{returns}' not found in {step_name} step result. Falling back to 'data'.")
        return step_result

    def _apply_meta_save_as(self, meta:dict, result:Any, step_name:str):
        save_as = meta.get("save_as")
        if save_as:
            self.context[save_as] = result
            if meta.get("log"):
                log.info(f'[Step Executor] Saved {step_name} result to context as: {save_as}')

    def _process_step_result(self, meta:dict, original_input:Any, step_result:Any, step_name:str) -> Any:
        # apply "save_as"
        self._apply_meta_save_as(meta, step_result, step_name)
        # apply "returns"
        processed_result = self._apply_meta_returns(meta, original_input, step_result, step_name)
        # apply "log"
        if meta.get("log"):
            log.info(f'[Step Executor] {step_name} results: {step_result}')

        return processed_result


    ### Context Resolution
    def _resolve_context_placeholders(self, data: Any, config: Any, sources=["result", "context", "task", "websocket"]) -> Any:
        # Merge context with 'result'
        if "result" in sources:
            config = processing.resolve_placeholders(config, {"result": data}, log_prefix='[StepExecutor]', log_suffix='from prior step "result"')
        if "context" in sources:
            config = processing.resolve_placeholders(config, self.context, log_prefix='[StepExecutor]', log_suffix='from saved context')
        if "task" in sources and self.task:
            config = processing.resolve_placeholders(config, vars(self.task.vars), log_prefix='[StepExecutor]', log_suffix=f'from Task "{self.task.name}" context')
        if "websocket" in sources and self.endpoint and self.endpoint.client.ws_config:
            ws_context = self.endpoint.client.ws_config.get_context()
            config = processing.resolve_placeholders(config, ws_context, log_prefix='[StepExecutor]', log_suffix=f'from "{self.endpoint.client.name}" Websocket context')
        return config
    

    ### Steps execution
    async def _step_for_each(self, data: Any, config: dict) -> list:
        """
        Runs a sub-StepExecutor for each item in a list or each key-value pair in a dict.

        Returns:
            list: A list of results, one per item processed.
        """
        source = config.get("in")
        alias = config.get("as", "item")
        steps = config.get("steps")

        if not steps or not isinstance(steps, list):
            raise ValueError("[StepExecutor] 'for_each' step requires a 'steps' list.")

        # Determine iterable: from context (by string) or directly as list/dict
        items = self.context.get(source) if isinstance(source, str) else source

        if isinstance(items, list):
            iterable = enumerate(items)
            get_context = lambda idx, val: {
                alias: val,
                f"{alias}_index": idx,
            }
        elif isinstance(items, dict):
            iterable = enumerate(items.items())
            get_context = lambda idx, pair: {
                f"{alias}_key": pair[0],
                f"{alias}_value": pair[1],
                f"{alias}_index": idx,
            }
        else:
            raise TypeError(f"[StepExecutor] 'for_each' expected list or dict but got {type(items).__name__}")

        results = []

        for index, item in iterable:
            sub_executor = StepExecutor(steps, task=self.task, ictx=self.ictx, endpoint=self.endpoint, response=self.response)

            sub_executor.context = {
                **self.context,
                **get_context(index, item),
            }

            value = item if isinstance(items, list) else item[1]
            result = await sub_executor.run(value)
            results.append(result)

        return results

    async def _step_group(self, data: Any, config: list[list[dict]]) -> list:
        """
        Executes multiple step sequences in parallel, each defined as a list of steps.

        Each item in the config is a list of steps (a sub-workflow), which is executed
        in parallel with others.

        Returns:
            list: The list of results from each parallel sub-sequence.
        """
        if not isinstance(config, list) or not all(isinstance(group, list) for group in config):
            raise ValueError("step_group config must be a list of lists of steps")

        async def run_subgroup(steps: list[dict], index: int):
            sub_executor = StepExecutor(steps, task=self.task, ictx=self.ictx, endpoint=self.endpoint)
            sub_executor.response = self.response
            sub_executor.context = self.context.copy()
            return await sub_executor.run(data)

        tasks = [run_subgroup(steps, idx) for idx, steps in enumerate(config)]
        results = await asyncio.gather(*tasks)
        return results

    def _step_pass(self, data: Any, config: Any):
        return data
    
    def _step_format(self, data: Any, config: str):
        if not isinstance(config, str):
            log.warning(f"[StepExecutor] 'format' step expects a string key, got: {type(config).__name__}")
            return data
        resolved = self._resolve_context_placeholders(data, config)
        if isinstance(resolved, str):
            return valueparser.parse_value(resolved)
        return resolved

    def _step_return(self, data: Any, config: str) -> Any:
        """
        Returns a value from the context using the given key (config).
        Example: - return: image_list
        """
        if not isinstance(config, str):
            raise ValueError(f"[StepExecutor] 'return' step expects a string key, got: {type(config).__name__}")
        if config not in self.context:
            raise KeyError(f"[StepExecutor] Context does not contain key '{config}'")
        return self.context[config]
    
    ### API RELATED STEPS
    def resolve_api_payload(self, data: Any, payload: Any):
        # Resolve from all sources except Task
        payload = self._resolve_context_placeholders(data, payload, sources=["result", "context", "websocket"])
        # Resolve Task placeholders
        if self.task:
            payload = self.task.override_payload(payload)
        return payload

    def resolve_api_input(self, data:Any, config:dict, step_name:str, default:Any|None=None, endpoint:Endpoint|None=None):
        input_data = config.pop('input_data', default)
        init_payload = config.pop('init_payload', False)
        if not endpoint: # Websocket
            return default
        # init_payload overrides input_data
        if init_payload:
            log.info(f'[StepExecutor] Step "{step_name}": Fetching payload for "{endpoint.name}" and trying to update placeholders with internal variables.')
            input_data = endpoint.get_payload()
            # Resolves context data more cleanly for Task variables
            input_data = self.resolve_api_payload(data, input_data)

        elif input_data is not None:
            log.info(f'[StepExecutor] Step "{step_name}": Sending "input_data" to "{endpoint.name}". If unwanted, update your step definition with "input_data: null".')
        else:
            pass

        return input_data
    
    def resolve_api_names(self, config:dict, step_name:str):
        client_name = config.pop("client_name", None) or config.pop("client", None)
        if not client_name:
            raise ValueError(f'[StepExecutor] API "client_name" was not included in "{step_name}" response handling step')
        use_ws = config.pop("use_ws", False)
        endpoint_name = config.pop("endpoint_name", None) or config.pop("endpoint", None)
        if not endpoint_name and not use_ws:
            raise ValueError(f'[StepExecutor] API "endpoint_name" was not included in "{step_name}" response handling step')
        return client_name, endpoint_name, use_ws

    async def _step_get_api_ws_config(self, data: Any, config: Union[str, dict]) -> Any:
        config['use_ws'] = True
        api:API = await get_api()
        client_name, _, _ = self.resolve_api_names(config, 'get_api_ws_config')
        api_client:APIClient = api.get_client(client_name=client_name, strict=True)
        if api_client.ws_config is None:
            raise RuntimeError(f'[StepExecutor] API client "{client_name}" does not have a websocket config to get.')
        return api_client.ws_config.get_context()

    async def _step_get_api_payload(self, data: Any, config: Union[str, dict]) -> Any:
        api:API = await get_api()
        client_name, endpoint_name, use_ws = self.resolve_api_names(config, 'get_api_payload')
        api_client:APIClient = api.get_client(client_name=client_name, strict=True)
        endpoint:Endpoint = api_client.get_endpoint(endpoint_name=endpoint_name, strict=True)
        payload = endpoint.get_payload()
        # Resolves context data more cleanly for Task variables
        return self.resolve_api_payload(data, payload)

    async def _step_call_api(self, data: Any, config: Union[str, dict]) -> Any:
        api:API = await get_api()
        client_name, endpoint_name, use_ws = self.resolve_api_names(config, 'call_api')
        api_client:APIClient = api.get_client(client_name=client_name, strict=True)
        endpoint:Endpoint = api_client.get_endpoint(endpoint_name=endpoint_name, strict=True)
        self.endpoint = endpoint # Helps to resolve API related context

        input_data = self.resolve_api_input(data, config, step_name='call_api', default=data, endpoint=endpoint)

        response = await endpoint.call(input_data=input_data, **config)
        if not isinstance(response, APIResponse):
            return response
        return response.body

    async def _step_track_progress(self, data: Any, config: dict) -> list[dict]:
        """
        Polls an endpoint while sending a progress Embed to discord.
        """
        ictx = self.ictx # to send discord Embed to channel
        api:API = await get_api()
        client_name, endpoint_name, use_ws = self.resolve_api_names(config, 'track_progress')
        api_client:APIClient = api.get_client(client_name=client_name, strict=True)

        completion_config = config.pop("completion_condition", None)
        completion_condition = None
        if completion_config:
            completion_condition = processing.build_completion_condition(completion_config, self.context)

        # Resolve polling method
        endpoint:Optional[Endpoint] = None
        if not use_ws:
            endpoint = api_client.get_endpoint(endpoint_name=endpoint_name, strict=True)

        config['input_data'] = self.resolve_api_input(data, config, step_name='track_progress', default=None, endpoint=endpoint)

        return await api_client.track_progress(endpoint=endpoint,
                                               use_ws=use_ws,
                                               ictx=ictx,
                                               completion_condition=completion_condition,
                                               **config)

    async def _step_poll_api(self, data: Any, config: dict) -> list[dict]:
        """
        Polls an endpoint.
        """
        api:API = await get_api()
        client_name, endpoint_name, use_ws = self.resolve_api_names(config, 'poll_api')
        api_client:APIClient = api.get_client(client_name=client_name, strict=True)
        endpoint:Endpoint = api_client.get_endpoint(endpoint_name=endpoint_name, strict=True)

        return_values = config.pop("return_values", {})
        interval = config.pop("interval", 1.0)
        duration = config.pop("duration", -1)
        num_yields = config.pop("num_yields", -1)

        config['input_data'] = self.resolve_api_input(data, config, step_name='poll_api', default=data, endpoint=endpoint)

        results = []
        try:
            async for result in endpoint.poll(return_values=return_values,
                                            interval=interval,
                                            duration=duration,
                                            num_yields=num_yields,
                                            **config):
                results.append(result)
        except Exception as e:
            log.error(f"[StepExecutor] Error in 'poll_api' step: {e}")

        return results

    async def _step_prompt_user(self, data, config):
        """
        Prompts the user for input via Discord interaction.

        "prompt": "Please upload an image or reply with text",
        "type": "text" | "file",
        "timeout": 60
        """
        ictx = self.ictx
        if not ictx:
            raise RuntimeError("[StepExecutor] Cannot prompt user: 'ictx' (interaction context) is not set")
        
        from discord import Message, Attachment
        ictx:Message

        prompt_text = config.get("prompt", "Please respond.")
        expected_type = config.get("type", "text")
        timeout = config.get("timeout", 60)

        # Open DM channel
        dm_channel = await ictx.author.create_dm()

        # Send the prompt via DM
        await dm_channel.send(prompt_text)

        def check(msg: Message):
            return msg.author.id == ictx.author.id and msg.channel.id == dm_channel.id

        # Wait for the response
        try:
            client.waiting_for[ictx.author.id] = True
            msg: Message = await client.wait_for("message", check=check, timeout=timeout)

            if expected_type == "text":
                return msg.content.strip()
            elif expected_type == "file":
                if msg.attachments:
                    attachment: Attachment = msg.attachments[0]
                    file_bytes = await attachment.read()
                    filename = attachment.filename

                    kind = filetype.guess(file_bytes)
                    mime_type = kind.mime if kind else 'application/octet-stream'
                    mime_category = mime_type.strip().lower().split('/')[0]
                    # Prepare a file-like object
                    file_obj = io.BytesIO(file_bytes)
                    file_obj.name = filename

                    file = {mime_category: {"file": file_obj, "filename": filename, "content_type": mime_type}}

                    return {"file": file, "bytes": file_bytes, "file_format": mime_type, "filename": filename}
                else:
                    raise ValueError("[StepExecutor] Expected file attachment but none provided.")
            else:
                raise ValueError(f"[StepExecutor] Unknown input type: {expected_type}")

        except asyncio.TimeoutError:
            raise TimeoutError("[StepExecutor] User did not respond in time.")
        finally:
            client.waiting_for.pop(ictx.author.id, None)

    async def _step_send_content(self, data: Any, config: str):
        """
        Send text or files to discord. Currently supports string or list of strings.

        Returns:
          None

        """
        if not isinstance(config, str) and not isinstance(config, list):
            log.error("[StepExecutor] Step 'send_content' did not receive valid content (must be string or list of strings)")
            return None

        resolved_content = processing.resolve_content_to_send(config)
        await processing.send_content_to_discord(ictx=self.ictx, **resolved_content)
        return None

    def _step_set_key(self, data: dict|list, config: Union[str, dict]) -> Any:
        path = config.get('path')
        value = config.get('value')
        return set_key(data, path, value)

    def _step_extract_key(self, data: dict|list, config: Union[str, dict]) -> Any:
        extracted = extract_key(data, config)
        return extracted

    def _step_extract_values(self, data: Any, config: dict[str, Union[str, dict]]) -> dict[str, Any]:
        result = {}
        for key, path_config in config.items():
            result[key] = extract_key(data, path_config)
        return result

    def _step_decode_base64(self, data, config):
        if isinstance(data, str):
            if "," in data:
                data = data.split(",", 1)[1]
            return base64.b64decode(data)
        raise TypeError("[StepExecutor] Expected base64 string for 'decode_base64' step")

    def _step_type(self, data, to_type):
        type_map = {"int": int, "float": float, "str": str, "bool": bool}
        return type_map[to_type](data)
    
    def _step_cast(self, data, config: dict):
        type_map = {
            "int": int,
            "float": float,
            "str": str,
            "bool": lambda x: str(x).lower() in ("true", "1", "yes")
        }

        if not isinstance(data, dict):
            raise TypeError(f"[StepExecutor] 'cast' step requires a dict input, got {type(data).__name__}")

        result = data.copy()
        for key, type_name in config.items():
            if key not in result:
                log.warning(f"[StepExecutor] 'cast' step: key '{key}' not found in data")
                continue
            if type_name not in type_map:
                raise ValueError(f"[StepExecutor] 'cast' step: unsupported type '{type_name}'")
            try:
                result[key] = type_map[type_name](result[key])
            except Exception as e:
                log.warning(f"[StepExecutor] Failed to cast '{key}' to {type_name}: {e}")

        return result

    def _step_map(self, data, config: dict):
        """
        Transforms each item in a list using a mapping config.
        Example config:
            {
                "as": "dict",          # or "value", "tuple", "custom"
                "key": "name",         # Used if "as" == "dict"
            }
        """
        if not isinstance(data, list):
            raise TypeError("[StepExecutor] 'map' step requires list input")

        transform_type = config.get("as", "value")

        if transform_type == "dict":
            key = config.get("key")
            if not key:
                raise ValueError("[StepExecutor] 'map' step with 'dict' transform requires 'key'")
            return [{key: item} for item in data]

        elif transform_type == "tuple":
            return [(item,) for item in data]

        # elif transform_type == "custom":
        #     # Optional: allow arbitrary callable by name (e.g., "lambda x: {'name': x}")
        #     expr = config.get("lambda")
        #     if not expr:
        #         raise ValueError("[StepExecutor] 'map' step with 'custom' transform requires 'lambda'")
        #     func = eval(expr)  # ⚠️ Only safe in trusted environments!
        #     return [func(item) for item in data]

        # elif transform_type == "value":
        #     return data  # no-op

        else:
            raise ValueError(f"[StepExecutor] Unknown transform type: {transform_type}")

    def _step_regex(self, data, pattern):
        match = re.search(pattern, data)
        if not match:
            log.warning("[StepExecutor] No regex match found")
            return data
        return match.group(1) if match.lastindex else match.group(0)

    def _step_eval(self, data, expression):
        # TODO: Expand eval step
        return eval(expression, {"data": data})

    def _step_add_pnginfo(self, data: Any, config: dict) -> Image.Image:
        """
        Adds PngInfo metadata to an image using values from context or data.

        Config:
            image (str, optional): Context key to retrieve the image object.
            metadata (str, optional): Context key to retrieve metadata string.

            If only one is provided, the other is assumed to come from the `data` parameter.

        Returns:
            Image.Image: Image with PngInfo metadata injected.
        """
        image_key = config.get("image")
        metadata_key = config.get("metadata")

        if image_key and metadata_key:
            image = self.context.get(image_key)
            metadata = self.context.get(metadata_key)
        elif image_key:
            image = self.context.get(image_key)
            metadata = data
        elif metadata_key:
            metadata = self.context.get(metadata_key)
            image = data
        else:
            raise ValueError("[StepExecutor] 'add_pnginfo' step requires at least one of 'image' or 'metadata' in config")

        if not isinstance(image, Image.Image):
            raise TypeError(f"[StepExecutor] Expected a PIL.Image.Image for image, got {type(image).__name__}")
        if not isinstance(metadata, str):
            raise TypeError(f"[StepExecutor] Expected a string for metadata, got {type(metadata).__name__}")

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", metadata)

        # Attach pnginfo to image
        image.info["pnginfo"] = pnginfo

        return image
    
    async def _step_load_data_file(self, data: Any, path: str):
        from pathlib import Path
        from modules.utils_shared import config
        file_path = Path(path)
        if not file_path.is_absolute() and not file_path.exists():
            corrected_path = Path(shared_path.dir_user) / file_path
            if corrected_path.exists():
                file_path = corrected_path
        if not config.path_allowed(file_path):
            raise RuntimeError(f"[StepExecutor] Tried loading a file which is not in config.yaml 'allowed_paths': {file_path}")
        return load_file(file_path, {})

    async def _step_save(self, data: Any, config: dict):
        """
        Save input data to a file and return either path, original data, or metadata.

        Config options:
        - file_format: Explicit format (e.g. 'json', 'jpg').
        - file_name: Optional file name (without extension).
        - file_path: Relative directory inside output_dir.
        - returns: dict containing:
                "path" (str) - full path to file
                "format" (str) - file format
                "name" (str) -  filename
                "data" - file data
        """
        return await processing.save_any_file(data=data,
                                              file_format=config.get('file_format'),
                                              file_name=config.get('file_name'),
                                              file_path=config.get('file_path', ''),
                                              use_timestamp=config.get('timestamp', True),
                                              response=self.response,
                                              msg_prefix='[StepExecutor] ')

    async def _step_call_comfy(self, data: Any, config: dict):
        config['use_ws'] = True
        if config.get('payload'):
            config['input_data'] = config.pop('payload')

        api:API = await get_api()
        client_name, endpoint_name, _ = self.resolve_api_names(config, 'call_comfy')
        comfy_client:APIClient = api.get_client(client_name=client_name, strict=True)
        if not isinstance(comfy_client, ImgGenClient_Comfy):
            raise RuntimeError(f'[StepExecutor] API Client "{client_name}" is not ComfyUI. Cannot run step "call_comfy".')

        endpoint:Optional[Endpoint] = None
        if endpoint_name:
            endpoint = comfy_client.get_endpoint(endpoint_name=endpoint_name, strict=True)
        self.endpoint = endpoint # Helps to resolve API related context

        payload:dict = self.resolve_api_input(data, config, step_name='call_api', default=data, endpoint=endpoint)

        results = await comfy_client._execute_prompt(payload, endpoint, self.ictx, self.task, **config)

        return results


# async def _process_file_input(self, path: str, input_type: str):
#     if input_type == "text":
#         async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
#             return await f.read()

#     elif input_type == "base64":
#         async with aiofiles.open(path, mode='rb') as f:
#             raw = await f.read()
#             return base64.b64encode(raw).decode('utf-8')

#     elif input_type == "file":
#         # Special behavior — this must be handled outside JSON.
#         raise ValueError("File upload input type 'file' should be used with multipart/form-data.")

#     elif input_type == "raw":
#         async with aiofiles.open(path, mode='rb') as f:
#             return await f.read()

#     elif input_type == "url":
#         return path  # Just return the path (expected to be a full URL)

#     else:
#         raise ValueError(f"Unknown input type: {input_type}")
