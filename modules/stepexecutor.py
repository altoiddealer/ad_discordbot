import asyncio
import re
from PIL import Image, PngImagePlugin
import base64
import copy
from modules.typing import CtxInteraction, APIRequestCancelled
from typing import Any, Optional, Union
from modules.utils_shared import client, shared_path, is_tgwui_integrated, load_file, get_api
from modules.utils_misc import valueparser, set_key, extract_key, safe_copy, deep_merge, process_attachment
import modules.utils_processing as processing
from modules.apis import APIResponse, Endpoint, ImgGenClient_Comfy, ImgGenClient
from modules.presets_workflows import bot_workflows

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

async def call_stepexecutor(name:str=None, steps:list=None, input_data:Any=None, task=None, context:dict=None, prefix='Running '):
    if name:
        processing_steps = bot_workflows.get_workflow_steps_for(name)
        prefix = f'Running Workflow "{name}" with '
    elif steps:
        processing_steps = steps
    else:
        raise RuntimeError('[StepExecutor] Received a config without any workflow name or steps provided.')

    num_steps = len(processing_steps)
    log.info(f'[StepExecutor] {prefix}{num_steps} processing steps')
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
        self.context: dict[str, Any] = context if context is not None else {}
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

            except APIRequestCancelled as e:
                if e.cancel_event:
                    e.cancel_event.clear()
                log.info(e)
                if self.task:
                    await self.task.embeds.edit_or_send('img_gen', str(e), " ")
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
            config = processing.resolve_placeholders(config, {"context": self.context, **self.context}, log_prefix='[StepExecutor]', log_suffix='from saved context')
        if "task" in sources and self.task:
            config = processing.resolve_placeholders(config, vars(self.task.vars), log_prefix='[StepExecutor]', log_suffix=f'from Task "{self.task.name}" context')
        if "websocket" in sources and self.endpoint and self.endpoint.client.ws_config:
            ws_context = self.endpoint.client.ws_config.get_context()
            config = processing.resolve_placeholders(config, ws_context, log_prefix='[StepExecutor]', log_suffix=f'from "{self.endpoint.client.name}" Websocket context')
        return config

    def clone(self, steps:list|None=None, context:dict|None=None) -> "StepExecutor":
        steps = steps or []
        sub_context = context or safe_copy(self.context)

        return StepExecutor(steps,
                            response=self.response,
                            task=self.task,
                            ictx=self.ictx,
                            endpoint=self.endpoint,
                            context=sub_context)

    async def _step_if(self, data: Any, config: dict):
        """
        Execute the steps in this 'if' block if the condition evaluates True.

        Config format:
            value1: status
            operator: ==
            value2: active
            steps:
            - send_content: "**Status**: Active"
        """
        if not isinstance(config, dict):
            raise ValueError("[StepExecutor] 'if' step requires a dict config")

        steps = config.pop("steps", None)
        if not steps or not isinstance(steps, list):
            raise ValueError("[StepExecutor] 'if' step requires a 'steps' list")
        
        config['context'] = self.context
        if processing.evaluate_condition(**config):
            sub_executor:StepExecutor = self.clone(steps)
            branch_result = await sub_executor.run(data)

            self.context = deep_merge(self.context, sub_executor.context)

            return branch_result

        return data


    async def _step_if_group(self, data: Any, config: list[dict]):
        """
        Execute a chain of if/elif/else steps encapsulated in one step.

        Each item in config is a dict with one key: "if", "elif", or "else".
        Example:
            - if:
                value1: status
                operator: ==
                value2: active
                steps:
                - send_content: "**Status**: Active"
            - elif:
                value1: status
                operator: ==
                value2: pending
                steps:
                - send_content: "**Status**: Pending"
            - else:
                steps:
                - send_content: "**Status**: Unknown"
        """
        if not isinstance(config, list):
            raise ValueError("[StepExecutor] 'if_group' config must be a list of dicts")

        branch_executed = False
        result = data

        for branch in config:
            if not isinstance(branch, dict) or len(branch) != 1:
                raise ValueError(f"[StepExecutor] Invalid if_group branch: {branch}")

            branch_type, branch_cfg = next(iter(branch.items()))
            branch_type = branch_type.lower()

            if branch_type not in ("if", "elif", "else"):
                raise ValueError(f"[StepExecutor] Unsupported branch in if_group: {branch_type}")

            # Determine if branch should execute
            execute_branch = False
            if branch_type in ("if", "elif"):
                if branch_type == "if" or not branch_executed:
                    value1 = branch_cfg.get('value1')
                    operator = branch_cfg.get('operator')
                    value2 = branch_cfg.get('value2')
                    execute_branch = processing.evaluate_condition(value1, operator, value2, self.context)
            elif branch_type == "else":
                execute_branch = not branch_executed

            if execute_branch:
                branch_executed = True
                if isinstance(branch_cfg, dict):
                    branch_steps = branch_cfg.get("steps", [])
                elif isinstance(branch_cfg, list):
                    branch_steps = branch_cfg
                else:
                    raise ValueError(f"[StepExecutor] '{branch_type}' steps must be a list in 'if_group' step.")
                sub_executor:StepExecutor = self.clone(branch_steps)
                result = await sub_executor.run(result)

                self.context = deep_merge(self.context, sub_executor.context)
                break  # Exit after first branch executes

        return result

    ### Steps execution
    async def _step_for_each(self, data: Any, config: dict) -> list:
        """
        Runs a sub-StepExecutor for each item in a list or each key-value pair in a dict.

        Context updates from each iteration will merge into the root context.
        Returns a list of results, one per item processed.
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
            get_context = lambda idx, val: {alias: val,
                                            f"{alias}_index": idx}
            iter_keys = [alias, f"{alias}_index"]
            get_value = lambda item: item
        elif isinstance(items, dict):
            iterable = enumerate(items.items())
            get_context = lambda idx, pair: {f"{alias}_key": pair[0],
                                             f"{alias}_value": pair[1],
                                             f"{alias}_index": idx}
            iter_keys = [f"{alias}_key", f"{alias}_value", f"{alias}_index"]
            get_value = lambda item: item[1]
        else:
            raise TypeError(f"[StepExecutor] 'for_each' expected list or dict but got {type(items).__name__}")

        results = []
        root_context = copy.deepcopy(self.context)

        for index, item in iterable:
            # Add iteration variables to context
            item_context = {**root_context,
                            **get_context(index, item)}
            
            sub_executor:StepExecutor = self.clone(steps, context=item_context)
            result = await sub_executor.run(get_value(item))
            results.append(result)

            # Remove iteration variables
            for key in iter_keys:
                item_context.pop(key, None)

            self.context = deep_merge(self.context, item_context)

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
            subgroup_context = copy.deepcopy(self.context)
            sub_executor:StepExecutor = self.clone(steps, context=subgroup_context)
            result = await sub_executor.run(data)
            context_updates = deep_merge(context_updates, subgroup_context)
            return result, subgroup_context

        tasks = [run_subgroup(steps, idx) for idx, steps in enumerate(config)]
        results_with_contexts = await asyncio.gather(*tasks)

        # Separate results and contexts
        results = []
        all_contexts = []
        for result, subgroup_context in results_with_contexts:
            results.append(result)
            all_contexts.append(subgroup_context)

        # Merge all subgroup contexts into the root context
        for subgroup_context in all_contexts:
            self.context = deep_merge(self.context, subgroup_context)

        return results

    def _step_offload(self, data: Any, config: Any):
        if self.task:
            self.task.release_semaphore()
        else:
            log.warning('[StepExecutor] step "offload" had no effect because current execution is not part of a "Task"')
        return data

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

    def resolve_api_input(self, data: Any, config: dict, step_name: str, default: Any = None, endpoint: Endpoint | None = None):
        input_data = config.pop("input_data", default)
        init_payload = config.pop("init_payload", False)

        if not endpoint:
            input_data = default

        if init_payload and endpoint:
            log.info(f'[StepExecutor] Step "{step_name}": Using init_payload from endpoint "{endpoint.name}"')
            input_data = endpoint.get_payload()

        return self.resolve_api_payload(data, input_data)

    def resolve_api_config(self, config: dict, step_name: str, use_ws_default=False):
        client_name = config.pop("client_name", None) or config.pop("client", None)
        if not client_name:
            raise ValueError(f'[StepExecutor] API "client_name" was not included in "{step_name}" step')

        use_ws = config.pop("use_ws", use_ws_default)

        endpoint_name = config.pop("endpoint_name", None) or config.pop("endpoint", None)
        if not endpoint_name and not use_ws:
            raise ValueError(f'[StepExecutor] API "endpoint_name" was not included in "{step_name}" step')

        return client_name, endpoint_name, use_ws

    async def get_api_client_and_endpoint(self, config: dict, step_name: str, allow_ws_only=False):
        api = await get_api()
        client_name, endpoint_name, use_ws = self.resolve_api_config(config, step_name)

        client = api.get_client(client_name=client_name, strict=True)

        endpoint = None
        if endpoint_name:
            endpoint = client.get_endpoint(endpoint_name=endpoint_name, strict=True)
        elif not use_ws and not allow_ws_only:
            raise ValueError(f'[StepExecutor] Endpoint not specified for "{step_name}" step')

        return client, endpoint, use_ws

    async def _step_get_api_ws_config(self, data: Any, config: Union[str, dict]) -> Any:
        config['use_ws'] = True
        client, _, _ = await self.get_api_client_and_endpoint(config, 'get_api_ws_config', allow_ws_only=True)

        if not hasattr(client, 'ws_config') or client.ws_config is None:
            raise RuntimeError(f'[StepExecutor] API client does not have a websocket config to get.')

        return client.ws_config.get_context()

    async def _step_get_api_payload(self, data: Any, config: Union[str, dict]) -> Any:
        client, endpoint, _ = await self.get_api_client_and_endpoint(config, 'get_api_payload')
        payload = endpoint.get_payload()
        return self.resolve_api_payload(data, payload)

    async def _step_call_api(self, data: Any, config: Union[str, dict]) -> Any:
        client, endpoint, _ = await self.get_api_client_and_endpoint(config, 'call_api')
        self.endpoint = endpoint

        input_data = self.resolve_api_input(data, config, step_name='call_api', default=data, endpoint=endpoint)
        log.info(f'[StepExecutor] Calling API: {client.name}')
        response = await endpoint.call(input_data=input_data, **config)

        return response.body if isinstance(response, APIResponse) else response

    async def _step_track_progress(self, data: Any, config: dict) -> list[dict]:
        """ Polls an endpoint while sending a progress Embed to discord. """
        client, endpoint, use_ws = await self.get_api_client_and_endpoint(config, 'track_progress', allow_ws_only=True)

        completion_config = config.pop("completion_condition", None)
        completion_condition = None
        if completion_config:
            completion_condition = processing.build_completion_condition(completion_config, self.context)

        config["input_data"] = self.resolve_api_input(data, config, step_name='track_progress', default=None, endpoint=endpoint)

        return await client.track_progress(
            endpoint=endpoint,
            use_ws=use_ws,
            ictx=self.ictx,
            completion_condition=completion_condition,
            **config
        )

    async def _step_poll_api(self, data: Any, config: dict) -> list[dict]:
        """ Polls an endpoint. """
        client, endpoint, _ = await self.get_api_client_and_endpoint(config, 'poll_api')

        return_values = config.pop("return_values", {})
        interval = config.pop("interval", 1.0)
        duration = config.pop("duration", -1)
        num_yields = config.pop("num_yields", -1)

        config["input_data"] = self.resolve_api_input(data, config, step_name='poll_api', default=data, endpoint=endpoint)

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

    async def _step_upload_files(self, data: Any, config: Union[str, dict]) -> Any:
        """
        Uploads a file or files to an endpoint.

        Returns the response body (one file) or list of response bodies (multiple files)

        """
        client, endpoint, _ = await self.get_api_client_and_endpoint(config, 'upload_files')
        self.endpoint = endpoint
        input_data = config.pop("input_data", data)
        return await endpoint.upload_files(input_data, **config)

    async def _step_run_workflow(self, data, config):
        config['input_data'] = config.pop('input_data', data)
        config['task'] = self.task
        return await call_stepexecutor(**config)

    async def _step_prompt_user(self, data, config):
        """
        Prompts the user for input via Discord interaction.

        Config Parameters:
        - "prompt" (str): Message to show the user.
        - "type" (str): One of "text", "file", or "select".
        - "timeout" (int): Time in seconds to wait for user input (default: 60).
        - "options" (required for "select" type):
            - List of primitive values: [1, 2, 3.5, "Label"]
            - Dict of label-value pairs: {"Production": "prod", "Dev": "dev"}
            - List of objects: [{"label": "Production", "value": "prod"}, ...]
        """
        ictx = self.ictx
        if not ictx:
            raise RuntimeError("[StepExecutor] Cannot prompt user: 'ictx' (interaction context) is not set")

        prompt_text = config.get("prompt", "Please respond.")
        expected_type = config.get("type", "text")
        timeout = config.get("timeout", 60)

        from discord import Message, Attachment, Interaction, TextStyle
        from modules.utils_discord import SelectOptionsView, DynamicModal, get_user_ctx_inter

        user = get_user_ctx_inter(ictx)
        is_interaction = isinstance(ictx, Interaction)

        # Handle select menu
        if expected_type == "select":
            options_raw = config.get("options")
            if not options_raw or not isinstance(options_raw, (list, dict)):
                raise ValueError("[StepExecutor] 'select_options' must be a list or dict for 'prompt_user' step, when type is 'select'.")

            # Case: dict {label: value}
            if isinstance(options_raw, dict):
                display_names = list(options_raw.keys())
                value_lookup = options_raw

            # Case: list of dicts or primitives
            elif isinstance(options_raw, list):
                if all(isinstance(opt, dict) and "label" in opt and "value" in opt for opt in options_raw):
                    # List of dicts: [{label: ..., value: ...}]
                    display_names = [opt["label"] for opt in options_raw]
                    value_lookup = {opt["label"]: opt["value"] for opt in options_raw}
                else:
                    # List of primitives: [int, str, float, ...]
                    display_names = [str(opt) for opt in options_raw]
                    value_lookup = {str(opt): opt for opt in options_raw}

            else:
                raise ValueError("[StepExecutor] Unrecognized structure for 'select_options'.")

            view = SelectOptionsView(all_items=display_names,
                                     custom_id_prefix='step_select',
                                     placeholder_prefix='Choices: ',
                                     unload_item=None,
                                     warned=False)
            
            if is_interaction:
                msg = await ictx.followup.send(content=prompt_text, view=view, ephemeral=True)
            else:
                channel = await user.create_dm()
                msg = await channel.send(content=prompt_text, view=view)

            try:
                await asyncio.wait_for(view.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                await msg.edit(content="[StepExecutor] Timed out waiting for your selection.", view=None)
                raise TimeoutError("[StepExecutor] User did not respond in time.")

            selected_item = view.get_selected()
            await msg.delete()

            return value_lookup.get(selected_item)
        
        ### CAN'T SEND MODAL AFTER DELAY - KEEPING FOR FUTURE REFERENCE
        # if expected_type == "text":
        #     if is_interaction:
        #         modal_fields_config = config.get("modal_fields")

        #         modal_fields = []
        #         key_mapping = {}  # Maps custom_id => user-defined key (or fallback)

        #         if modal_fields_config and isinstance(modal_fields_config, list):
        #             for i, field in enumerate(modal_fields_config):
        #                 key = field.get("key") or f"field_{i}"
        #                 custom_id = f"modal_input_{key}"
        #                 key_mapping[custom_id] = key

        #                 modal_fields.append({
        #                     "label": field.get("label", f"Field {i + 1}"),
        #                     "placeholder": field.get("placeholder", ""),
        #                     "style": field.get("style", TextStyle.short),
        #                     "required": field.get("required", True),
        #                     "max_length": field.get("max_length", 2000),
        #                     "custom_id": custom_id,
        #                 })
        #         else:
        #             # Fallback to a single field
        #             custom_id = "modal_input_default"
        #             key_mapping[custom_id] = "response"
        #             modal_fields = [{
        #                 "label": prompt_text,
        #                 "placeholder": "Type your answer...",
        #                 "style": TextStyle.paragraph,
        #                 "required": True,
        #                 "max_length": 2000,
        #                 "custom_id": custom_id,
        #             }]

        #         modal = DynamicModal(title="Your Response", fields=modal_fields)
        #         await ictx.response.send_modal(modal)

        #         try:
        #             await asyncio.wait_for(modal.wait(), timeout=timeout)

        #             responses = {key_mapping[child.custom_id]: modal.responses.get(child.custom_id, "")
        #                          for child in modal.children}

        #             return next(iter(responses.values())) if len(responses) == 1 else responses

        #         except asyncio.TimeoutError:
        #             raise TimeoutError("[StepExecutor] User did not respond in time for 'prompt_user' step.")
        #         except Exception as e:
        #             raise RuntimeError(f"[StepExecutor] Modal processing failed unexpectedly: {e}")

        # Handle all other scenarios via DM
        channel = await user.create_dm()

        # Send basic text or file prompt
        await channel.send(prompt_text)

        def check(msg: Message):
            return msg.author.id == user.id and msg.channel.id == channel.id

        # Wait for the response
        try:
            client.waiting_for[user.id] = True
            msg: Message = await client.wait_for("message", check=check, timeout=timeout)

            if expected_type == "text":
                return msg.content.strip()

            elif expected_type == "file":
                if msg.attachments:
                    attachment: Attachment = msg.attachments[0]
                    file_dict = process_attachment(attachment)
                    return file_dict
                    #  {"file": file,
                    #   "bytes": file_bytes,
                    #   "file_format": mime_type,
                    #   "filename": filename}
                else:
                    raise ValueError("[StepExecutor] Expected file attachment but none provided.")
            else:
                raise ValueError(f"[StepExecutor] Unknown input type: {expected_type}")

        except asyncio.TimeoutError:
            raise TimeoutError("[StepExecutor] User did not respond in time.")
        finally:
            client.waiting_for.pop(user.id, None)

    async def _step_send_content(self, data: Any, config: str):
        """
        Send text or files to discord. Currently supports string or list of strings.

        Returns:
          None

        """
        if not isinstance(config, str) and not isinstance(config, list):
            log.error("[StepExecutor] Step 'send_content' did not receive valid content (must be string or list of strings)")
            return None

        resolved_content = processing.collect_content_to_send(config)
        await processing.send_content_to_discord(ictx=self.ictx, **resolved_content)
        return None

    def _step_set_context(self, data: Any, config: dict[str, Any]) -> Any:
        """ Sets key values in Context dict """
        for key, value in config.items():
            self.context[key] = value
        return data

    def _step_set_key(self, data: dict|list, config: Union[str, dict]) -> Any:
        """ Sets a value in the input dict or list, using dot bracket notation """
        path = config.get('path')
        value = config.get('value')
        return set_key(data, path, value)

    def _step_extract_key(self, data: dict|list, config: Union[str, dict]) -> Any:
        """ Gets a value in the input dict or list, using dot bracket notation """
        extracted = extract_key(data, config)
        return extracted

    def _step_extract_values(self, data: Any, config: dict[str, Union[str, dict]]) -> dict[str, Any]:
        """ Gets key values from Context dict using dot bracket notation. Returns dict. """
        result = {}
        for key, path_config in config.items():
            result[key] = extract_key(data, path_config)
        return result

    def _step_encode_base64(self, data, config):
        return base64.b64encode(data).decode('utf-8')

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

    def _step_dict(self, data, config: dict):
        if not isinstance(config, dict):
            raise ValueError("[StepExecutor] 'dict' step required to be formatted as a dict.")
        return config

    def _step_list(self, data, config: list):
        if not isinstance(config, list):
            raise ValueError("[StepExecutor] 'list' step required to be formatted as a list.")
        return config

    def _step_regex(self, data, pattern):
        match = re.search(pattern, data)
        if not match:
            log.warning("[StepExecutor] No regex match found")
            return data
        return match.group(1) if match.lastindex else match.group(0)

    # def _step_eval(self, data, expression):
    #     # TODO: Expand eval step
    #     return eval(expression, {"data": data})

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
        file = load_file(file_path, {})
        import json
        with open("payload_before.json", "w") as f:
            json.dump(file, f)
        return file

    async def _step_save(self, data: Any, config: dict):
        """
        Save input data to a file and return either path, original data, or metadata.

        Config options:
        - file_format: Explicit format (e.g. 'json', 'jpg').
        - file_name: Optional file name (without extension).
        - file_path: Relative directory inside output_dir.
        - use_timestamp: Adds a timestamp to the file.
        - overwrite: Whether to overwrite an existing file.

        returns: dict containing:
        - "file_path" (str) - full path to file
        - "file_format" (str) - file format without leading period
        - "file_name" (str) - filename including extension
        - "file_data" - original or decoded data as applicable
        """
        return await processing.save_any_file(data=data,
                                              file_format=config.get('file_format'),
                                              file_name=config.get('file_name'),
                                              file_path=config.get('file_path', ''),
                                              use_timestamp=config.get('timestamp', True),
                                              overwrite=config.get('overwrite', False),
                                              response=self.response,
                                              msg_prefix='[StepExecutor] ')

    async def _step_call_imggen(self, data: Any, config: dict):
        if config.get("payload"):
            config["input_data"] = config.pop("payload")

        client, endpoint, _ = await self.get_api_client_and_endpoint(config, 'call_imggen')
        self.endpoint = endpoint

        if not isinstance(client, ImgGenClient):
            raise RuntimeError(f'[StepExecutor] API Client is not an ImgGen client. Cannot run step "call_imggen".')

        payload = self.resolve_api_input(data, config, step_name='call_imggen', default=data, endpoint=endpoint)
        log.info(f'[StepExecutor] Calling ImgGen API: {client.name}')
        return await client._main_imggen(self.task, payload, endpoint, **config)

    async def _step_call_comfy(self, data: Any, config: dict):
        config['use_ws'] = True
        if config.get("payload"):
            config["input_data"] = config.pop("payload")

        client, endpoint, _ = await self.get_api_client_and_endpoint(config, 'call_comfy', allow_ws_only=True)
        self.endpoint = endpoint

        if not isinstance(client, ImgGenClient_Comfy):
            raise RuntimeError(f'[StepExecutor] API Client is not ComfyUI. Cannot run step "call_comfy".')

        payload = self.resolve_api_input(data, config, step_name='call_comfy', default=data, endpoint=endpoint)
        import json
        with open("payload_after.json", "w") as f:
            json.dump(payload, f)
        log.info(f'[StepExecutor] Calling ComfyUI (API: {client.name})')
        return await client._execute_prompt(payload, endpoint, self.ictx, self.task, **config)

    async def _step_free_comfy_memory(self, data: Any, config: dict):
        config["use_ws"] = True
        client, _, _ = await self.get_api_client_and_endpoint(config, 'free_comfy_memory', allow_ws_only=True)

        if not isinstance(client, ImgGenClient_Comfy):
            raise RuntimeError('[StepExecutor] API Client is not ComfyUI. Cannot run step "free_comfy_memory".')

        unload_models = config.get("unload_models", True)
        free_memory = config.get("free_memory", True)

        if not unload_models and not free_memory:
            log.warning("[StepExecutor] step 'free_comfy_memory' expected either 'unload_models' or 'free_memory'.")
            return data

        return await client._free_memory(unload_models, free_memory)
    
    async def _step_comfy_delete_nodes(self, data: Any, config: dict):
        if not config.get("input_data"):
            config["input_data"] = config.pop("payload", data)

        payload = config['input_data']
        delete_nodes = config['delete_nodes']
        delete_until = config.get('delete_until', [])

        processing.comfy_delete_and_reroute_nodes(payload, delete_nodes, delete_until)

        return payload

    async def _step_load_llmmodel(self, data: Any, config: str|dict):
        tgwui_enabled = False
        if is_tgwui_integrated:
            from modules.utils_tgwui import tgwui_shared_module, tgwui_utils_module, get_tgwui_functions, tgwui
            tgwui_enabled = tgwui.enabled

        if not tgwui_enabled:
            log.warning("[StepExecutor] TGWUI currently disabled (step 'load_llmmodel').")
            return data

        new_llmmodel = config if isinstance(config, str) else config.get('name', 'None')
        current_llmmodel = tgwui_shared_module.model_name

        if new_llmmodel == 'None' and current_llmmodel == 'None':
            log.warning("[StepExecutor] No LLM model currently loaded to unload.")
            return data
        
        # Unload current LLM Model
        if new_llmmodel == 'None':
            unload_model_func = get_tgwui_functions('unload_model')
            unload_model_func()
            log.info(f"[StepExecutor] LLM model unloaded.")
            # Reload the LLM on next LLM Gen request
            tgwui.lazy_load_llm = True
            tgwui_shared_module.model_name = current_llmmodel
        # Load a new LLM Model
        else:
            all_llmmodels = tgwui_utils_module.get_available_models()
            if new_llmmodel not in all_llmmodels:
                raise ValueError(f"[StepExecutor] LLM Model '{new_llmmodel}' not in available models!")
            tgwui_shared_module.model_name = new_llmmodel
            from modules.utils_shared import bot_database
            bot_database.update_was_warned('no_llmmodel')         # Reset warning message
            loader = tgwui.get_llm_model_loader(new_llmmodel)     # Try getting loader from user-config.yaml to prevent errors
            await tgwui.load_llm_model(loader)                    # Load an LLM model if specified
            log.info(f"[StepExecutor] LLM model loaded: {new_llmmodel}.")

        return data

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
