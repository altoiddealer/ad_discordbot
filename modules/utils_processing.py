import base64
import os
import uuid
import aiofiles
import re
import json
import yaml
import aiofiles
from PIL import Image, PngImagePlugin
from pathlib import Path
from datetime import datetime
from pathlib import Path
from pydub import AudioSegment
import io
import mimetypes
from typing import Any, Union, Optional, Callable
from modules.utils_misc import extract_key, normalize_mime_type, guess_format_from_headers, guess_format_from_data, is_base64
from modules.utils_shared import config, shared_path, get_api
from modules.utils_discord import send_long_message
from modules.apis import API, Endpoint, apisettings
from discord import File
from modules.typing import CtxInteraction, FILE_INPUT

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

async def save_any_file(data: Any,
                        file_format:Optional[str]=None,
                        file_name:Optional[str]=None,
                        file_path:str='',
                        use_timestamp:bool=True,
                        response = None,
                        msg_prefix:str = '',
                        overwrite:bool = False):
    """
    Save input data to a file and returns dict.

    Arguments:
    - file_format Explicit format (e.g. 'json', 'jpg').
    - file_name: Optional file name (without extension).
    - file_path: Relative directory inside output_dir.
    - response: optional APIResponse object (if data type is APIResponse.body)
    - msg_prefix: to prefix logging messages

    Returns dict containing:
    - file_path: full file path string
    - file_format: format without leading period
    - file_name: file name including extension
    - file_data: original or decoded data when applicable
    """
    from modules.apis import APIResponse
    response:Optional[APIResponse] = response

    from discord import Attachment
    if isinstance(data, Attachment):
        file_name = Path(data.filename).stem
        data = await data.read()

    # 1. Setup file path & naming
    from modules.utils_shared import shared_path

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = file_name or timestamp
    if file_name != timestamp and use_timestamp == True:
        file_name = f'{file_name}_{timestamp}'
    file_path = Path(file_path)
    output_path = shared_path.output_dir / file_path
    if not config.path_allowed(output_path):
        raise RuntimeError(f"Tried saving to a path which is not in config.yaml 'allowed_paths': {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. Resolve file_format
    if not file_format:
        file_format = guess_format_from_data(data)
        if not file_format and isinstance(response, APIResponse):
            file_format = guess_format_from_headers(response.headers)
        log.info(f'{msg_prefix}Guessed output file format: "{file_format}"')
    file_format = normalize_mime_type(file_format) # Normalize if MIME type like 'image/png'

    full_path = output_path / f"{file_name}.{file_format}"
    file_name = f"{file_name}.{file_format}"

    if not overwrite and full_path.exists():
        log.warning(f"{msg_prefix}File already exists (skipped): ...{full_path.parent.name}/{full_path.name}")
        return {"file_path": str(full_path),
                "file_format": file_format,
                "file_name": file_name,
                "file_data": data}

    binary_formats = {"jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff", "mp3", "wav", "ogg", "flac",
                      "mp4", "webm", "avi", "mov", "mkv", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
                      "zip", "rar", "7z", "tar", "gz", "bz2", "exe", "dll", "iso", "bin", "dat"}

    # 3. Base64 decoding if applicable
    if isinstance(data, str) and is_base64(data):
        try:
            data = base64.b64decode(data)
            log.info(f"{msg_prefix}Detected base64 input; decoded to binary.")
        except Exception as e:
            log.error(f"{msg_prefix}Failed to decode base64 string: {e}")
            raise

    # 4. Select write mode
    mode = "wb" if file_format in binary_formats else "w"

    # 5. Save logic
    try:
        # 5a. Special case: Handle PIL images with optional PngInfo
        if isinstance(data, Image.Image) and file_format.lower() in {"png", "jpeg", "jpg", "webp"}:
            pnginfo = data.info.get("pnginfo") if file_format.lower() == "png" else None
            format_map = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}
            data.save(full_path, format=format_map.get(file_format.lower(), file_format.upper()), pnginfo=pnginfo)
            log.info(f"{msg_prefix}Saved image using PIL to {full_path}")
            return {"file_path": str(full_path),
                    "file_format": file_format,
                    "file_name": file_name,
                    "file_data": data}

        # 5b. Proceed with async file saving for everything else
        async with aiofiles.open(full_path, mode) as f:
            if file_format == "json":
                if isinstance(data, (dict, list)):
                    await f.write(json.dumps(data, indent=2))
                else:
                    raise TypeError(f"{msg_prefix}JSON format requires dict or list.")
            elif file_format == "yaml":
                if isinstance(data, (dict, list)):
                    await f.write(yaml.dump(data))
                else:
                    raise TypeError(f"{msg_prefix}YAML format requires dict or list.")
            elif file_format == "csv":
                if isinstance(data, list) and all(isinstance(row, (list, tuple)) for row in data):
                    csv_content = "\n".join([",".join(map(str, row)) for row in data])
                    await f.write(csv_content)
                else:
                    raise TypeError(f"{msg_prefix}CSV format requires list of lists/tuples.")
            elif mode == "w":
                if not isinstance(data, (str, int, float)):
                    raise TypeError(f"{msg_prefix}Text format requires str/number, got {type(data).__name__}")
                await f.write(str(data))
            elif mode == "wb":
                if isinstance(data, bytes):
                    await f.write(data)
                elif isinstance(data, str):
                    await f.write(data.encode())
                elif hasattr(data, "read"):  # e.g., BytesIO or file-like
                    await f.write(data.read())
                else:
                    raise TypeError(f"{msg_prefix}Binary format requires bytes or str, got {type(data).__name__}")

    except Exception as e:
        log.error(f"{msg_prefix}Failed to save data as {file_format}: {e}")
        raise

    log.info(f"{msg_prefix}Saved data to {full_path}")

    return {"file_path": str(full_path),
            "file_format": file_format,
            "file_name": file_name,
            "file_data": data}

def resolve_placeholders(config: Any, context: dict, log_prefix: str = '', log_suffix: str = '') -> Any:
    formatted_keys = []

    def _stringify(value):
        if isinstance(value, bytes):
            return "cannot_be_stringified"
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value)
            except TypeError:
                return str(value)
        if value is None:
            return ""
        return str(value)

    def _extract_from_context(path: str):
        try:
            if re.match(r'^[a-zA-Z_]\w*$', path):  # simple key
                value = context.get(path)
            else:
                value = extract_key(context, path)
            if value is not None:
                formatted_keys.append(path.split('.')[0])
            return value
        except ValueError:
            return None

    def _resolve(config: Any) -> Any:
        if isinstance(config, str):
            # If it is an exact placeholder
            stripped = config.strip()
            if re.fullmatch(r"\{[^\{\}]+\}", stripped):
                key_path = stripped[1:-1]
                value = _extract_from_context(key_path)
                return value if value is not None else config

            # Otherwise, do a regex-based substitution manually
            def replacer(match):
                key_path = match.group(1)
                val = _extract_from_context(key_path)
                stringified = _stringify(val) if val is not None else match.group(0)
                
                if stringified == "cannot_be_stringified":
                    raise ValueError(f"[ValueParser] Cannot stringify value for key '{key_path}'. "
                                     f"Detected unserializable object (e.g., bytes or custom type) during string interpolation."
                                     "(Don't try to format bytes / other unserializable types via string formatting strategy)")
                
                return stringified

            formatted = re.sub(r'\{([^\{\}]+)\}', replacer, config)
            return formatted

        elif isinstance(config, dict):
            return {k: _resolve(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [_resolve(item) for item in config]

        return config

    result = _resolve(config)

    if formatted_keys:
        unique_keys = sorted(set(formatted_keys))
        prefix = f"{log_prefix} " if log_prefix else ""
        suffix = f" {log_suffix}" if log_suffix else ""
        log.info(f'{prefix}Formatted the following keys{suffix}: {", ".join(unique_keys)}')

    return result

def build_completion_condition(condition_config: dict, context_vars: dict = None) -> Callable[[dict], bool]:
    """
    Builds a callable that checks if a websocket message meets a user-defined condition.
    If only a key's presence is important, value can be "*" or Ellipsis (...)

    Example input:
    {
        "type": "executed",
        "data": {
            "prompt_id": "{prompt_id}"
        }
    }
    """
    from copy import deepcopy

    # Fill context vars into placeholders like "{prompt_id}"
    def resolve_placeholders(value):
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            key = value.strip("{}")
            return context_vars.get(key, value)
        return value

    # Deep-copy and resolve context
    condition = deepcopy(condition_config or {})
    if context_vars:
        for k, v in condition.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    condition[k][sub_k] = resolve_placeholders(sub_v)
            else:
                condition[k] = resolve_placeholders(v)

    # Return the actual checker function
    def condition_func(msg: dict) -> bool:
        try:
            for key, expected in condition.items():
                actual = msg.get(key)
                if isinstance(expected, dict):
                    if not isinstance(actual, dict):
                        return False
                    for sub_key, sub_expected in expected.items():
                        if sub_expected in [Ellipsis, "*"]:
                            if sub_key not in actual:
                                return False
                        elif actual.get(sub_key) != sub_expected:
                            return False

                else:
                    if expected in [Ellipsis, "*"]:
                        if key not in msg:
                            return False
                    elif actual != expected:
                        return False
            return True
        except Exception:
            return False

    return condition_func

def split_files_by_size(normalized_files: list[FILE_INPUT],
                        max_discord_size=10 * 1024 * 1024) -> tuple[list, list]:
    small_files = []
    large_files = []

    for file in normalized_files:
        if file["file_size"] < max_discord_size:
            small_files.append(file)
        else:
            large_files.append(file)

    return small_files, large_files

def normalize_file_inputs(input_data:Union[dict, bytes, str],
                          filename:str='file.bin') -> list[FILE_INPUT]:
    input_list = input_data if isinstance(input_data, list) else [input_data]
    normalized = []

    for item in input_list:
        file_obj = None
        should_close = False
        mime_type = 'application/octet-stream'
        file_size = None

        if isinstance(item, dict):
            data = item.get("bytes") or item.get("file_data")
            file_path = item.get("file_path")
            filename = item.get("file_name", filename)

            if data:
                mime_type = guess_format_from_data(data, default="application/octet-stream")
                file_obj = io.BytesIO(data)
                file_obj.name = filename
                file_size = len(data)

            elif file_path:
                filename = os.path.basename(file_path)
                mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                file_obj = open(file_path, "rb")
                file_size = os.path.getsize(file_path)
                should_close = True

            else:
                raise ValueError("Dict input must contain 'bytes'/'file_data' or 'file_path'.")

        elif isinstance(item, bytes):
            mime_type = guess_format_from_data(item, default="application/octet-stream")
            file_obj = io.BytesIO(item)
            file_obj.name = filename
            file_size = len(item)

        elif isinstance(item, str):
            filename = os.path.basename(item)
            mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            file_obj = open(item, "rb")
            file_size = os.path.getsize(item)
            should_close = True

        else:
            raise TypeError(f"Unsupported input type: {type(item)}")

        normalized.append({"file_obj": file_obj,
                           "filename": filename,
                           "mime_type": mime_type,
                           "file_size": file_size,
                           "should_close": should_close})

    return normalized

def collect_content_to_send(all_content) -> dict:
    resolved_content = {'text': [],
                        'audio': [],
                        'files': []}

    def add_file(item):
        if isinstance(item, Path):
            resolved_content['files'].append(str(item))
        elif isinstance(item, (str, bytes, io.BytesIO)):
            resolved_content['files'].append(item)
        elif isinstance(item, dict) and ('file_path' in item or \
                                        (('file_data' in item or 'bytes' in item) and 'file_name' in item)):
            resolved_content['files'].append(item)

    def sort_content(content):
        # Handle strings and Path objects as potential file paths
        if isinstance(content, (str, Path)):
            target_path = Path(content)
            if not target_path.is_absolute():
                try_paths = [shared_path.dir_user / target_path,
                             shared_path.output_dir / target_path]
                for path in try_paths:
                    if path.exists():
                        target_path = path
                        break

            if target_path.exists() and target_path.is_file():
                filepath = str(target_path.resolve())
                if target_path.suffix.lower() in [".mp3", ".wav"]:
                    resolved_content['audio'].append(filepath)
                else:
                    resolved_content['files'].append(filepath)
            else:
                if isinstance(content, str):
                    resolved_content['text'].append(content)
                else:
                    log.warning(f"Unresolved Path object: {content}")

        elif isinstance(content, (bytes, io.BytesIO)):
            add_file(content)

        elif isinstance(content, dict):
            add_file(content)

        else:
            log.warning(f"Unsupported content type: {type(content).__name__}")

    all_content = [all_content] if not isinstance(all_content, list) else all_content
    for item in all_content:
        if isinstance(item, (str, bytes, io.BytesIO, dict, Path)):
            sort_content(item)

    return resolved_content

async def send_content_to_discord(task = None,
                                  ictx: CtxInteraction|None = None,
                                  text: dict|None = None,
                                  audio: dict|None = None,
                                  files: Any | FILE_INPUT | list[FILE_INPUT] | None = None,
                                  vc = None,
                                  normalize: bool = True):
    ictx = task.ictx if task else ictx
    if not ictx:
        raise RuntimeError('A discord interaction is required for send_context_to_discord()')
    if text is None and task:
        text = task.extra_text
    if audio is None and task:
        audio = task.extra_audio
    if files is None and task:
        files = task.extra_files

    try:
        if text:
            # header = "**__Extra text__**:\n"
            delimiter = "\n--------------------------------------------\n"
            joined_text = delimiter.join(text)
            # all_extra_text = header + joined_text
            await send_long_message(ictx.channel, joined_text)

        if audio:
            for audio_fp in audio:
                if vc and ictx:
                    await vc.process_audio_file(ictx, audio_fp)
                else:
                    files.append(audio_fp)

        if files:
            if normalize:
                # NOTE: bytes + BytesIO objects **WILL FAIL unless provided in a dict including "file_name"**
                files = normalize_file_inputs(files)

            files: list[FILE_INPUT]
            small, large = files, None
            upload_large_files_ep = None

            api:API = await get_api()
            if config.discord.get('upload_large_files', False) and api.upload_large_files and api.upload_large_files.enabled:
                upload_large_files_ep = api.get_misc_function_endpoint(func_key="upload_large_files",
                                                                       task_key="post_upload")
                small, large = split_files_by_size(files)
                if large and not isinstance(upload_large_files_ep, Endpoint):
                    log.error(f"The bot is configured to upload large files that exceed discord's 10MB limit, but endpoint was not found.")
                    small.extend(large)
                    large = None

            if small:
                discord_files = [File(f["file_obj"], filename=f["filename"]) for f in small]
                if len(discord_files) == 1:
                    await ictx.channel.send(file=discord_files[0])
                else:
                    await ictx.channel.send(files=discord_files)

            if large and isinstance(upload_large_files_ep, Endpoint):
                url_strings:list = await upload_large_files_ep.upload_files(normalized_inputs=large)
                uploaded_files_msg = "**__Uploaded files exceeding Discord 10MB limit:__**\n"
                uploaded_files_msg += '\n'.join(f"<{url}>" for url in url_strings)
                await send_long_message(ictx.channel, uploaded_files_msg)

    except Exception as e:
        error_message = str(e)
        if "413 Payload Too Large" in error_message or "Request entity too large" in error_message:
            log.error("Failed to send files to discord because payload too large. Consider enabling/configuring the 'upload_large_files' feature.")
        else:
            log.error(f"An error occurred while trying to send extra results: {error_message}")
        raise

def decode_base64(data, _config=None):
    return base64.b64decode(data) if isinstance(data, str) else data

async def encode_to_base64(path):
    async with aiofiles.open(path, mode='rb') as f:
        raw = await f.read()
        return base64.b64encode(raw).decode('utf-8')

def save_base64(base64_str, output_format="png", save_to="./", prefix="file"):
    binary_data = base64.b64decode(base64_str)
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.{output_format}"
    filepath = os.path.join(save_to, filename)
    if not config.path_allowed(filepath):
        raise RuntimeError(f"Tried saving to a path which is not in config.yaml 'allowed_paths': {filepath}")
    with open(filepath, "wb") as f:
        f.write(binary_data)
    return filepath

def extract_and_save_base64(data:dict, key:str, output_format:str, save_to:str, prefix="file"):
    base64_str = data.get(key)
    if base64_str:
        return save_base64(base64_str, output_format, save_to, prefix)
    return None

def type(data, to_type):
    type_map = {"int": int, "float": float, "str": str, "bool": bool}
    return type_map[to_type](data)

def detect_audio_format(data: bytes) -> str:
    if data.startswith(b'ID3') or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    elif data.startswith(b'RIFF') and b'WAVE' in data[8:16]:
        return "wav"
    else:
        return "unknown"

def save_audio_bytes(
    audio_bytes: bytes,
    output_path: str,
    file_prefix: str|None = '',
    input_format: str = "mp3",  # or "wav", "ogg", etc.
    output_format: str = "mp3",  # or "wav", etc.
) -> str:
    """
    Save raw audio bytes to a file, optionally converting format.

    Args:
        audio_bytes: The raw audio bytes.
        output_path: Path to save the output file (without extension).
        input_format: Format of the input bytes (e.g. 'mp3', 'wav').
        output_format: Desired output format.

    Returns:
        The final file path.
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=input_format)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_prefix = file_prefix or "audio"
        output_dir = Path(output_path)
        if not config.path_allowed(output_dir):
            raise RuntimeError(f"Tried saving to a path which is not in config.yaml 'allowed_paths': {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        output_file = output_dir / f"{file_prefix}_{timestamp}.{output_format}"
        audio.export(output_file, format=output_format)
        return str(output_file)
    except Exception as e:
        log.exception(f"Failed to save audio to {output_format}: {e}")
        raise

def extract_filepath(response_json, key):
    return response_json.get(key)  # for APIs that return file path directly
