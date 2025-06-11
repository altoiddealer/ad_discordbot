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
from typing import Any, Optional, Callable
from modules.utils_misc import extract_key, normalize_mime_type, guess_format_from_headers, guess_format_from_data, is_base64, valueparser
from modules.utils_shared import config

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

async def save_any_file(data: Any,
                        file_format:Optional[str]=None,
                        file_name:Optional[str]=None,
                        file_path:str='',
                        use_timestamp:bool=True,
                        response = None,
                        msg_prefix:str = ''):
    """
    Save input data to a file and returns dict.

    - file_format: Explicit format (e.g. 'json', 'jpg').
    - file_name: Optional file name (without extension).
    - file_path: Relative directory inside output_dir.
    - response: optional APIResponse object (if data type is APIResponse.body)
    - msg_prefix: to prefix logging messages
    """
    from modules.apis import APIResponse
    response:Optional[APIResponse] = response

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
    if file_format:
        file_format = normalize_mime_type(file_format) # Normalize if MIME type like 'image/png'
    else:
        if isinstance(response, APIResponse):
            file_format = guess_format_from_headers(response.headers)
        if not file_format:
            file_format = guess_format_from_data(data)
        if file_format:
            file_format = normalize_mime_type(file_format)
            log.info(f'{msg_prefix}Guessed output file format: "{file_format}"')

    full_path = output_path / f"{file_name}.{file_format}"
    file_name = f"{file_name}.{file_format}"

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
            return {"path": str(full_path),
                    "format": file_format,
                    "name": file_name,
                    "data": data}

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

    return {"path": str(full_path),
            "format": file_format,
            "name": file_name,
            "data": data}

def resolve_placeholders(config: Any, context: dict, log_prefix: str = '', log_suffix: str = '') -> Any:
    formatted_keys = []

    def _stringify(value):
        if isinstance(value, bytes):
            return "placeholder"
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
            value = extract_key(context, path)
            formatted_keys.append(path.split('.')[0])  # only log the root key
            return value
        except ValueError:
            return None

    def _resolve(config: Any) -> Any:
        if isinstance(config, str):
            stripped = config.strip()
            # Exact placeholder
            if (stripped.startswith("{") and stripped.endswith("}") and
                    stripped.count("{") == 1 and stripped.count("}") == 1):
                key_path = stripped[1:-1]
                value = _extract_from_context(key_path)
                return value if value is not None else config

            # Partial format with possible multiple keys
            matches = re.findall(r'\{([^\}]+)\}', config)
            formatted_context = {k: _stringify(_extract_from_context(k)) for k in matches}
            try:
                formatted = config.format(**formatted_context)
                if formatted != config:
                    for k in matches:
                        formatted_keys.append(k.split('.')[0])
                    try:
                        parsed = valueparser.parse_value(formatted)
                        return parsed
                    except Exception:
                        return formatted
            except KeyError:
                return config

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
                        if actual.get(sub_key) != sub_expected:
                            return False
                else:
                    if actual != expected:
                        return False
            return True
        except Exception:
            return False

    return condition_func


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
