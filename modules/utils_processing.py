import base64
import os
import uuid
import aiofiles
import re
from datetime import datetime
from pathlib import Path
from pydub import AudioSegment
import io
from modules.typing import Any

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

def extract_nested_value(data: Any, path: str | list[str]) -> Any:
    if isinstance(path, str):
        # Split by dots, handle [index] if present
        path = re.split(r'\.(?![^\[]*\])', path)

    current = data
    for part in path:
        # Handle list indexing e.g. "images[0]"
        match = re.match(r"^([^\[\]]+)(?:\[(\d+)\])?$", part)
        if not match:
            raise KeyError(f"Invalid key part: '{part}'")

        key, idx = match.groups()

        if isinstance(current, dict):
            current = current.get(key)
        else:
            raise KeyError(f"Expected dict, got {type(current)} at '{key}'")

        if idx is not None:
            if isinstance(current, list):
                index = int(idx)
                current = current[index]
            else:
                raise KeyError(f"Expected list at '{key}', got {type(current)}")

    return current

async def save(data, config):
    path = Path(config.get("file_path", "./saved"))
    path.mkdir(parents=True, exist_ok=True)
    file_name = config.get("file_name", "output")
    ext = config.get("file_format", "txt")
    full_path = path / f"{file_name}.{ext}"
    mode = "wb" if isinstance(data, bytes) else "w"

    async with aiofiles.open(full_path, mode) as f:
        await f.write(data if isinstance(data, str) else data.decode())
    log.info(f"Saved response to {full_path}")
    return data

def extract_keys(data, keys):
    keys = keys.split(".") if isinstance(keys, str) else keys
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            raise ValueError("Cannot extract key from non-dict response")
    return data

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
    with open(filepath, "wb") as f:
        f.write(binary_data)
    return filepath

def extract_and_save_base64(response_json, key, output_format, save_to, prefix="file"):
    base64_str = response_json.get(key)
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
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        output_file = output_dir / f"{file_prefix}_{timestamp}.{output_format}"
        audio.export(output_file, format=output_format)
        return str(output_file)
    except Exception as e:
        log.exception(f"Failed to save audio to {output_format}: {e}")
        raise

def extract_filepath(response_json, key):
    return response_json.get(key)  # for APIs that return file path directly
