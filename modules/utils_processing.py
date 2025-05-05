import base64
import os
import uuid
import aiofiles
import re
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

def extract_filepath(response_json, key):
    return response_json.get(key)  # for APIs that return file path directly
