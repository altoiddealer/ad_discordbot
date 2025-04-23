import base64
import os
import uuid
import aiofiles

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

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
