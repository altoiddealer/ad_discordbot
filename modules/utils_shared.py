import asyncio
import os
import logging

task_semaphore = asyncio.Semaphore(1)

class SharedPath:
    dir_root = 'ad_discordbot'
    dir_internal = os.path.join(dir_root, 'internal')
    os.makedirs(dir_internal, exist_ok=True)

    # Internal
    active_settings = os.path.join(dir_internal, 'activesettings.yaml')
    starboard = os.path.join(dir_internal, 'starboard_messages.yaml')

    # Configs
    config = os.path.join(dir_root, 'config.yaml')
    base_settings = os.path.join(dir_root, 'dict_base_settings.yaml')
    cmd_options = os.path.join(dir_root, 'dict_cmdoptions.yaml')
    img_models = os.path.join(dir_root, 'dict_imgmodels.yaml')
    tags = os.path.join(dir_root, 'dict_tags.yaml')

shared_path = SharedPath()
