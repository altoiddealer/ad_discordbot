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

    # Configs
    config = os.path.join(dir_root, 'config.yaml')
    base_settings = os.path.join(dir_root, 'dict_base_settings.yaml')
    cmd_options = os.path.join(dir_root, 'dict_cmdoptions.yaml')
    img_models = os.path.join(dir_root, 'dict_imgmodels.yaml')
    tags = os.path.join(dir_root, 'dict_tags.yaml')
    starboard = os.path.join(dir_root, 'starboard_messages.yaml')

shared_path = SharedPath()



_old_active = os.path.join(shared_path.dir_root, 'activesettings.yaml')
if os.path.isfile(_old_active):
    logging.info(f'Migrating file to "{shared_path.active_settings}"')
    os.rename(_old_active, shared_path.active_settings)
    