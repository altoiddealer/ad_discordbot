# altoiddealer's Discord Bot

A Discord Bot for chatting with LLMs using txt-generation-webui API.

This bot stands apart from many other ones due to a variety of features I coded in:

-"Dynamic Context" activated by user configured trigger phrases to autmoatically swap character context, state settings, add instruct to prompt, alter history handling, with option to remove trigger phrase from user prompt

-All settings can be modified on-the-fly in active_settings.yaml. Settings changed via commands are immediately reflected in active_settings.yaml.

-Plenty of customization due to a variety of Settings commands paired with easy to manage .yaml dictionaries.

-Continue and Regenerate responses with /cont and /regen commands - both work fluidly with very long messages exceeding Discords message size limitation.

-Get image response from bot via user configured trigger phrases

-Modify Stable Diffusion payload settings automatically via user configured phrases

-Powerful /image command that includes support for ControlNet and Reactor (face swap)

-Change image models and relavent payload settings via /imgmodel command

-Simple Starboard feature, easy to set up.

-Feature to include current settings in a dedicated channel

-AND MORE.

# Installation

1. Move "/ad_discordbot/" and "bot.py" to the oobabooga directory /text-generation-webui/
     /text-generation-webui/ad_discordbot/(included files)
     /text-generation-webui/bot.py
2. Run the cmd file in oobabooga directory (ex: cmd_windows.bat)
3. cd text-generation-webui
4. pip install discord
5. pip install OpenCV-Python (required for /image command)

See included example characters for reference.
