# altoiddealer's Discord Bot

A Discord Bot for chatting with LLMs using txt-generation-webui API.

<img width="561" alt="Screenshot 2023-09-22 214847" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/e5e60178-afd0-41af-aab9-911a11f12160">


This bot stands apart from many other ones due to a variety of features I coded in:

-"Dynamic Context" activated by user configured trigger phrases to autmoatically swap character context, state settings, add instruct to prompt, alter history handling, with option to remove trigger phrase from user prompt

-All settings can be modified on-the-fly in active_settings.yaml. Settings changed via commands are immediately reflected in active_settings.yaml.

-Plenty of customization due to a variety of Settings commands paired with easy to manage .yaml dictionaries.

-Continue and Regenerate responses with /cont and /regen commands - both work fluidly with very long messages exceeding Discords message size limitation.

-Get image response from bot via user configured trigger phrases

-Modify Stable Diffusion payload settings automatically via user configured phrases

-Send your image prompt directly to A1111 with powerful /image command that includes support for ControlNet and Reactor (face swap)

-Change image models and relavent payload settings via /imgmodel command

-Simple Starboard feature, easy to set up.

-Feature to include current settings in a dedicated channel

-AND MORE.

<img width="817" alt="Screenshot 2023-09-22 220802" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/b2f2bd96-ed12-4eed-b842-68171c62a8e5">


# Installation

1. Install oobabooga's text-generation-webui
   
2. Create a Discord bot account, invite it to your server, and note its authentication token.
3. 
   https://discordpy.readthedocs.io/en/stable/discord.html

4. Move "/ad_discordbot/" and "bot.py" to the oobabooga directory /text-generation-webui/
   
     /text-generation-webui/ad_discordbot/(included files)
   
     /text-generation-webui/bot.py
   
5. Run the cmd file in oobabooga directory (ex: cmd_windows.bat)
   
   cd text-generation-webui
   
   pip install discord
   
   pip install OpenCV-Python (required for /image command)


# Running the bot

1. Run the cmd file in oobabooga directory (ex: cmd_windows.bat)

   cd text-generation-webui
   python bot.py (args)
   
   # Example
   python bot.py --loader exllama --model airoboros-l2-13b-gpt4-2.0-GPTQ


   

See included example characters for reference.
