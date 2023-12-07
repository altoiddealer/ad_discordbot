**12/7/2023: coqui_tts Extension support now added!**
**11/20/2023: Big update - "imgloras" are now "imgtags".**

If you manually apply updates to your existing config.py, dict .yaml files, etc, please carefully [compare the changes](https://github.com/altoiddealer/ad_discordbot/commit/96ff8e6698e98b25cd00837f529278275ecb6c51).


# altoiddealer's Discord Bot

A Discord Bot for chatting with LLMs using txt-generation-webui API.

<img width="560" alt="Screenshot 2023-09-22 224716" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/5ffb5037-29d2-4a2f-9966-3ef6bd35b1f9">


This bot stands apart from many other ones due to a variety of custom features:

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

1. **Install oobabooga's [text-generation-webui](https://github.com/oobabooga/text-generation-webui)**

2. **[Create a Discord bot account](https://discordpy.readthedocs.io/en/stable/discord.html), invite it to your server, and note its authentication token**.

3. Clone this repository into **/text-generation-webui/**
   ```
   git clone https://github.com/altoiddealer/ad_discordbot
   ```
4. Move **bot.py** out of subdirectory **/ad_discordbot/** -> into the directory **/text-generation-webui/**

   ```
   /text-generation-webui/bot.py
   
   /text-generation-webui/ad_discordbot/(remaining files)
   ```
5. **Add the bot token (from Step 2) into **/ad_discordbot/config.py**
   
6. **Run the .cmd file** in text-generation-webui directory (**ex: cmd_windows.bat**), and performing the following commands:
   ```
   pip install discord
   ```

# Running the bot

1. **Run the .cmd file** in text-generation-webui directory (**ex: cmd_windows.bat**)
   ```
   python bot.py (args)
   ```

   **EXAMPLE LAUNCH COMMAND:**
   ```
   python bot.py --loader exllama --model airoboros-l2-13b-gpt4-2.0-GPTQ
   ```
2. Use [command](https://github.com/altoiddealer/ad_discordbot/wiki/commands) **/character** to choose a character.

# Updating the bot

1. **Open a cmd window** in **/ad_discordbot/** and **git pull**
   ```
   git pull
   ```

2. **IF bot.py has changed:**
  
   Move **bot.py** out of subdirectory **/ad_discordbot/** -> into the directory **/text-generation-webui/** (*overwriting old version*)

   ```
   /text-generation-webui/bot.py
   
   /text-generation-webui/ad_discordbot/(remaining files)
   ```

3. **IF files have changed that you modified:**
  
   You will need to backup files you modified, and update them to match new changes**

   **Example**: config.py gets updated with a new feature.
   
   Solution is to either:

   * Update your existing config.py with the new feature, OR
     
   * Update the new config.py with your settings.

# Getting responses from the bot

* @ mention the bot

* Use [command](https://github.com/altoiddealer/ad_discordbot/wiki/commands) **/main** to set a main channel. **The bot won't need to be @ mentioned in main channels.**

* If you enclose your text in parenthesis (like this), the bot will not respond.

# Getting image responses from the bot

* @ mention it including a trigger phrase defined in config.py, and the bot will reply with a Stable Diffusion prompt and image based on your prompt.

* Use **/image** command to use your own prompt with advanced options

# Getting TTS responses from the bot (**currently coqui_tts only**)

1. **Run the .cmd file** in text-generation-webui directory (**ex: cmd_windows.bat**), and performing the following commands:
   ```
   pip install pynacl
   ```
   
   Linux / Mac:
   ```
   pip install -r extensions/coqui_tts/requirements.txt
   ```
   
   Windows:
   ```
   pip install -r extensions\coqui_tts\requirements.txt
   ```

2. Ensure that your bot has sufficient permissions to use the Voice channel and/or upload files

3. Configure **config.py** in the section **discord** > **tts_settings**

4. The necessary model file(s) should download on first launch of the bot.  If not, then first launch textgen-webui normally and enable the extension.

5. **Your characters can have their own settings including voices!  See example character M1nty for usage**
