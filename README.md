# altoiddealer's Discord Bot

### Uniting LLM ([text-generation-webui](https://github.com/oobabooga/text-generation-webui)) and Img Gen ([A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) / [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)) for chat & professional use.

---

<img width="560" alt="Screenshot 2023-09-22 224716" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/5ffb5037-29d2-4a2f-9966-3ef6bd35b1f9">

---

### For support / discussion, visit the dedicated [ad_discordbot channel](https://discord.com/channels/1089972953506123937/1154970156108365944) on the TextGenWebUI Discord server.

---

**[What's new](#whats-new)   |   [Features](#features)   |   [Installation](#installation)   |   [Usage](#usage)   |   [Updating](#updating)**

---

## What's new:

<details>
  <summary>click to expand</summary>
  
   **04/01/2024:** Pretty massive update. Be sure to update textgen-webui, and take care updating settings files.
   
    - Overhauled SD WebUI extension support (ControlNet, layerdiffuse, ReActor) to be much more powerful and manageable.
    
      - ControlNet support received a massive update in particular... multi-ControlNet is even supported!
      - These extensions each have a simple primary Tag to activate and apply.
      - **ALL** of their parameters are now easily controlled by the Tags system.
      
    - Added a new method to create Tags on-the-fly using this syntax: [[key:value]] or [[key1:value1 | key2:value2]]. These go into immediate effect, particularly useful for controlling the extension settings.
    
    - 4 older parameters were recently removed from textgen-webui - mirrored in this bot update.

   ---
   
   **03/07/2024:** [layerdiffuse](https://github.com/layerdiffusion/sd-forge-layerdiffuse) support added to Tags feature!

   <img width="603" alt="Screenshot 2024-03-07 104132" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/30975294-d8b4-4ae1-a53d-7f3616d8a22c">

   ---
   
   **02/13/2024:** Major update introducing the Tags feature. Take care migrating your existing settings

   <img width="1403" alt="Screenshot 2024-03-07 104231" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/89aae51f-2abe-43c2-bade-a2ea395be2da">

   ---
   
   **12/11/2023:** New "/speak" command! Silero and ElevenLabs TTS extensions now supported!

   <img width="595" alt="Screenshot 2024-03-07 104326" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/c7b42c39-c29f-4d69-af25-fff5a4d9dbcf">

   ---
   
   **12/8/2023:** TTS Support, and Character Specific Extension Settings now added!

   <img width="1152" alt="Screenshot 2024-03-07 104503" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/5bbd15bc-f181-4f49-bece-b633bdc7412d">
   
</details>

## Features:

- **Robust "Tags" system to manipulate bot behavior persistently or via trigger phrases, including:**
  - Trigger Text and/or Image response
  - Image censoring settings (None / Spoiler / Block)
  - Powerful [ControlNet](https://github.com/Mikubill/sd-webui-controlnet), [ReActor](https://github.com/Gourieff/sd-webui-reactor), and [layerdiffuse](https://github.com/layerdiffusion/sd-forge-layerdiffuse) integration
  - Automatically apply [loractl scaling](https://github.com/cheald/sd-webui-loractl) (Currently A1111 only)
  - Swapping / Changing LLM characters (ei: "draw... " can trigger character tailored for image prompting)
  - Swapping / Changing LLM models and Img models
  - Modifying LLM state (temperature, repetition penalty, etc) and image gen settings (width, height, cfg scale, etc)
  - Keep things spicy by factoring random variations to LLM state / img gen settings
  - Modifying the user's prompt and LLM's reply with delete/insert/replace text, img LORAs, etc
  - Manipulate LLM history (Suppress history, limit history, save or do-no-save reply, etc)
  - **New Feature:** Flexible, Instant-Tags by using syntax [[key:value]] or [[key1:value1 | key2:value2 ]]
  - New features being added frequently and easily due to the framework of the "Tags" system

- **TTS Support!**
  - [alltalk_tts](https://github.com/erew123/alltalk_tts), coqui_tts, silero_tts, and elevenlabs_tts
  - Bot can speak on Voice channel, upload copy of audio file, or both!
  - Per-character TTS settings! Give each character a unique voice!

- **Sophisticated function to send text responses over Discord's 2,000 character limit**
  - "chunks" messages by looking back to nearest line break or sentence completion.
  - Preserves syntax between chunks such as **bold**, *italic*, and even `code formatting`

- **Commands!**
  - **/helpmenu** - Display information message
  - **/character** - Change character
  - **/main** - Toggle if Bot always replies, per channel
  - **/image** - Allows more controlled image prompting (positive prompt, neg prompt, Size settings, **ControlNet**, **ReActor**)
  - **/speak** - Bot can speak any text, using any voices (including user attach .mp3 or .wav for alltalk_tts)!
  - **/imgmodel** - Change A1111 model & img_payload settings
  - **/llmmodel** - Change LLM model

- **Dynamic settings handling:**
  - Core bot settings managed in **`config.py`** (bot behavior, discord features, extensions, etc.)
  - The "Tags" system is configured in **`dict_tags.yaml`** (global Tags, default Tag params, Tag presets, etc.)
  - Foundational layer of user settings configured in **`base_settings.yaml`**.
  - Character files can include custom Tags, TTS settings, LLM state parameters, and special behaviors, which prioritize over basesettings.
  - Custom Image models settings defined in **`dict_imgmodels.yaml`** (Tags, payload params) which prioritize over basesettings.
  - All user settings commit to **`activesettings.yaml`**, which serves as a dashboard or for manually testing new settings on-the-fly.

- **Automatic Img model changing:**
  - Adjustable duration and mode (random / cycle)
  - Smart filters and settings to auto-update relavent settings (SD1.5, SDXL, Turbo, etc)

- **Continue and Regenerate text replies via Context Menu (right click on the reply)**

- **All tasks queue up and process elagently - go ahead, spam it with requests!**
- 
- **Built in Starboard feature**

- **Feature to post current settings in a dedicated channel**

- **ALWAYS MORE TO COME**

---

<img width="817" alt="Screenshot 2023-09-22 220802" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/b2f2bd96-ed12-4eed-b842-68171c62a8e5">

---

## Installation

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
---

### Running the bot

1. **Run the .cmd file** in text-generation-webui directory (**ex: cmd_windows.bat**)
   ```
   python bot.py (args)
   ```

   **EXAMPLE LAUNCH COMMAND:**
   ```
   python bot.py --loader exllama --model airoboros-l2-13b-gpt4-2.0-GPTQ
   ```
2. Use [command](https://github.com/altoiddealer/ad_discordbot/wiki/commands) **/character** to choose a character.

---

## Usage:

### Getting responses from the bot:

* @ mention the bot

* Use [command](https://github.com/altoiddealer/ad_discordbot/wiki/commands) **/main** to set a main channel. **The bot won't need to be @ mentioned in main channels.**

* If you enclose your text in parenthesis (like this), the bot will not respond.

### Getting image responses from the bot

(**A1111 or sd-webui-forge must be running!**)

* By default, starting your request with "draw " or "generate " will trigger an image response via the Tags system (see **`dict_tags.yaml`**)

* Use **/image** command to use your own prompt with advanced options

### Getting TTS responses from the bot (Tested: alltalk_tts, coqui_tts, silero_tts, elevenlabs_tts)

1. **Run the .cmd file** in text-generation-webui directory (**ex: cmd_windows.bat**), and performing the following commands:
   
   Required for bot to join a discord voice channel:
   ```
   pip install pynacl
   ```

2. **Install your TTS extension**.

   **Follow the specific instructions for your TTS extension!!**
  
   Example instructions for **coqui_tts**:
   
   **Run the .cmd file** in text-generation-webui directory (**ex: cmd_windows.bat**), and performing the following commands:
   
   Linux / Mac:
   ```
   pip install -r extensions/coqui_tts/requirements.txt
   ```
   
   Windows:
   ```
   pip install -r extensions\coqui_tts\requirements.txt
   ```

5. Ensure that your bot has sufficient permissions to use the Voice channel and/or upload files (From your bot invite/Discord Developer portal, and your Discord server/channel settings)

6. Configure **config.py** in the section **discord** > **tts_settings**

7. If necessary, model file(s) should download on first launch of the bot.  If not, then first launch textgen-webui normally and enable the extension.

8. **Your characters can have their own settings including voices!  See example character M1nty for usage**

---

## Updating

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
