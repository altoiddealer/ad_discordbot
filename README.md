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

   **04/19/2024:** Overhauled Img Model handling. Now "API" Method only.
   
  At first, there was only the '.YAML method' - each model required its own definition.
  
  Later, fetching models via API became a secondary option.
  
  Now, I noticed that all the improvements to API method have made the 'YAML method' obsolete:
  
    - Filter / Exclusion settings to control models that get loaded
    - Sophisticated calculations for 'Sizes' menu in '/image' command
    - Apply model settings and 'Tags' based on intelligent filter matching
    - Now, additional check for 'exact_match' if necessary.

  **Please take care migrating to this change**:

    - Fetch the new version of 'dict_imgmodels.yaml' which is now the settings panel for the API method.
    - Migrate your 'imgmodel' settings from config.py
    - Delete the whole 'imgmodels' block in config.py!

  ---
    
  
   **04/18/2024:** New Feature: Dynamic Prompting.

   Works ~~exactly~~ _**very similarly**_ to the SD WebUI extension [sd-dynamic-prompts](https://github.com/adieyal/sd-dynamic-prompts)

   **[Read up on it here!](https://github.com/altoiddealer/ad_discordbot/wiki/dynamic-prompting)**

  <img width="959" alt="Screenshot 2024-04-20 202457" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/6d1c0498-fc4d-4869-807e-904392ab44d2">

  ---


   **04/16/2024:** Enhanced Flows and "Instant Tags". Many other improvements.

    - Changed the 'Logging Level' from DEBUG to INFO - LESS SPAM!
    - Performance may be more optimized... all settings were being stored in the discord client object.
      Now, they are stored in a dedicated class object.
    - Characters can now be omitted from /character command with new parameter (see M1nty example char)
    - The feature to create tags instantly from your text has been upgraded.
      ANY tag values can be created including dictionaries, lists, sublists... anything.
    - The SD API "Guess imgmodel params" feature has much better success rate now.
    - "Flows" feature can now use variables for tag values.
    - Added a new "Flows" example in 'dict_tags.yaml'
    - Added new forge-couple param.
    - Better error handling when failure to change Img model
    - Fixed img prompt insertions from Tags sometimes creating a line break.

  <img width="1616" alt="Screenshot_2024-04-16_144136" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/141a03ae-0412-4cb0-a462-816c29eea857">

  ---
  
   **04/12/2024:** Changed user images dir. New Tags. Enhanced Image Selection.

     - All images now go into a root 'user_images' folder.
       There are no longer separate root folders for ReActor, ControlNet, Img2Img, Inpainting masks, etc.
       Users can organize their images in 'user_images' however they wish - just include path in Tags values.

     - New tags:
       - 'img2img'. Previous commit added img2img to /images cmd - now, it's also a Tag.
       - 'img2img_mask' (inpainting)...  and now also added to /image command!
       - 'send_user_image' - can send a local, non-AI generated image! Can be triggered to send an image after LLM Gen and/or after Img Gen.
       
    - Enhanced image selection:
      When processing images from tags (ReActor, ControlNet, etc etc), if the value is a directory (does not include .jpg/.png/.txt),
      the function will recursively attempt to select a random image - if no images are in the directory, it will try picking a random directory, and so on,
      until an image is found or reaches an empty directory (error).  So rather than just picking a random image from a folder, it can now pick from a random folder.
      
  <img width="1166" alt="example" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/1f4e1297-b69e-4c85-b525-d34c568d5477">
  
  ---

   **04/10/2024:** Upgraded '/image' cmd. Added Tags. Added [sd-forge-couple](https://github.com/Haoming02/sd-forge-couple) extension support.
   
    - Upgraded the /image command:
    
      - ControlNet and ReActor now only appear in the select options if enabled in config.py
      - 'img2img' has been added. If an image is attached, it will prompt for the Denoise Strength.
      - ControlNet now follows up asking for model/map if an image is attached, to simplify the main menu.
      
    - Added 'sd_output_dir' tag, so now you can control the image save location in your tag definitions.
    
    - Added extension support for SD Forge Couple.  Currently only useful for '/image' command unless you can get the LLM to reply with the correct format (I'm working on that!)

  <img width="658" alt="Screenshot_2024-04-10_140521" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/c5aa7146-92b5-43d2-87d7-8887320a45d8">

   ---
  
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

3. Clone this repository to a safe location. (**NOT into /text-generation-webui/**)
   ```
   git clone https://github.com/altoiddealer/ad_discordbot
   ```
3. **Make a copy** of the cloned repository, into **/text-generation-webui/**

   `.../text-generation-webui/ad_discordbot/`
  
4. Move **bot.py** out of subdirectory **/ad_discordbot/** -> into the directory **/text-generation-webui/**

   `/text-generation-webui/bot.py`
   
   `/text-generation-webui/ad_discordbot/(remaining files)`

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

1. **Open a cmd window** in the **/ad_discordbot/** cloned repository (See step 3 in Installation) and **git pull**
   ```
   git pull
   ```

2. **IF bot.py has changed:**
  
   Copy and Replace **bot.py** -> into the directory **/text-generation-webui/** (*overwriting old version*)

   `.../text-generation-webui/bot.py`
   
   /text-generation-webui/ad_discordbot/(remaining files)

3. **IF other files have changed ('config.py', 'dict_X.yaml', etc:**
  
   **You will need to compare changes, and either:**
   - migrate the changes into your active setup, OR
   - migrate values from your active setup into the fresh new files

   **Example**: config.py gets updated with a new feature.
   
   Solution is to either:

   * Update your existing config.py with the new feature, OR
     
   * Make a copy of the new config.py and update it with your settings.
