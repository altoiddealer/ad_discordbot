# altoiddealer's Discord Bot

### Uniting [text-generation-webui](https://github.com/oobabooga/text-generation-webui) and Image Generation ([A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) / [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) / [ReForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) / [ComfyUI](https://github.com/comfyanonymous/ComfyUI) / [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) / And More!) for casual fun, creativity & professional use.

- Both **text-generation-webui** and **Image Generation** are **optional.**
- The features of both can be independently enabled/disabled in the main config file.

---

<img width="560" alt="Screenshot 2023-09-22 224716" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/5ffb5037-29d2-4a2f-9966-3ef6bd35b1f9">

---

### For support / discussion, visit the dedicated [ad_discordbot channel](https://discord.com/channels/1089972953506123937/1154970156108365944) on the TextGenWebUI Discord server.

---

**[What's new](#whats-new)   |   [Features](#features)   |   [Installation](#installation)   |   [Usage](#usage)   |   [Updating](#updating)**

---

## Accouncements:

**11/12/2025:** - New Main Feature: **Custom Commands!** ([more details](https://github.com/altoiddealer/ad_discordbot/wiki/custom-commands))

## What's new:

<details>
  <summary>click to expand</summary>

   **11/12/2025:** **Custom Slash AND Context Commands!**

   [Custom Slash Commands](https://github.com/altoiddealer/ad_discordbot/wiki/custom-commands#slash-commands) was silently some time ago, actually.

   Now, this has been expanded to also include [Custom Context Commands](https://github.com/altoiddealer/ad_discordbot/wiki/custom-commands#context-commands)

   Paired with the [API system](https://github.com/altoiddealer/ad_discordbot/wiki/apis) and [StepExecutor](https://github.com/altoiddealer/ad_discordbot/wiki/stepexecutor), these commands are extremely powerful and satisfy limitless use cases.
   
  ---

   **08/31/2025:** **Integrated STT Processing!** + **Multimodal TextGen Support!**

   Thanks to the effort of [marcos33998](https://github.com/marcos33998) STT has been added as a native feature.

   <img width="1054" height="170" alt="Screenshot 2025-08-31 082318" src="https://github.com/user-attachments/assets/f3a591ba-6bbc-414d-9ae7-d10c1b1e5a91" />

   [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui) recently added [Multimodal Support](https://github.com/oobabooga/text-generation-webui/wiki/Multimodal-Tutorial) (models which can "see images").

   The bot has been updated to include discord image attachments in the payload, when the LLM Model has multimodal capabilities.

   <img width="851" height="431" alt="Screenshot 2025-08-31 082418" src="https://github.com/user-attachments/assets/087bda88-2aa0-46d9-9b19-13f71c6aa829" />
   
  ---

   **06/19/2025:** Added **SwarmUI Support!**
   
  ---

   **06/10/2025:** Major Feature: **Universal API System!**. **ComfyUI Support!**

   **Note:** This update includes a major restructuring of **`config.yaml`** (sorry!) - this structure should remain in place for the forseeable future.

   After 3 months, I'm finally unvieling a new API system that is intended to obsolete hardcoded limitations for APIs (*only* Automatic1111 / Forge / etc.)

   Theoretically, **any** API can now be used for the main bot functions for image generation and TTS.
   
   **For the time being**, an integrated Text Generation WebUI install is still the only support TextGen for "main functions".  Other TTS APIs (including TGWUI) CAN be defined and used via the Tags system.

   This is a very well thought out, flexible, deeply integrated, internally-highly-complex system that is absolutely overkill for a project named "altoiddealer's discordbot".

   Please review the Wiki, comments in the config files, examples folder - or [ask me questions directly in discord](https://discord.com/channels/1089972953506123937/1154970156108365944).
   
  ---

   **04/04/2025:** Major rewrite of core install logic (can now be installed as a Standalone!) **Update Wizards!**

   This bot has always required text-generation-webui to function.  Now, the bot may installed as a standalone, or optionally with TGWUI integration.

   This update also replaces the Update scripts with Update Wizards.  From the wizard it is possible to change between Standalone and TGWUI integration.

   <img width="662" alt="Screenshot 2025-04-04 215229" src="https://github.com/user-attachments/assets/03a20ac8-db09-4fbf-91b4-8376d8bc059e" />

   <img width="609" alt="Screenshot 2025-04-04 215515" src="https://github.com/user-attachments/assets/4593ecdb-3e44-487e-a213-884fb627af80" />
   
  ---

   **08/19/2024:** Major User Settings Enhancement: **Per-Server Settings!**

   There is now an option to enable "Per-Server" settings:

   - Settings for new servers will be initialized as a copy of the "main" settings
   - All settings are managed separately (*EXCLUDING* `config.yaml`)
   - There are sub-options for "per-character settings" and "per-server ImdModels"
   - Note that `basesettings.yaml` applies to ALL settings.
   
  ---

   **08/12/2024:** New **/prompt** command! Enhanced **Post Active Settings** feature!

   There is now a **/prompt** command to add some advanced options to your message request.

   Also, the **Post Active Settings** feature received a nice overhaul and is definitely worth trying out!

   Requires using yet another new command, **/set_server_settings_channel**, in each server that you would like to share settings in.

   This feature will automatically post a copy of your settings to the channel (*EXCLUDING* `config.yaml`)

   <img width="689" alt="Screenshot 2024-08-12 221633" src="https://github.com/user-attachments/assets/ba4be59f-69d9-494b-86d4-f2a7b010df33">
   
  ---

   **08/02/2024:** New Behavior - Streaming Responses!

   - Base Behavior / Character Behavior now have option to stream responses!

   - You no longer have to wait for the entire message to generate!

   - Actually works with all the other crazy features!
   
  ---

   **07/24/2024:** New [Behaviors](https://github.com/altoiddealer/ad_discordbot/wiki/behaviors) to bring your characters to life!

   Some new settings appeared recently... which were slightly buggy.  Today, I announce that they are working very well!

   These new behaviors can make your character more humanistic (or just be like a computer program, by default)

    - Character can go idle, taking more time to respond
    - Pause to "read messages" before writing
    - Simulate typing at different speeds (does not actually generate text slower)
    - *DOES NOT* throttle the bot's performance! All tasks running full speed on the backend!
    - These behaviors will continue to improve in the coming days/weeks

  ![351922166-c98883f4-143e-40af-b641-544d23c7452e](https://github.com/user-attachments/assets/da26fc19-3ba1-4c90-a15a-ea09e3efcac2)

  ---

   **07/02/2024:** New Behaviors "Spontaneous Messaging"

    - Character can now be silently prompted to say something after inactivity
    - A few settings control the behavior
    - It can also serve as an "Auto-prompter" generating unlimited responses

  ---

   **06/18/2024:** Per-Server Voice Channels (TTS)

    - New command to set voice channels: /set_server_voice_channel
    - The context menu commands (regen / continue / edit history) vastly improved
    - The bot now reacts to messages to identify their status (hidden, etc)

  ---

   **06/14/2024:** Direct Messages Support. Image model changes.

    - The bot now supports Direct Messages, which have their own history!
    - Most commands are disabled in DMs.
    - Direct messaging can be disabled via config.yaml
    - Img model "override settings" (checkpoint, vae, clip skip) are no longer saved.

  ---

  **06/11/2024:** More Context Menu Commands!

  <img width="285" alt="Screenshot 2024-06-11 155926" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/92d1ef4f-56c2-46aa-be9e-5ce69284900e">

  ---

   **06/06/2024:** Context commands are now AMAZING

    - "Continue" and "Regen" have been revised to work AMAZINGLY well!
    - New "Edit History" command has been added, that also works AMAZINGLY well!
    - You'll have to try these for yourself to see just how perfect they work.

  Right click any message to invoke these context commands

  <img width="299" alt="Screenshot 2024-06-06 155152" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/bae6b141-dc70-4ad4-a533-5f239e9da0c6">

  ---

   **06/04/2024:** Launchers! Updaters! New History Manager!

   Shoutout to @Artificiangel who coded an **amazing** new History Manager. Great new features are coming!

    - We now have launcher and updater scripts (...look familiar?)
    - New CMD_FLAGS.txt to add your custom launch flags (...look familiar?)
    - bot.py no longer has to be moved! Correct location is now in main 'ad_discordbot' folder
    - New history manager is much more flexible and unlocks new possibilities

  ---

   **05/28/2024:** Per-Channel History!

    - New setting 'per_channel_history' enables all channels to have their own chat history.
    - Custom logging format puts chat history, server name, and channel name under each channel.id key.
    - A utility .bat file is now included to split these custom logs into normal logs, if needed.
    - New '/announce/ command allows channels to be assigned as Announcement channels.
      Announcement channels will receive Model / Character change announcements
      instead of interaction channels.

  ---

   **05/22/2024:** Big Update - Easier to Update Moving Forward

   Quick shoutout to @Artificiangel who has recently joined development and made **stunning contributions**.

    - The directory '/internal/' which contains persistent settings (not intended to be modified by users)
      is no longer part of the bot package.  Instead, '/internal/' and its contents are created dynamically if missing.
    - User settings are now present in a '/settings_templates/' which will be automatically copied into the root directory,
      if not done manually by users.  This allows the bot to be easily updated without conflicts due to modified files.

  ---

   **05/16/2024:** Significant Changes to File Structure

    - The main bot script has grown massive, so it is now split to modules (new '/modules/' subdirectory)
    - activesettings.yaml is now in an '/internal/' subdirectory. Your settings will migrate automatically.
    - 'bot.db' has been superceded by a 'database.yaml' file. Your settings will migrate automatically.
    - Note: Changes to 'dict_base_settings.yaml' and 'activesettings.yaml' are just comment updates

  ---

   **04/29/2024:** Huge Improvement to ControlNet in /image Command

    - Options are now dynamically filtered and populated using the '/controlnet/control_type' API endpoint
    - Sets almost identical default options as in SD WebUI interface
    - This took a lot of time and effort, please try it out!

  <img width="690" alt="Screenshot 2024-04-29 145440" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/bd399667-8d1e-4dd8-9b36-074b9c4a3e54">

</details>

## Features:

- **Chat History for each channel!**
  - Each channel has its chat history maintained seperately and cleanly.
  - History is gracefully loaded for all channels on startup.
  - **`/reset_conversation`** will reset history in interaction channel only.

- **Robust ["Tags" system](https://github.com/altoiddealer/ad_discordbot/wiki/tags) to manipulate bot behavior persistently or via trigger phrases, including:**
  - Trigger Text and/or Image response
  - Image censoring settings (None / Spoiler / Block)
  - Powerful [ControlNet](https://github.com/Mikubill/sd-webui-controlnet), [ReActor](https://github.com/Gourieff/sd-webui-reactor), [Forge Couple](https://github.com/Haoming02/sd-forge-couple), and [layerdiffuse](https://github.com/layerdiffusion/sd-forge-layerdiffuse) integration
  - Automatically apply [loractl scaling](https://github.com/cheald/sd-webui-loractl) (Currently ReForge/A1111 only)
  - Swapping / Changing LLM characters (ei: "draw... " can trigger character tailored for image prompting)
  - Swapping / Changing LLM models and Img models
  - Modifying LLM state (temperature, repetition penalty, etc) and image gen settings (width, height, cfg scale, etc)
  - Keep things spicy by factoring random variations to LLM state / img gen settings
  - Modifying the user's prompt and LLM's reply with delete/insert/replace text, img LORAs, etc
  - Manipulate LLM history (Suppress history, limit history, save or do-no-save reply, etc)
  - **New Feature:** Flexible, Instant-Tags by using syntax [[key:value]] or [[key1:value1 | key2:value2 ]]
  - New features being added frequently and easily due to the framework of the ["Tags" system](https://github.com/altoiddealer/ad_discordbot/wiki/tags)

- **TTS Support!**
  - Any TTS software with an API is supported.
  - TGWUI TTS extensions are also supported:  [alltalk_tts](https://github.com/erew123/alltalk_tts), coqui_tts, silero_tts, [edge_tts](https://github.com/Unorthodox-oddball/text-generation-webui-edge-tts) and elevenlabs_tts
  - Bot can speak on Voice channel, upload copy of audio file, or both!
  - Per-character TTS settings! Give each character a unique voice!

- **Sophisticated function to send text responses over Discord's 2,000 character limit**
  - "chunks" messages by looking back to nearest line break or sentence completion.
  - Preserves syntax between chunks such as **bold**, *italic*, and even `code formatting`

- **[Commands](https://github.com/altoiddealer/ad_discordbot/wiki/commands)!**
  - **/character** - Change character
  - **/reset_conversation** - Starts a new conversation in the current channel.
  - **/image** - Allows more controlled image prompting (positive prompt, neg prompt, Size settings, **ControlNet**, **ReActor**)
  - **/speak** - Bot can speak any text, using any voices (including user attach .mp3 or .wav for alltalk_tts)!
  - **/imgmodel** - Change image model along with any custom settings for it
  - **/llmmodel** - Change LLM model
  - **/set_X_for_server** - A number of commands to assign a specific channel in your servers (Voice channel for TTS, Announcement channel, Settings channel, Starboard channel, etc).
  - Plus many more!

- **Dynamic settings handling:**
  - Core bot settings managed in **`config.yaml`** (bot behavior, discord features, extensions, etc.)
  - The ["Tags" system](https://github.com/altoiddealer/ad_discordbot/wiki/tags) is configured in **`dict_tags.yaml`** (global Tags, default Tag params, Tag presets, etc.)
  - Foundational layer of user settings configured in **`base_settings.yaml`**.
  - Character files can include custom Tags, TTS settings, LLM state parameters, and special behaviors, which prioritize over basesettings.
  - Custom Image models settings defined in **`dict_imgmodels.yaml`** (Tags, payload params) which prioritize over basesettings.
  - All user settings commit to **`internal/activesettings.yaml`**, which serves as a dashboard to view the bot's current state.

- **Automatic Img model changing:**
  - Adjustable duration and mode (random / cycle)
  - Smart filters and settings to auto-update relavent settings (SD1.5, SDXL, Turbo, etc)
 
- **Powerful Context Menu commands** (right-click on a message):

  <img width="285" alt="338729061-92d1ef4f-56c2-46aa-be9e-5ce69284900e" src="https://github.com/user-attachments/assets/8964bf42-e7b5-43b1-aed5-5ebafcc79ee2">
  
  - Can target **any** message in chat history (not only the most recent one!), making **Continue** & **Regenerate** more powerful than native TGWUI.
  - Custom **Regenerate** methods:
    - **regenerate replace** works as everyone is familiar with (generates a new reply and replaces the original).
    - **regenerate create** makes a new generation while "hiding" the previous response. These can be easily toggled with **toggle as hidden** command, so you can choose your favorite reply!
  - **edit history** allows you to edit any message in history
  - **toggle as hidden** will hide or reveal the user/bot reply exchange in history.
  - The above are restricted to your own / the bot's messages (can't target other users' messages).

- **All tasks queue up and process elagently - go ahead, spam it with requests!**

- **Built in Starboard feature**

- **Feature to post current settings in a dedicated channel**

- **ALWAYS MORE TO COME**

---

<img width="817" alt="Screenshot 2023-09-22 220802" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/b2f2bd96-ed12-4eed-b842-68171c62a8e5">

---

## Installation

1. **OPTIONAL FOR TGWUI INTEGRATION:** Install oobabooga's [text-generation-webui](https://github.com/oobabooga/text-generation-webui)

2. **[Create a Discord bot account](https://discordpy.readthedocs.io/en/stable/discord.html), invite it to your server, and note its authentication token**.

   !!! **IMPORTANT: You must allow your bot to have "Privileged Intents" for "`MESSAGE_CONTENT`"** !!!
   
   !!! **[Enabling Privileged Intents](https://discord.com/developers/docs/topics/gateway#enabling-privileged-intents)** !!!

4. Clone this repository anywhere.
   
   **FOR TGWUI INTEGRATION:** clone into **/text-generation-webui/**
   
   ```
   git clone https://github.com/altoiddealer/ad_discordbot
   ```

5. **Run the launcher** for your OS (**ex: start_windows.bat**)

6. **Enter your bot token (from Step 2) into the CMD window**

   <img width="559" alt="Screenshot 2024-05-25 100216" src="https://github.com/altoiddealer/ad_discordbot/assets/1613484/ff5207ec-954e-4e71-975d-ce171c1c42c8">

7. **The bot should now be up and running!**
  
   If a Welcome message does not appear in your main channel, you can use `/helpmenu` to see it.

   A number of user settings files will appear (copied in from **`/user_settings/`**) where you can customize the bot.

   `config.yaml` , `dict_api_settings.yaml`, `dict_base_settings.yaml` , `dict_cmdoptions.yaml` , `dict_imgmodels.yaml` , `dict_tags.yaml`

---

### Running the bot

1. **Run the launcher** for your OS (**ex: start_windows.bat**)

  - **Optionally** add launch flags to **CMD_FLAGS.txt**

   **EXAMPLE CMD Flags:**
   ```
    --loader exllama --model airoboros-l2-13b-gpt4-2.0-GPTQ
   ```

2. In Discord UI, use [command](https://github.com/altoiddealer/ad_discordbot/wiki/commands) **/character** to choose a character.

---

## Usage:

### Getting responses from the bot:

* @ mention the bot

* Use [command](https://github.com/altoiddealer/ad_discordbot/wiki/commands) **/main** to set a main channel. **The bot won't need to be @ mentioned in main channels.**

* If you enclose your text in parenthesis (like this), the bot will not respond.

### Getting image responses from the bot

**An "ImgGen" API must be configured and enabled - See [APIs Configuration](https://github.com/altoiddealer/ad_discordbot/wiki/apis#configuration)**

* By default, starting your request with "draw " or "generate " will trigger an image response via the Tags system (see **`dict_tags.yaml`**)

* Use **/image** command to use your own prompt with advanced options

### Getting TTS responses from the bot

1. a. **A "TTSGen" API can be used if configured and enabled - See [APIs Configuration](https://github.com/altoiddealer/ad_discordbot/wiki/apis#configuration)**

   b. **Alternatively**: If bot is TGWUI enabled, TGWUI TTS extensions can also be used (Tested: alltalk_tts, coqui_tts, silero_tts, elevenlabs_tts)

      <details>
      <summary>TGWUI TTS Extension Usage Instructions</summary>

      **Install your TTS extension**.

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

      If necessary, model file(s) should download on first launch of the bot.  If not, then first launch textgen-webui normally and enable the extension.

      </details>

2. Ensure that your bot has sufficient permissions to access the Voice channel and/or upload files (From your bot invite/Discord Developer portal, and your Discord server/channel settings)

3. Configure the **ttsgen** section in **config.yaml**

4. **Your characters can have their own settings including voices!  See example character M1nty for usage**

---

## Updating

1. **Run the update-wizard** for your OS (**ex: update_wizard_windows.bat**)

<img width="609" alt="Screenshot 2025-04-04 215515" src="https://github.com/user-attachments/assets/f33ff332-babd-4508-8b73-58aca743b5c3" />
