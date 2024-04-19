discord = {
    # **BOT TOKEN IS REQUIRED** (https://discordpy.readthedocs.io/en/stable/discord.html)
    'TOKEN': "YOUR BOT TOKEN",
    # ** Currently only 'alltalk_tts', 'coqui_tts', 'silero_tts', and 'elevenlabs_tts' have been tested. Other tts extensions may work. **
    'tts_settings': {      # REQUIRES: 'pip install pynacl' in textgen-webui venv for bot to join a voice channel
        'extension': '',    # '' = Disabled. Ex: 'alltalk_tts' (Loads automatically. Don't include in '--extensions' launch flag)
        'api_key': '',      # May be required for some tts extensions (ex: elevenlabs_tts)
        'voice_channel': 11111111111111111111,   # ** Bot will need voice & channel permissions. **
        'play_mode': 0,         # 0 = use voice channel / 1 = upload file to chat channel / 2 = both (use voice & upload file)
        'mp3_bit_rate': 128,    # If play_mode = 1 or 2, and the output (.wav) exceeds 8MB (discord limit), it will convert to .mp3 before uploading.
        'save_mode': 0          # 0 = save outputs / 1 = try deleting outputs. Note: Can't delete outputs uploaded to channel. (../extensions/coqui_tts/outputs)
    },
    'post_active_settings': {                        # when changing settings via commands (ei: /imgmodel),
        'enabled': False,                            # a copy of the settings will be posted/updated in dedicated channel
        'target_channel_id': 11111111111111111111},  # **Bot will need permission to message this channel.**
    'starboard': {                          # bot will automatically send a copy of image to starboard channel.
        'enabled': False,
        'min_reactions': 2,
        'emoji_specific': False,
        'react_emojis': ['✅', '👏'],
        'target_channel_id': 11111111111111111111
    }   # YOU CAN GET CHANNEL ID BY ENABLING DEVELOPER MODE IN YOUR DISCORD ACCOUNT
}       # THEN SIMPLY RIGHT CLICK CHANNEL > GET CHANNEL ID

# Dynamic Prompting - allows all the same syntax and usage as the SD WebUI extension 'sd-dynamic-prompts'.
dynamic_prompting_enabled = True # For usage see here: 'https://github.com/adieyal/sd-dynamic-prompts/blob/main/docs/SYNTAX.md'

sd = {
    'SD_URL': "http://127.0.0.1:7860",   # Default URL for A1111 API. Adjust if you have issues connecting.
    # Bot has integrations with the following extensions. **Only enable them if they are installed AND enabled in A1111.**
    'extensions': {
        'controlnet_enabled': False,    # Requires: sd-webui-controlnet (https://github.com/Mikubill/sd-webui-controlnet) AND configuring 'dict_cmdoptions.yaml'
        'forgecouple_enabled': False,   # Requires: sd-forge-couple (https://github.com/Haoming02/sd-forge-couple). **Only works for Forge, not A1111**
        'layerdiffuse_enabled': False,  # Requires: sd-forge-layerdiffuse (https://github.com/layerdiffusion/sd-forge-layerdiffuse). **Only works for Forge, not A1111**
        'reactor_enabled': False,       # Requires: sd-webui-reactor (https://github.com/Gourieff/sd-webui-reactor) AND adding face images to '/swap_faces/'
        'lrctl': {                      # Requires: sd-webui-loractl (https://github.com/cheald/sd-webui-loractl)
            'enabled': False,   # This extension enhances LORA processing by allowing adjustable weights during image generation. Weight is scaled linearly between step definitions.
            'min_loras': 2,     # Minimum number of loras in the prompt to trigger the following weight scaling definitions.
            'lora_1_scaling': '1.0@0,0.5@0.4',  # format is (% of existing lora weight)@(step percent),(more step definitions). These are not static weights- these are multiplied by the existing LORA weight. '1.0@0' = begin generation with existing LORA weight.
            'lora_2_scaling': '0.5@0,1.0@0.4',  # You can define as many steps as you want. ex1: '0.5@0' would just cut the LORA weight in half. ex2: '0.0@0,1.0@0.5,0.0@1.0' would look like a pyramid on the graph.
            # More scaling definitions can be added ex: 'lora_3_scaling' etc
            'additional_loras_scaling': '0.4@0.0,0.8@0.5'   # Applies to LORAs when there are more than specifically defined. If 4 loras are found in prompt, but only defined up to lora_2_scaling, then the last 2 LORAs would be scaled by this definition.
        }
    }
}