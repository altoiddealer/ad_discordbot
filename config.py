discord = {
    # **BOT TOKEN IS REQUIRED**
    'TOKEN': "YOUR BOT TOKEN",
    'change_username_with_character': True, # When changing username OR avatar, you will not be able
    'change_avatar_with_character': True,   # to use /character cmd for 10 minutes (Discord policy)
    'char_name': "M1nty",                 # default bot name (**Required if change_username_with_character = False**)
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

sd = {
    'A1111': "http://127.0.0.1:7860",   # Default URL for A1111 API. Adjust if you have issues connecting.
    # Bot has integrations with the following extensions. **Only enable them if they are installed AND enabled in A1111.**
    'extensions': {
        'controlnet_enabled': False,    # Requires: sd-webui-controlnet (https://github.com/Mikubill/sd-webui-controlnet) AND configuring 'dict_cmdoptions.yaml'
        'reactor_enabled': False,       # Requires: sd-webui-reactor (https://github.com/Gourieff/sd-webui-reactor) AND adding face images to '/swap_faces/'
        'layerdiffuse_enabled': False,  # Requires: sd-forge-layerdiffuse (https://github.com/layerdiffusion/sd-forge-layerdiffuse). **Only works for sd-webui-forge, not A1111**
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

# Imgmodels can be switched by using /imgmodels command, or enabling 'auto_change_models' (or manually updating activesettings.yaml)
imgmodels = {
    # There are 2 methods: A1111 API (simple, less customization), or 'dict_imgmodels.yaml' (high customization)
    'get_imgmodels_via_api': {      # Settings for A1111 API method
        'enabled': True,            # True = get models via A1111 API (simple, less customization). False = use 'dict_imgmodels.yaml' (high customization).
        'guess_model_res': True,    # Option to update imgmodel data using filesize and/or filtering text in filename.
        'presets': [                # 'guess_model_data' definitions. Add presets as desired, sorted in ascending order for 'max_filesize'.
            # For SD 1.5 models
            {'max_filesize': 6.0, 'exclude': ['xl'],    # filter methods: 'max_filesize' expressed in GB / 'filter' must be in filename / 'exclude' must not be in filename
                'width': 512, 'height': 512,            # width and height MUST be specified for each preset. They do not have to be 1:1 ratio (square).
                'tags': [                               # Any number of 'tags' can be applied! (*each 'tag' is a comma-separated dictionary in the 'tags' list*)
                    {'tag_preset_name': 'SD15 Tags'}    # Using a 'tag_preset_name' from 'dict_tags.yaml' is a great idea!
                ]
            },
            # For SDXL Turbo models
            {'max_filesize': 100.0, 'filter': ['turbo'],
                'width': 1024, 'height': 1024,
                'tags': [
                    {'tag_preset_name': 'SDXL Turbo Payload'},
                    {'tag_preset_name': 'SDXL Tags'}
                ]
            },
            # For SDXL 1.0 models
            {'max_filesize': 100.0, 'exclude': ['turbo'],
                'width': 1024, 'height': 1024,
                'tags': [
                    {'tag_preset_name': 'SDXL Payload'},
                    {'tag_preset_name': 'SDXL Tags'}
                ]
            }
        ]
    },

    # Omit models which include matching text.
    'exclude': ['inpaint', 'refiner', 'base'],
    # Only consider models which include matching text. '' = No filtering (loads all models)
    'filter': [''], # Example: 'xl' = likely just SDXL models. Can match multiple such as ['xl', '15']

    # Feature to periodically switch imgmodels. Behavior is affected by setting for 'get_imgmodels_via_api'
    'auto_change_imgmodels': {
        'enabled': False,
        'mode': 'random',       # 'random' = picks model at random / 'cycle' = sequential order
        'frequency': 2.0,       # How often to change models in hours. 0.5 = 30 minutes
        'channel_announce': ''  # If a channel is specified, it will announce/update as configured below. '' = Don't announce/update topic.
    }
}