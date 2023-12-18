discord = {
    # **BOT TOKEN IS REQUIRED**
    'TOKEN': "YOUR BOT TOKEN",
    'change_username_with_character': True, # When changing username OR avatar, you will not be able
    'change_avatar_with_character': True,   # to use /character cmd for 10 minutes (Discord policy)
    'char_name': "M1nty",                 # default bot name (**Required if change_username_with_character = False**)
    # ** Currently only 'coqui_tts', 'silero_tts', and 'elevenlabs_tts' have been tested. Other tts extensions may work. **
    'tts_settings': {      # REQUIRES: 'pip install pynacl' in textgen-webui venv for bot to join a voice channel
        'extension': '',    # '' = Disabled. Ex: 'coqui_tts' (Loads automatically. Don't include in '--extensions' launch flag)
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
    # Bot has integrations with the following extensions. **Only enable them if they are installed AND active in A1111.**
    'extensions': {
        'controlnet_enabled': False,    # Requires: sd-webui-controlnet AND configuring ad_discordbot/dict_cmdoptions.yaml
        'reactor_enabled': False,        # Requires: sd-webui-reactor
        'lrctl': {                     # Requires: sd-webui-loractl (https://github.com/cheald/sd-webui-loractl)
            'enabled': False,   # This extension enhances LORA processing by allowing adjustable weights during image generation. Weight is scaled linearly between step definitions.
            'min_loras': 2,     # Minimum number of loras in the prompt to trigger the following weight scaling definitions.
            'lora_1_scaling': '1.0@0,0.5@0.5',  # format is (% of existing lora weight)@(step percent),(more step definitions). These are not static weights- these are multiplied by the existing LORA weight. '1.0@0' = begin generation with existing LORA weight.
            'lora_2_scaling': '0.5@0,1.0@0.5',  # You can define as many steps as you want. ex1: '0.5@0' would just cut the LORA weight in half. ex2: '0.0@0,1.0@0.5,0.0@1.0' would look like a pyramid on the graph.
            # More scaling definitions can be added ex: 'lora_3_scaling' etc
            'additional_loras_scaling': '0.4@0.0,0.8@0.5'   # Applies to LORAs when there are more than specifically defined. If 4 loras are found in prompt, but only defined up to lora_2_scaling, then the last 2 LORAs would be scaled by this definition.
        }
    },
    # Feature to compare the image prompt to defined trigger phrases and determine if image should be censored
    'nsfw_censoring': {
        'enabled': False,
        'mode': 0,   # 0 = blur image / 1 = block image from being generated
        'triggers': ['nude', 'erotic']
    },
    # Feature to add randomness to payload parameters. For each image request, a random value from each range will be merged with your current settings (activesettings.yaml).
    'param_variances': {
        'enabled': False,
        'presets': [
            {'cfg_scale': (-2,2), 'steps': (0,10), 'hr_scale': (-0.15,0.15)}
        ]
    } 
}

# Former default behavior of textgen-webui (but not anymore). Use this setting to update 'context' + 'greeting'.
replace_char_names = {
    'replace_user': '{{user}}', # replace every instance with user's username. '' = don't replace anything
    'replace_char': '{{char}}'   # replace every instance with bot's character name. '' = don't replace anything
}

# Slips in a message about the current time before your context.
tell_bot_time = {
    'enabled': True,
    'mode': 0,                      # 0 = text requests only / 1 = image requests only / 2 = both
    'message': "It is now {}\n",    # datetime is inserted at {}
    'time_offset': 0.0 # 0 = today's date (system time). -0.5 shifts the current date to be 12 hours ago. 100000 sets the date to be 100000 days in the future.
}

# Imgmodels can be switched by using /imgmodels command, or enabling 'auto_change_models' (or manually updating activesettings.yaml)
imgmodels = {
    # There are 2 methods: A1111 API (simple, less customization), or 'dict_imgmodels.yaml' (high customization)
    'get_imgmodels_via_api': {      # Settings for A1111 API method
        'enabled': True,            # True = get models via A1111 API (simple, less customization). False = use 'dict_imgmodels.yaml' (high customization).
        'guess_model_res': True,    # Option to update payload size based on selected imgmodel filesize.
        'presets': [                # Defininitions for if 'guess_model_res' = True.  Add presets as desired, sorted in ascending order.
            {'max_filesize': 6.0,                               # 'max_filesize' expressed in GB.
                'width': 512, 'height': 512, 'enable_hr': True, # Any defined 'payload' options will be updated.
                'imgtag_name': 'SD15 ImgTags'},                  # If you specify an imgtag_name from 'dict_imgtags.yaml', those will also be swapped in.
            {'max_filesize': 100.0,
                'width': 1024, 'height': 1024, 'enable_hr': False, 
                'imgtag_name': 'SDXL ImgTags'}
        ]
    },

    # Omit models which include matching text.
    'exclude': ['inpaint', 'refiner'],
    # Only consider models which include matching text. '' = No filtering (loads all models)
    'filter': [''], # Example: 'xl' = likely just SDXL models. Can match multiple such as ['xl', '15']

    # Feature to periodically switch imgmodels. Behavior is affected by setting for 'get_imgmodels_via_api'
    'auto_change_models': {
        'enabled': False,
        'mode': 'random',       # 'random' = picks model at random / 'cycle' = sequential order
        'frequency': 1.0,       # How often to change models, in hours. 0.5 = 30 minutes
        'channel_announce': ''  # If a channel is specified, it will announce/update as configured below. '' = Don't announce/update topic.
    },

    # Options to update topic / announce new img model. When using /imgmodel, it will use the channel where the command was executed.
    'update_topic': {
        'enabled': True,
        'topic_prefix': "**Current Image Model:** ",
        'include_url': True         # URL only applies if 'get_imgmodels_via_api': False
    },
    'announce_in_chat': {
        'enabled': True,
        'reply_prefix': "**Model Loaded:** ",
        'include_url': True,        # URL only applies if 'get_imgmodels_via_api': False
        'include_params': False     # If True, lists all the current A1111 parameters
    }
}

# Feature to modify settings / prompts from trigger phrases
imgprompt_settings = {
    'trigger_search_mode': 'userllm',   # What to compare triggers against. 'user' = user prompt only / 'llm' = bot reply only / 'userllm' = search all text
    'insert_imgtags_in_prompt': True,     # ImgTag handling. True = insert positive_prompt after matches found in prompt / False = append all to end of prompt.

    # Trigger an image response with words/phrases.
    'trigger_img_gen_by_phrase': {
        'enabled': True,        # If you want phrases removed from your prompt, use dynamic_context configuration for that.
        'on_prefix_only': True, # if True, image response only occurs when prompt begins with trigger phrase.
        'triggers': ['draw', 'generate an image', 'generate a picture', 'generate a photo', 'take a photo', 'take a picture', 'take another picture', 'take a selfie', 'take another selfie', 'take a self portrait']
    },

    # Modifies payload settings if prompt includes trigger phrases
    'trigger_img_params_by_phrase': {
        'enabled': True,
        'presets': [
            {'triggers': ['vertical', 'selfie', 'self portrait'],
                'width': 896, 'height': 1152},
            {'triggers': ['landscape'],
                'width': 1152, 'height': 896}
        ]
    },

    # Modifies payload settings if prompt includes trigger phrases
    'trigger_faces_by_phrase': {
        'enabled': True,
        'presets': [
            {'triggers': ['alfred neuman', 'mad magazine guy'],
                'face_swap': 'neuman.png'}, # face_swap can be used for Reactor extension. Valid file types: .png, .jpg, .txt (containing base64 string)
            {'triggers': ['donald trump'],
                'face_swap': 'trump.txt'}
        ]
    }
}

# Feature to swap in custom character/settings when your prompt includes pre-defined trigger phrases
dynamic_context = {
    'enabled': True,
    'print_results': True, # Whether to print results to console
    # Define as many presets as you want (copy/paste, edit).
    # If multiple presets are matched in user's prompt, the highest one on this list will be prioritized.
    # Similarly for "remove_trigger_phrase", phrases are removed from left to right (put your verbose triggers first)
    'presets': [
        {'triggers': ['draw', 'generate'],
            'on_prefix_only': True,             # Only trigger if user prompt begins with the phrase.
            'swap_character': 'M1nty-SDXL',   # filename of character in /characters. Character's llmstate params will be included. '' = don't swap character.
            'remove_trigger_phrase': True,      # if True, removes the phrase from your prompt so the LLM does not see it
            'load_history': -1,     # 0 = default (all history included) / -1 = prompt excludes chat history / > 1 = llm only sees this many recent exchanges.
            'save_history': False,  # whether to save this interaction to history or not.
            'add_instruct': '{}'    # adds an instruction. {} is where your prompt is inserted.
        },
        {'triggers': ['take a photo', 'take a picture', 'take another picture'],
            'on_prefix_only': False,
            'swap_character': '',
            'remove_trigger_phrase': False,
            'load_history': 0,
            'save_history': True,
            'add_instruct': '[SYSTEM] You have been tasked with generating an image: "{}". Describe the image in vivid detail as if you were describing it to a blind person. The description in your response will be sent to an image generation API.'
        },
        {'triggers': ['selfie', 'self portrait'],
            'on_prefix_only': False,
            'swap_character': '',
            'remove_trigger_phrase': False,
            'load_history': 0,
            'save_history': True,
            'add_instruct': '[SYSTEM] You have been tasked with taking a selfie: "{}". Include your appearance, your current state of clothing, your surroundings and what you are doing right now.'
        }
    ]
}