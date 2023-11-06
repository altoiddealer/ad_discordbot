discord = {
    # **BOT TOKEN IS REQUIRED**
    'TOKEN': "YOUR BOT TOKEN",
    'change_username_with_character': True, # When changing username OR avatar, you will not be able
    'change_avatar_with_character': True,   # to use /character cmd for 10 minutes (Discord policy)
    'char_name': "M1nty",                 # default bot name (**Required if change_username_with_character = False**)
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
        'reactor_enabled': False        # Requires: sd-webui-reactor
    },
    # Feature to compare the image prompt to defined trigger phrases and determine if image should be censored
    'nsfw_censoring': {
        'enabled': False,
        'mode': 0,   # 0 = blur image / 1 = block image from being generated
        'triggers': ['nude', 'erotic']
    }
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
                'imglora_name': 'SD15 Loras'},                  # If you specify an imglora_name from 'dict_imgloras.yaml', those will also be swapped in.
            {'max_filesize': 100.0,
                'width': 1024, 'height': 1024, 'enable_hr': False, 
                'imglora_name': 'SDXL Loras'}
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
    'insert_loras_in_prompt': True,     # ImgLora handling. True = insert positive_prompt after matches found in prompt / False = append all to end of prompt.

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