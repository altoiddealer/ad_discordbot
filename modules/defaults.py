from typing import Optional

class Behavior:
    def __init__(self):
        self.reply_to_itself = 0.0
        self.chance_to_reply_to_other_bots = 0.5
        self.reply_to_bots_when_addressed = 0.3
        self.only_speak_when_spoken_to = True
        self.ignore_parentheses = True
        self.go_wild_in_channel = True
        self.conversation_recency = 600
        self.user_conversations = {}
        # Chance for bot reply to be sent in chunks to discord chat
        self.chance_to_stream_reply = 0.0
        self.stream_reply_triggers = ['\n', '.']
        # Behaviors to be more like a computer program or humanlike
        self.maximum_typing_speed = -1
        self.responsiveness = 1.0
        self.msg_size_affects_delay = False
        self.max_reply_delay = 30.0
        self.response_delay_values = []   # self.response_delay_values and self.response_delay_weights
        self.response_delay_weights = []  # are calculated from the 3 settings above them via build_response_weights()
        self.text_delay_values = []       # similar to response_delays, except weights need to be made for each message
        self.current_response_delay:Optional[float] = None # If bot status is idle (not "online"), the next message will set the delay. When bot is online, resets to None.
        self.idle_range = []
        self.idle_weights = []
        # Spontaneous messaging
        self.spontaneous_msg_chance = 0.0
        self.spontaneous_msg_max_consecutive = 1
        self.spontaneous_msg_min_wait = 10.0
        self.spontaneous_msg_max_wait = 60.0
        self.spontaneous_msg_prompts = []

    def get_vars(self):
        return {k:v for k,v in vars(self).items() if not k.startswith('_')}


# Sub-classes under a main class 'Settings'
class ImgModel:
    def __init__(self):
        self.tags = []
        self.imgmodel_name = '' # label used for /imgmodel command
        self.override_settings = {}
        self.payload = {'alwayson_scripts': {}}

    def get_vars(self):
        return {k:v for k,v in vars(self).items() if not k.startswith('_')}


class LLMContext:
    def __init__(self):
        self.context = 'The following is a conversation with an AI Large Language Model. The AI has been trained to answer questions, provide recommendations, and help with decision making. The AI follows user requests. The AI thinks outside the box.'
        self.extensions = {}
        self.greeting = '' # 'How can I help you today?'
        self.name = 'AI'
        self.use_voice_channel = False
        self.bot_in_character_menu = True
        self.tags = []

    def get_vars(self):
        return {k:v for k,v in vars(self).items() if not k.startswith('_')}


class LLMState:
    def __init__(self):
        self.text = ''
        self.state = {
            # These are defaults for 'Midnight Enigma' preset
            'preset': '',
            'grammar_string': '',
            'add_bos_token': True,
            'auto_max_new_tokens': False,
            'ban_eos_token': False,
            'character_menu': '',
            'chat_generation_attempts': 1,
            'chat_prompt_size': 2048,
            'custom_stopping_strings': '',
            'custom_system_message': '',
            'custom_token_bans': '',
            'do_sample': True,
            'dry_multiplier': 0,
            'dry_base': 1.75,
            'dry_allowed_length': 2,
            'dry_sequence_breakers': '"\\n", ":", "\\"", "*"',
            'dynamic_temperature': False,
            'dynatemp_low': 1,
            'dynatemp_high': 1,
            'dynatemp_exponent': 1,
            'encoder_repetition_penalty': 1,
            'epsilon_cutoff': 0,
            'eta_cutoff': 0,
            'frequency_penalty': 0,
            'greeting': '',
            'guidance_scale': 1,
            'history': {'internal': [], 'visible': []},
            'max_new_tokens': 512,
            'max_tokens_second': 0,
            'max_updates_second': 0,
            'min_p': 0.00,
            'mirostat_eta': 0.1,
            'mirostat_mode': 0,
            'mirostat_tau': 5,
            'mode': 'chat',
            'name1': '',
            'name1_instruct': '',
            'name2': '',
            'name2_instruct': '',
            'negative_prompt': '',
            'no_repeat_ngram_size': 0,
            'penalty_alpha': 0,
            'presence_penalty': 0,
            'prompt_lookup_num_tokens': 0,
            'repetition_penalty': 1.18,
            'repetition_penalty_range': 1024,
            'sampler_priority': [],
            'seed': -1.0,
            'skip_special_tokens': True,
            'stop_at_newline': False,
            'stopping_strings': '',
            'stream': True,
            'temperature': 0.98,
            'temperature_last': False,
            'tfs': 1,
            'top_a': 0,
            'top_k': 100,
            'top_p': 0.37,
            'truncation_length': 2048,
            'turn_template': '',
            'typical_p': 1,
            'user_bio': '',
            'chat_template_str': '''{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{- name1 + ': ' + message['content'] + '\\n'-}}\n        {%- else -%}\n            {{- name2 + ': ' + message['content'] + '\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}''',
            'instruction_template_str': '''{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not ns.found -%}\n    {{- '' + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + '\\n\\n' -}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- '' + message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{-'### Instruction:\\n' + message['content'] + '\\n\\n'-}}\n        {%- else -%}\n            {{-'### Response:\\n' + message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{-'### Response:\\n'-}}\n{%- endif -%}''',
            'chat-instruct_command': '''Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>'''
            }
        self.regenerate = False
        self._continue = False

    def get_vars(self):
        return {k:v for k,v in vars(self).items() if not k.startswith('_')}
