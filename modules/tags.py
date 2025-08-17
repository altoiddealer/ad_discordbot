# from datetime import timedelta
# import time
import copy
from itertools import product
import random
import re
import traceback
from modules.database import BaseFileMemory
from modules.utils_discord import get_user_ctx_inter, is_direct_message
from modules.typing import TAG, TAG_LIST, TAG_LIST_DICT, CtxInteraction, Union
from modules.utils_shared import shared_path, flows_queue, patterns
from typing import Any, Optional, Tuple, Callable
from modules.utils_misc import valueparser

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

class PersistentTags():
    def __init__(self) -> None:
        # {discord.channel.id: [(repeat, 'tag name'), (repeat, 'tag name')], discord.channel.id: ...}
        self.llm_ptags: dict[int, list[tuple[int, str]]] = {}
        self.img_ptags: dict[int, list[tuple[int, str]]] = {}

    def get_ptags_for(self, match_phase:str, ictx:CtxInteraction|None=None):
        ptag_repeats:list = []
        ptag_names:list = []
        # get and return applicable persistent tags
        phase_ptags:dict = getattr(self, f'{match_phase}_ptags', {})
        tupled_ptags = phase_ptags.get(ictx.channel.id) if ictx else None
        if tupled_ptags:
            for tupled_ptag in tupled_ptags:
                repeat, ptag_name = tupled_ptag
                ptag_repeats.append(repeat)
                ptag_names.append(ptag_name)
        return tupled_ptags, ptag_repeats, ptag_names
    
    def append_tag_name_to(self, match_phase:str, channel_id:int, repeat:int, tag_name:str):
        phase_ptags:dict = getattr(self, f'{match_phase}_ptags', {})
        channel_ptags:list = phase_ptags.setdefault(channel_id, [])
        channel_ptags.append( (repeat, tag_name) )

persistent_tags = PersistentTags()

class BaseTags(BaseFileMemory):
    def __init__(self) -> None:
        self._global_tag_keys: dict
        self.tags: list
        self._tag_presets: list
        super().__init__(shared_path.tags, version=1, missing_okay=True)
        self.tags = self.update_tags(self.tags)

    def load_defaults(self, data: dict):
        self._global_tag_keys = data.pop('global_tag_keys', {})
        self.tags = data.pop('base_tags', [])
        self._tag_presets = data.pop('tag_presets', [])

    def expand_triggers(self, all_tags:list) -> list:
        def expand_value(value:str) -> str:
            # Split the value on commas
            parts = value.split(',')
            expanded_values = []
            for part in parts:
                # Check if the part contains curly brackets
                if '{' in part and '}' in part:
                    # Use regular expression to find all curly bracket groups
                    group_matches = patterns.in_curly_brackets.findall(part)
                    permutations = list(product(*[group_match.split('|') for group_match in group_matches]))
                    # Replace each curly bracket group with permutations
                    for perm in permutations:
                        expanded_part = part
                        for part_match in group_matches:
                            expanded_part = expanded_part.replace('{' + part_match + '}', perm[group_matches.index(part_match)], 1)
                        expanded_values.append(expanded_part)
                else:
                    expanded_values.append(part)
            return ','.join(expanded_values)

        try:
            for tag in all_tags:
                for key in tag:
                    if key.startswith('trigger'):
                        tag[key] = expand_value(tag[key])

        except Exception as e:
            log.error(f"Error expanding tags: {e}")

        return all_tags

    # Unpack tag presets and add global tag keys
    def update_tags(self, tags:Optional[list]) -> list:
        if not isinstance(tags, list):
            log.warning('''One or more "tags" are improperly formatted. Please ensure each tag is formatted as a list item designated with a hyphen (-)''')
            return []
        try:
            updated_tags = []
            for tag in tags:
                if 'tag_preset_name' in tag:
                    # Find matching tag preset in tag_presets
                    for preset in self._tag_presets:
                        if 'tag_preset_name' in preset and preset['tag_preset_name'] == tag['tag_preset_name']:
                            # Merge corresponding tag presets
                            updated_tags.extend(preset.get('tags', []))
                            tag.pop('tag_preset_name', None)
                            break
                if tag:
                    updated_tags.append(tag)
            # Add global tag keys to each tag item
            for tag in updated_tags:
                for key, value in self._global_tag_keys.items():
                    if key not in tag:
                        tag[key] = value
            updated_tags = self.expand_triggers(updated_tags) # expand any simplified trigger phrases
            return updated_tags

        except Exception as e:
            log.error(f"Error loading tag presets: {e}")
            return tags

base_tags = BaseTags()

class Tags():
    def __init__(self, ictx:CtxInteraction|None=None):
        self.ictx = ictx
        self.user = get_user_ctx_inter(self.ictx) if self.ictx else None # Union[discord.User, discord.Member]
        self.tags_initialized = False
        # tags lists
        self.char_tags: TAG_LIST = []
        self.imgmodel_tags: TAG_LIST = []
        self.matches:list = []
        self.matches_names:list = []
        self.unmatched = {'user': [], 'llm': [], 'userllm': []}
        self.tag_trumps:set = set([])
        self.llm_censor_tags: TAG_LIST = []
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        Instances of Tags() are initialized with empty defaults.

        The values will populate automatically when calling 'match_tags()',
        or they may be initialized on demand using 'init()'
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    async def init_tags(self, text:str, settings:dict) -> str:
        try:
            self.tags_initialized = True
            base_tags_obj = BaseTags()
            base_tags: TAG_LIST      = getattr(base_tags_obj, "tags", [])
            char_tags: TAG_LIST      = settings['llmcontext'].get('tags', []) # character specific tags
            imgmodel_tags: TAG_LIST  = settings['imgmodel'].get('tags', []) # imgmodel specific tags
            tags_from_text           = self.get_tags_from_text(text)
            flow_step_tags: TAG_LIST = []
            if flows_queue.qsize() > 0:
                flow_step_tags = [await flows_queue.get()]
            # merge tags to one list (Priority is important!!)
            all_tags: TAG_LIST = tags_from_text + flow_step_tags + char_tags + imgmodel_tags + base_tags
            self.sort_tags(all_tags) # sort tags into phases (user / llm / userllm)
        except Exception as e:
            log.error(f"Error getting tags: {e}")

    def untuple(self, tag_value:tuple|TAG) -> TAG:
        if isinstance(tag_value, tuple):
            return tag_value[0]
        return tag_value
    
    def get_name_for(self, tag:TAG) -> str:
        return tag.get('name', '')
    
    def get_print_for(self, str_or_tag:str|TAG) -> str:
        tag_name = str_or_tag if isinstance(str_or_tag, str) else self.get_name_for(str_or_tag)
        return f'tag "{tag_name}"' if tag_name else 'tag'

    def get_name_print_for(self, tag:TAG) -> tuple[str, str]:
        """ if tag has name: returns 'name', 'tag "name"' ; else: returns '', 'tag' """
        tag_name = self.get_name_for(tag)
        tag_print = self.get_print_for(tag_name)
        return tag_name, tag_print
    

    # Check for discord conditions
    def pass_discord_check(self, key, value) -> bool:
        # Must have an interaction to check
        if not key in ['for_guild_ids_only', 'for_channel_ids_only'] or self.ictx is None: # TODO 'for_role_ids_only' (requires 'members' intent)
            return True

        # Adjust values
        if isinstance(value, int):
            id_values_list = [value]
        elif isinstance(value, list):
            id_values_list = value
        else:
            log.error(f"Error: Value for '{key}' in tags must be an integer or a list of integers (a valid discord ID, or [list, of, IDs]).")
            return False

        # check each value against the interaction
        for id_value in id_values_list:
            if key == 'for_guild_ids_only' and not is_direct_message(self.ictx) and id_value != self.ictx.guild.id:
                return False
            elif key == 'for_channel_ids_only' and id_value != self.ictx.channel.id:
                return False
            # TODO 'for_role_ids_only' (requires 'members' intent)
            # elif key == 'for_role_ids_only' and self.user and not is_direct_message(self.ictx):
            #     member = self.ictx.guild.get_member(self.user.id)
            #     role_ids = [role.id for role in member.roles]
            #     if id_value not in role_ids:
            #         return False
        return True

    # Check if 'random' key exists and handle its value
    def pass_random_check(self, key, value) -> bool:
        if not isinstance(value, (int, float)):
            log.error("Error: Value for 'random' in tags should be float value (ex: 0.8).")
            return False
        if not random.random() < value:
            return False
        return True

    def sort_tags(self, all_tags: TAG_LIST):
        for tag in all_tags:
            # check initial conditions
            for key, value in tag.items():
                if key == 'random' and not self.pass_random_check(key, value):
                    break # Skip this tag
                if not self.pass_discord_check(key, value):
                    break # Skip this tag
            # sort tags that passed condition for further processing
            else:
                search_mode = tag.get('search_mode', 'userllm')  # Default to 'userllm' if 'search_mode' is not present
                if 'llm_censoring' in tag and search_mode != 'user' and tag['llm_censoring'] == True:
                    self.llm_censor_tags.append(tag)
                if search_mode in self.unmatched:
                    self.unmatched[search_mode].append({k: v for k, v in tag.items() if k != 'search_mode'})
                else:
                    log.warning(f"Ignoring unknown search_mode: {search_mode}")


    def get_tags_from_text(self, text: str) -> Tuple[str, TAG_LIST]:
        """
        Extracts embedded tags from the input text and parses them into structured dictionaries.

        Returns:
            - Modified text with tags removed
            - List of parsed tag dictionaries
        """
        tags_from_text = []
        matches = patterns.instant_tags.findall(text)

        # Remove tags from text / prompt
        self.text = patterns.instant_tags.sub('', text).strip()
        self.prompt = patterns.instant_tags.sub('', text).strip()

        for match in matches:
            tag_dict = {}
            tag_pairs = match.split('|')

            for pair in tag_pairs:
                key, value = self._parse_key_value(pair)
                if key in ('run_workflow', 'call_api', 'for_guild_ids_only', 'for_channel_ids_only'):
                    log.warning(f'[TAGS] Ignored "tags from text" (AKA Instant Tags) for disallowed tag type: {key}')
                    continue
                if key is not None:
                    tag_dict[key] = value

            if tag_dict:
                tags_from_text.append(tag_dict)

        return tags_from_text

    def _parse_key_value(self, pair: str) -> Tuple[str, Any]:
        """Splits 'key:value' string and parses the value using the ValueParser."""
        try:
            key_value = pair.split(':', 1)
            if len(key_value) != 2:
                return None, None
            key = key_value[0].strip()
            value_str = key_value[1].strip()
            value = valueparser.parse_value(value_str)
            return key, value
        except Exception as e:
            print(f"Error parsing key-value pair '{pair}': {e}")
            return None, None

    async def match_img_tags(self, img_prompt:str, settings:dict):
        try:
            # Unmatch any previously matched tags which try to insert text into the img_prompt
            matches_:TAG_LIST = self.matches # type: ignore
            for tag in matches_[:]:  # Iterate over a copy of the list
                # extract text insertion key pairs from previously matched tags
                if tag.get('imgtag_matched_early'):
                    new_tag = {}
                    tag_copy = copy.copy(tag)
                    for key, value in tag_copy.items(): # Iterate over a copy of the tag
                        if (key in ["matched_trigger", "imgtag_matched_early", "case_sensitive", "on_prefix_only", "search_mode", "img_text_joining", "phase"]
                            or key.startswith(('trigger', 'positive_prompt', 'negative_prompt'))):
                            new_tag[key] = value
                            if not key == 'phase':
                                del tag[key] # Remove the key from the original tag
                    self.unmatched['userllm'].append(new_tag) # append to unmatched list
                    # Remove tag items from original list that became an empty list
                    if not tag:
                        self.matches.remove(tag)
            # match tags for 'img' phase.
            await self.match_tags(img_prompt, settings, phase='img')
            # Rematch any previously matched tags that failed to match text in img_prompt
            for tag in self.unmatched['userllm'][:]:  # Iterate over a copy of the list
                if tag.get('imgtag_matched_early') and tag.get('imgtag_uninserted'):
                    self.matches.append(tag)
                    self.unmatched['userllm'].remove(tag)

        except Exception as e:
            log.error(f"Error matching tags for img phase: {e}")

    def process_tag_insertions(self, prompt:str) -> str:
        try:
            # iterate over a copy of the matches, preserving the structure of the original matches list
            tuple_matches = copy.deepcopy(self.matches) # type: ignore
            tuple_matches: list[tuple[dict, int, int]] = [item for item in tuple_matches if isinstance(item, tuple)]  # Filter out only tuples
            tuple_matches.sort(key=lambda x: -x[1])  # Sort the tuple matches in reverse order by their second element (start index)
            for item in tuple_matches:
                tag, start, end = item # unpack tuple
                phase = tag.get('phase', 'user')
                if phase == 'llm':
                    insert_text = tag.pop('insert_text', None)
                    insert_method = tag.pop('insert_text_method', 'after')  # Default to 'after'
                    join = tag.pop('text_joining', ' ')
                else:
                    insert_text = tag.get('positive_prompt', None)
                    insert_method = tag.pop('positive_prompt_method', 'after')  # Default to 'after'
                    join = tag.pop('img_text_joining', ' ')
                if insert_text is None:
                    log.error(f"Error processing matched tag {item}. Skipping this tag.")
                else:
                    if insert_method == 'replace':
                        if insert_text == '':
                            prompt = prompt[:start] + prompt[end:].lstrip()
                        else:
                            prompt = prompt[:start] + insert_text + prompt[end:]
                    elif insert_method == 'after':
                        prompt = prompt[:end] + join + insert_text + prompt[end:]
                    elif insert_method == 'before':
                        prompt = prompt[:start] + insert_text + join + prompt[start:]
            # clean up the original matches list
            updated_matches = []
            for tag in self.matches:
                tag_dict:TAG = self.untuple(tag)
                phase = tag_dict.get('phase', 'user')
                if phase == 'llm':
                    tag_dict.pop('insert_text', None)
                    tag_dict.pop('insert_text_method', None)
                    tag_dict.pop('text_joining', None)
                else:
                    tag_dict.pop('img_text_joining', None)
                    tag_dict.pop('positive_prompt_method', None)
                updated_matches.append(tag_dict)
            self.matches = updated_matches
            return prompt
        except Exception as e:
            log.error(f"Error processing LLM prompt tags: {e}")
            return prompt

    def check_only_with_tags_for(self, tag_dict:TAG) -> bool:
        tag_name, _ = self.get_name_print_for(tag_dict)
        only_with_tags = tag_dict.get('only_with_tags', [])
        # check condition for 'only_with_tags'
        if only_with_tags:
            if isinstance(only_with_tags, str):
                only_with_tags = [only_with_tags]
            if not any(trigger_tag in self.matches_names for trigger_tag in only_with_tags):
                # remove this tag name from matched names list
                if tag_name and tag_name in self.matches_names:
                    self.matches_names.remove(tag_name)
                return False

        return True

    def process_tag_trumps(self, matches:list) -> TAG_LIST:
        try:
            passed_only_with_tags_check = []
            passed_trumps_check = []
            passed_all_checks = []

            # collect 'trump' values while applying logic for any 'only_with_tags'
            for tag in matches:
                tag_dict:TAG = self.untuple(tag)
                only_with_tags_pass = self.check_only_with_tags_for(tag_dict)
                if not only_with_tags_pass:
                    continue
                # Collect all 'trump' parameters
                if 'trumps' in tag_dict:
                    for param in tag_dict['trumps'].split(','):
                        stripped_param = param.strip().lower()
                        self.tag_trumps.update([stripped_param])

                passed_only_with_tags_check.append(tag)

            # Iterate over all tags that passed trigger tags check, and remove 'trumped' tags
            for tag in passed_only_with_tags_check:
                tag_dict:TAG = self.untuple(tag)
                tag_name, tag_print = self.get_name_print_for(tag_dict)
                # Collect all trigger phrases from all trigger keys
                all_triggers = []
                for key in tag_dict:
                    if key.startswith('trigger'):
                        triggers = [trigger.strip().lower() for trigger in tag_dict[key].split(',')]
                        all_triggers.extend(triggers)
                # Check if any trigger is in the trump parameters set
                trumped_trigger = None
                for trigger in all_triggers:
                    if trigger in self.tag_trumps:
                        trumped_trigger = trigger
                        break
                if trumped_trigger:
                    log.info(f'''[TAGS] A {tag_print} was trumped by another tag for phrase: "{trumped_trigger}".''')
                    # remove tag name from matched names list
                    if tag_name and tag_name in self.matches_names:
                        self.matches_names.remove(tag_name)
                    continue
                passed_trumps_check.append(tag)

            # Finally, confirm no 'only_with_tags' were trumped
            for tag in passed_trumps_check:
                tag_dict:TAG = self.untuple(tag)
                only_with_tags_pass = self.check_only_with_tags_for(tag_dict)
                if not only_with_tags_pass:
                    continue
                passed_all_checks.append(tag)

            return passed_all_checks

        except Exception as e:
            log.error(f"Error processing matched tags: {e}")
            return matches

    async def match_tags(self, search_text:str, settings:dict, phase:str='llm'):
        if not self.tags_initialized:
            await self.init_tags(search_text, settings)
        try:
            # Remove 'llm' tags if pre-LLM phase, to be added back to unmatched tags list at the end of function
            if phase == 'llm':
                llm_tags = self.unmatched.pop('llm', []) if 'user' in self.unmatched else [] # type: ignore

            # Gather any applicable "persistent tags"
            updated_ptags = []
            channel_ptags, ptag_repeats, ptag_names = persistent_tags.get_ptags_for(phase, self.ictx)

            # Iterate over copies of tags lists and apply tag matching logic
            updated_matches:TAG_LIST = list(copy.deepcopy(self.matches)) # type: ignore
            updated_unmatched:TAG_LIST_DICT = dict(copy.deepcopy(self.unmatched)) # type: ignore
            for list_name, unmatched_list in self.unmatched.items(): # type: ignore
                unmatched_list: TAG_LIST

                for tag in unmatched_list:

                    # Collect any 'name' parameter in tag
                    tag_name, tag_print = self.get_name_print_for(tag)

                    def match(tag_copy:dict|None=None):
                        # Remove tag from unmatched list
                        updated_unmatched[list_name].remove(tag)
                        # Collect name
                        if tag_name:
                            self.matches_names.append(tag_name)
                        # Capture match phase and add to matches list
                        if tag_copy is None:
                            tag['phase'] = phase
                            updated_matches.append(tag)
                        else:
                            tag_copy['phase'] = phase
                            updated_matches.append(tag_copy)

                    def is_persistent():
                        if tag_name in ptag_names:
                            tag_index = ptag_names.index(tag_name) # get the index
                            repeat = ptag_repeats[tag_index] # identity the corresponding repeat value
                            repeat -= 1                      # decrement repeat
                            log.info(f"[TAGS] Persistent {tag_print} was automatically applied.")
                            if repeat != 0:
                                log.info(f"[TAGS] Remaining persistency: {repeat if repeat >= 0 else 'Forever'}")
                                updated_ptags.append( (repeat, tag_name) ) # continue persisting the tag
                            return True
                        return False

                    # Collect list of all key pairs in tag dict that begin with 'trigger'
                    trigger_keys = [key for key in tag if key.startswith('trigger')]

                    # Match tags without trigger keys
                    if not trigger_keys:
                        match()
                        continue

                    # Match tags previously triggered with persistency
                    if is_persistent():
                        tag_copy = copy.deepcopy(tag)
                        tag_copy.pop('persist') # don't re-trigger persistency
                        match(tag_copy)
                        continue

                    case_sensitive = tag.get('case_sensitive', False)
                    all_triggers_matched = True
                    trigger_match = None
                    matched_trigger = None

                    # iterate over trigger keys in reverse so first trigger definition may be used for text insertions
                    for key in reversed(trigger_keys):
                        triggers = [t.strip() for t in tag[key].split(',')]

                        for index, trigger in enumerate(triggers):
                            trigger_regex = r'\b[^\w]*{}\b'.format(re.escape(trigger))
                            if case_sensitive:
                                trigger_match = re.search(trigger_regex, search_text)
                            else:
                                trigger_match = re.search(trigger_regex, search_text, flags=re.IGNORECASE)

                            if trigger_match:
                                # Check any on_prefix_only condition
                                if tag.get('on_prefix_only', False) and trigger_match.start() != 0:
                                    # Revert trigger match if on_prefix_only is unsatisfied
                                    if len(trigger_keys) == 1:
                                        trigger_match = None
                                        if index == len(triggers) - 1:
                                            all_triggers_matched = False
                                            break
                                        else:
                                            continue

                                # retain the matched trigger phrase, from first trigger definition due to reverse()
                                matched_trigger = str(trigger)
                                break

                            if not trigger_match:
                                # If last trigger phrase in the trigger key
                                if index == len(triggers) - 1:
                                    # If Tag was previously matched in 'user' text, but not in 'llm' text.
                                    if (phase=='img') and ('imgtag_matched_early' in tag):
                                        tag['imgtag_uninserted'] = True
                                        updated_matches.append(tag) # Will be suffixed to image prompt instead of inserted

                                    all_triggers_matched = False
                                    break

                        # stop checking trigger keys if any unmatched
                        if not all_triggers_matched:
                            break

                    if all_triggers_matched:
                        # If all triggers matched in search text
                        updated_unmatched[list_name].remove(tag)
                        tag['matched_trigger'] = matched_trigger
                        tag['phase'] = phase
                        # Collect name
                        if tag_name:
                            self.matches_names.append(tag_name)
                        # Append as a tuple with start/end indexes for any subsequent text insertions
                        if (('insert_text' in tag and phase == 'llm') or ('positive_prompt' in tag and phase == 'img')):
                            updated_matches.append((tag, trigger_match.start(), trigger_match.end()))
                            continue

                        # Retries finding a text insertion index during the Img phase
                        if phase == 'llm' and 'positive_prompt' in tag:
                            tag['imgtag_matched_early'] = True
                        updated_matches.append(tag)
                        continue
                                
            if updated_matches:
                updated_matches = self.process_tag_trumps(updated_matches) # type: ignore # trump tags
            # Add LLM sublist back to unmatched tags list if LLM phase
            if phase == 'llm':
                updated_unmatched['llm'] = llm_tags
            if 'user' in updated_unmatched:
                del updated_unmatched['user'] # Remove after first phase. Controls the 'llm' tag processing at function start.

            self.matches = updated_matches
            self.unmatched = updated_unmatched

            # Update persistent tags list for current phase/channel
            if isinstance(channel_ptags, list):
                channel_ptags.clear()
                channel_ptags.extend(updated_ptags)

        except Exception as e:
            log.error(f"Error matching tags: {e}")
            traceback.print_exc()
