import copy
from typing import Optional
from modules.utils_shared import is_tgwui_integrated, config
if is_tgwui_integrated:
    from modules.utils_tgwui import tgwui

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

# TODO Isolate independant TTS handling from TGWUI extension handling

class TTS:
    def __init__(self):
        self.enabled:bool = False
        self.api_mode:bool = False
        self.settings:dict = config.ttsgen
        self.api_name:Optional[str] = self.settings.get('api_name')
        self.api_url:Optional[str] = self.settings.get('api_url')
        self.api_get_voices_endpoint:Optional[str] = ''
        self.api_generate_endpoint:Optional[str] = ''


    # Toggles TTS on/off
    async def apply_toggle_tts(self, settings, toggle:str='on', tts_sw:bool=False):
        try:
            #settings:"Settings" = get_settings(ictx)
            llmcontext_dict = vars(settings.llmcontext)
            extensions:dict = copy.deepcopy(llmcontext_dict.get('extensions', {}))
            if toggle == 'off' and extensions.get(tgwui.tts_extension, {}).get('activate'):
                extensions[tgwui.tts_extension]['activate'] = False
                await tgwui.update_extensions(extensions)
                # Return True if subsequent apply_toggle_tts() should enable TTS
                return True
            if tts_sw:
                extensions[tgwui.tts_extension]['activate'] = True
                await tgwui.update_extensions(extensions)
        except Exception as e:
            log.error(f'[TTS] An error occurred while toggling the TTS on/off: {e}')
        return False

tts = TTS()

class VoiceClients:
    def __init__(self):
        self.guild_vcs:dict = {}
        self.expected_state:dict = {}
        self.queued_tts:list = []

    def is_connected(self, guild_id):
        if self.guild_vcs.get(guild_id):
            return self.guild_vcs[guild_id].is_connected()
        return False
    
    def should_be_connected(self, guild_id):
        return self.expected_state.get(guild_id, False)

    # Try loading character data regardless of mode (chat/instruct)
    async def restore_state(self):
        for guild_id, should_be_connected in self.expected_state.items():
            try:
                if should_be_connected and not self.is_connected(guild_id):
                    voice_channel = client.get_channel(bot_database.voice_channels[guild_id])
                    self.guild_vcs[guild_id] = await voice_channel.connect()
                elif not should_be_connected and self.is_connected(guild_id):
                    await self.guild_vcs[guild_id].disconnect()
            except Exception as e:
                log.error(f'[Voice Clients] An error occurred while restoring voice channel state for guild ID "{guild_id}": {e}')

    async def toggle_voice_client(self, guild_id, toggle:str=None):
        try:
            if toggle == 'enabled' and not self.is_connected(guild_id):
                if bot_database.voice_channels.get(guild_id):
                    voice_channel = client.get_channel(bot_database.voice_channels[guild_id])
                    self.guild_vcs[guild_id] = await voice_channel.connect()
                    self.expected_state[guild_id] = True
                else:
                    log.warning(f'[Voice Clients] "{tts.client}" enabled, but a valid voice channel is not set for this server.')
                    log.info('[Voice Clients] Use "/set_server_voice_channel" to select a voice channel for this server.')
            if toggle == 'disabled':
                if self.is_connected(guild_id):
                    await self.guild_vcs[guild_id].disconnect()
                    self.expected_state[guild_id] = False
        except Exception as e:
            log.error(f'[Voice Clients] An error occurred while toggling voice channel for guild ID "{guild_id}": {e}')

    async def voice_channel(self, guild_id:int, vc_setting:bool=True):
        try:
            # Start voice client if configured, and not explicitly deactivated in character settings
            if tts.enabled and vc_setting == True and int(tts.settings.get('play_mode', 0)) != 1 and not self.guild_vcs.get(guild_id):
                try:
                    if tts.client and tts.client in shared.args.extensions:
                        await self.toggle_voice_client(guild_id, 'enabled')
                    else:
                        if not bot_database.was_warned('char_tts'):
                            bot_database.update_was_warned('char_tts')
                            log.warning('[Voice Clients] No "tts_client" is specified in config.yaml')
                except Exception as e:
                    log.error(f"[Voice Clients] An error occurred while connecting to voice channel: {e}")
            # Stop voice client if explicitly deactivated in character settings
            if self.guild_vcs.get(guild_id) and self.guild_vcs[guild_id].is_connected():
                if vc_setting is False:
                    log.info("[Voice Clients] New context has setting to disconnect from voice channel. Disconnecting...")
                    await self.toggle_voice_client(guild_id, 'disabled')
        except Exception as e:
            log.error(f"[Voice Clients] An error occurred while managing channel settings: {e}")

    def after_playback(self, guild_id, file, error):
        if error:
            log.info(f'[Voice Clients] Message from audio player: {error}, output: {error.stderr.decode("utf-8")}')
        # Check save mode setting
        if int(tts.settings.get('save_mode', 0)) > 0:
            try:
                os.remove(file)
            except Exception:
                pass
        # Check if there are queued tasks
        if self.queued_tts:
            # Pop the first task from the queue and play it
            next_file = self.queued_tts.pop(0)
            source = discord.FFmpegPCMAudio(next_file)
            self.guild_vcs[guild_id].play(source, after=lambda e: self.after_playback(guild_id, next_file, e))

    async def play_in_voice_channel(self, guild_id, file):
        if not self.guild_vcs.get(guild_id):
            log.warning(f"[Voice Clients] tts response detected, but bot is not connected to a voice channel in guild ID {guild_id}")
            return
        # Queue the task if audio is already playing
        if self.guild_vcs[guild_id].is_playing():
            self.queued_tts.append(file)
        else:
            # Otherwise, play immediately
            source = discord.FFmpegPCMAudio(file)
            self.guild_vcs[guild_id].play(source, after=lambda e: self.after_playback(guild_id, file, e))

    async def toggle_playback_in_voice_channel(self, guild_id, action='stop'):
        if self.guild_vcs.get(guild_id):          
            guild_vc:discord.VoiceClient = self.guild_vcs[guild_id]
            if action == 'stop' and guild_vc.is_playing():
                guild_vc.stop()
                log.info(f"TTS playback was stopped for guild {guild_id}")
            elif (action == 'pause' or action == 'toggle') and guild_vc.is_playing():
                guild_vc.pause()
                log.info(f"TTS playback was paused in guild {guild_id}")
            elif (action == 'resume' or action == 'toggle') and guild_vc.is_paused():
                guild_vc.resume()
                log.info(f"TTS playback resumed in guild {guild_id}")

    def detect_format(self, file_path):
        try:
            audio = AudioSegment.from_wav(file_path)
            return 'wav'
        except:
            pass  
        try:
            audio = AudioSegment.from_mp3(file_path)
            return 'mp3'
        except:
            pass
        return None

    async def upload_tts_file(self, channel:discord.TextChannel, tts_resp:str|None=None, bot_hmessage:HMessage|None=None):
        file = tts_resp
        filename = os.path.basename(file)
        original_ext = os.path.splitext(filename)[1]
        correct_ext = original_ext
        detected_format = self.detect_format(file)
        if detected_format is None:
            raise ValueError(f"Could not determine the audio file format for file: {file}")
        if original_ext != f'.{detected_format}':
            correct_ext = f'.{detected_format}'
            new_filename = os.path.splitext(filename)[0] + correct_ext
            new_file_path = os.path.join(os.path.dirname(file), new_filename)
            os.rename(file, new_file_path)
            file = new_file_path

        mp3_filename = os.path.splitext(filename)[0] + '.mp3'
        
        bit_rate = int(tts.settings.get('mp3_bit_rate', 128))
        with io.BytesIO() as buffer:
            if file.endswith('wav'):
                audio = AudioSegment.from_wav(file)
            elif file.endswith('mp3'):
                audio = AudioSegment.from_mp3(file)
            else:
                log.error('TTS generated unsupported file format:', file)
            audio.export(buffer, format="mp3", bitrate=f"{bit_rate}k")
            mp3_file = File(buffer, filename=mp3_filename)
            
            sent_message = await channel.send(file=mp3_file)
            # if bot_hmessage:
            #     bot_hmessage.update(audio_id=sent_message.id)

    async def process_tts_resp(self, ictx:CtxInteraction, tts_resp:Optional[str]=None, bot_hmessage:Optional[HMessage]=None):
        play_mode = int(tts.settings.get('play_mode', 0))
        # Upload to interaction channel
        if play_mode > 0:
            await self.upload_tts_file(ictx.channel, tts_resp, bot_hmessage)
        # Play in voice channel
        connected = self.guild_vcs.get(ictx.guild.id)
        if not is_direct_message(ictx) and play_mode != 1 and self.guild_vcs.get(ictx.guild.id):
            await bg_task_queue.put(self.play_in_voice_channel(ictx.guild.id, tts_resp)) # run task in background
        if bot_hmessage:
            bot_hmessage.update(spoken=True)

voice_clients = VoiceClients()
