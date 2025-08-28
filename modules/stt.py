import numpy as np
import asyncio
import glob
import os
import wave
import time
import threading
from discord.ext import voice_recv
from modules.utils_shared import client, config, stt_blacklist

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702

class AudioConfig:
    audio_config = config.stt.get('audio_config', {})
    SPEECH_VOLUME_THRESHOLD = audio_config.get('speech_volume_threshold', 1)
    CHUNK_SILENCE_THRESHOLD = audio_config.get('chunk_silence_threshold', 0.5)
    FINAL_SILENCE_THRESHOLD = audio_config.get('final_silence_threshold', 0.8)
    SAMPLE_RATE             = audio_config.get('sample_rate', 48000)
    CHANNELS                = audio_config.get('channels', 2)
    SAMPLE_WIDTH            = audio_config.get('sample_width', 2)

    # Volume normalization parameters, adjust if needed
    TARGET_RMS = audio_config.get('target_rms', 1400)
    MAX_GAIN = audio_config.get('max_gain', 10)
    MIN_RMS = audio_config.get('min_rms', 100)


class ASRManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # config
            whisper_config = config.stt.get('whisper_config', {})
            self.model_name = whisper_config.get('model_name', 'small')
            self.device = whisper_config.get('device', 'cpu')
            self.cuda_visible_devices = whisper_config.get('cuda_visible_devices', '0')
            self.whisper_model = None
            self.lock = threading.Lock()
            self._initialized = True

    def initialize(self):
        with self.lock:
            if not self.whisper_model:
                import whisper
                os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
                self.whisper_model = whisper.load_model(self.model_name, device=self.device)

    def transcribe(self, audio_file_path):
        self.initialize()  # Ensure models are loaded
        with self.lock:
            result = self.whisper_model.transcribe(audio_file_path)
            return result["text"]

# Singleton instance
asr_manager = ASRManager()


class TranscriberSink(voice_recv.AudioSink):
    def __init__(self, loop, guild_id, stt_channel, min_audio_duration=0.5):
        super().__init__()
        self.loop = loop
        self.guild_id = guild_id  # Store guild ID to tie sink to specific guild
        self.stt_channel = stt_channel
        self.user_states = {}  # Dictionary to manage state for each user
        self.wrap_messages = config.stt.get('wrap_messages', False)

        if self.wrap_messages: #  Initialize shared buffers if wrapping messages
            self.combined_text_buffer = []
            self.combined_pending_transcriptions = []

        # Text buffering and timing will be per-user, managed in UserState
        # Audio configuration
        self.SPEECH_VOLUME_THRESHOLD = AudioConfig.SPEECH_VOLUME_THRESHOLD
        self.CHUNK_SILENCE_THRESHOLD = AudioConfig.CHUNK_SILENCE_THRESHOLD
        self.FINAL_SILENCE_THRESHOLD = AudioConfig.FINAL_SILENCE_THRESHOLD
        self.SAMPLE_RATE = AudioConfig.SAMPLE_RATE
        self.CHANNELS = AudioConfig.CHANNELS
        self.SAMPLE_WIDTH = AudioConfig.SAMPLE_WIDTH

        # Volume normalization parameters
        self.TARGET_RMS = AudioConfig.TARGET_RMS
        self.MAX_GAIN = AudioConfig.MAX_GAIN
        self.MIN_RMS = AudioConfig.MIN_RMS

        # Minimum audio duration
        self.MIN_AUDIO_DURATION = min_audio_duration

        # Start silence monitoring task
        self.silence_check_task = loop.create_task(self.silence_monitor())

    def _current_time(self):
        return self.loop.time()

    def wants_opus(self):
        return False

    def write(self, user, data):
        if user.id in stt_blacklist.blacklisted_ids:  # Skip if user is blacklisted
            return
        user_id = user.id
        # Handle all users, not just one
        if user_id not in self.user_states:
            self.user_states[user_id] = self.UserState(self, user_id)
        user_state = self.user_states[user_id]
        #log.info(f"[Guild {self.guild_id}] Received audio data for user {user_id}: length={len(data.pcm)}bytes")
        packet = data.pcm  # Use PCM data from VoiceData
        asyncio.run_coroutine_threadsafe(user_state.on_audio_packet(packet), self.loop)

    def cleanup(self):
        """Clean up resources."""
        for user_id in self.user_states:
            for chunk in glob.glob(f"{self.guild_id}_{user_id}_chunk*.wav"):
                try:
                    os.remove(chunk)
                except Exception as e:
                    log.error(f"🧹 Final cleanup failed: {str(e)}")

    class UserState:
        def __init__(self, sink, user_id):
            self.sink = sink
            self.user_id = user_id
            self.audio_buffer = bytearray()
            self.lock = asyncio.Lock()
            self.text_buffer = []
            self.pending_transcriptions = []
            self.last_packet_time = 0
            self.first_packet_time = 0
            self.last_packet_received = 0
            self.speaking = False
            self.finalizing = False
            self.chunk_counter = 0  # For unique file naming per chunk
            self.pause_sent = False  # Added for pause trigger control

        async def on_audio_packet(self, data):
            """Process each audio packet."""
            async with self.lock:
                packet_receive_time = self.sink._current_time()
                raw_data = bytes(data)

                # Track packet timing
                if not self.speaking:
                    self.first_packet_time = packet_receive_time
                self.last_packet_received = packet_receive_time

                process_start = self.sink._current_time()
                audio_data = np.frombuffer(raw_data, dtype=np.int16)

                if len(audio_data) == 0:
                    return

                # Handle odd-length buffers
                if len(audio_data) % self.sink.CHANNELS != 0:
                    audio_data = audio_data[:-(len(audio_data) % self.sink.CHANNELS)]

                samples = audio_data.reshape(-1, self.sink.CHANNELS)

                # Calculate RMS for each channel
                with np.errstate(divide='ignore', invalid='ignore'):
                    rms_left = np.sqrt(np.mean(np.square(samples[:, 0].astype(np.float32))))
                    rms_right = np.sqrt(np.mean(np.square(samples[:, 1].astype(np.float32))))
                max_rms = max(rms_left, rms_right)

                current_time = self.sink._current_time()
                process_time = current_time - process_start
                log.debug(f"[Guild {self.sink.guild_id}]📦 Packet processed for user {self.user_id} in {process_time:.4f}s | RMS: {max_rms:.1f}")

                if max_rms > self.sink.SPEECH_VOLUME_THRESHOLD:
                    if not self.speaking:
                        self.speaking = True
                        log.info(f"[Guild {self.sink.guild_id}]🎤 Speech START for user {self.user_id}")
                    self.audio_buffer.extend(raw_data)  # Append raw data without normalization
                    self.last_packet_time = current_time
                    self.finalizing = False
                else:
                    if self.speaking:
                        self.silence_start_time = current_time

        async def _process_chunk(self, current_time):
            """Process an audio chunk and transcribe it if it meets the minimum duration."""
            chunk_collection_time = self.last_packet_received - self.first_packet_time
            log.info(f"[Guild {self.sink.guild_id}]⏳ Packet collection duration for user {self.user_id}: {chunk_collection_time:.4f}s")

            num_frames = len(self.audio_buffer) // (self.sink.CHANNELS * self.sink.SAMPLE_WIDTH)
            duration = num_frames / self.sink.SAMPLE_RATE

            # Skip if audio duration is less than the minimum
            if duration < self.sink.MIN_AUDIO_DURATION:
                log.info(f"[Guild {self.sink.guild_id}]Audio chunk too short for user {self.user_id}: {duration:.2f}s < {self.sink.MIN_AUDIO_DURATION:.2f}s, skipping")
                self.audio_buffer.clear()
                self.speaking = False
                return

            file_start = self.sink._current_time()
            raw_filename = f"{self.sink.guild_id}_{self.user_id}_chunk{self.chunk_counter}_{int(current_time * 1000)}_raw.wav"
            norm_filename = f"{self.sink.guild_id}_{self.user_id}_chunk{self.chunk_counter}_{int(current_time * 1000)}.wav"
            self.chunk_counter += 1

            # Save raw audio to temporary file
            with wave.open(raw_filename, 'wb') as wf:
                wf.setnchannels(self.sink.CHANNELS)
                wf.setsampwidth(self.sink.SAMPLE_WIDTH)
                wf.setframerate(self.sink.SAMPLE_RATE)
                wf.writeframes(self.audio_buffer)

            # Read raw audio and normalize
            with wave.open(raw_filename, 'rb') as wf:
                audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).reshape(-1, self.sink.CHANNELS)

            # Calculate average RMS
            with np.errstate(divide='ignore', invalid='ignore'):
                rms_left = np.sqrt(np.mean(np.square(audio_data[:, 0].astype(np.float32))))
                rms_right = np.sqrt(np.mean(np.square(audio_data[:, 1].astype(np.float32))))
            avg_rms = max(rms_left, rms_right)

            # Apply normalization
            if avg_rms > self.sink.MIN_RMS:
                gain = self.sink.TARGET_RMS / avg_rms
                gain = min(gain, self.sink.MAX_GAIN)  # Cap gain to avoid distortion
                if gain < 1.0:  # Reduce volume if RMS is above target
                    gain = max(gain, 1.0 / self.sink.MAX_GAIN)  # Cap reduction to avoid excessive lowering

                normalized_left = audio_data[:, 0].astype(np.float32) * gain
                normalized_right = audio_data[:, 1].astype(np.float32) * gain

                normalized_left = np.clip(normalized_left, -32768, 32767)
                normalized_right = np.clip(normalized_right, -32768, 32767)

                normalized_samples = np.column_stack((normalized_left, normalized_right))
                normalized_audio_data = normalized_samples.astype(np.int16).tobytes()
            else:
                normalized_audio_data = self.audio_buffer  # Use raw data if below MIN_RMS

            # Save normalized audio
            with wave.open(norm_filename, 'wb') as wf:
                wf.setnchannels(self.sink.CHANNELS)
                wf.setsampwidth(self.sink.SAMPLE_WIDTH)
                wf.setframerate(self.sink.SAMPLE_RATE)
                wf.writeframes(normalized_audio_data)

            self.audio_buffer.clear()
            self.speaking = False

            log.info(
                f"🔊 Audio chunk ready for user {self.user_id} | "
                f"Duration: {duration:.2f}s | "
                f"Total latency: {current_time - self.first_packet_time:.4f}s"
            )

            async def transcribe_task():
                task_start = self.sink._current_time()
                log.info(f"⏳ START Transcription for user {self.user_id} | File: {norm_filename}")

                try:
                    process_start = self.sink._current_time()
                    transcription = await asyncio.to_thread(asr_manager.transcribe, norm_filename)
                    process_time = self.sink._current_time() - process_start
                    rtf = duration / process_time if duration > 0 else 0

                    stripped_transcription = transcription.strip()
                    if stripped_transcription:  # Check if not empty (length >= 1)
                        if len(stripped_transcription) == 3 and stripped_transcription in ["???", "..."]:
                            log.info(f"Discarding transcription for user {self.user_id}: {stripped_transcription}")
                        else:
                            if self.sink.wrap_messages: # Append to combined buffer if wrap_messages is True
                                self.sink.combined_text_buffer.append((self.user_id, self.first_packet_time, stripped_transcription))
                            else: # Append to user's own buffer if wrap_messages is False
                                self.text_buffer.append(stripped_transcription)
                            log.info(
                                f"✅ Transcription COMPLETE for user {self.user_id} | "
                                f"RTF: {rtf:.2f} | "
                                f"Total time: {self.sink._current_time() - task_start:.4f}s | "
                                f"Text: {stripped_transcription[:50]}..."
                            )
                    else:
                        log.warning(f"🚫 Empty transcription for user {self.user_id}")

                except Exception as e:
                    log.error(f"❌ FAILED for user {self.user_id} | Error: {str(e)}")
                finally:
                    try:
                        os.remove(raw_filename)
                        os.remove(norm_filename)
                    except Exception as e:
                        log.error(f"🧹 Cleanup failed for user {self.user_id}: {str(e)}")

            task = asyncio.create_task(transcribe_task())
            if self.sink.wrap_messages: # Append to combined pending tasks if wrapping
                self.sink.combined_pending_transcriptions.append((self.user_id, task))
            else: # Append to user's own pending tasks if not wrapping
                self.pending_transcriptions.append(task)

        async def _log_transcription(self, member, transcription):
            print(f"{member.display_name}: {transcription}") # console output
            transcription_dir = "chatbot_transcriptions" # Directory to store transcription files
            os.makedirs(transcription_dir, exist_ok=True) # Create directory if it doesn't exist
            timestamp = int(time.time()) # Get current timestamp
            filename = os.path.join(transcription_dir, f"transcription_{member.id}_{timestamp}.txt") # Unique filename
            try:
                with open(filename, "w") as file:
                    file.write(f"{member.display_name}: {transcription}")
                log.info(f"📝 Transcription saved to file: {filename}")
            except Exception as e:
                log.error(f"🔥 Error writing transcription to file {filename}: {e}")

        async def _send_transcription(self, member, channel, transcription):
            guild, channel, member = self._get_discord_objs()
            message = f"{member.display_name}: {transcription}"
            await channel.send(message)

        async def _get_discord_objs(self):
            guild, channel, member = None, None, None
            guild = await client.fetch_guild(self.sink.guild_id)
            if guild:
                channel = await client.fetch_channel(self.sink.stt_channel)
                member = guild.get_member(self.user_id)
            return guild, channel, member

        async def _finalize_transcription(self):
            """Finalize and send the transcription."""
            self.finalizing = True
            log.info(f"📭 Finalizing transcription sequence for user {self.user_id}")

            final_start = self.sink._current_time()
            if not self.sink.wrap_messages: # Finalize if not wrapping
                if self.pending_transcriptions:
                    log.info(f"⏳ Awaiting {len(self.pending_transcriptions)} tasks for user {self.user_id}")
                    await asyncio.gather(*self.pending_transcriptions, return_exceptions=True)

                if self.text_buffer:
                    full_transcription = " ".join(self.text_buffer)
                    total_time = self.sink._current_time() - final_start
                    log.info(f"📨 Sending result for user {self.user_id} | Chunks: {len(self.text_buffer)} | Time: {total_time:.4f}s")
                    guild, channel, member = self._get_discord_objs()
                    if member:
                        #await self._log_transcription(member, full_transcription)
                        if channel:
                            await self._send_transcription(member, channel, full_transcription)

                    self.text_buffer.clear()

                self.pending_transcriptions.clear()
            # Finalization for wrapped messages is handled in silence_monitor
            self.finalizing = False


    async def _send_wrapped_message(self): # Send wrapped message
        """Sends a combined message for all users when wrap_messages is True."""
        log.info(f"📭 Finalizing wrapped transcription sequence for all users")

        if self.combined_pending_transcriptions:
            log.info(f"⏳ Awaiting {len(self.combined_pending_transcriptions)} combined tasks")
            await asyncio.gather(*[task for _, task in self.combined_pending_transcriptions], return_exceptions=True)

        if self.combined_text_buffer:
            self.combined_text_buffer.sort(key=lambda entry: entry[1]) # Sort by start_time

            full_message = ""
            for user_entry in self.combined_text_buffer:
                user_id, start_time, text_chunk = user_entry
                if text_chunk:
                    guild = client.guilds[0] # Assuming bot is in at least one guild
                    print("GUILD:", guild)
                    member = guild.get_member(user_id)
                    if member:
                        full_message += f"{member.display_name}: {text_chunk}\n" #Add each chunk on a new line

            if full_message.strip(): # Check if the message is not empty
                log.info(f"📨 Sending wrapped result | Users: {len(self.combined_text_buffer)}")
                print(f"{full_message.strip()}") # console output

        self.combined_text_buffer.clear()
        self.combined_pending_transcriptions.clear()


    async def silence_monitor(self):
        """Monitor silence and process audio chunks for all users."""
        log.info("🔇 Starting silence monitor")
        while True:
            await asyncio.sleep(0.01)
            current_time = self._current_time()
            shortest_silence_duration = float('inf') # Added for wrapped messages

            for user_state in self.user_states.values():
                async with user_state.lock:
                    silence_duration = current_time - user_state.last_packet_time

                    if user_state.finalizing:
                        continue

                    if silence_duration >= self.CHUNK_SILENCE_THRESHOLD and user_state.audio_buffer:
                        await user_state._process_chunk(current_time)

                    if user_state.speaking: # Track shortest silence only for speaking users
                        shortest_silence_duration = min(shortest_silence_duration, silence_duration)

                    if not self.wrap_messages: # Finalize individually if not wrapping
                        if silence_duration >= self.FINAL_SILENCE_THRESHOLD and user_state.text_buffer:
                            await user_state._finalize_transcription()


            if self.wrap_messages: # Handle wrapped messages finalization
                if shortest_silence_duration >= self.FINAL_SILENCE_THRESHOLD and self.combined_text_buffer: # Check shortest silence for all users
                    await self._send_wrapped_message()
