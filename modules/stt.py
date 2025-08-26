import numpy as np
import asyncio
import wave
import os
import glob
import discord
from discord.ext import voice_recv
from utils.asr import asr_manager
from config_stt import AudioConfig
from utils import black_list
import config_stt
import time
from modules.utils_shared import client

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

class TranscriberSink(voice_recv.AudioSink):
    def __init__(self, loop:asyncio.BaseEventLoop, guild_id, min_audio_duration=0.5):
        super().__init__()
        self.loop = loop
        self.guild_id = guild_id  # Store guild ID to tie sink to specific guild
        self.user_states = {}  # Dictionary to manage state for each user
        self.wrap_messages = config_stt.WRAP_MESSAGES

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

        # Minimum audio duration
        self.MIN_AUDIO_DURATION = min_audio_duration

        # Start silence monitoring task
        self.silence_check_task = loop.create_task(self.silence_monitor())

    def _current_time(self):
        return self.loop.time()

    def wants_opus(self):
        return False

    def write(self, user, data):
        if user.id in black_list.blacklist:  # Skip if user is blacklisted
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
        self.silence_check_task.cancel()
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
            filename = f"{self.sink.guild_id}_{self.user_id}_chunk{self.chunk_counter}_{int(current_time * 1000)}.wav"
            self.chunk_counter += 1

            # Save raw audio to temporary file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.sink.CHANNELS)
                wf.setsampwidth(self.sink.SAMPLE_WIDTH)
                wf.setframerate(self.sink.SAMPLE_RATE)
                wf.writeframes(self.audio_buffer)

            self.audio_buffer.clear()
            self.speaking = False

            log.info(
                f"🔊 Audio chunk ready for user {self.user_id} | "
                f"Duration: {duration:.2f}s | "
                f"Total latency: {current_time - self.first_packet_time:.4f}s"
            )

            async def transcribe_task():
                task_start = self.sink._current_time()
                log.info(f"⏳ START Transcription for user {self.user_id} | File: {filename}")

                try:
                    process_start = self.sink._current_time()
                    transcription = await asyncio.to_thread(asr_manager.transcribe, filename)
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
                        os.remove(filename)
                    except Exception as e:
                        log.error(f"🧹 Cleanup failed for user {self.user_id}: {str(e)}")

            task = asyncio.create_task(transcribe_task())
            if self.sink.wrap_messages: # Append to combined pending tasks if wrapping
                self.sink.combined_pending_transcriptions.append((self.user_id, task))
            else: # Append to user's own pending tasks if not wrapping
                self.pending_transcriptions.append(task)

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
                    guild = client.guilds[0] # Assuming bot is in at least one guild
                    member = guild.get_member(self.user_id)
                    if member:
                        print(f"{member.display_name}: {full_transcription}") # console output
                        transcription_dir = "chatbot_transcriptions" # Directory to store transcription files
                        os.makedirs(transcription_dir, exist_ok=True) # Create directory if it doesn't exist
                        timestamp = int(time.time()) # Get current timestamp
                        filename = os.path.join(transcription_dir, f"transcription_{member.id}_{timestamp}.txt") # Unique filename
                        try:
                            with open(filename, "w") as file:
                                file.write(f"{member.display_name}: {full_transcription}")
                            log.info(f"📝 Transcription saved to file: {filename}")
                        except Exception as e:
                            log.error(f"🔥 Error writing transcription to file {filename}: {e}")
                        # --- File writing logic ends here ---
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
