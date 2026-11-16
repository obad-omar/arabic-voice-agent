"""
Dermeen: a real-time Arabic AI voice agent that answers phone calls.

Asterisk opens one AudioSocket TCP connection per call and streams the caller's
audio (8 kHz, 16-bit, mono). The agent streams its own voice back in the same format.

  Ears:  Azure Speech, continuous recognition with Saudi Arabic + English auto-detect
  Brain: Google Gemini, streamed, so speech starts before the full answer is ready
  Voice: Azure Neural TTS

Environment: AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, GEMINI_API_KEY,
             GEMINI_MODEL, AGENT_VOICE, AGENT_VOICE_EN, GREETING, PORT (all optional except the keys)
"""
import asyncio
import json
import logging
import os
import re
import struct
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("dermeen")

HERE = Path(__file__).parent
PORT = int(os.getenv("PORT", "9092"))
AZ_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZ_REGION = os.getenv("AZURE_SPEECH_REGION", "uaenorth")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
VOICE = os.getenv("AGENT_VOICE", "ar-SA-ZariyahNeural")
VOICE_EN = os.getenv("AGENT_VOICE_EN", "en-US-AvaMultilingualNeural")

GREETING = os.getenv("GREETING", "هلا والله، معك درمين، مساعدتك الذكية. وش أقدر أخدمك؟")
END_TAG = "[END]"  # the model appends this when the conversation is over

# AudioSocket frame: kind (1 byte) + length (2 bytes, big-endian) + payload
KIND_HANGUP, KIND_UUID, KIND_AUDIO, KIND_ERROR = 0x00, 0x01, 0x10, 0xFF
FRAME_BYTES = 320  # 20 ms of 8 kHz 16-bit audio
FRAME_SECONDS = 0.02

SENTENCE_END = re.compile(r"(.+?[\.\!\?؟،,:\n])\s*", re.S)


def system_prompt() -> str:
    prompt = (HERE / "prompt.md").read_text(encoding="utf-8")
    knowledge = (HERE / "knowledge.md").read_text(encoding="utf-8")
    return prompt + "\n\n# Knowledge base\n" + knowledge


def frame(kind: int, payload: bytes = b"") -> bytes:
    return struct.pack(">BH", kind, len(payload)) + payload


def split_frames(pcm: bytes):
    for i in range(0, len(pcm), FRAME_BYTES):
        chunk = pcm[i:i + FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk += b"\x00" * (FRAME_BYTES - len(chunk))
        yield chunk


def take_sentences(buffer: str):
    """Cut the streamed text into speakable sentences. Returns (sentences, remainder)."""
    # Always cut at a full stop; cut at a comma only once the piece is long,
    # so the voice does not sound choppy.
    sentences, pos, start = [], 0, 0
    for m in SENTENCE_END.finditer(buffer):
        piece = buffer[start:m.end()].strip()
        hard_stop = piece[-1:] in ".!?؟\n"
        if len(piece) >= 2 and (hard_stop or len(piece) >= 25):
            sentences.append(piece)
            start = pos = m.end()
    return sentences, buffer[pos:]


def is_english(text: str) -> bool:
    letters = re.findall(r"[A-Za-z؀-ۿ]", text)
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) > 0.6


class Call:
    def __init__(self, reader, writer, system: str, genai_client):
        import azure.cognitiveservices.speech as speechsdk
        self.sdk = speechsdk
        self.reader, self.writer = reader, writer
        self.loop = asyncio.get_running_loop()
        self.system = system
        self.genai = genai_client
        self.audio_out: asyncio.Queue = asyncio.Queue()
        self.speaking = False
        self.history = []
        self.transcript = []
        self.turn_task = None
        self.hangup_when_done = False
        self.closed = False
        self.call_id = "?"
        self.started = time.time()

    # ---------- Ears: speech-to-text ----------
    def start_listening(self):
        sdk = self.sdk
        fmt = sdk.audio.AudioStreamFormat(samples_per_second=8000, bits_per_sample=16, channels=1)
        self.mic = sdk.audio.PushAudioInputStream(fmt)
        cfg = sdk.SpeechConfig(subscription=AZ_KEY, region=AZ_REGION)
        cfg.set_property(sdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous")
        cfg.set_property(sdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "700")
        langs = sdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["ar-SA", "en-US"])
        self.recognizer = sdk.SpeechRecognizer(
            speech_config=cfg,
            audio_config=sdk.audio.AudioConfig(stream=self.mic),
            auto_detect_source_language_config=langs,
        )
        self.recognizer.recognizing.connect(
            lambda e: self.loop.call_soon_threadsafe(self.on_partial, e.result.text))
        self.recognizer.recognized.connect(
            lambda e: self.loop.call_soon_threadsafe(self.on_final, e.result.text)
            if e.result.reason == sdk.ResultReason.RecognizedSpeech else None)
        self.recognizer.start_continuous_recognition_async()

    def on_partial(self, text: str):
        # Barge-in: the caller started talking, so stop speaking right away.
        if self.speaking and len(text.strip()) >= 3:
            self.stop_talking()

    def on_final(self, text: str):
        text = text.strip()
        if not text or self.closed:
            return
        log.info("[%s] caller: %s", self.call_id, text)
        self.transcript.append({"role": "caller", "text": text})
        self.stop_talking()
        self.turn_task = asyncio.create_task(self.reply(text))

    def stop_talking(self):
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
        while not self.audio_out.empty():
            self.audio_out.get_nowait()
        self.speaking = False

    # ---------- Brain: streamed LLM reply ----------
    async def reply(self, text: str):
        from google.genai import types
        self.history.append(types.Content(role="user", parts=[types.Part(text=text)]))
        full, buffer = "", ""
        try:
            stream = await self.genai.aio.models.generate_content_stream(
                model=MODEL,
                contents=self.history,
                config=types.GenerateContentConfig(
                    system_instruction=self.system,
                    temperature=0.4,
                    max_output_tokens=400,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            async for chunk in stream:
                piece = chunk.text or ""
                full += piece
                buffer += piece
                sentences, buffer = take_sentences(buffer)
                for s in sentences:
                    await self.say(s)
            if buffer.strip():
                await self.say(buffer.strip())
        except asyncio.CancelledError:
            full += " …(interrupted by caller)"
            raise
        except Exception as exc:  # keep the call alive even if one reply fails
            log.exception("[%s] reply failed: %s", self.call_id, exc)
            await self.say("المعذرة، ما وضح لي. ممكن تعيد؟")
        finally:
            clean = full.replace(END_TAG, "").strip()
            if clean:
                self.history.append(types.Content(role="model", parts=[types.Part(text=clean)]))
                self.transcript.append({"role": "agent", "text": clean})
                log.info("[%s] agent: %s", self.call_id, clean)
            if END_TAG in full:
                self.hangup_when_done = True

    # ---------- Voice: text-to-speech ----------
    def synthesize(self, text: str) -> bytes:
        sdk = self.sdk
        cfg = sdk.SpeechConfig(subscription=AZ_KEY, region=AZ_REGION)
        cfg.set_speech_synthesis_output_format(sdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm)
        cfg.speech_synthesis_voice_name = VOICE_EN if is_english(text) else VOICE
        result = sdk.SpeechSynthesizer(speech_config=cfg, audio_config=None).speak_text_async(text).get()
        if result.reason != sdk.ResultReason.SynthesizingAudioCompleted:
            log.warning("[%s] TTS failed: %s", self.call_id, result.reason)
            return b""
        return result.audio_data

    async def say(self, text: str):
        text = text.replace(END_TAG, "").strip()
        if not text:
            return
        pcm = await asyncio.to_thread(self.synthesize, text)
        for chunk in split_frames(pcm):
            self.audio_out.put_nowait(chunk)

    async def send_audio(self):
        """Stream the agent's voice in real time, one 20 ms frame at a time."""
        next_at = self.loop.time()
        while not self.closed:
            try:
                chunk = await asyncio.wait_for(self.audio_out.get(), timeout=0.2)
            except asyncio.TimeoutError:
                self.speaking = False
                if self.hangup_when_done and (not self.turn_task or self.turn_task.done()):
                    log.info("[%s] agent ended the call", self.call_id)
                    self.writer.write(frame(KIND_HANGUP))
                    await self.writer.drain()
                    return
                next_at = self.loop.time()
                continue
            self.speaking = True
            self.writer.write(frame(KIND_AUDIO, chunk))
            await self.writer.drain()
            next_at += FRAME_SECONDS
            await asyncio.sleep(max(0, next_at - self.loop.time()))

    # ---------- Call lifecycle ----------
    async def run(self):
        from google.genai import types
        self.start_listening()
        sender = asyncio.create_task(self.send_audio())
        self.history.append(types.Content(role="model", parts=[types.Part(text=GREETING)]))
        self.transcript.append({"role": "agent", "text": GREETING})
        await self.say(GREETING)
        try:
            while True:
                header = await self.reader.readexactly(3)
                kind, length = struct.unpack(">BH", header)
                payload = await self.reader.readexactly(length) if length else b""
                if kind == KIND_AUDIO:
                    self.mic.write(payload)
                elif kind == KIND_UUID:
                    self.call_id = payload.hex()[:8]
                    log.info("[%s] new call", self.call_id)
                elif kind in (KIND_HANGUP, KIND_ERROR):
                    break
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.closed = True
            self.stop_talking()
            sender.cancel()
            self.mic.close()
            self.recognizer.stop_continuous_recognition_async()
            log.info("[%s] call ended (%d s) %s", self.call_id, time.time() - self.started,
                     json.dumps(self.transcript, ensure_ascii=False))
            self.writer.close()


async def main():
    if not (AZ_KEY and GEMINI_KEY):
        log.warning("AZURE_SPEECH_KEY or GEMINI_API_KEY is not set; waiting until they are configured")
        await asyncio.Event().wait()

    from google import genai
    client = genai.Client(api_key=GEMINI_KEY)
    system = system_prompt()

    async def handle(reader, writer):
        try:
            await Call(reader, writer, system, client).run()
        except Exception:
            log.exception("call failed")
            writer.close()

    server = await asyncio.start_server(handle, "0.0.0.0", PORT)
    log.info("agent ready on port %d (voice %s, model %s)", PORT, VOICE, MODEL)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
