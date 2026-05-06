import json, os, re, time, random, threading, queue, subprocess, tempfile
from collections import deque

import cv2
import numpy as np
import sounddevice as sd
import vosk
from dotenv import load_dotenv
from google import genai
from ultralytics import YOLOE

try:
    import torch
    torch.set_num_threads(3)
except Exception:
    pass

vosk.SetLogLevel(-1)

MIC_DEVICE = None
MIC_NATIVE_RATE = 48000
MIC_SAMPLE_RATE = 16000
MIC_BLOCKSIZE = 12000

AUDIO_OUT_DEVICE = "default"

VOSK_MODEL_PATH = "/opt/vosk_models/vosk-model-small-en-us-0.15"
PIPER_VOICE = "en_US-ryan-low"
PIPER_DATA_DIR = os.path.expanduser("~/piper_voices")
PIPER_LENGTH_SCALE = 1.10

CAMERA_PROCESS_SIZE = (1200, 1200)
SHOW_PREVIEW = True

DETECT_EVERY_N_FRAMES = 4
OBJECT_HOLD_SECONDS = 1.5
OBJECT_COOLDOWN_SECONDS = 3.0
DETECT_CONFIDENCE_THRESHOLD = 0.10
TRIGGER_CONFIDENCE_THRESHOLD = 0.10

TRIGGER_OBJECTS = {"vase", "sword", "pharaoh mask", "mona lisa"}
OBJECT_PRIORITY = {"pharaoh mask": 1, "mona lisa": 2, "crown": 3, "vase": 4}

WAKE_WORDS = ("atlas", "helmet", "guide", "assistant")
EXIT_WORDS = ("goodbye", "good bye", "exit", "quit", "stop program")

STT_MIN_WORDS = 3
STT_MIN_SECONDS = 1.0
VOSK_MIN_CONF = 0.55
POST_SPEAK_SETTLE_SECONDS = 0.60
MEMORY_TURNS = 10

GEMINI_MODEL_PRIMARY = "gemini-2.5-flash-lite"
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash-lite"

GREETING = "Hi, I'm Atlas, your museum guide. You can ask me anything, or stop in front of something and I'll tell you about it."

ACK_FIRST_TRY_PHRASES = [
    "One moment please, let me think.",
    "Let me think for a moment, please wait.",
]
ACK_SECOND_TRY_PHRASE = "Sorry, one second, let me think."
FAILURE_PHRASE = "There's a problem with the connection right now. Please try again in a moment."


def _aplay_command(wav_path):
    cmd = ["aplay", "-q"]
    if AUDIO_OUT_DEVICE:
        cmd += ["-D", AUDIO_OUT_DEVICE]
    cmd.append(wav_path)
    return cmd


def _piper_synthesize(text, out_path):
    try:
        subprocess.run(
            [
                "python3", "-m", "piper",
                "--model", PIPER_VOICE,
                "--data-dir", PIPER_DATA_DIR,
                "--length-scale", str(PIPER_LENGTH_SCALE),
                "--output-file", out_path,
            ],
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception as e:
        print("[Piper] synth failed:", e)
        return False


class MuseumHelmet:
    def __init__(self):
        load_dotenv()

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.gemini_api_key)

        if not os.path.isdir(VOSK_MODEL_PATH):
            raise RuntimeError(f"Vosk model not found at {VOSK_MODEL_PATH}")

        print("[STT] Loading Vosk model...")
        self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
        print("[STT] Vosk model loaded.")

        self.model_path = "yoloe-11s-seg.pt"
        self.prompt_names = [
            "vase", "mona lisa", "cement", "head",
            "pharaoh mask", "royal crown", "bone",
            "statue", "diamonds", "shiny", "sword"
        ]

        self.detect_confidence_threshold = DETECT_CONFIDENCE_THRESHOLD
        self.model_imgsz = 192

        self.last_seen_object = None
        self.object_first_seen_time = None
        self.last_object_trigger_time = {name: 0.0 for name in self.prompt_names}
        self.last_terminal_objects = None

        self.utterance_queue = queue.Queue()
        self.request_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_busy_event = threading.Event()

        self.speak_start_time = 0.0
        self.last_speak_end_time = 0.0

        self._proc_lock = threading.Lock()
        self._piper_proc = None
        self._aplay_proc = None

        self.memory = deque(maxlen=MEMORY_TURNS * 2 + 5)
        self.memory_lock = threading.Lock()

        self._ack_first_try_wavs = []
        self._ack_second_try_wav = None
        self._failure_wav = None
        self._ack_tempdir = tempfile.mkdtemp(prefix="atlas_ack_")

        self.system_prompt = """
You are Atlas, a warm AI museum guide inside a wearable helmet.
Speak naturally and briefly, usually 1 to 2 short sentences.
Answer educational questions about art, history, culture, artifacts, museums, science, geography, mythology, religion, and general knowledge.
Never use markdown, bullets, emojis, asterisks, underscores, or backticks.
Everything you write will be spoken aloud, so write plain spoken prose only.
"""

    def _prepare_ack_wavs(self):
        print("[Piper] Pre-rendering acknowledgment audio...")
        for i, phrase in enumerate(ACK_FIRST_TRY_PHRASES):
            path = os.path.join(self._ack_tempdir, f"ack_first_{i}.wav")
            if _piper_synthesize(phrase, path):
                self._ack_first_try_wavs.append(path)

        path = os.path.join(self._ack_tempdir, "ack_second.wav")
        if _piper_synthesize(ACK_SECOND_TRY_PHRASE, path):
            self._ack_second_try_wav = path

        path = os.path.join(self._ack_tempdir, "failure.wav")
        if _piper_synthesize(FAILURE_PHRASE, path):
            self._failure_wav = path

        print("[Piper] Acknowledgment audio ready.")

    def _sanitize_for_tts(self, text):
        text = re.sub(r"[*_`~]", "", text or "")
        text = re.sub(r"^\s*[-•]+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+[.)]\s*", "", text, flags=re.MULTILINE)
        return re.sub(r"\s+", " ", text).strip()

    def _play_cached_wav(self, wav_path):
        if not wav_path or not os.path.exists(wav_path):
            return
        with self._proc_lock:
            self._aplay_proc = subprocess.Popen(_aplay_command(wav_path))
        self._aplay_proc.wait()
        with self._proc_lock:
            self._aplay_proc = None

    def _speak_full(self, text):
        text = self._sanitize_for_tts(text)
        if not text:
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            self._piper_proc = subprocess.Popen(
                [
                    "python3", "-m", "piper",
                    "--model", PIPER_VOICE,
                    "--data-dir", PIPER_DATA_DIR,
                    "--length-scale", str(PIPER_LENGTH_SCALE),
                    "--output-file", wav_path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if self._piper_proc.stdin:
                self._piper_proc.stdin.write(text.encode("utf-8"))
                self._piper_proc.stdin.close()

            self._piper_proc.wait()
            self._piper_proc = None

            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                self._aplay_proc = subprocess.Popen(_aplay_command(wav_path))
                self._aplay_proc.wait()
                self._aplay_proc = None
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def say_blocking(self, text):
        print(text)
        self.is_busy_event.set()
        self.speak_start_time = time.time()
        try:
            self._speak_full(text)
        finally:
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()
              def _hard_stop_all_audio(self):
        for p in (self._piper_proc, self._aplay_proc):
            if p and p.poll() is None:
                p.terminate()

    def _memory_append(self, role, text):
        with self.memory_lock:
            self.memory.append((role, text))

    def _memory_as_transcript(self):
        with self.memory_lock:
            items = list(self.memory)
        return "\n".join(f"{r}: {t}" for r, t in items) or "(no prior turns)"

    def _gemini_try_once(self, model, prompt):
        chunks = []
        stream = self.client.models.generate_content_stream(model=model, contents=prompt)
        for chunk in stream:
            if getattr(chunk, "text", None):
                chunks.append(chunk.text)
        return "".join(chunks).strip()

    def _handle_request(self, kind, text):
        if kind == "user":
            self._memory_append("Visitor", text)
            prompt = f"{self.system_prompt}\nConversation:\n{self._memory_as_transcript()}\nVisitor: {text}\nAnswer briefly."
        else:
            self._memory_append("Camera", text)
            prompt = f"{self.system_prompt}\nThe visitor is looking at {text}. Explain it briefly."

        self.is_busy_event.set()
        try:
            response = self._gemini_try_once(GEMINI_MODEL_PRIMARY, prompt)
            response = self._sanitize_for_tts(response)
            if response and response.upper() != "SKIP":
                print(response)
                self._speak_full(response)
                self._memory_append("Atlas", response)
        except Exception as e:
            print("[Gemini] error:", e)
        finally:
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()

    def _gemini_worker(self):
        while not self.stop_event.is_set():
            try:
                req = self.request_queue.get(timeout=0.2)
                self._handle_request(req["kind"], req["text"])
            except queue.Empty:
                pass

    def _listen_forever(self):
        audio_q = queue.Queue()
        decim = MIC_NATIVE_RATE // MIC_SAMPLE_RATE

        def callback(indata, frames, time_info, status):
            if self.is_busy_event.is_set():
                return
            samples = np.frombuffer(bytes(indata), dtype=np.int16)
            audio_q.put(samples[::decim].tobytes())

        while not self.stop_event.is_set():
            try:
                rec = vosk.KaldiRecognizer(self.vosk_model, MIC_SAMPLE_RATE)
                with sd.RawInputStream(
                    samplerate=MIC_NATIVE_RATE,
                    blocksize=MIC_BLOCKSIZE,
                    dtype="int16",
                    channels=1,
                    device=MIC_DEVICE,
                    callback=callback,
                ):
                    print("[STT] Listening...")
                    while not self.stop_event.is_set():
                        data = audio_q.get()
                        if rec.AcceptWaveform(data):
                            result = json.loads(rec.Result())
                            text = (result.get("text") or "").strip().lower()
                            if text:
                                print("[Heard]:", text)
                                self.utterance_queue.put(text)
            except Exception as e:
                print("[STT] error:", e)
                time.sleep(0.5)

    def camera_worker(self):
        gst = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
            "nvvidconv flip-method=2 ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink"
        )

        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            print("[Camera] GStreamer failed. Trying /dev/video0 raw fallback.")
            cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

        if not cap.isOpened():
            print("[Camera] failed to open")
            return

        print("[Camera] opened")
        model = YOLOE(self.model_path)
        model.set_classes(self.prompt_names)
        print("[Camera] YOLOE loaded")

        frame_idx = 0

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, CAMERA_PROCESS_SIZE)
            frame_idx += 1

            if frame_idx % DETECT_EVERY_N_FRAMES == 0:
                results = model.predict(frame, imgsz=self.model_imgsz, verbose=False)
                result = results[0]
                frame = result.plot(boxes=True, masks=False)

            cv2.imshow("ATLAS Live Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop_event.set()
                break

        cap.release()
        cv2.destroyAllWindows()

    def start(self):
        self._prepare_ack_wavs()

        threading.Thread(target=self.camera_worker, daemon=True).start()
        threading.Thread(target=self._listen_forever, daemon=True).start()
        threading.Thread(target=self._gemini_worker, daemon=True).start()

        self.say_blocking(GREETING)

        try:
            while not self.stop_event.is_set():
                try:
                    text = self.utterance_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if any(w in text for w in EXIT_WORDS):
                    self.say_blocking("Goodbye.")
                    break

                self.request_queue.put({"kind": "user", "text": text})

        except KeyboardInterrupt:
            print("shutting down")
        finally:
            self.stop_event.set()
            self._hard_stop_all_audio()


if __name__ == "__main__":
    helmet = MuseumHelmet()
    helmet.start()
