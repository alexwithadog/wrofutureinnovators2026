"""
ATLAS - Museum Helmet (JRAG1)
==============================

Changes from JET8:
  1. Whisper language detection locked to en/es/fr/it only.
     Anything else gets rejected as noise (no more accidental Arabic).
  2. System prompt now tells Gemini that Whisper may mishear words and
     to use context to recover (e.g., "task" -> "mask of Tutankhamun").
  3. Whisper model size lowered to "tiny" for faster CPU inference.
  4. Headless mode confirmed: SHOW_CAMERA_WINDOW=False if no display.
  5. Visitor profile flow at startup: ATLAS asks age, profession,
     and interest level by voice. Profile is injected into every
     Gemini prompt so responses adapt to the visitor.
  6. Reboot triggers "reboot", "new visitor", or "start over" wipe
     memory and re-run the profile questions for a new visitor.
  7. Gemini system prompt now tells it to occasionally end a
     response with a short curious question back to the visitor,
     making the conversation feel two-way.
"""

import json
import os
import re
import signal
import sys
import time
import random
import threading
import queue
import subprocess
import tempfile
from collections import deque

import cv2
import numpy as np
import sounddevice as sd

from dotenv import load_dotenv
from google import genai
from faster_whisper import WhisperModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas.rag import RAG

from motor_controller import MotorController


MIC_DEVICE = 1
MIC_CHANNELS = 2
MIC_SAMPLE_RATE = 16000
MIC_NATIVE_RATE = 48000
MIC_BLOCKSIZE = 4800

AUDIO_OUT_DEVICE: str | None = "plughw:0,0"

PIPER_DATA_DIR = os.path.expanduser("~/piper_voices")
PIPER_LENGTH_SCALE = 1.00

ACK_CACHE_DIR = os.path.expanduser("~/.atlas_ack_cache")

EV3_MAC = "2C:6B:7D:7B:AE:02"
YOLO_TO_SLOT = {
    "mona_lisa":    "slot_1",
    "starry_night": "slot_2",
    "pharaoh_mask": "slot_3",
}
MOTOR_LOWER_DELAY_SECONDS = 5.0

WHISPER_MODEL_SIZE = "tiny"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"
WHISPER_ALLOWED_LANGS = {"en", "fr", "es", "it"}
DEFAULT_LANG_CODE = "en"

WHISPER_LANG_TO_KEY = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
}

VAD_ENERGY_THRESHOLD = 500
VAD_SILENCE_DURATION = 0.8
VAD_MIN_UTTERANCE_DURATION = 0.5
VAD_MAX_UTTERANCE_DURATION = 15.0

POST_SPEAK_SETTLE_SECONDS = 0.60

LANGUAGES: dict[str, dict] = {
    "english": {
        "piper_voice": "en_US-ryan-low",
        "exit_phrases": ["goodbye", "good bye", "exit", "quit", "stop program", "see you"],
        "ack_first": ["Let me think.", "One moment please.", "Good question, give me a second."],
        "ack_second": "Sorry, one second, let me think.",
        "failure": "There's a problem with the connection right now. Please try again in a moment.",
        "exit_phrase": "Goodbye.",
        "greeting": "Hi, I'm Atlas, your museum guide. Before we start, I'd love to get to know you a little.",
        "ask_age": "How old are you?",
        "ask_profession": "What do you do? For example, are you a student, a teacher, a tourist, or something else?",
        "ask_interest": "How would you describe your interest in art and history? Beginner, enthusiast, or expert?",
        "profile_thanks": "Great, thank you. Now feel free to ask me anything about art, history, or culture.",
        "reboot_ack": "Of course. Let me start fresh.",
        "didnt_catch": "Sorry, I didn't catch that. Could you say it again?",
        "gemini_directive": "Respond ONLY in English.",
        "display_label": "English",
    },
    "french": {
        "piper_voice": "fr_FR-siwis-medium",
        "exit_phrases": ["au revoir", "salut", "a bientot", "termine", "quitte", "arrete le programme"],
        "ack_first": ["Laisse-moi reflechir.", "Un instant, s'il vous plait.", "Bonne question, un moment."],
        "ack_second": "Desole, un moment, je reflechis.",
        "failure": "Il y a un probleme de connexion en ce moment. Veuillez reessayer dans un instant.",
        "exit_phrase": "Au revoir.",
        "greeting": "Bonjour, je suis Atlas, votre guide de musee. Avant de commencer, j'aimerais faire votre connaissance.",
        "ask_age": "Quel age avez-vous?",
        "ask_profession": "Que faites-vous dans la vie? Par exemple, etes-vous etudiant, enseignant, touriste, ou autre chose?",
        "ask_interest": "Comment decririez-vous votre interet pour l'art et l'histoire? Debutant, passionne, ou expert?",
        "profile_thanks": "Parfait, merci. Maintenant, n'hesitez pas a me poser n'importe quelle question sur l'art, l'histoire ou la culture.",
        "reboot_ack": "Bien sur. Je recommence depuis le debut.",
        "didnt_catch": "Desole, je n'ai pas bien compris. Pouvez-vous repeter?",
        "gemini_directive": "Respond ONLY in French.",
        "display_label": "Francais",
    },
    "spanish": {
        "piper_voice": "es_MX-claude-high",
        "exit_phrases": ["adios", "hasta luego", "hasta la vista", "salir", "termina el programa"],
        "ack_first": ["Dejame pensar.", "Un momento, por favor.", "Buena pregunta, un momento."],
        "ack_second": "Disculpe, un momento, estoy pensando.",
        "failure": "Hay un problema de conexion ahora mismo. Por favor, intentelo de nuevo en un momento.",
        "exit_phrase": "Adios.",
        "greeting": "Hola, soy Atlas, su guia del museo. Antes de empezar, me gustaria conocerle un poco.",
        "ask_age": "Cuantos anos tienes?",
        "ask_profession": "A que te dedicas? Por ejemplo, eres estudiante, profesor, turista, u otra cosa?",
        "ask_interest": "Como describirias tu interes por el arte y la historia? Principiante, aficionado, o experto?",
        "profile_thanks": "Perfecto, gracias. Ahora, sientete libre de preguntarme cualquier cosa sobre arte, historia o cultura.",
        "reboot_ack": "Por supuesto. Empezamos de nuevo.",
        "didnt_catch": "Disculpa, no te entendi bien. Podrias repetirlo?",
        "gemini_directive": "Respond ONLY in Spanish.",
        "display_label": "Espanol",
    },
    "italian": {
        "piper_voice": "it_IT-paola-medium",
        "exit_phrases": ["arrivederci", "ciao", "addio", "esci", "termina", "ferma il programma"],
        "ack_first": ["Fammi pensare.", "Un momento, per favore.", "Buona domanda, un attimo."],
        "ack_second": "Scusa, un momento, sto pensando.",
        "failure": "C'e un problema di connessione in questo momento. Per favore, riprova tra un attimo.",
        "exit_phrase": "Arrivederci.",
        "greeting": "Ciao, sono Atlas, la tua guida del museo. Prima di iniziare, vorrei conoscerti un po'.",
        "ask_age": "Quanti anni hai?",
        "ask_profession": "Cosa fai nella vita? Per esempio, sei uno studente, un insegnante, un turista, o qualcos'altro?",
        "ask_interest": "Come descriveresti il tuo interesse per l'arte e la storia? Principiante, appassionato, o esperto?",
        "profile_thanks": "Perfetto, grazie. Ora, sentiti libero di chiedermi qualsiasi cosa sull'arte, la storia o la cultura.",
        "reboot_ack": "Certo. Ricomincio da capo.",
        "didnt_catch": "Scusa, non ho capito bene. Puoi ripetere?",
        "gemini_directive": "Respond ONLY in Italian.",
        "display_label": "Italiano",
    },
}

DEFAULT_LANGUAGE = "english"

WAKE_WORDS = ("atlas", "helmet", "guide", "assistant")

REBOOT_PHRASES = (
    "reboot", "new visitor", "start over", "restart",
    "nouveau visiteur", "recommence", "redemarre",
    "nuevo visitante", "empieza de nuevo", "reinicia",
    "nuovo visitatore", "ricomincia", "riavvia",
)

MEMORY_TURNS = 10

ENABLE_YOLO = False
YOLO_WEIGHTS_PATH = "yolo26n.pt"
YOLO_IMGSZ = 480
DETECT_EVERY_N_FRAMES = 2
OBJECT_HOLD_SECONDS = 2.0
OBJECT_COOLDOWN_SECONDS = 8.0
TRIGGER_OBJECTS = set(YOLO_TO_SLOT.keys())
DETECT_CONFIDENCE_THRESHOLD = 0.15
TRIGGER_CONFIDENCE_THRESHOLD = 0.15

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_ROTATION = None
CAMERA_PROCESS_SIZE = (1280, 720)

GEMINI_MODEL_PRIMARY = "gemini-2.5-flash"
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash-lite"
ACK_DELAY_SECONDS = 1.5

RAG_SCORE_THRESHOLD = 0.25
RAG_TOP_K = 1

YOLO_TO_ARTWORK_ID = {
    "mona_lisa":    "mona_lisa",
    "starry_night": "starry_night",
    "pharaoh_mask": "pharaoh_mask",
}

PROFILE_MAX_RETRIES = 2
PROFILE_DEFAULT = {
    "age": "adult",
    "profession": "curious visitor",
    "interest": "interested",
}


def _has_display() -> bool:
    if os.environ.get("DISPLAY"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    return False


def _strip_accents(text: str) -> str:
    import unicodedata
    text = (text or "").lower().strip()
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _is_reboot_phrase(text: str) -> bool:
    norm = _strip_accents(text)
    for phrase in REBOOT_PHRASES:
        if _strip_accents(phrase) in norm:
            return True
    return False


def _piper_synthesize(voice: str, text: str, out_path: str) -> bool:
    try:
        subprocess.run(
            ["python3", "-m", "piper", "--model", voice, "--data-dir", PIPER_DATA_DIR,
             "--length-scale", str(PIPER_LENGTH_SCALE), "--output-file", out_path],
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except subprocess.CalledProcessError:
        return False


def _ensure_ack_wav(voice: str, text: str, cache_path: str) -> bool:
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return True
    return _piper_synthesize(voice, text, cache_path)


def _format_sheet_for_prompt(sheet: dict) -> str:
    return (
        f"Title: {sheet.get('title', '')}\n"
        f"Artist: {sheet.get('artist', '')}\n"
        f"Year: {sheet.get('year', '')}\n"
        f"Description: {sheet.get('long_description', '')}\n"
        f"Historical context: {sheet.get('historical_context', '')}\n"
        f"Themes: {', '.join(sheet.get('themes', []))}\n"
        f"Anecdote: {sheet.get('anecdote', '')}\n"
    )


def _rms_energy(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


class _RebootRequested(Exception):
    pass


class MuseumHelmet:
    def __init__(self):
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Put it in .env.")
        self.client = genai.Client(api_key=self.gemini_api_key)

        print("[RAG] Loading sheets and embedding model...")
        self.rag = RAG()
        print(f"[RAG] Collection size: {self.rag.collection.count()} sheet(s)")

        print(f"[STT] Loading Whisper '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE} ({WHISPER_COMPUTE})...")
        self.whisper = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        print("[STT] Whisper ready.")

        self.current_language = DEFAULT_LANGUAGE
        self.language_lock = threading.Lock()

        self.has_display = _has_display()
        if self.has_display:
            print("[Display] Display detected.")
        else:
            print("[Display] No display detected, running headless.")

        self.yolo_model = None
        if ENABLE_YOLO:
            try:
                from ultralytics import YOLO
                print(f"[YOLO] Loading {YOLO_WEIGHTS_PATH}...")
                self.yolo_model = YOLO(YOLO_WEIGHTS_PATH)
                try:
                    self.yolo_model.to("cuda")
                    print("[YOLO] Loaded on CUDA.")
                except Exception as e:
                    print(f"[YOLO] WARNING: could not move to CUDA: {e}")
            except Exception as e:
                print(f"[YOLO] Failed to load: {e}. Continuing without detection.")
                self.yolo_model = None

        self.last_seen_object = None
        self.object_first_seen_time = None
        self.last_object_trigger_time: dict[str, float] = {name: 0.0 for name in YOLO_TO_SLOT.keys()}
        self.last_terminal_objects = None

        self.utterance_queue: queue.Queue = queue.Queue()
        self.request_queue: queue.Queue = queue.Queue()

        self.stop_event = threading.Event()
        self.is_busy_event = threading.Event()
        self.in_profile_flow = threading.Event()
        self.profile_inbox: queue.Queue = queue.Queue()

        self.speak_start_time = 0.0
        self.last_speak_end_time = 0.0

        self._proc_lock = threading.Lock()
        self._piper_proc: subprocess.Popen | None = None
        self._aplay_proc: subprocess.Popen | None = None

        self.memory: deque = deque(maxlen=MEMORY_TURNS * 2 + 5)
        self.memory_lock = threading.Lock()

        self.visitor_profile: dict[str, str] = {}
        self.profile_lock = threading.Lock()

        self._ack_wavs: dict[str, dict] = {}
        os.makedirs(ACK_CACHE_DIR, exist_ok=True)

        self.motor = MotorController(server_mac=EV3_MAC)
        self._currently_raised_slot: str | None = None
        self._motor_lock = threading.Lock()
        self._last_motor_activity = 0.0

        self.system_prompt = """
You are Atlas, an AI museum guide embedded in a wearable helmet, speaking directly to a visitor in front of an exhibit.

Personality & Style
Speak warmly, naturally, and conversationally, like a real human guide.
Avoid sounding robotic, scripted, or like a textbook.
Keep responses concise: usually 1-2 short sentences, 3 only if clarity really needs it.
Prefer short back-and-forth interaction over long explanations.
Adjust energy depending on the subject.

Adapting to the visitor
A visitor profile block will be provided. ADAPT your tone, vocabulary, and depth to the visitor:
- If the visitor is young (under 12), use simple words, fun analogies, and light humor.
- If the visitor is a teenager, be engaging and a bit playful, but still informative.
- If the visitor is an adult beginner or tourist, be welcoming and explain context without assuming prior knowledge.
- If the visitor is an enthusiast or expert (art historian, museum guide, etc.), use richer vocabulary, deeper context, and skip basic explanations.
- Use the profession to tailor examples (for a teacher, mention pedagogical angles; for a scientist, mention technique and materials).

Asking the visitor questions back
You are HAVING A CONVERSATION, not delivering a lecture. About every other response, end your answer with a short, natural follow-up question that invites the visitor to share their thoughts. Examples: "What stands out to you about it?", "Have you seen anything like this before?", "What part interests you most?". Vary the questions, do NOT use the same one twice in a row, and skip the question entirely if the visitor seems to be in a hurry or asking rapid-fire factual questions.

Tolerating speech recognition errors
The visitor's words come from a speech-to-text system that sometimes mishears similar-sounding words. If a sentence almost makes sense but has an obviously wrong word, infer what was meant from context. Examples:
- "task of Tutankhamun" -> they mean MASK of Tutankhamun
- "starry knight" -> they mean Starry NIGHT (the painting)
- "moaner Lisa" or strange spellings of Mona Lisa -> they mean Mona Lisa
- "fair away" or "fer away" -> "far away"
Never ask the visitor to repeat unless the sentence is genuinely incomprehensible. Just answer the most likely question.

What you will answer
You are an educational and cultural guide first. ANSWER any reasonable question about art, history, culture, artifacts, artworks, artists, architecture, literature, mythology, religion, science, nature, geography, historical events, historical figures, museums, and general knowledge an educated museum guide would know, whether or not the subject is physically in front of the visitor.

Only gently redirect for things clearly unrelated to education or culture: personal advice, medical advice, financial advice, live sports scores, current news, directions to specific addresses, or explicit political debate.

Knowledge source
When a [Museum sheet] block is included in the prompt, treat it as the ground truth for that specific artwork. Prefer its facts over your general knowledge. Do not quote it verbatim, paraphrase it into natural spoken language.

When no [Museum sheet] is present, answer from your general knowledge as best you can.

Style rules when answering
Give clear, simple, meaningful explanations.
When explaining an object: say what it is, why it matters, and one interesting detail.
If unsure, acknowledge uncertainty calmly while still giving helpful context.

Vision & Context Awareness
Treat [Camera] notes as context about what the visitor is looking at right now.
If something may be misidentified, acknowledge uncertainty and still provide helpful context.
Vary phrasing to avoid sounding repetitive.

Privacy & Safety
Never mention storing, tracking, or saving personal data.
"""

        self.formatting_rules = """
CRITICAL OUTPUT FORMAT RULES, these are read aloud by a text-to-speech engine:
- Do NOT use asterisks, underscores, backticks, or any markdown.
- Do NOT use bold, italics, or any emphasis markers.
- Do NOT use bullet points, numbered lists, or dashes for lists.
- Do NOT use headers, titles, or section labels.
- Do NOT use emoji.
- Write ONLY plain spoken prose, continuous sentences, like a person talking.
"""

    def _motor_raise(self, yolo_class_name: str) -> None:
        slot = YOLO_TO_SLOT.get(yolo_class_name)
        if slot is None:
            return
        if not self.motor.connected:
            return
        with self._motor_lock:
            print(f"[Motor] raise {yolo_class_name!r} -> {slot}")
            ok = self.motor.raise_picture(slot)
            if ok:
                self._currently_raised_slot = slot
                self._last_motor_activity = time.time()

    def _motor_lower_all(self) -> None:
        if not self.motor.connected:
            return
        with self._motor_lock:
            if self._currently_raised_slot is None:
                return
            print("[Motor] lower all")
            ok = self.motor.lower_all()
            if ok:
                self._currently_raised_slot = None

    def _motor_idle_watcher(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(0.5)
            if self.stop_event.is_set():
                break
            with self._motor_lock:
                raised = self._currently_raised_slot
                last_activity = self._last_motor_activity
                last_speak_end = self.last_speak_end_time
                busy = self.is_busy_event.is_set()
            if busy:
                continue
            if raised is None:
                continue
            if last_speak_end <= 0:
                continue
            idle_since = max(last_speak_end, last_activity)
            if (time.time() - idle_since) >= MOTOR_LOWER_DELAY_SECONDS:
                self._motor_lower_all()

    def _prepare_ack_wavs(self) -> None:
        print(f"[Piper] Checking acknowledgment cache at {ACK_CACHE_DIR} ...")
        rendered = 0
        cached = 0
        for lang_name, cfg in LANGUAGES.items():
            voice = cfg["piper_voice"]
            self._ack_wavs[lang_name] = {"first_try": []}
            for i, phrase in enumerate(cfg["ack_first"]):
                path = os.path.join(ACK_CACHE_DIR, f"ack_first_{lang_name}_{i}.wav")
                already = os.path.exists(path) and os.path.getsize(path) > 0
                if _ensure_ack_wav(voice, phrase, path):
                    self._ack_wavs[lang_name]["first_try"].append(path)
                    if already:
                        cached += 1
                    else:
                        rendered += 1
            for kind, src_key in [("second_try", "ack_second"), ("failure", "failure"), ("greeting", "greeting"), ("exit_phrase", "exit_phrase")]:
                path = os.path.join(ACK_CACHE_DIR, f"{kind}_{lang_name}.wav")
                already = os.path.exists(path) and os.path.getsize(path) > 0
                if _ensure_ack_wav(voice, cfg[src_key], path):
                    self._ack_wavs[lang_name][kind] = path
                    if already:
                        cached += 1
                    else:
                        rendered += 1
        print(f"[Piper] Acknowledgments ready. Cached: {cached}, newly rendered: {rendered}.")

    def _play_cached_wav(self, wav_path: str) -> None:
        if not wav_path or not os.path.exists(wav_path):
            return
        aplay_cmd = ["aplay", "-q"]
        if AUDIO_OUT_DEVICE:
            aplay_cmd += ["-D", AUDIO_OUT_DEVICE]
        aplay_cmd.append(wav_path)
        try:
            with self._proc_lock:
                self._aplay_proc = subprocess.Popen(aplay_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._aplay_proc.wait()
        finally:
            with self._proc_lock:
                self._aplay_proc = None

    def _memory_append(self, role: str, text: str) -> None:
        with self.memory_lock:
            self.memory.append((role, text))

    def _memory_clear(self) -> None:
        with self.memory_lock:
            self.memory.clear()

    def _memory_as_transcript(self) -> str:
        with self.memory_lock:
            items = list(self.memory)
        lines = []
        for role, text in items:
            if role == "user":
                lines.append(f"Visitor: {text}")
            elif role == "assistant":
                lines.append(f"Guide: {text}")
            elif role == "camera":
                lines.append(f"[Camera] Visitor is now looking at: {text}")
        return "\n".join(lines) if lines else "(no prior turns)"

    def _profile_block(self) -> str:
        with self.profile_lock:
            p = dict(self.visitor_profile)
        if not p:
            return ""
        return (
            "[Visitor profile]\n"
            f"Age: {p.get('age', 'unknown')}\n"
            f"Profession: {p.get('profession', 'unknown')}\n"
            f"Interest level: {p.get('interest', 'unknown')}\n"
            "[End of profile]\n"
        )

    def _get_active_language(self) -> str:
        with self.language_lock:
            return self.current_language

    def _set_active_language(self, lang_key: str) -> None:
        if lang_key not in LANGUAGES:
            return
        with self.language_lock:
            if self.current_language != lang_key:
                print(f"[Lang] auto-detected: {self.current_language} -> {lang_key}")
                self.current_language = lang_key

    def _get_active_voice(self) -> str:
        return LANGUAGES[self._get_active_language()]["piper_voice"]

    def _get_active_gemini_directive(self) -> str:
        return LANGUAGES[self._get_active_language()]["gemini_directive"]

    def _is_exit_phrase(self, text: str) -> bool:
        text = (text or "").lower().strip()
        cfg = LANGUAGES[self._get_active_language()]
        for phrase in cfg.get("exit_phrases", []):
            if phrase in text:
                return True
        return False

    def _sanitize_for_tts(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r"[*_`~]", "", text)
        text = re.sub(r"^\s*[-]+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+[.)]\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _speak_full(self, text: str) -> None:
        text = self._sanitize_for_tts(text)
        if not text:
            return
        voice = self._get_active_voice()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name
        try:
            piper_cmd = ["python3", "-m", "piper", "--model", voice, "--data-dir", PIPER_DATA_DIR,
                         "--length-scale", str(PIPER_LENGTH_SCALE), "--output-file", wav_path]
            with self._proc_lock:
                self._piper_proc = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                if self._piper_proc.stdin:
                    self._piper_proc.stdin.write(text.encode("utf-8"))
                    self._piper_proc.stdin.close()
            except Exception:
                pass
            self._piper_proc.wait()
            with self._proc_lock:
                self._piper_proc = None
            if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                return
            aplay_cmd = ["aplay", "-q"]
            if AUDIO_OUT_DEVICE:
                aplay_cmd += ["-D", AUDIO_OUT_DEVICE]
            aplay_cmd.append(wav_path)
            with self._proc_lock:
                self._aplay_proc = subprocess.Popen(aplay_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._aplay_proc.wait()
            with self._proc_lock:
                self._aplay_proc = None
        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    def _hard_stop_all_audio(self) -> None:
        with self._proc_lock:
            for p in (self._piper_proc, self._aplay_proc):
                if p and p.poll() is None:
                    try:
                        p.terminate()
                    except Exception:
                        pass

    def say_blocking(self, text: str) -> None:
        print(f"[TTS] {text}")
        self.is_busy_event.set()
        self.speak_start_time = time.time()
        try:
            self._speak_full(text)
        finally:
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()

    def _gemini_try_once(self, model: str, prompt: str) -> str:
        chunks: list[str] = []
        stream = self.client.models.generate_content_stream(model=model, contents=prompt)
        for chunk in stream:
            delta = getattr(chunk, "text", None)
            if delta:
                chunks.append(delta)
        return "".join(chunks).strip()

    def _gemini_request_with_retries(self, prompt: str, ack_enabled: bool) -> tuple[str, str]:
        result_holder: dict = {}

        def attempt(model, key):
            try:
                result_holder[key] = ("ok", self._gemini_try_once(model, prompt))
            except Exception as e:
                result_holder[key] = ("err", str(e))

        t1 = threading.Thread(target=attempt, args=(GEMINI_MODEL_PRIMARY, "t1"), daemon=True)
        t1.start()
        t1_start = time.time()
        ack_played = False
        while t1.is_alive():
            if (ack_enabled and not ack_played and (time.time() - t1_start) >= ACK_DELAY_SECONDS):
                first_try_wavs = self._ack_wavs.get(self._get_active_language(), {}).get("first_try", [])
                if first_try_wavs:
                    print("[ack] (let me think)")
                    self._play_cached_wav(random.choice(first_try_wavs))
                ack_played = True
            time.sleep(0.05)
        t1.join(timeout=0.1)
        status, payload = result_holder.get("t1", ("err", "unknown"))
        if status == "ok" and payload:
            return (payload, "ok")
        print(f"[Gemini] primary attempt 1 failed/empty: {payload[:120] if payload else '(empty)'}")
        if ack_enabled:
            second_wav = self._ack_wavs.get(self._get_active_language(), {}).get("second_try")
            if second_wav:
                print("[ack] (sorry, one second)")
                self._play_cached_wav(second_wav)
        try:
            text = self._gemini_try_once(GEMINI_MODEL_PRIMARY, prompt)
            if text:
                return (text, "ok")
            print("[Gemini] primary attempt 2 returned empty.")
        except Exception as e:
            print(f"[Gemini] primary attempt 2 failed: {e}")
        try:
            print(f"[Gemini] falling back to {GEMINI_MODEL_FALLBACK}")
            text = self._gemini_try_once(GEMINI_MODEL_FALLBACK, prompt)
            if text:
                return (text, "ok")
            print("[Gemini] fallback returned empty.")
        except Exception as e:
            print(f"[Gemini] fallback failed: {e}")
        return ("", "failed")

    def _retrieve_rag_context(self, query: str) -> str:
        results = self.rag.search(query, n=RAG_TOP_K)
        if not results:
            return ""
        top = results[0]
        if top["score"] < RAG_SCORE_THRESHOLD:
            print(f"[RAG] Top result below threshold ({top['score']:.3f}): {top['id']}, skipping.")
            return ""
        print(f"[RAG] Injecting {top['id']} (score={top['score']:.3f})")
        return ("\n[Museum sheet - use this as ground truth]\n"
                f"{_format_sheet_for_prompt(top['sheet'])}\n"
                "[End of museum sheet]\n")

    def _retrieve_rag_by_id(self, artwork_id: str) -> str:
        sheet = self.rag.sheet_by_id(artwork_id)
        if sheet is None:
            return ""
        print(f"[RAG] Injecting {artwork_id} (direct lookup)")
        return ("\n[Museum sheet - use this as ground truth]\n"
                f"{_format_sheet_for_prompt(sheet)}\n"
                "[End of museum sheet]\n")

    _skip_instructions = """
Bystander filter:
If the visitor's latest line looks like random background chatter, off-topic noise (not a real question), or clearly not directed at you, reply with exactly:
SKIP
and nothing else.

Otherwise, answer normally as the museum guide.
"""

    def _build_user_prompt(self, user_text: str) -> str:
        directive = self._get_active_gemini_directive()
        rag_block = self._retrieve_rag_context(user_text)
        profile_block = self._profile_block()
        return f"""{self.system_prompt}

LANGUAGE INSTRUCTION: {directive}

{self.formatting_rules}

{profile_block}
Conversation so far (most recent last):
{self._memory_as_transcript()}
{rag_block}
Visitor's latest line: {user_text}

{self._skip_instructions}

If you answer, keep it to 1-2 short sentences. Warm and conversational, plain prose.
Use prior turns when relevant so follow-ups feel natural.
About every other response, end with a short curious question back to the visitor (skip if they seem in a hurry).
{directive}
"""

    def _build_object_prompt(self, object_name: str) -> str:
        directive = self._get_active_gemini_directive()
        artwork_id = YOLO_TO_ARTWORK_ID.get(object_name)
        if artwork_id:
            rag_block = self._retrieve_rag_by_id(artwork_id)
        else:
            rag_block = self._retrieve_rag_context(object_name)
        profile_block = self._profile_block()
        return f"""{self.system_prompt}

LANGUAGE INSTRUCTION: {directive}

{self.formatting_rules}

{profile_block}
Conversation so far (most recent last):
{self._memory_as_transcript()}
{rag_block}
Camera event:
The visitor has been steadily looking at an object detected as "{object_name}" for at least {OBJECT_HOLD_SECONDS} seconds.

Task:
Give a short, natural museum-guide explanation of "{object_name}".
Do NOT mention detection or observation; just speak as if you noticed it yourself.
Keep it to 1-2 short sentences. If unsure, use soft uncertainty.
This is NOT a bystander event, never reply SKIP for a camera event.
{directive}
"""

    def _handle_request(self, kind: str, text: str) -> None:
        if kind == "user":
            prompt = self._build_user_prompt(text)
            self._memory_append("user", text)
            ack_enabled = True
        elif kind == "object":
            prompt = self._build_object_prompt(text)
            self._memory_append("camera", text)
            ack_enabled = False
        else:
            return
        print("[Gemini] thinking ...")
        self.is_busy_event.set()
        try:
            response, status = self._gemini_request_with_retries(prompt, ack_enabled)
        except Exception as e:
            print(f"[Gemini] unexpected error: {e}")
            response, status = "", "failed"
        if status == "failed":
            failure_wav = self._ack_wavs.get(self._get_active_language(), {}).get("failure")
            if failure_wav:
                self._play_cached_wav(failure_wav)
            if kind == "user":
                with self.memory_lock:
                    if self.memory and self.memory[-1] == ("user", text):
                        self.memory.pop()
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()
            return
        if not response:
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()
            return
        if kind == "user":
            first_token = (response.split(None, 1)[0].strip().rstrip(".").upper() if response else "")
            if first_token == "SKIP":
                print("[Gemini] SKIP - bystander noise, staying silent.")
                with self.memory_lock:
                    if self.memory and self.memory[-1] == ("user", text):
                        self.memory.pop()
                self.is_busy_event.clear()
                self.last_speak_end_time = time.time()
                return
        sanitized = self._sanitize_for_tts(response)
        print(f"[TTS] {sanitized}")
        self.speak_start_time = time.time()
        try:
            self._speak_full(sanitized)
        finally:
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()
        self._memory_append("assistant", sanitized)

    def _gemini_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                req = self.request_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle_request(req.get("kind"), req.get("text", ""))

    def _listen_forever(self) -> None:
        if MIC_NATIVE_RATE % MIC_SAMPLE_RATE != 0:
            raise RuntimeError("MIC_NATIVE_RATE must be integer multiple of MIC_SAMPLE_RATE.")
        decim = MIC_NATIVE_RATE // MIC_SAMPLE_RATE
        audio_buffer: list[np.ndarray] = []
        is_speaking = False
        utt_start_time: float | None = None
        last_voice_time: float | None = None

        def callback(indata, frames, time_info, status):
            nonlocal is_speaking, utt_start_time, last_voice_time
            if self.is_busy_event.is_set():
                return
            if (time.time() - self.last_speak_end_time) < POST_SPEAK_SETTLE_SECONDS:
                return
            samples = np.frombuffer(bytes(indata), dtype=np.int16)
            if MIC_CHANNELS > 1:
                samples = samples.reshape(-1, MIC_CHANNELS)
                samples = samples[:, 0].copy()
            if decim > 1:
                samples = samples[::decim]
            energy = _rms_energy(samples)
            now = time.time()
            if energy >= VAD_ENERGY_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    utt_start_time = now
                    audio_buffer.clear()
                audio_buffer.append(samples)
                last_voice_time = now
            elif is_speaking:
                audio_buffer.append(samples)
                if (last_voice_time is not None and (now - last_voice_time) >= VAD_SILENCE_DURATION):
                    duration = (now - utt_start_time) if utt_start_time else 0.0
                    is_speaking = False
                    if duration >= VAD_MIN_UTTERANCE_DURATION:
                        utterance = np.concatenate(audio_buffer)
                        self.utterance_queue.put({"audio": utterance, "duration": duration})
                    audio_buffer.clear()
                    utt_start_time = None
                    last_voice_time = None
                elif (utt_start_time is not None and (now - utt_start_time) >= VAD_MAX_UTTERANCE_DURATION):
                    is_speaking = False
                    utterance = np.concatenate(audio_buffer)
                    self.utterance_queue.put({"audio": utterance, "duration": now - utt_start_time})
                    audio_buffer.clear()
                    utt_start_time = None
                    last_voice_time = None

        print(f"[STT] Listening on device {MIC_DEVICE} ({MIC_CHANNELS}ch) @ {MIC_NATIVE_RATE} -> {MIC_SAMPLE_RATE} Hz")
        try:
            with sd.RawInputStream(
                samplerate=MIC_NATIVE_RATE,
                blocksize=MIC_BLOCKSIZE,
                dtype="int16",
                channels=MIC_CHANNELS,
                device=MIC_DEVICE,
                callback=callback,
            ):
                while not self.stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            print(f"[STT] Listener error: {e}")

    def _transcribe_audio(self, audio_int16: np.ndarray, force_language: str | None = None) -> tuple[str, str]:
        audio_float = audio_int16.astype(np.float32) / 32768.0
        try:
            kwargs = {"beam_size": 5, "vad_filter": False}
            if force_language:
                kwargs["language"] = force_language
            segments, info = self.whisper.transcribe(audio_float, **kwargs)
            text = " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            print(f"[STT] Whisper error: {e}")
            return ("", DEFAULT_LANG_CODE)
        lang_code = force_language or info.language
        if WHISPER_ALLOWED_LANGS and lang_code not in WHISPER_ALLOWED_LANGS:
            print(f"[STT] Detected unsupported lang '{lang_code}', rejecting as noise.")
            return ("", DEFAULT_LANG_CODE)
        return (text.strip(), lang_code)

    def _transcribe_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.utterance_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            audio_int16 = item["audio"]
            duration = item.get("duration", 0.0)
            if self.in_profile_flow.is_set():
                self.profile_inbox.put({"audio": audio_int16, "duration": duration})
                continue
            text, lang_code = self._transcribe_audio(audio_int16)
            if not text:
                continue
            lang_key = WHISPER_LANG_TO_KEY.get(lang_code, DEFAULT_LANGUAGE)
            self._set_active_language(lang_key)
            print(f"\n[Heard] ({lang_code}, dur={duration:.1f}s): {text}")
            self._process_transcribed(text.lower().strip())

    def _process_transcribed(self, text: str) -> None:
        if not text:
            return
        if _is_reboot_phrase(text):
            self._do_reboot()
            return
        if self._is_exit_phrase(text):
            exit_wav = self._ack_wavs.get(self._get_active_language(), {}).get("exit_phrase")
            if exit_wav:
                self.is_busy_event.set()
                self.speak_start_time = time.time()
                try:
                    self._play_cached_wav(exit_wav)
                finally:
                    self.is_busy_event.clear()
                    self.last_speak_end_time = time.time()
            self._motor_lower_all()
            self.stop_event.set()
            return
        for w in WAKE_WORDS:
            if text.startswith(w + " ") or text == w:
                text = text[len(w):].strip()
                break
        if not text:
            return
        self.request_queue.put({"kind": "user", "text": text})

    def _wait_for_profile_utterance(self, timeout_seconds: float = 30.0) -> np.ndarray | None:
        try:
            item = self.profile_inbox.get(timeout=timeout_seconds)
            return item["audio"]
        except queue.Empty:
            return None

    def _drain_profile_inbox(self) -> None:
        while not self.profile_inbox.empty():
            try:
                self.profile_inbox.get_nowait()
            except queue.Empty:
                break

    def _ask_and_capture(self, prompt_text: str, retries: int = PROFILE_MAX_RETRIES) -> str | None:
        for attempt in range(retries + 1):
            self.say_blocking(prompt_text)
            self._drain_profile_inbox()
            audio = self._wait_for_profile_utterance(timeout_seconds=30.0)
            if audio is None:
                if attempt < retries:
                    self.say_blocking(LANGUAGES[self._get_active_language()]["didnt_catch"])
                    continue
                return None
            text, lang_code = self._transcribe_audio(audio)
            if not text:
                if attempt < retries:
                    self.say_blocking(LANGUAGES[self._get_active_language()]["didnt_catch"])
                    continue
                return None
            if _is_reboot_phrase(text):
                raise _RebootRequested()
            lang_key = WHISPER_LANG_TO_KEY.get(lang_code, DEFAULT_LANGUAGE)
            self._set_active_language(lang_key)
            print(f"[Profile heard] ({lang_code}): {text}")
            return text.strip()
        return None

    def _run_profile_flow(self) -> None:
        self.in_profile_flow.set()
        self._drain_profile_inbox()
        new_profile: dict[str, str] = {}
        try:
            self.say_blocking(LANGUAGES[self._get_active_language()]["greeting"])
            age_text = self._ask_and_capture(LANGUAGES[self._get_active_language()]["ask_age"])
            new_profile["age"] = age_text or PROFILE_DEFAULT["age"]
            prof_text = self._ask_and_capture(LANGUAGES[self._get_active_language()]["ask_profession"])
            new_profile["profession"] = prof_text or PROFILE_DEFAULT["profession"]
            interest_text = self._ask_and_capture(LANGUAGES[self._get_active_language()]["ask_interest"])
            new_profile["interest"] = interest_text or PROFILE_DEFAULT["interest"]
            with self.profile_lock:
                self.visitor_profile = new_profile
            print(f"[Profile] Collected: {new_profile}")
            self.say_blocking(LANGUAGES[self._get_active_language()]["profile_thanks"])
        except _RebootRequested:
            print("[Profile] Reboot during profile flow, restarting.")
            self.in_profile_flow.clear()
            self._do_reboot()
            return
        finally:
            self.in_profile_flow.clear()

    def _do_reboot(self) -> None:
        print("[Reboot] Resetting for new visitor.")
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                break
        self._memory_clear()
        with self.profile_lock:
            self.visitor_profile = {}
        self._motor_lower_all()
        with self.language_lock:
            self.current_language = DEFAULT_LANGUAGE
        self.say_blocking(LANGUAGES[self._get_active_language()]["reboot_ack"])
        self._run_profile_flow()

    def camera_worker(self) -> None:
        cap = None
        try:
            print(f"[Camera] Opening /dev/video{CAMERA_INDEX} via V4L2 ...")
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
            if not cap.isOpened():
                print("[Camera] Failed to open. Disabling.")
                return
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, test_frame = cap.read()
            if not ret:
                print("[Camera] Could not read first frame, disabling.")
                return
            print(f"[Camera] Open OK, frame shape: {test_frame.shape}")
            frame_idx = 0
            last_annotated = None
            last_fps = 0.0
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                if CAMERA_ROTATION is not None:
                    frame = cv2.rotate(frame, CAMERA_ROTATION)
                frame = cv2.resize(frame, CAMERA_PROCESS_SIZE)
                frame_idx += 1
                detections: list[dict] = []
                if (ENABLE_YOLO and self.yolo_model is not None and frame_idx % DETECT_EVERY_N_FRAMES == 0):
                    try:
                        t0 = time.time()
                        results = self.yolo_model.predict(frame, imgsz=YOLO_IMGSZ, verbose=False, device="cuda")
                        result = results[0]
                        last_annotated = result.plot(boxes=True, masks=False)
                        boxes = result.boxes
                        if (boxes is not None and boxes.cls is not None and boxes.conf is not None):
                            for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                                if conf < DETECT_CONFIDENCE_THRESHOLD:
                                    continue
                                cls_index = int(cls_id)
                                name = result.names.get(cls_index, str(cls_index))
                                detections.append({"name": str(name).lower(), "confidence": float(conf)})
                        inference_time = (time.time() - t0) * 1000.0
                        last_fps = 1000.0 / inference_time if inference_time > 0 else 0.0
                        self._maybe_trigger_object_explanation(detections)
                    except Exception as e:
                        print(f"[YOLO] Inference error: {e}")
                if self.has_display:
                    display = last_annotated if last_annotated is not None else frame.copy()
                    lang_label = f"Lang: {self._get_active_language()}"
                    fps_label = f"YOLO FPS: {last_fps:.1f}" if ENABLE_YOLO else "YOLO: OFF"
                    motor_label = f"Raised: {self._currently_raised_slot}" if self._currently_raised_slot else "Raised: -"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(display, lang_label, (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    ts = cv2.getTextSize(fps_label, font, 0.8, 2)[0]
                    cv2.putText(display, fps_label, (display.shape[1] - ts[0] - 10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(display, motor_label, (10, 60), font, 0.6, (200, 200, 255), 2, cv2.LINE_AA)
                    try:
                        cv2.imshow("ATLAS Museum Helmet", display)
                        if cv2.waitKey(1) == ord("q"):
                            self.stop_event.set()
                            break
                    except cv2.error:
                        print("[Display] cv2.imshow failed mid-run, disabling preview.")
                        self.has_display = False
        except Exception as e:
            print("Camera worker error:", e)
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            try:
                if self.has_display:
                    cv2.destroyAllWindows()
            except Exception:
                pass

    def _maybe_trigger_object_explanation(self, detections: list[dict]) -> None:
        if not ENABLE_YOLO:
            return
        if self.in_profile_flow.is_set():
            return
        current_time = time.time()
        if detections:
            summary = ", ".join(f"{d['name']} ({d['confidence']:.2f})" for d in detections)
            if summary != self.last_terminal_objects:
                print(f"[Camera detected]: {summary}")
                self.last_terminal_objects = summary
        else:
            if self.last_terminal_objects is not None:
                print("[Camera detected]: none")
                self.last_terminal_objects = None
        triggerable = [d for d in detections if d["name"] in TRIGGER_OBJECTS and d["confidence"] >= TRIGGER_CONFIDENCE_THRESHOLD]
        still_seeing = [d for d in detections if d["name"] in TRIGGER_OBJECTS and d["confidence"] >= DETECT_CONFIDENCE_THRESHOLD]
        if not still_seeing:
            self.last_seen_object = None
            self.object_first_seen_time = None
            return
        if not triggerable:
            return
        dominant = max(triggerable, key=lambda d: d["confidence"])
        dominant_name = dominant["name"]
        if dominant_name != self.last_seen_object:
            self.last_seen_object = dominant_name
            self.object_first_seen_time = current_time
            return
        if self.object_first_seen_time is None:
            self.object_first_seen_time = current_time
            return
        held_long_enough = (current_time - self.object_first_seen_time) >= OBJECT_HOLD_SECONDS
        off_cooldown = (current_time - self.last_object_trigger_time.get(dominant_name, 0.0)) >= OBJECT_COOLDOWN_SECONDS
        if held_long_enough:
