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
import sounddevice as sd  # type: ignore

from dotenv import load_dotenv # type: ignore
from google import genai
from faster_whisper import WhisperModel # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas.rag import RAG

from motor_controller import MotorController


MIC_DEVICE = 1
MIC_CHANNELS = 2
MIC_SAMPLE_RATE = 16000
MIC_NATIVE_RATE = 48000
MIC_BLOCKSIZE = 2400

AUDIO_OUT_DEVICE: str | None = "plughw:0,0"

PIPER_DATA_DIR = os.path.expanduser("~/piper_voices")
PIPER_LENGTH_SCALE = 0.92
PIPER_CACHE_TAG = f"ls{str(PIPER_LENGTH_SCALE).replace('.', 'p')}"

ACK_CACHE_DIR = os.path.expanduser("~/.atlas_ack_cache")
PRECACHE_ALL_LANGUAGES = True
STARTUP_WAIT_FOR_TERMINAL = True
STARTUP_MOTOR_WAIT_SECONDS = 8.0
STARTUP_CAMERA_WAIT_SECONDS = 5.0
STARTUP_GEMINI_WARMUP_TIMEOUT_SECONDS = 6.0

EV3_MAC = "2C:6B:7D:7B:AE:02"
YOLO_TO_SLOT = {
    "starry_night": "slot_1",
    "mona_lisa":    "slot_2",
    "pharaoh_mask": "slot_3",
}

YOLO_CLASS_ALIASES = {
    "mona_lisa": "mona_lisa",
    "mona lisa": "mona_lisa",
    "monalisa": "mona_lisa",
    "starry_night": "starry_night",
    "starry night": "starry_night",
    "starrynight": "starry_night",
    "pharaoh_mask": "pharaoh_mask",
    "pharaoh mask": "pharaoh_mask",
    "pharaohmask": "pharaoh_mask",
    "tutankhamun": "pharaoh_mask",
    "tutankhamun_mask": "pharaoh_mask",
    "tutankhamun mask": "pharaoh_mask",
    "mask_of_tutankhamun": "pharaoh_mask",
    "mask of tutankhamun": "pharaoh_mask",
    "objects": "pharaoh_mask",
}

ARTWORK_SPOKEN_NAMES = {
    "mona_lisa": "Mona Lisa",
    "starry_night": "The Starry Night",
    "pharaoh_mask": "the Mask of Tutankhamun",
}

MOTOR_LOWER_DELAY_SECONDS = 5.0

WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"
WHISPER_BEAM_SIZE = 5
WHISPER_PROFILE_BEAM_SIZE = 5
WHISPER_LANGUAGE_LOCK_AFTER_PROFILE = True
WHISPER_ALLOWED_LANGS = {"en", "fr", "es", "it"}
FRENCH_RESCUE_ENABLED = True
FRENCH_RESCUE_BEAM_SIZE = 5
DEFAULT_LANG_CODE = "en"

WHISPER_LANG_TO_KEY = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
}

VAD_ENERGY_THRESHOLD = 500
VAD_SILENCE_DURATION = 0.75
VAD_MIN_UTTERANCE_DURATION = 0.5
VAD_MAX_UTTERANCE_DURATION = 15.0

POST_SPEAK_SETTLE_SECONDS = 0.35

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
        "language_changed": "Sure, I'll speak English now.",
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
        "language_changed": "Bien sur, je parlerai en francais maintenant.",
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
        "language_changed": "Claro, hablare en espanol ahora.",
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
        "language_changed": "Certo, parlero in italiano adesso.",
        "reboot_ack": "Certo. Ricomincio da capo.",
        "didnt_catch": "Scusa, non ho capito bene. Puoi ripetere?",
        "gemini_directive": "Respond ONLY in Italian.",
        "display_label": "Italiano",
    },
}

DEFAULT_LANGUAGE = "english"

CACHED_PHRASE_KEYS = [
    ("second_try", "ack_second"),
    ("failure", "failure"),
    ("greeting", "greeting"),
    ("ask_age", "ask_age"),
    ("ask_profession", "ask_profession"),
    ("ask_interest", "ask_interest"),
    ("profile_thanks", "profile_thanks"),
    ("language_changed", "language_changed"),
    ("reboot_ack", "reboot_ack"),
    ("didnt_catch", "didnt_catch"),
    ("exit_phrase", "exit_phrase"),
]

WAKE_WORDS = ("atlas", "helmet", "guide", "assistant")

REBOOT_PHRASES = (
    "reboot", "new visitor", "start over", "restart",
    "nouveau visiteur", "recommence", "redemarre",
    "nuevo visitante", "empieza de nuevo", "reinicia",
    "nuovo visitatore", "ricomincia", "riavvia",
)

PROFILE_SKIP_PHRASES = (
    "skip", "skip profile", "skip setup", "skip questions", "test mode",
)

LANGUAGE_SWITCH_MARKER = "LANGUAGE_SWITCH:"
LANGUAGE_SWITCH_TARGETS = {
    "french": (
        "francais", "francaise", "francois", "french",
    ),
    "english": (
        "anglais", "anglaise", "english",
    ),
    "spanish": (
        "espagnol", "espagnole", "espanol", "espanole", "spanish",
    ),
    "italian": (
        "italien", "italienne", "italiano", "italiana", "italian",
    ),
}
LANGUAGE_SWITCH_VERBS = (
    "parle", "parlez", "parler", "reponds", "repondez", "repondre",
    "speak", "talk", "answer", "respond", "switch", "change",
    "habla", "hablar", "responde", "responder",
    "parla", "parlare", "rispondi", "rispondere",
)

LANGUAGE_HINTS = {
    "french": (
        "bonjour", "salut", "merci", "francais", "parle", "parlez", "reponds",
        "qui", "quoi", "quel", "quelle", "comment", "pourquoi", "est ce",
        "joconde", "masque", "pharaon", "toutankhamon", "etoile", "etoilee",
        "nuit etoilee",
    ),
    "spanish": (
        "hola", "gracias", "espanol", "habla", "hablar", "quien", "que",
        "como", "por que", "estrella", "noche estrellada", "mascara",
    ),
    "italian": (
        "ciao", "grazie", "italiano", "parla", "parlare", "chi", "che",
        "come", "perche", "gioconda", "notte stellata", "maschera",
    ),
}

MEMORY_TURNS = 10

ENABLE_YOLO = True
YOLO_WEIGHTS_PATH = "best.pt"
YOLO_IMGSZ = 416
DETECT_EVERY_N_FRAMES = 1
OBJECT_HOLD_SECONDS = 2.0
OBJECT_COOLDOWN_SECONDS = 7.0
TRIGGER_OBJECTS = set(YOLO_TO_SLOT.keys())
DETECT_CONFIDENCE_THRESHOLD = 0.10
TRIGGER_CONFIDENCE_THRESHOLD = 0.24
TRIGGER_CONFIDENCE_BY_OBJECT = {
    "mona_lisa": 0.24,
    "starry_night": 0.24,
    "pharaoh_mask": 0.45,
}
CENTER_PRIORITY_WEIGHT = 0.40
CENTER_ACTIVE_THRESHOLD = 0.50

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_ROTATION = cv2.ROTATE_180
CAMERA_PROCESS_SIZE = (1280, 720)

GEMINI_MODEL_PRIMARY = "gemini-2.5-flash"
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash-lite"
GEMINI_MODEL_CAMERA = GEMINI_MODEL_FALLBACK
ACK_DELAY_SECONDS = 1.5

RAG_SCORE_THRESHOLD = 0.25
RAG_TOP_K = 1

EXHIBIT_CONTEXT = """
[Current exhibit]
Atlas is guiding a small museum display with exactly these three featured artworks or artifacts:
- Mona Lisa, by Leonardo da Vinci.
- The Starry Night, by Vincent van Gogh.
- The Mask of Tutankhamun, also called the pharaoh mask or Egyptian mask.
If the visitor asks about "the star one", "starry", "l'etoile", "la nuit etoilee", "the mask", "le masque", "Mona", or "la Joconde", map that to the matching item above and answer confidently.
[End of current exhibit]
"""

YOLO_TO_ARTWORK_ID = {
    "mona_lisa":    "mona_lisa",
    "starry_night": "starry_night",
    "pharaoh_mask": "pharaoh_mask",
}

ARTWORK_QUERY_ALIASES = {
    "mona_lisa": (
        "mona", "lisa", "mona lisa", "monalisa", "joconde", "jaconde",
        "gioconda", "la gioconda",
    ),
    "starry_night": (
        "starry", "starry night", "starry knight", "star", "stars",
        "etoile", "etoiles", "etoilee", "nuit etoilee", "nuit etoile",
        "nuit des etoiles", "van gogh", "vangogh", "vincent",
    ),
    "pharaoh_mask": (
        "mask", "masque", "pharaoh", "pharaon", "tutankhamun", "toutankhamon",
        "toutankamon", "tutankamon", "egyptian", "egyptien", "egypte",
    ),
}

PROFILE_MAX_RETRIES = 1
PROFILE_LISTEN_TIMEOUT_SECONDS = 12.0
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


def _normalize_yolo_class(raw_name: str) -> str | None:
    name = _strip_accents(raw_name)
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    spaced = name.replace("_", " ")
    return YOLO_CLASS_ALIASES.get(name) or YOLO_CLASS_ALIASES.get(spaced)


def _artwork_spoken_name(artwork_id: str) -> str:
    return ARTWORK_SPOKEN_NAMES.get(artwork_id, artwork_id.replace("_", " "))


def _object_threshold(thresholds: dict[str, float], object_name: str, default: float) -> float:
    return thresholds.get(object_name, default)


def _detect_artwork_query(text: str) -> str | None:
    norm = _strip_accents(text)
    norm = re.sub(r"[^a-z0-9]+", " ", norm).strip()
    if not norm:
        return None
    for artwork_id, aliases in ARTWORK_QUERY_ALIASES.items():
        for alias in aliases:
            alias_norm = _strip_accents(alias)
            if re.search(r"\b" + re.escape(alias_norm) + r"\b", norm):
                return artwork_id
    return None


def _parse_age(text: str) -> int | None:
    norm = _strip_accents(text)
    norm = re.sub(r"[^a-z0-9]+", " ", norm)
    match = re.search(r"\b(\d{1,3})\b", norm)
    if match:
        age = int(match.group(1))
        return age if 0 < age < 120 else None

    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20,
        "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4,
        "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9,
        "dix": 10, "onze": 11, "douze": 12, "treize": 13,
        "quatorze": 14, "quinze": 15, "seize": 16, "dix sept": 17,
        "dix huit": 18, "dix neuf": 19, "vingt": 20,
        "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
        "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
        "diez": 10, "once": 11, "doce": 12, "trece": 13,
        "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
        "dieciocho": 18, "diecinueve": 19, "veinte": 20,
        "uno": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5,
        "sei": 6, "sette": 7, "otto": 8, "nove": 9, "dieci": 10,
        "undici": 11, "dodici": 12, "tredici": 13, "quattordici": 14,
        "quindici": 15, "sedici": 16, "diciassette": 17,
        "diciotto": 18, "diciannove": 19, "venti": 20,
    }
    for word, value in sorted(number_words.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(r"\b" + re.escape(word) + r"\b", norm):
            return value
    return None


def _is_reboot_phrase(text: str) -> bool:
    norm = _strip_accents(text)
    for phrase in REBOOT_PHRASES:
        if _strip_accents(phrase) in norm:
            return True
    return False


def _is_profile_skip_phrase(text: str) -> bool:
    norm = _strip_accents(text)
    for phrase in PROFILE_SKIP_PHRASES:
        if re.search(r"\b" + re.escape(_strip_accents(phrase)) + r"\b", norm):
            return True
    return False


def _detect_language_switch(text: str) -> str | None:
    norm = _strip_accents(text)
    norm = re.sub(r"[^a-z0-9]+", " ", norm).strip()
    if not norm:
        return None
    has_switch_intent = any(re.search(r"\b" + re.escape(verb) + r"\b", norm) for verb in LANGUAGE_SWITCH_VERBS)
    has_polite_request = any(phrase in norm for phrase in (
        "please", "s il vous plait", "stp",
        "peux tu", "pouvez vous", "por favor", "per favore",
    ))
    for lang_key, aliases in LANGUAGE_SWITCH_TARGETS.items():
        if not any(re.search(r"\b" + re.escape(alias) + r"\b", norm) for alias in aliases):
            continue
        if has_switch_intent or has_polite_request:
            return lang_key
        for alias in aliases:
            if re.search(r"\b(en|in|a|to) " + re.escape(alias) + r"\b", norm):
                return lang_key
        if norm in aliases:
            return lang_key
    return None


def _infer_language_from_text(text: str) -> str | None:
    norm = _strip_accents(text)
    norm = re.sub(r"[^a-z0-9]+", " ", norm).strip()
    if not norm:
        return None
    scores: dict[str, int] = {}
    for lang_key, hints in LANGUAGE_HINTS.items():
        score = 0
        for hint in hints:
            hint_norm = _strip_accents(hint)
            if re.search(r"\b" + re.escape(hint_norm) + r"\b", norm):
                score += 2 if " " in hint_norm else 1
        scores[lang_key] = score
    best_lang, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score >= 2:
        return best_lang
    if best_score == 1 and _detect_artwork_query(norm):
        return best_lang
    return None


def _language_key_to_code(lang_key: str) -> str | None:
    for code, key in WHISPER_LANG_TO_KEY.items():
        if key == lang_key:
            return code
    return None


def _parse_language_switch_marker(text: str) -> str | None:
    if not text:
        return None
    cleaned = _strip_accents(text)
    marker = _strip_accents(LANGUAGE_SWITCH_MARKER)
    if not cleaned.startswith(marker):
        return None
    target = cleaned[len(marker):].strip().split(None, 1)[0] if len(cleaned) > len(marker) else ""
    return target if target in LANGUAGES else None


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


def _startup_ok(label: str, ok: bool, detail: str = "") -> None:
    status = "OK" if ok else "WARN"
    suffix = f" - {detail}" if detail else ""
    print(f"[Startup] {status}: {label}{suffix}")


class _RebootRequested(Exception):
    pass


class _ProfileSkipRequested(Exception):
    pass


class MuseumHelmet:
    def __init__(self):
        print("\n[Startup] ATLAS boot check")
        print(f"[Startup] Working directory: {os.getcwd()}")
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        _startup_ok("Gemini API key", bool(self.gemini_api_key), "set" if self.gemini_api_key else "missing GEMINI_API_KEY")
        _startup_ok("YOLO weights", os.path.exists(YOLO_WEIGHTS_PATH), YOLO_WEIGHTS_PATH)
        _startup_ok("Piper voice directory", os.path.isdir(PIPER_DATA_DIR), PIPER_DATA_DIR)
        _startup_ok("Phrase cache directory", os.path.isdir(ACK_CACHE_DIR), ACK_CACHE_DIR)
        _startup_ok("Artwork sheets directory", os.path.isdir(os.path.join("data", "artworks")), os.path.join("data", "artworks"))
        print(
            "[Startup] Config: "
            f"camera={CAMERA_INDEX}, mic={MIC_DEVICE}, ev3={EV3_MAC}, "
            f"yolo_imgsz={YOLO_IMGSZ}, detect_every={DETECT_EVERY_N_FRAMES}"
        )
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Put it in .env.")
        self.client = genai.Client(api_key=self.gemini_api_key)

        print(f"[STT] Loading Whisper '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE} ({WHISPER_COMPUTE})...")
        self.whisper = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        print("[STT] Whisper ready.")

        self.current_language = DEFAULT_LANGUAGE
        self.language_lock = threading.Lock()
        self.profile_language_locked = False

        self.has_display = _has_display()
        if self.has_display:
            print("[Display] Display detected.")
        else:
            print("[Display] No display detected, running headless.")

        self.rag = None
        self.rag_lock = threading.Lock()
        self.rag_ready = threading.Event()
        self.rag_failed = False

        self.yolo_model = None
        self.yolo_device = "cuda"
        self.yolo_lock = threading.Lock()
        self.yolo_ready = threading.Event()
        self.yolo_failed = False

        self.last_seen_object = None
        self.object_first_seen_time = None
        self.last_object_trigger_time: dict[str, float] = {name: 0.0 for name in YOLO_TO_SLOT.keys()}
        self.last_terminal_objects = None

        self.utterance_queue: queue.Queue = queue.Queue()
        self.request_queue: queue.Queue = queue.Queue()

        self.stop_event = threading.Event()
        self.is_busy_event = threading.Event()
        self.in_profile_flow = threading.Event()
        self.demo_ready_event = threading.Event()
        self.camera_ready_event = threading.Event()
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

Multilingual switching
Atlas supports English, French, Spanish, and Italian. If the visitor asks to change language, speak in another language, or says something like "Parle-moi en francais", do not refuse and do not explain that you are set to another language. If the system has not already handled the switch, reply with exactly one of these markers and nothing else:
LANGUAGE_SWITCH:english
LANGUAGE_SWITCH:french
LANGUAGE_SWITCH:spanish
LANGUAGE_SWITCH:italian
If the LANGUAGE INSTRUCTION says to respond in French, Spanish, or Italian, fully answer in that language even if the speech transcript contains a few English-looking words from recognition errors.
Never say that this experience is English-only.

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

    def _motor_raise_all(self) -> None:
        if not self.motor.connected:
            return
        with self._motor_lock:
            if self._currently_raised_slot is None:
                return
            print("[Motor] raise all")
            ok = self.motor.raise_all()
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
                self._motor_raise_all()

    def _load_rag(self):
        with self.rag_lock:
            if self.rag is not None:
                return self.rag
            if self.rag_failed:
                return None
            try:
                print("[RAG] Loading sheets and embedding model...")
                self.rag = RAG()
                print(f"[RAG] Collection size: {self.rag.collection.count()} sheet(s)")
            except Exception as e:
                self.rag_failed = True
                print(f"[RAG] Failed to load: {e}. Continuing without museum sheets.")
            finally:
                self.rag_ready.set()
            return self.rag

    def _start_rag_background(self) -> None:
        threading.Thread(target=self._load_rag, daemon=True).start()

    def _load_yolo(self):
        if not ENABLE_YOLO:
            self.yolo_ready.set()
            return None
        with self.yolo_lock:
            if self.yolo_model is not None or self.yolo_failed:
                return self.yolo_model
            try:
                from ultralytics import YOLO
                print(f"[YOLO] Loading {YOLO_WEIGHTS_PATH}...")
                model = YOLO(YOLO_WEIGHTS_PATH)
                try:
                    model.to("cuda")
                    self.yolo_device = "cuda"
                    print("[YOLO] Loaded on CUDA.")
                except Exception as e:
                    self.yolo_device = "cpu"
                    print(f"[YOLO] WARNING: could not move to CUDA: {e}")
                    print("[YOLO] Using CPU for detection.")
                self.yolo_model = model
            except Exception as e:
                self.yolo_failed = True
                print(f"[YOLO] Failed to load: {e}. Continuing without detection.")
            finally:
                self.yolo_ready.set()
            return self.yolo_model

    def _start_yolo_background(self) -> None:
        threading.Thread(target=self._load_yolo, daemon=True).start()

    def _warm_up_yolo(self) -> None:
        if not ENABLE_YOLO or self.yolo_model is None:
            return
        try:
            print("[YOLO] Warming up inference...")
            dummy = np.zeros((CAMERA_PROCESS_SIZE[1], CAMERA_PROCESS_SIZE[0], 3), dtype=np.uint8)
            t0 = time.time()
            self.yolo_model.predict(dummy, imgsz=YOLO_IMGSZ, verbose=False, device=self.yolo_device)
            print(f"[YOLO] Warmup ready in {time.time() - t0:.2f}s.")
        except Exception as e:
            print(f"[YOLO] Warmup skipped: {e}")

    def _warm_up_whisper(self) -> None:
        try:
            print("[STT] Warming up Whisper decode...")
            silence = np.zeros(int(MIC_SAMPLE_RATE * 0.6), dtype=np.int16)
            t0 = time.time()
            self._transcribe_audio(silence, force_language=DEFAULT_LANG_CODE)
            print(f"[STT] Whisper warmup ready in {time.time() - t0:.2f}s.")
        except Exception as e:
            print(f"[STT] Whisper warmup skipped: {e}")

    def _warm_up_gemini(self) -> None:
        print(f"[Gemini] Warming up {GEMINI_MODEL_CAMERA} connection...")
        result_holder: dict[str, str] = {}

        def run():
            try:
                result_holder["text"] = self._gemini_try_once(
                    GEMINI_MODEL_CAMERA,
                    "Reply with exactly: OK",
                )
            except Exception as e:
                result_holder["error"] = str(e)

        t0 = time.time()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=STARTUP_GEMINI_WARMUP_TIMEOUT_SECONDS)
        if thread.is_alive():
            print("[Gemini] Warmup still pending; continuing startup.")
            return
        if "error" in result_holder:
            print(f"[Gemini] Warmup skipped: {result_holder['error']}")
            return
        print(f"[Gemini] Warmup ready in {time.time() - t0:.2f}s.")

    def _preload_competition_runtime(self) -> None:
        print("[Startup] Competition preload: caching phrases and loading models before demo start.")
        t0 = time.time()
        self._prepare_ack_wavs()
        self._load_rag()
        self._load_yolo()
        self._warm_up_yolo()
        self._warm_up_whisper()
        self._warm_up_gemini()
        print(f"[Startup] Competition preload complete in {time.time() - t0:.2f}s.")

    def _wait_for_terminal_start(self) -> None:
        if not STARTUP_WAIT_FOR_TERMINAL:
            return
        print("\n[Startup] ATLAS is fully preloaded.")
        print("[Startup] Press Enter when the helmet is on and you want ATLAS to start talking.")
        try:
            input("> ")
        except EOFError:
            print("[Startup] No terminal input available; starting now.")

    def _prepare_ack_wavs(self) -> None:
        print(f"[Piper] Checking local phrase cache at {ACK_CACHE_DIR} ...")
        rendered = 0
        cached = 0
        if PRECACHE_ALL_LANGUAGES:
            languages_to_prepare = list(LANGUAGES.keys())
        else:
            languages_to_prepare = [DEFAULT_LANGUAGE]
        for lang_name in languages_to_prepare:
            cfg = LANGUAGES[lang_name]
            voice = cfg["piper_voice"]
            self._ack_wavs[lang_name] = {"first_try": []}
            for i, phrase in enumerate(cfg["ack_first"]):
                path = os.path.join(ACK_CACHE_DIR, f"ack_first_{lang_name}_{i}_{PIPER_CACHE_TAG}.wav")
                already = os.path.exists(path) and os.path.getsize(path) > 0
                if _ensure_ack_wav(voice, phrase, path):
                    self._ack_wavs[lang_name]["first_try"].append(path)
                    if already:
                        cached += 1
                    else:
                        rendered += 1
            for kind, src_key in CACHED_PHRASE_KEYS:
                path = os.path.join(ACK_CACHE_DIR, f"{kind}_{lang_name}_{PIPER_CACHE_TAG}.wav")
                already = os.path.exists(path) and os.path.getsize(path) > 0
                if _ensure_ack_wav(voice, cfg[src_key], path):
                    self._ack_wavs[lang_name][kind] = path
                    if already:
                        cached += 1
                    else:
                        rendered += 1
        print(f"[Piper] Local phrases ready. Cached: {cached}, newly rendered: {rendered}.")

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
        if WHISPER_LANGUAGE_LOCK_AFTER_PROFILE and self.profile_language_locked:
            return
        with self.language_lock:
            if self.current_language != lang_key:
                print(f"[Lang] auto-detected: {self.current_language} -> {lang_key}")
                self.current_language = lang_key

    def _lock_profile_language(self, lang_key: str) -> None:
        if lang_key not in LANGUAGES:
            return
        with self.language_lock:
            if self.current_language != lang_key:
                print(f"[Lang] profile locked: {self.current_language} -> {lang_key}")
            else:
                print(f"[Lang] profile locked: {lang_key}")
            self.current_language = lang_key
            self.profile_language_locked = True

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

    def say_phrase_blocking(self, phrase_key: str) -> None:
        lang = self._get_active_language()
        text = LANGUAGES[lang].get(phrase_key, "")
        lang_cache = self._ack_wavs.setdefault(lang, {"first_try": []})
        wav_path = lang_cache.get(phrase_key)
        if not wav_path and text:
            voice = LANGUAGES[lang]["piper_voice"]
            path = os.path.join(ACK_CACHE_DIR, f"{phrase_key}_{lang}_{PIPER_CACHE_TAG}.wav")
            if _ensure_ack_wav(voice, text, path):
                lang_cache[phrase_key] = path
                wav_path = path
        print(f"[TTS] {text}")
        self.is_busy_event.set()
        self.speak_start_time = time.time()
        try:
            if wav_path:
                self._play_cached_wav(wav_path)
            else:
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

    def _gemini_request_with_retries(
        self,
        prompt: str,
        ack_enabled: bool,
        primary_model: str = GEMINI_MODEL_PRIMARY,
        fallback_model: str = GEMINI_MODEL_FALLBACK,
    ) -> tuple[str, str]:
        result_holder: dict = {}

        def attempt(model, key):
            try:
                result_holder[key] = ("ok", self._gemini_try_once(model, prompt))
            except Exception as e:
                result_holder[key] = ("err", str(e))

        t1 = threading.Thread(target=attempt, args=(primary_model, "t1"), daemon=True)
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
        print(f"[Gemini] {primary_model} attempt 1 failed/empty: {payload[:120] if payload else '(empty)'}")
        if ack_enabled:
            second_wav = self._ack_wavs.get(self._get_active_language(), {}).get("second_try")
            if second_wav:
                print("[ack] (sorry, one second)")
                self._play_cached_wav(second_wav)
        try:
            text = self._gemini_try_once(primary_model, prompt)
            if text:
                return (text, "ok")
            print(f"[Gemini] {primary_model} attempt 2 returned empty.")
        except Exception as e:
            print(f"[Gemini] {primary_model} attempt 2 failed: {e}")
        if fallback_model != primary_model:
            try:
                print(f"[Gemini] falling back to {fallback_model}")
                text = self._gemini_try_once(fallback_model, prompt)
                if text:
                    return (text, "ok")
                print("[Gemini] fallback returned empty.")
            except Exception as e:
                print(f"[Gemini] fallback failed: {e}")
        return ("", "failed")

    def _retrieve_rag_context(self, query: str) -> str:
        rag = self._load_rag()
        if rag is None:
            return ""
        results = rag.search(query, n=RAG_TOP_K)
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
        rag = self._load_rag()
        if rag is None:
            return ""
        sheet = rag.sheet_by_id(artwork_id)
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
        artwork_id = _detect_artwork_query(user_text)
        if artwork_id:
            rag_block = self._retrieve_rag_by_id(artwork_id)
        else:
            rag_block = self._retrieve_rag_context(user_text)
        profile_block = self._profile_block()
        return f"""{self.system_prompt}

LANGUAGE INSTRUCTION: {directive}

{self.formatting_rules}

{EXHIBIT_CONTEXT}

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
        spoken_name = _artwork_spoken_name(artwork_id or object_name)
        if artwork_id:
            rag_block = self._retrieve_rag_by_id(artwork_id)
        else:
            rag_block = self._retrieve_rag_context(object_name)
        profile_block = self._profile_block()
        return f"""You are Atlas, a warm museum guide speaking aloud to a visitor.

LANGUAGE INSTRUCTION: {directive}

{self.formatting_rules}

{EXHIBIT_CONTEXT}

{profile_block}
{rag_block}
Camera event:
The visitor has been steadily looking at "{spoken_name}" for at least {OBJECT_HOLD_SECONDS} seconds.

Task:
Give a short, natural museum-guide explanation of "{spoken_name}".
Do NOT mention detection or observation; just speak as if you noticed it yourself.
Keep it to 1 short sentence, or 2 very short sentences at most.
Use the museum sheet as ground truth if present.
{directive}
"""

    def _handle_request(self, kind: str, text: str, created_at: float | None = None) -> None:
        if created_at is not None:
            print(f"[Timing] {kind} queue wait: {time.time() - created_at:.2f}s")
        if kind == "user":
            prompt = self._build_user_prompt(text)
            self._memory_append("user", text)
            ack_enabled = True
            primary_model = GEMINI_MODEL_PRIMARY
        elif kind == "object":
            prompt = self._build_object_prompt(text)
            self._memory_append("camera", text)
            ack_enabled = False
            primary_model = GEMINI_MODEL_CAMERA
        else:
            return
        print(f"[Gemini] thinking with {primary_model} ...")
        self.is_busy_event.set()
        gemini_start = time.time()
        try:
            response, status = self._gemini_request_with_retries(prompt, ack_enabled, primary_model=primary_model)
        except Exception as e:
            print(f"[Gemini] unexpected error: {e}")
            response, status = "", "failed"
        print(f"[Timing] {kind} Gemini: {time.time() - gemini_start:.2f}s")
        if status == "failed":
            failure_wav = self._ack_wavs.get(self._get_active_language(), {}).get("failure")
            if failure_wav:
                self._play_cached_wav(failure_wav)
            if kind == "object":
                self._motor_raise_all()
            if kind == "user":
                with self.memory_lock:
                    if self.memory and self.memory[-1] == ("user", text):
                        self.memory.pop()
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()
            return
        if not response:
            if kind == "object":
                self._motor_raise_all()
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()
            return
        if kind == "user":
            target_language = _parse_language_switch_marker(response)
            if target_language:
                print(f"[Lang] Gemini switch marker -> {target_language}")
                self._lock_profile_language(target_language)
                self.say_phrase_blocking("language_changed")
                with self.memory_lock:
                    if self.memory and self.memory[-1] == ("user", text):
                        self.memory.pop()
                self.is_busy_event.clear()
                self.last_speak_end_time = time.time()
                return
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
        tts_start = time.time()
        try:
            self._speak_full(sanitized)
        finally:
            if kind == "object":
                self._motor_raise_all()
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()
            print(f"[Timing] {kind} TTS/playback: {time.time() - tts_start:.2f}s")
        self._memory_append("assistant", sanitized)

    def _gemini_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                req = self.request_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle_request(req.get("kind"), req.get("text", ""), req.get("created_at"))

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
                        self.utterance_queue.put({"audio": utterance, "duration": duration, "created_at": time.time()})
                    audio_buffer.clear()
                    utt_start_time = None
                    last_voice_time = None
                elif (utt_start_time is not None and (now - utt_start_time) >= VAD_MAX_UTTERANCE_DURATION):
                    is_speaking = False
                    utterance = np.concatenate(audio_buffer)
                    self.utterance_queue.put({"audio": utterance, "duration": now - utt_start_time, "created_at": time.time()})
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

    def _transcribe_audio(
        self,
        audio_int16: np.ndarray,
        force_language: str | None = None,
        beam_size: int | None = None,
    ) -> tuple[str, str]:
        audio_float = audio_int16.astype(np.float32) / 32768.0
        t0 = time.time()
        try:
            kwargs = {
                "beam_size": beam_size or WHISPER_BEAM_SIZE,
                "vad_filter": False,
                "condition_on_previous_text": False,
            }
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
        print(f"[Timing] STT: {time.time() - t0:.2f}s")
        return (text.strip(), lang_code)

    def _should_try_french_rescue(self, text: str, lang_code: str) -> bool:
        if not FRENCH_RESCUE_ENABLED or lang_code == "fr":
            return False
        active_lang = self._get_active_language()
        if active_lang == "french":
            return True
        if _detect_artwork_query(text) is not None:
            return False
        norm = _strip_accents(text)
        return any(word in norm for word in ("joconde", "francais", "etoile", "masque"))

    def _rescue_french_transcription(self, audio_int16: np.ndarray, text: str, lang_code: str) -> tuple[str, str]:
        if not self._should_try_french_rescue(text, lang_code):
            return (text, lang_code)
        fr_text, fr_lang = self._transcribe_audio(
            audio_int16,
            force_language="fr",
            beam_size=FRENCH_RESCUE_BEAM_SIZE,
        )
        if not fr_text:
            return (text, lang_code)
        if _detect_artwork_query(fr_text) or _detect_language_switch(fr_text) or self._get_active_language() == "french":
            print(f"[STT] French rescue accepted: {text!r} -> {fr_text!r}")
            return (fr_text, fr_lang)
        print(f"[STT] French rescue ignored: {fr_text!r}")
        return (text, lang_code)

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
            if not self.demo_ready_event.is_set():
                continue
            force_language = None
            if WHISPER_LANGUAGE_LOCK_AFTER_PROFILE and self.profile_language_locked:
                active_lang = self._get_active_language()
                for code, key in WHISPER_LANG_TO_KEY.items():
                    if key == active_lang:
                        force_language = code
                        break
            text, lang_code = self._transcribe_audio(audio_int16, force_language=force_language)
            if not text:
                continue
            text, lang_code = self._rescue_french_transcription(audio_int16, text, lang_code)
            lang_key = WHISPER_LANG_TO_KEY.get(lang_code, DEFAULT_LANGUAGE)
            inferred_lang = _detect_language_switch(text) or _infer_language_from_text(text)
            if inferred_lang and inferred_lang in LANGUAGES and inferred_lang != lang_key:
                inferred_code = _language_key_to_code(inferred_lang)
                if inferred_code:
                    print(f"[Lang] transcript hints override: {lang_key} -> {inferred_lang}")
                    lang_key = inferred_lang
                    lang_code = inferred_code
                    if self.profile_language_locked and inferred_lang != self._get_active_language():
                        self._lock_profile_language(inferred_lang)
            self._set_active_language(lang_key)
            print(f"\n[Heard] ({lang_code}, dur={duration:.1f}s): {text}")
            self._process_transcribed(text.lower().strip(), created_at=item.get("created_at"))

    def _process_transcribed(self, text: str, created_at: float | None = None) -> None:
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
            self._motor_raise_all()
            self.stop_event.set()
            return
        target_language = _detect_language_switch(text)
        if target_language:
            print(f"[Lang] explicit switch request -> {target_language}: {text}")
            self._lock_profile_language(target_language)
            self.say_phrase_blocking("language_changed")
            return
        for w in WAKE_WORDS:
            if text.startswith(w + " ") or text == w:
                text = text[len(w):].strip()
                break
        if not text:
            return
        self.request_queue.put({"kind": "user", "text": text, "created_at": created_at or time.time()})

    def _wait_for_profile_utterance(self, timeout_seconds: float = PROFILE_LISTEN_TIMEOUT_SECONDS) -> np.ndarray | None:
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

    def _ask_and_capture(self, phrase_key: str, retries: int = PROFILE_MAX_RETRIES) -> str | None:
        for attempt in range(retries + 1):
            self.say_phrase_blocking(phrase_key)
            self._drain_profile_inbox()
            audio = self._wait_for_profile_utterance()
            if audio is None:
                if attempt < retries:
                    self.say_phrase_blocking("didnt_catch")
                    continue
                return None
            text, lang_code = self._transcribe_audio(audio, beam_size=WHISPER_PROFILE_BEAM_SIZE)
            if not text:
                if attempt < retries:
                    self.say_phrase_blocking("didnt_catch")
                    continue
                return None
            if _is_reboot_phrase(text):
                raise _RebootRequested()
            if _is_profile_skip_phrase(text):
                raise _ProfileSkipRequested()
            lang_key = WHISPER_LANG_TO_KEY.get(lang_code, DEFAULT_LANGUAGE)
            self._lock_profile_language(lang_key)
            print(f"[Profile heard] ({lang_code}): {text}")
            return text.strip()
        return None

    def _run_profile_flow(self) -> None:
        self.in_profile_flow.set()
        self._drain_profile_inbox()
        new_profile: dict[str, str] = {}
        try:
            self.say_phrase_blocking("greeting")
            age_text = self._ask_and_capture("ask_age")
            new_profile["age"] = age_text or PROFILE_DEFAULT["age"]
            age_value = _parse_age(age_text or "")
            if age_value is not None and age_value < 20:
                new_profile["profession"] = "student"
                print("[Profile] Age under 20, assuming profession: student")
            else:
                prof_text = self._ask_and_capture("ask_profession")
                new_profile["profession"] = prof_text or PROFILE_DEFAULT["profession"]
            interest_text = self._ask_and_capture("ask_interest")
            new_profile["interest"] = interest_text or PROFILE_DEFAULT["interest"]
            with self.profile_lock:
                self.visitor_profile = new_profile
            print(f"[Profile] Collected: {new_profile}")
            self.say_phrase_blocking("profile_thanks")
        except _ProfileSkipRequested:
            with self.profile_lock:
                self.visitor_profile = dict(PROFILE_DEFAULT)
            print(f"[Profile] Skipped by voice command. Using defaults: {PROFILE_DEFAULT}")
            self.say_phrase_blocking("profile_thanks")
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
        self._motor_raise_all()
        with self.language_lock:
            self.current_language = DEFAULT_LANGUAGE
            self.profile_language_locked = False
        self.say_phrase_blocking("reboot_ack")
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
            self.camera_ready_event.set()
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
                        results = self.yolo_model.predict(frame, imgsz=YOLO_IMGSZ, verbose=False, device=self.yolo_device)
                        result = results[0]
                        last_annotated = result.plot(boxes=True, masks=False)
                        boxes = result.boxes
                        if (boxes is not None and boxes.cls is not None and boxes.conf is not None and boxes.xyxy is not None):
                            frame_h, frame_w = frame.shape[:2]
                            frame_cx = frame_w / 2.0
                            frame_cy = frame_h / 2.0
                            max_center_dist = ((frame_cx ** 2) + (frame_cy ** 2)) ** 0.5
                            for cls_id, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
                                if conf < DETECT_CONFIDENCE_THRESHOLD:
                                    continue
                                cls_index = int(cls_id)
                                raw_name = str(result.names.get(cls_index, str(cls_index))).lower()
                                canonical_name = _normalize_yolo_class(raw_name)
                                if canonical_name is None:
                                    continue
                                x1, y1, x2, y2 = [float(v) for v in xyxy]
                                box_cx = (x1 + x2) / 2.0
                                box_cy = (y1 + y2) / 2.0
                                center_dist = (((box_cx - frame_cx) ** 2) + ((box_cy - frame_cy) ** 2)) ** 0.5
                                center_dist_norm = min(1.0, center_dist / max_center_dist) if max_center_dist else 1.0
                                center_score = 1.0 - center_dist_norm
                                priority_score = ((1.0 - CENTER_PRIORITY_WEIGHT) * float(conf)) + (CENTER_PRIORITY_WEIGHT * center_score)
                                detections.append({
                                    "name": canonical_name,
                                    "raw_name": raw_name,
                                    "confidence": float(conf),
                                    "center_score": center_score,
                                    "priority_score": priority_score,
                                    "box_center": (box_cx, box_cy),
                                })
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
        if not self.demo_ready_event.is_set():
            return
        if self.in_profile_flow.is_set():
            return
        current_time = time.time()
        if detections:
            summary = ", ".join(
                f"{d['name']} raw={d.get('raw_name', d['name'])} "
                f"conf={d['confidence']:.2f} center={d.get('center_score', 0.0):.2f}"
                for d in detections
            )
            if summary != self.last_terminal_objects:
                print(f"[Camera detected]: {summary}")
                self.last_terminal_objects = summary
        else:
            if self.last_terminal_objects is not None:
                print("[Camera detected]: none")
                self.last_terminal_objects = None
        trigger_candidates = [
            d for d in detections
            if (
                d["name"] in TRIGGER_OBJECTS
                and d["confidence"] >= _object_threshold(TRIGGER_CONFIDENCE_BY_OBJECT, d["name"], TRIGGER_CONFIDENCE_THRESHOLD)
                and d.get("center_score", 0.0) >= CENTER_ACTIVE_THRESHOLD
            )
        ]

        if not trigger_candidates:
            self.last_seen_object = None
            self.object_first_seen_time = None
            return

        dominant = max(trigger_candidates, key=lambda d: d.get("priority_score", d["confidence"]))
        dominant_name = dominant["name"]

        if dominant_name != self.last_seen_object:
            self.last_seen_object = dominant_name
            self.object_first_seen_time = current_time
            print(f"[Camera hold]: started {dominant_name}")
            return
        if self.object_first_seen_time is None:
            self.object_first_seen_time = current_time
            return
        held_long_enough = (current_time - self.object_first_seen_time) >= OBJECT_HOLD_SECONDS
        off_cooldown = (current_time - self.last_object_trigger_time.get(dominant_name, 0.0)) >= OBJECT_COOLDOWN_SECONDS
        if held_long_enough:
            busy = self.is_busy_event.is_set()
            queue_empty = self.request_queue.empty()
            if not (off_cooldown and not busy and queue_empty):
                return
            print(
                f"[Camera trigger]: {dominant_name} "
                f"(hold={current_time - self.object_first_seen_time:.2f}s, "
                f"conf={dominant['confidence']:.2f}, center={dominant.get('center_score', 0.0):.2f})"
            )
            self.last_object_trigger_time[dominant_name] = current_time
            threading.Thread(target=self._motor_raise, args=(dominant_name,), daemon=True).start()
            self.request_queue.put({"kind": "object", "text": dominant_name, "created_at": current_time})
            self.object_first_seen_time = current_time

    def start(self) -> None:
        self._preload_competition_runtime()
        print("[Motor] Starting background connection to EV3 ...")
        self.motor.connect_in_background()

        print("[Startup] Starting camera, microphone, transcription, Gemini, and motor idle threads...")
        camera_thread = threading.Thread(target=self.camera_worker, daemon=True)
        camera_thread.start()
        stt_thread = threading.Thread(target=self._listen_forever, daemon=True)
        stt_thread.start()
        transcribe_thread = threading.Thread(target=self._transcribe_worker, daemon=True)
        transcribe_thread.start()
        worker_thread = threading.Thread(target=self._gemini_worker, daemon=True)
        worker_thread.start()
        motor_idle_thread = threading.Thread(target=self._motor_idle_watcher, daemon=True)
        motor_idle_thread.start()

        if self.camera_ready_event.wait(timeout=STARTUP_CAMERA_WAIT_SECONDS):
            print("[Startup] Camera thread ready.")
        else:
            print("[Startup] Camera was not ready before timeout; continuing anyway.")
        motor_wait_start = time.time()
        while (
            not self.motor.connected
            and (time.time() - motor_wait_start) < STARTUP_MOTOR_WAIT_SECONDS
            and not self.stop_event.is_set()
        ):
            time.sleep(0.1)
        if self.motor.connected:
            print("[Startup] EV3 motor link ready.")
        else:
            print("[Startup] EV3 motor link not ready yet; it will keep connecting in background.")

        self._wait_for_terminal_start()
        self.demo_ready_event.set()
        print("[Startup] Demo started. Beginning visitor profile flow.")
        self._run_profile_flow()

        try:
            while not self.stop_event.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n[Ctrl-C] shutting down.")
        finally:
            self.stop_event.set()
            self._hard_stop_all_audio()
            try:
                self._motor_raise_all()
            except Exception:
                pass


if __name__ == "__main__":
    helmet = MuseumHelmet()

    def _emergency_shutdown(signum, frame):
        print("\n[Shutdown] Forcing immediate exit...")
        helmet.stop_event.set()
        helmet._hard_stop_all_audio()
        try:
            helmet._motor_raise_all()
        except Exception:
            pass
        os.system("pkill -9 -f piper 2>/dev/null")
        os.system("pkill -9 aplay 2>/dev/null")
        os._exit(0)

    signal.signal(signal.SIGINT, _emergency_shutdown)
    signal.signal(signal.SIGTERM, _emergency_shutdown)

    helmet.start()
