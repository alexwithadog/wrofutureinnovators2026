"""
Warm ATLAS runtime caches before a demo.

Run this once on the Jetson after installing dependencies and before judging:

    python3 warmup_atlas.py

It downloads/loads model caches where needed and pre-renders short Piper
phrases so JRAG2.py does less work during the live demo.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PIPER_DATA_DIR = Path(os.path.expanduser("~/piper_voices"))
ACK_CACHE_DIR = Path(os.path.expanduser("~/.atlas_ack_cache"))

WHISPER_MODEL_SIZE = "tiny"
YOLO_WEIGHTS_PATH = ROOT / "best.pt"

LANGUAGES = {
    "english": {
        "piper_voice": "en_US-ryan-low",
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
    },
    "french": {
        "piper_voice": "fr_FR-siwis-medium",
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
    },
    "spanish": {
        "piper_voice": "es_MX-claude-high",
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
    },
    "italian": {
        "piper_voice": "it_IT-paola-medium",
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
    },
}

CACHED_PHRASE_KEYS = [
    ("second_try", "ack_second"),
    ("failure", "failure"),
    ("greeting", "greeting"),
    ("ask_age", "ask_age"),
    ("ask_profession", "ask_profession"),
    ("ask_interest", "ask_interest"),
    ("profile_thanks", "profile_thanks"),
    ("reboot_ack", "reboot_ack"),
    ("didnt_catch", "didnt_catch"),
    ("exit_phrase", "exit_phrase"),
]


def ok(label: str, detail: str = "") -> None:
    print(f"[warmup] OK: {label}{' - ' + detail if detail else ''}")


def warn(label: str, detail: str = "") -> None:
    print(f"[warmup] WARN: {label}{' - ' + detail if detail else ''}")


def synthesize(voice: str, text: str, path: Path) -> bool:
    if path.exists() and path.stat().st_size > 0:
        ok("cached phrase", path.name)
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "-m", "piper",
        "--model", voice,
        "--data-dir", str(PIPER_DATA_DIR),
        "--length-scale", "1.00",
        "--output-file", str(path),
    ]
    try:
        subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception as exc:
        warn("Piper phrase failed", f"{path.name}: {exc}")
        return False

    if path.exists() and path.stat().st_size > 0:
        ok("rendered phrase", path.name)
        return True
    warn("Piper phrase empty", path.name)
    return False


def warm_piper() -> None:
    ACK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not PIPER_DATA_DIR.is_dir():
        warn("Piper voice directory missing", str(PIPER_DATA_DIR))

    for lang_name, cfg in LANGUAGES.items():
        voice = cfg["piper_voice"]
        for i, phrase in enumerate(cfg["ack_first"]):
            synthesize(voice, phrase, ACK_CACHE_DIR / f"ack_first_{lang_name}_{i}.wav")
        for cache_key, src_key in CACHED_PHRASE_KEYS:
            synthesize(voice, cfg[src_key], ACK_CACHE_DIR / f"{cache_key}_{lang_name}.wav")


def warm_rag() -> None:
    try:
        from atlas.rag import RAG

        rag = RAG()
        count = rag.collection.count()
        result = rag.search("tell me about the mona lisa", n=1)
        ok("RAG", f"{count} sheets, test result={result[0]['id'] if result else 'none'}")
    except Exception as exc:
        warn("RAG warmup failed", str(exc))


def warm_whisper() -> None:
    try:
        from faster_whisper import WhisperModel

        WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        ok("Whisper", WHISPER_MODEL_SIZE)
    except Exception as exc:
        warn("Whisper warmup failed", str(exc))


def warm_yolo() -> None:
    if not YOLO_WEIGHTS_PATH.exists():
        warn("YOLO weights missing", str(YOLO_WEIGHTS_PATH))
        return
    try:
        from ultralytics import YOLO

        model = YOLO(str(YOLO_WEIGHTS_PATH))
        ok("YOLO model loaded", str(YOLO_WEIGHTS_PATH.name))
        try:
            model.to("cuda")
            ok("YOLO CUDA")
        except Exception as exc:
            warn("YOLO CUDA unavailable", str(exc))
    except Exception as exc:
        warn("YOLO warmup failed", str(exc))


def main() -> None:
    print("[warmup] ATLAS cache warmup")
    print(f"[warmup] repo: {ROOT}")
    warm_piper()
    warm_rag()
    warm_whisper()
    warm_yolo()
    print("[warmup] done")


if __name__ == "__main__":
    main()
