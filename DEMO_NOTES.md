# ATLAS Demo Notes

## Normal Startup



EV3_MAC = "2C:6B:7D:7B:AE:02"

Before a serious demo, warm the local caches:

```bash
python3 warmup_atlas.py
```

This pre-renders the short Piper phrases and loads the local model caches for RAG, Whisper, and YOLO.

1. Start the EV3 program first.
2. Wait for the EV3 screen to say `Waiting for BT...`.
3. Start ATLAS on the Jetson:

```bash
python3 JRAG2.py
```

Expected terminal milestones:

```text
[Startup] ATLAS boot check
[RAG] Collection size: 3 sheet(s)
[STT] Whisper ready.
[YOLO] Loaded on CUDA.
[Camera] Open OK
[STT] Listening ...
[Startup] Threads started. Beginning visitor profile flow.
```

## EV3 Quick Test

Use this before the full demo if motors seem wrong:

```bash
python3 motor_test.py
```

If EV3 does not connect:

1. Confirm the EV3 motor program is running, not just paired in Bluetooth settings.
2. Confirm the EV3 screen says `Waiting for BT...`.
3. Confirm `EV3_MAC` in `JRAG2.py` and `motor_test.py` matches the active brick.
4. Restart the EV3 program and run `python3 motor_test.py` again.

## Kill A Stuck Run

```bash
pkill -9 -f 'python.*JRAG2'
pkill -9 -f piper
pkill -9 -f aplay
```

If that fails:

```bash
ps aux | grep -E 'JRAG2|piper|aplay|python' | grep -v grep
kill -9 PID_HERE
```

Last resort:

```bash
sudo reboot
```

## Camera / YOLO Tuning

During testing, watch lines like:

```text
[Camera detected]: mona_lisa raw=monalisa conf=0.18 center=0.71
[Camera hold]: started mona_lisa
[Camera trigger]: mona_lisa (hold=2.02s, conf=0.28, center=0.74)
[Timing] object queue wait: 0.01s
[Timing] object Gemini: 1.42s
[Timing] object TTS/playback: 2.10s
```

Write down real values for each object:

```text
Mona Lisa:
Starry Night:
Mask:
```

Tune these in `JRAG2.py` only after seeing real logs:

```python
TRIGGER_CONFIDENCE_BY_OBJECT
CENTER_ACTIVE_THRESHOLD
OBJECT_HOLD_SECONDS
```

## Known File Roles

Keep visible:

```text
JRAG2.py
motor_controller.py
motor_test.py
best.pt
atlas/
data/artworks/
EV3/
```

Archive/noise:

```text
useless/
```

## Before Judging

Run this checklist:

```text
EV3 battery charged
Jetson battery charged
Speaker works
Mic level works
Camera index correct
Gemini API key in .env
best.pt present
EV3 motor program uploaded and running
At least one full demo run tested end-to-end
```
