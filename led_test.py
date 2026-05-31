#!/usr/bin/env python3
import os
# MUST be set BEFORE importing Jetson.GPIO
os.environ['JETSON_MODEL_NAME'] = 'JETSON_ORIN_NANO'

import Jetson.GPIO as GPIO
import time

RED_PIN   = 29
GREEN_PIN = 31
BLUE_PIN  = 33

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(RED_PIN,   GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(GREEN_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BLUE_PIN,  GPIO.OUT, initial=GPIO.LOW)

def set_color(r, g, b):
    GPIO.output(RED_PIN,   GPIO.HIGH if r else GPIO.LOW)
    GPIO.output(GREEN_PIN, GPIO.HIGH if g else GPIO.LOW)
    GPIO.output(BLUE_PIN,  GPIO.HIGH if b else GPIO.LOW)

try:
    print("RED on  (2s)")
    set_color(1, 0, 0); time.sleep(2)

    print("GREEN on  (2s)")
    set_color(0, 1, 0); time.sleep(2)

    print("BLUE on  (2s)")
    set_color(0, 0, 1); time.sleep(2)

    print("ALL on  (whitish) (2s)")
    set_color(1, 1, 1); time.sleep(2)

    print("OFF (1s)")
    set_color(0, 0, 0); time.sleep(1)

    print("Cycling R→G→B x3")
    for _ in range(3):
        set_color(1, 0, 0); time.sleep(0.4)
        set_color(0, 1, 0); time.sleep(0.4)
        set_color(0, 0, 1); time.sleep(0.4)

    print("Done.")

except KeyboardInterrupt:
    print("\nInterrupted")
finally:
    set_color(0, 0, 0)
    GPIO.cleanup()
