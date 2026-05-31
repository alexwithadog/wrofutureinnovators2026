#!/usr/bin/env python3
"""
One-pin sink-current LED test for the Jetson.

Use this if the KY-016 lights from the Jetson 3.3V pin, but does not light when
the color pin is driven directly by GPIO.

Wiring for this test:
  KY-016 R       -> Jetson physical pin 1 (3.3V)
  KY-016 - / GND -> Jetson physical pin 29

This uses GPIO pin 29 as the ground switch. When pin 29 is LOW, red should turn on.
When pin 29 is HIGH, red should turn off.
"""
import os
import sys
import time


SINK_PIN = 29
JETSON_MODEL_NAME = "JETSON_ORIN_NANO"


def main() -> int:
    os.environ.setdefault("JETSON_MODEL_NAME", JETSON_MODEL_NAME)
    try:
        import Jetson.GPIO as GPIO
    except Exception as e:
        print(f"Could not import Jetson.GPIO: {e}")
        return 1

    print("ATLAS sink-current LED test")
    print("Wiring:")
    print("  KY-016 R       -> Jetson pin 1 (3.3V)")
    print(f"  KY-016 - / GND -> Jetson pin {SINK_PIN}")
    print("LOW should turn red ON. HIGH should turn red OFF.")
    print("Press Ctrl-C to stop.")

    try:
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(SINK_PIN, GPIO.OUT, initial=GPIO.HIGH)
        while True:
            print("GPIO LOW: LED should be ON")
            GPIO.output(SINK_PIN, GPIO.LOW)
            time.sleep(2.0)
            print("GPIO HIGH: LED should be OFF")
            GPIO.output(SINK_PIN, GPIO.HIGH)
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nStopping sink test.")
    finally:
        try:
            GPIO.output(SINK_PIN, GPIO.HIGH)
            GPIO.cleanup(SINK_PIN)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
