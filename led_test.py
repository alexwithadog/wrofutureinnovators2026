#!/usr/bin/env python3
"""
Standalone KY-016 RGB LED test for the Jetson GPIO header.

Wiring expected:
  KY-016 R       -> Jetson physical pin 29
  KY-016 G       -> Jetson physical pin 31
  KY-016 B       -> Jetson physical pin 33
  KY-016 - / GND -> Jetson physical pin 30

Run:
  python3 led_test.py

If permissions fail:
  sudo python3 led_test.py
"""
import argparse
import sys
import time


RED_PIN = 29
GREEN_PIN = 31
BLUE_PIN = 33


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the ATLAS KY-016 RGB status LED.")
    parser.add_argument(
        "--common-anode",
        action="store_true",
        help="Invert GPIO logic if your RGB LED is common-anode instead of KY-016 common-cathode.",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=1.0,
        help="Seconds to hold each color.",
    )
    args = parser.parse_args()

    try:
        import Jetson.GPIO as GPIO
    except Exception as e:
        print(f"Could not import Jetson.GPIO: {e}")
        print("Run this on the Jetson, not on Windows/macOS.")
        return 1

    active = GPIO.LOW if args.common_anode else GPIO.HIGH
    inactive = GPIO.HIGH if args.common_anode else GPIO.LOW

    def level(on: bool):
        return active if on else inactive

    def set_color(name: str, r: bool, g: bool, b: bool) -> None:
        print(f"Showing {name}...")
        GPIO.output(RED_PIN, level(r))
        GPIO.output(GREEN_PIN, level(g))
        GPIO.output(BLUE_PIN, level(b))
        time.sleep(args.hold)

    print("ATLAS RGB LED test")
    print(f"Using BOARD pins: R={RED_PIN}, G={GREEN_PIN}, B={BLUE_PIN}, GND=30")
    print("Make sure the Jetson is wired with power off before running this.")
    print("Press Ctrl-C to stop.")

    try:
        GPIO.setmode(GPIO.BOARD)
        for pin in (RED_PIN, GREEN_PIN, BLUE_PIN):
            GPIO.setup(pin, GPIO.OUT, initial=inactive)

        while True:
            set_color("red", True, False, False)
            set_color("green", False, True, False)
            set_color("blue", False, False, True)
            set_color("yellow", True, True, False)
            set_color("purple", True, False, True)
            set_color("cyan", False, True, True)
            set_color("white", True, True, True)
            set_color("off", False, False, False)
            print("Cycle complete.\n")
    except KeyboardInterrupt:
        print("\nStopping LED test.")
    finally:
        try:
            GPIO.output(RED_PIN, inactive)
            GPIO.output(GREEN_PIN, inactive)
            GPIO.output(BLUE_PIN, inactive)
            GPIO.cleanup()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
