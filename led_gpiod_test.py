#!/usr/bin/env python3
"""
Standalone RGB LED test using libgpiod's gpioset command.

Use this if Jetson.GPIO imports but the physical BOARD pins do not toggle.

Current wiring expected:
  KY-016 R       -> Jetson physical pin 15
  KY-016 G       -> Jetson physical pin 18
  KY-016 B       -> Jetson physical pin 32
  KY-016 - / GND -> Jetson physical pin 30

Important:
  The gpiochip/line numbers below may need changing depending on Jetson pinmux.
  Run `sudo gpioinfo` and send the output if this does not light the LED.
"""
import argparse
import subprocess
import sys
import time


# First guess for the current Jetson header diagram:
# pin 15 GPIO27, pin 18 GPIO35, pin 32 GPIO09.
# These line numbers are intentionally editable from the command line.
DEFAULT_RED = "gpiochip0:103"
DEFAULT_GREEN = "gpiochip0:116"
DEFAULT_BLUE = "gpiochip0:106"


def parse_line(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Use format gpiochipN:LINE, for example gpiochip0:103")
    chip, line = value.split(":", 1)
    if not chip.startswith("gpiochip") or not line.isdigit():
        raise argparse.ArgumentTypeError("Use format gpiochipN:LINE, for example gpiochip0:103")
    return chip, line


def run_color(lines: dict[str, tuple[str, str]], rgb: tuple[int, int, int], hold: float) -> None:
    by_chip: dict[str, list[str]] = {}
    for name, on in zip(("red", "green", "blue"), rgb):
        chip, line = lines[name]
        by_chip.setdefault(chip, []).append(f"{line}={1 if on else 0}")
    procs = []
    try:
        for chip, assignments in by_chip.items():
            procs.append(subprocess.Popen(["gpioset", "--mode=wait", chip, *assignments]))
        time.sleep(hold)
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Test ATLAS RGB LED with gpioset.")
    parser.add_argument("--red", type=parse_line, default=parse_line(DEFAULT_RED))
    parser.add_argument("--green", type=parse_line, default=parse_line(DEFAULT_GREEN))
    parser.add_argument("--blue", type=parse_line, default=parse_line(DEFAULT_BLUE))
    parser.add_argument("--hold", type=float, default=1.0)
    args = parser.parse_args()

    if subprocess.run(["which", "gpioset"], stdout=subprocess.DEVNULL).returncode != 0:
        print("gpioset not found. Install gpiod first: sudo apt install gpiod")
        return 1

    lines = {"red": args.red, "green": args.green, "blue": args.blue}
    print("ATLAS gpioset LED test")
    print(f"red={args.red[0]} line {args.red[1]}")
    print(f"green={args.green[0]} line {args.green[1]}")
    print(f"blue={args.blue[0]} line {args.blue[1]}")
    print("Press Ctrl-C to stop.")

    colors = [
        ("red", (1, 0, 0)),
        ("green", (0, 1, 0)),
        ("blue", (0, 0, 1)),
        ("yellow", (1, 1, 0)),
        ("purple", (1, 0, 1)),
        ("cyan", (0, 1, 1)),
        ("white", (1, 1, 1)),
        ("off", (0, 0, 0)),
    ]
    try:
        while True:
            for name, rgb in colors:
                print(f"Showing {name}...")
                run_color(lines, rgb, args.hold)
            print("Cycle complete.\n")
    except KeyboardInterrupt:
        print("\nStopping gpioset LED test.")
        run_color(lines, (0, 0, 0), 0.2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
