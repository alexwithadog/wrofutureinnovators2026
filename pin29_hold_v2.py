import Jetson.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(29, GPIO.OUT, initial=GPIO.HIGH)

print("Pin 29 HIGH. Ctrl-C to exit.")
try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
