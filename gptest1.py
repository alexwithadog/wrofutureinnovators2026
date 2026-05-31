import Jetson.GPIO as GPIO

# Set the GPIO mode to BCM
GPIO.setmode(GPIO.BCM)

# Set the GPIO pin as an output
GPIO.setup(7, GPIO.OUT)

# Set the GPIO pin high
GPIO.output(7, GPIO.HIGH)
