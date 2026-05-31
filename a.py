import pyspacemouse

device = pyspacemouse.open()

while True:
    state = device.read()
    print(state)