import hid as hid
import serial2arduino as ardu
from numpy import interp

for device in hid.enumerate():
    print(f"0x{device['vendor_id']:04x}:0x{device['product_id']:04x} {device['product_string']}")


gamepad = hid.device()
gamepad.open(0x054c, 0x0268) #PS3
#gamepad.open(0x11ff, 0x3331) #Generico PC
gamepad.set_nonblocking(True)

while True:
    report = gamepad.read(32)
    if report:
        print(report)
        ardu.turn(interp(report[8], [0, 255], [45, 135]))

        ardu.throttle(interp(report[9], [0, 128], [58, 30]))

        #for i in (report):
        #    print('[' + bin(i) + '], ', end='')
        #print('')


