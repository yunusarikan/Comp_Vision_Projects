import keyboard
import time
import pydirectinput

print("Program başlatıldı.")

try:
    while True:
        pydirectinput.press('z')  # 'z' tuşuna bas ve bırakzzzz
        time.sleep(0.5)  # 1 saniye bekle
except KeyboardInterrupt:
    print("Program sonlandırıldı.")