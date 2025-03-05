import pyautogui
import pydirectinput
import time
import cv2
import numpy as np
import keyboard
import torch
from ultralytics import YOLO
from threading import Thread, Event
import math  

if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA kullanılıyor. Cihaz:", torch.cuda.get_device_name(0))
else:
    device = 'cpu'
    print("CUDA kullanılamıyor. CPU kullanılıyor.")
    
model = YOLO("C:/Users/YUYU/Desktop/VisionProject/best.pt")
model.to(device)

METIN_WINDOW = {
    'x': 0,
    'y': 0,
    'width': 1280,
    'height': 720
}

FORBIDDEN_SQUARE = {
    'x1': 10,
    'y1': 70,
    'x2': 70,
    'y2': 80
}

def is_point_in_forbidden_square(x, y):
    return (FORBIDDEN_SQUARE['x1'] <= x <= FORBIDDEN_SQUARE['x2'] and
            FORBIDDEN_SQUARE['y1'] <= y <= FORBIDDEN_SQUARE['y2'])

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def detect_objects():
    screenshot = pyautogui.screenshot(region=(METIN_WINDOW['x'], METIN_WINDOW['y'], METIN_WINDOW['width'], METIN_WINDOW['height']))
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    results = model(screenshot)
    detections = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]

            if class_id == 2:
                detections.append([x1, y1, x2, y2, confidence, class_id])

    return detections

def move_mouse_drastically():
    current_x, current_y = pyautogui.position()
    offset_x = -60
    offset_y = 0
    pyautogui.moveTo(current_x + offset_x, current_y + offset_y)

def click_metinstone_every_5_seconds():
    last_click_time = time.time()
    while True:
        if time.time() - last_click_time >= 5:
            detections = detect_objects()
            if len(detections) > 0:
                closest_stone = None
                closest_distance = float('inf')

                for detection in detections:
                    x1, y1, x2, y2, confidence, class_id = detection
                    stone_center_x = (x1 + x2) // 2
                    stone_center_y = (y1 + y2) // 2
                    distance = calculate_distance(METIN_WINDOW['width'] // 2, METIN_WINDOW['height'] // 2, stone_center_x, stone_center_y)

                    if distance < closest_distance:
                        closest_distance = distance
                        closest_stone = detection

                if closest_stone:
                    x1, y1, x2, y2, confidence, class_id = closest_stone
                    stone_center_x = METIN_WINDOW['x'] + (x1 + x2) // 2
                    stone_center_y = METIN_WINDOW['y'] + (y1 + y2) // 2

                    if is_point_in_forbidden_square(stone_center_x, stone_center_y):
                        continue

                    pyautogui.moveTo(stone_center_x, stone_center_y)
                    pydirectinput.leftClick()
                else:
                    print("No Metinstone found.")
            
            last_click_time = time.time()
        time.sleep(1)

def main_loop(stop_event):
    last_z_press_time = time.time()

    while not stop_event.is_set():
        if keyboard.is_pressed('q'):
            print("Stop")
            break

        current_time = time.time()
        if current_time - last_z_press_time >= 1:
            pydirectinput.press('z')
            last_z_press_time = current_time
            print("'z' tuşuna basıldı.")

def start_main_loop():
    if not stop_event.is_set():
        print("Program başlatıldı.")
        stop_event.clear()
        thread = Thread(target=main_loop, args=(stop_event,))
        thread.start()

def stop_main_loop():
    stop_event.set()
    print("Program durduruldu.")

if __name__ == "__main__":
    stop_event = Event()

    keyboard.add_hotkey('F8', start_main_loop)
    keyboard.add_hotkey('F9', stop_main_loop)

    click_thread = Thread(target=click_metinstone_every_5_seconds)
    click_thread.start()

    print("F8'e basarak programı başlatabilir ve F9'a basarak durdurabilirsiniz. Çıkmak için 'esc' tuşuna basın.")
    keyboard.wait('esc')
    stop_event.set()
    print("Program sonlandırıldı.")
