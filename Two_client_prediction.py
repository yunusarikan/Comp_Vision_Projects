import pyautogui
import pydirectinput
import time
import cv2
import numpy as np
import keyboard
import torch
from ultralytics import YOLO
from threading import Event
import math

if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA kullanılıyor. Cihaz:", torch.cuda.get_device_name(0))
else:
    device = 'cpu'
    print("CUDA kullanılamıyor. CPU kullanılıyor.")
    
model = YOLO("C:/Users/YUYU/Desktop/VisionProject/best.pt")
model.to(device)

CLIENT_REGIONS = {
    1: {'x': 0, 'y': 0, 'width': 960, 'height': 1080},
    2: {'x': 960, 'y': 0, 'width': 960, 'height': 1080}
}

FORBIDDEN_SQUARES = {
    1: {'x1': 10, 'y1': 70, 'x2': 70, 'y2': 80},
    2: {'x1': 970, 'y1': 70, 'x2': 1030, 'y2': 80}
}

def is_point_in_forbidden_square(client_id, x, y):
    square = FORBIDDEN_SQUARES[client_id]
    return (square['x1'] <= x <= square['x2'] and square['y1'] <= y <= square['y2'])

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def detect_objects(client_id):
    region = CLIENT_REGIONS[client_id]
    screenshot = pyautogui.screenshot(region=(region['x'], region['y'], region['width'], region['height']))
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    results = model(screenshot)
    detections = []
    hp_bar_detected = False

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]

            if class_id == 1:
                hp_bar_detected = True

            if class_id == 2:
                detections.append([x1, y1, x2, y2, confidence, class_id])

    return detections, hp_bar_detected

def move_mouse_drastically(client_id):
    current_x, current_y = pyautogui.position()
    offset_x = -60 if client_id == 1 else 60
    offset_y = 0
    pyautogui.moveTo(current_x + offset_x, current_y + offset_y)

def is_mouse_in_client_region(client_id):
    current_x, current_y = pyautogui.position()
    region = CLIENT_REGIONS[client_id]
    return (region['x'] <= current_x < region['x'] + region['width'] and
            region['y'] <= current_y < region['y'] + region['height'])

def process_client(client_id):
    screen_center_x = CLIENT_REGIONS[client_id]['width'] // 2
    screen_center_y = CLIENT_REGIONS[client_id]['height'] // 2

    detections, hp_bar_detected = detect_objects(client_id)

    if hp_bar_detected:
        print(f"Client {client_id}: HP_bar class detected. Skipping collection...")
        return

    if len(detections) > 0:
        closest_stone = None
        closest_distance = float('inf')

        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            stone_center_x = (x1 + x2) // 2
            stone_center_y = (y1 + y2) // 2
            distance = calculate_distance(screen_center_x, screen_center_y, stone_center_x, stone_center_y)

            if distance < closest_distance:
                closest_distance = distance
                closest_stone = detection

        if closest_stone:
            x1, y1, x2, y2, confidence, class_id = closest_stone
            stone_center_x = CLIENT_REGIONS[client_id]['x'] + (x1 + x2) // 2
            stone_center_y = CLIENT_REGIONS[client_id]['y'] + (y1 + y2) // 2

            if is_point_in_forbidden_square(client_id, stone_center_x, stone_center_y):
                print(f"Client {client_id}: Metinstone detected at ({stone_center_x}, {stone_center_y}) is in the forbidden square. Skipping...")
                return

            pyautogui.moveTo(stone_center_x, stone_center_y)
            pydirectinput.rightClick()
            pydirectinput.press('z')
            time.sleep(0.1)

            pydirectinput.leftClick()

            print(f"Client {client_id}: Closest Metinstone detected at ({stone_center_x}, {stone_center_y}) with confidence {confidence:.2f}")
            time.sleep(0.1)
            pydirectinput.press('z')
    else:
        if not is_mouse_in_client_region(client_id):
            region = CLIENT_REGIONS[client_id]
            pyautogui.moveTo(region['x'] + region['width'] // 2, region['y'] + region['height'] // 2)

        pydirectinput.mouseDown(button='right')
        move_mouse_drastically(client_id)
        pydirectinput.mouseUp(button='right')
        print(f"Client {client_id}: No Metinstone found. Pressing and holding right click while moving mouse drastically.")

def main_loop(stop_event):
    while not stop_event.is_set():
        if keyboard.is_pressed('q'):
            print("Stop")
            break

        process_client(1)

        process_client(2)

if __name__ == "__main__":
    stop_event = Event()

    def start_main_loop():
        if not stop_event.is_set():
            print("Program başlatıldı.")
            stop_event.clear()
            main_loop(stop_event)

    def stop_main_loop():
        stop_event.set()
        print("Program durduruldu.")

    keyboard.add_hotkey('F8', start_main_loop)
    keyboard.add_hotkey('F9', stop_main_loop)

    print("F8'e basarak programı başlatabilir ve F9'a basarak durdurabilirsiniz. Çıkmak için 'esc' tuşuna basın.")
    keyboard.wait('esc')
    stop_event.set()
    print("Program sonlandırıldı.")