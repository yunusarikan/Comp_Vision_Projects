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
    
# Load the trained YOLOv5 model
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
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

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


def move_mouse_drastically():
    current_x, current_y = pyautogui.position()

    
    offset_x = -60
    offset_y = 0
    pyautogui.moveTo(current_x + offset_x, current_y + offset_y)


def main_loop(stop_event):
    last_hp_bar_state = False 

    screen_center_x = METIN_WINDOW['width'] // 2  # 640
    screen_center_y = METIN_WINDOW['height'] // 2  # 360

    while not stop_event.is_set():
        if keyboard.is_pressed('q'):
            print("Stop")
            break

        
        detections, hp_bar_detected = detect_objects()

        
        if hp_bar_detected:
            last_hp_bar_state = True
            print("HP_bar class detected. Skipping collection...")
            time.sleep(1)
            continue

        if last_hp_bar_state and not hp_bar_detected:
            print("HP_bar class disappeared. Performing collection once...")
            pydirectinput.press('z') 
            time.sleep(0.5)
            last_hp_bar_state = False

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

                stone_center_x = METIN_WINDOW['x'] + (x1 + x2) // 2
                stone_center_y = METIN_WINDOW['y'] + (y1 + y2) // 2

                if is_point_in_forbidden_square(stone_center_x, stone_center_y):
                    print(f"Metinstone detected at ({stone_center_x}, {stone_center_y}) is in the forbidden square. Skipping...")
                    continue  

                
                pyautogui.moveTo(stone_center_x, stone_center_y)
                pydirectinput.leftClick()

                print(f"Closest Metinstone detected at ({stone_center_x}, {stone_center_y}) with confidence {confidence:.2f}")
                time.sleep(1) 
                pydirectinput.press('z')
        else:
            pydirectinput.mouseDown(button='right')
            move_mouse_drastically()
            pydirectinput.mouseUp(button='right')
            print("No Metinstone found. Pressing and holding right click while moving mouse drastically to the left.")

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