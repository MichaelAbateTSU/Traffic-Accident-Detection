from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

import time
import datetime
import numpy as np

from ultralytics import YOLO
import cv2
import os
import shutil

# -------- YOLO SETUP --------
model = YOLO("yolov8n.pt")

# -------- FOLDERS --------
print("Clearing old images...")
if os.path.exists("captures"):
    shutil.rmtree("captures")
if os.path.exists("captures_annotated"):
    shutil.rmtree("captures_annotated")

os.makedirs("captures")
os.makedirs("captures_annotated")
print("✓ Folders ready for new captures\n")

# -------- SELENIUM SETUP --------
browser = webdriver.Firefox()
browser.maximize_window()

url = "https://smartway.tn.gov/allcams/camera/4186"
browser.get(url)

# wait for player to load
wait = WebDriverWait(browser, 15)
time.sleep(3)

# (optional) fullscreen the browser window
browser.fullscreen_window()
time.sleep(1)

# -------- CLICK THE VIDEO FULLSCREEN BUTTON --------
try:
    # wait for the fullscreen button to be present & clickable
    fs_button = wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.vjs-fullscreen-control"))
        # or: (By.CSS_SELECTOR, "button[title='Fullscreen']")
    )

    # sometimes controls hide until you hover over the video, so move mouse first
    ActionChains(browser).move_to_element(fs_button).perform()
    time.sleep(0.5)

    fs_button.click()
    print("✓ Clicked video fullscreen button")
except Exception as e:
    print("⚠ Could not click fullscreen button:", e)

time.sleep(2)  # let it go fullscreen

# -------- CAPTURE LOOP --------
for i in range(5):
    time.sleep(2)

    t = datetime.datetime.now()
    filename = f"{t.year}-{t.month}-{t.day}-{t.hour}-{t.minute}-{t.second}.png"
    filepath = os.path.join("captures", filename)

    # screenshot the now-fullscreen video/browser
    png_bytes = browser.get_screenshot_as_png()
    image = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)

    cv2.imwrite(filepath, image)
    print(f"Saved screenshot: {filepath}")

    # YOLO inference
    results = model(image)
    result = results[0]

    annotated_frame = result.plot()
    annotated_path = os.path.join("captures_annotated", filename)
    cv2.imwrite(annotated_path, annotated_frame)
    print(f"Saved annotated image: {annotated_path}")

    boxes = result.boxes
    print(f"\nDetections in frame {i+1}:")
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_name = result.names[cls]
        print(f"  Class: {class_name} (ID: {cls}), "
              f"Confidence: {conf:.2f}, "
              f"Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
    print()

browser.quit()
print("\n✓ Processing complete!")
print("✓ Raw screenshots in: captures/")
print("✓ Annotated images in: captures_annotated/")
