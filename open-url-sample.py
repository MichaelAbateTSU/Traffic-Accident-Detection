from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import datetime
import pyautogui

browser = webdriver.Firefox()
# Maximize the browser window
browser.maximize_window()
url = "https://smartway.tn.gov/allcams/camera/3245"
browser.get( url )
time.sleep(5)

for i in range(5):
    time.sleep(2)
    t = datetime.datetime.now()
    s = str(t.year) + '-' + str(t.month) + '-' +str(t.day) \
        + '-' +str(t.hour) + '-' +str(t.minute)+ '-' +str(t.second) \
        + '.png'

    pyautogui.screenshot( s )

browser.quit()
