#!/usr/bin/python3
import asyncio
from datetime import datetime
import evdev
import os
import re
import signal
import subprocess
import sys
import time

# Picamera2 and OpenCV (OpenCV is now only used for OCR binarization and Overlays)
from picamera2 import Picamera2, Preview
from libcamera import controls, Transform
import cv2
import numpy as np

# --- CONFIGURATION CONSTANTS ---
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
ROTATION = 0

CONTRAST = 1
BRIGHTNESS = 0.2
SATURATION = 1
SHARPNESS = 1
DISTANCE_TO_SURFACE_CM = None # Set to e.g., 24.5 for fixed focus

DEFAULT_FACTOR = 1.5
SCALE_FACTORS = [DEFAULT_FACTOR, 3, 4.5, 8]
factor = SCALE_FACTORS[0]
# The zoom level where we switch from binned to hardware crop
ZOOM_THRESHOLD = 4.0 
is_high_zoom_mode = False  # Tracks current state

# Audio and OCR/TTS Settings
AUDIO = 'aplay'
OCR_LANG = 'eng'
# Piper requires the exact model filename you downloaded
PIPER_MODEL = 'de_DE-thorsten-medium.onnx' 

ENABLE_OVERLAY = True
OVERLAY_DURATION_S = 3

# --- SHADER DEFINITIONS ---
# These run directly on the GPU for zero-latency processing
SHADER_NORMAL = None 

SHADER_INVERT = """
#version 100
precision mediump float;
varying vec2 texcoord;
uniform sampler2D tex;
void main() {
    vec4 color = texture2D(tex, texcoord);
    gl_FragColor = vec4(1.0 - color.rgb, color.a);
}
"""

SHADER_YELLOW_BLACK = """
#version 100
precision mediump float;
varying vec2 texcoord;
uniform sampler2D tex;
void main() {
    vec4 color = texture2D(tex, texcoord);
    float luma = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    float inverted = 1.0 - luma;
    gl_FragColor = vec4(inverted, inverted, 0.0, color.a);
}
"""

SHADER_HIGH_CONTRAST = """
#version 100
precision mediump float;
varying vec2 texcoord;
uniform sampler2D tex;
void main() {
    vec4 color = texture2D(tex, texcoord);
    float luma = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    float contrast = smoothstep(0.3, 0.7, luma);
    gl_FragColor = vec4(vec3(contrast), color.a);
}
"""

COLOR_MODES = [SHADER_NORMAL, SHADER_INVERT, SHADER_YELLOW_BLACK, SHADER_HIGH_CONTRAST]
color_mode_index = 0
current_preview_mode = 'DRM'

# --- GLOBALS ---
camera = None
bg_process = None
screen = (SCREEN_WIDTH, SCREEN_HEIGHT)

# --- HELPER FUNCTIONS ---
def get_best_preview_mode():
    if os.getenv('WAYLAND_DISPLAY') or os.getenv('DISPLAY'):
        print("Desktop/Cage environment detected. Using QTGL with Shaders.")
        return 'QTGL'
    print("No desktop detected. Falling back to DRM (Terminal). Shaders disabled.")
    return 'DRM'

def screen_resolution_fbset():
    try:
        result = subprocess.run(['fbset'], capture_output=True)
        output = result.stdout.decode('utf-8')
        m = re.search('mode "([0-9]+)x([0-9]+)"', output)
        if m:
            return int(m.group(1)), int(m.group(2))
    except:
        pass
    return SCREEN_WIDTH, SCREEN_HEIGHT

def overlay(text, duration_s=OVERLAY_DURATION_S):
    global camera
    if ENABLE_OVERLAY and camera:
        # Create an empty transparent buffer
        buffer = np.zeros((100, 300, 4), dtype=np.uint8)
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(buffer, text, (10, 50), font, 1.5, (0, 255, 255, 255), 3)
            camera.set_overlay(buffer, x=50, y=50)
            
            # Clear overlay after duration
            if duration_s > 0:
                asyncio.get_event_loop().call_later(duration_s, lambda: camera.set_overlay(np.zeros((1, 1, 4), dtype=np.uint8)))

def quit():
    global devices, camera
    print("Cleaning up...")
    asyncio.get_event_loop().stop()
    if camera:
        camera.stop_preview()
        camera.close()
    for dev in devices:
        try: dev.ungrab()
        except: pass

def power_off():
    print("System shutting down via keypress...")
    quit()
    subprocess.run(['sudo', 'shutdown', '-h', 'now'])

def handle_sigterm(signum, frame):
    print("Caught termination signal. Shutting down gracefully...")
    quit()

signal.signal(signal.SIGTERM, handle_sigterm)

# --- CAMERA CONTROLS ---
def color_mode():
    global color_mode_index, camera, current_preview_mode
    color_mode_index = (color_mode_index + 1) % len(COLOR_MODES)
    shader = COLOR_MODES[color_mode_index]
    
    if current_preview_mode == 'QTGL':
        camera.stop_preview()
        camera.start_preview(Preview.QTGL, gl_fragment_shader=shader)
        overlay(f'Color Mode: {color_mode_index}')
    else:
        overlay('Shaders require Cage/Desktop')

def scale(new_factor):
    global camera, factor, screen
    factor = max(1, new_factor)
    screen_w, screen_h = screen
    screen_ratio = screen_w / screen_h
    
    try:
        x, y, camera_w, camera_h = camera.camera_properties['ScalerCropMaximum']
        crop_w = int(camera_w / factor)
        crop_h = min(int(crop_w / screen_ratio), camera_h)
        window = (x, y, crop_w, crop_h)
        camera.set_controls({'ScalerCrop': window})
        overlay(f'Zoom: {factor:.2f}x')
        
        if 'AfMode' in camera.camera_controls and DISTANCE_TO_SURFACE_CM is None:
            camera.set_controls({'AfMode': controls.AfModeEnum.Auto, 'AfMetering': controls.AfMeteringEnum.Windows})
            camera.set_controls({'AfWindows': [window]})
            camera.autofocus_cycle()
    except Exception as e:
        print(f"Warning: Scale failed: {e}")

def next_factor():
    global factor
    same_or_less = [v for v in SCALE_FACTORS if v <= factor]
    if len(same_or_less) == 0:
        factor = DEFAULT_FACTOR
    else:
        closest_factor = max(same_or_less)
        i = SCALE_FACTORS.index(closest_factor)
        factor = SCALE_FACTORS[(i + 1) % len(SCALE_FACTORS)]
    scale(factor)

def zoom(change_by):
    global factor
    scale(factor + change_by)

def contrast_ctrl(multiply_by):
    global camera
    if hasattr(camera, 'camera_controls') and 'Contrast' in camera.camera_controls:
        min_c, max_c, _ = camera.camera_controls['Contrast']
        if not hasattr(contrast_ctrl, 'val'): contrast_ctrl.val = CONTRAST
        val = max(min_c, min(max_c, contrast_ctrl.val * multiply_by))
        contrast_ctrl.val = val
        camera.set_controls({'Contrast': val})
        overlay(f'Contrast: {val:.2f}')

def brightness(change_by):
    global camera
    if hasattr(camera, 'camera_controls') and 'Brightness' in camera.camera_controls:
        min_b, max_b, _ = camera.camera_controls['Brightness']
        if not hasattr(brightness, 'val'): brightness.val = BRIGHTNESS
        val = max(min_b, min(max_b, brightness.val + change_by))
        brightness.val = val
        camera.set_controls({'Brightness': val})
        overlay(f'Bright: {val:.2f}')

# --- OCR & TTS PIPELINE ---
def capture_for_ocr():
    global camera
    # Grab high-res array from camera hardware
    frame = camera.capture_array()
    
    # Process with OpenCV for maximum text clarity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cv2.imwrite('/home/pi/tmp.jpg', binarized)

def readout():
    global bg_process
    
    # Piper TTS Pipeline
    cmd = f'tesseract /home/pi/tmp.jpg /home/pi/tmp -l {OCR_LANG} && cat /home/pi/tmp.txt | ./piper --model /home/pi/{PIPER_MODEL} --output_file /home/pi/tmp.wav && {AUDIO} /home/pi/tmp.wav'
    
    if bg_process != None and bg_process.poll() == None:
        os.killpg(os.getpgid(bg_process.pid), signal.SIGTERM)
        overlay('Readout Stopped')
    else:
        subprocess.call(f'{AUDIO} plop.wav', shell=True)
        overlay('Scanning...')
        capture_for_ocr()
        overlay('Reading...')
        bg_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)

# --- INIT ---
def init_camera(width, height):
    global current_preview_mode
    try:
        picam2 = Picamera2()
        transform = Transform(hflip=1, vflip=1) if ROTATION == 180 else Transform()
        
        # Use 720p internal resolution for massive performance boost, relying on hardware upscaling
        config = picam2.create_preview_configuration(
            main={'size': (1920, 1080), 'format': 'XRGB8888'}, 
            transform=transform
        )
        picam2.configure(config)
        
        current_preview_mode = get_best_preview_mode()
        
        if current_preview_mode == 'QTGL':
            picam2.start_preview(Preview.QTGL, gl_fragment_shader=COLOR_MODES[color_mode_index])
        else:
            picam2.start_preview(Preview.DRM)

        picam2.start()
        
        # Force 30fps to keep sensor in fast binned mode
        picam2.set_controls({
            'Brightness': BRIGHTNESS,
            'Contrast': CONTRAST,
            'Saturation': SATURATION,
            'Sharpness': SHARPNESS,
            #'FrameDurationLimits': (33333, 33333) 
        })
        
        if DISTANCE_TO_SURFACE_CM is None:
            picam2.set_controls({'AfMode': controls.AfModeEnum.Continuous})
        else:
            picam2.set_controls({'AfMode': controls.AfModeEnum.Manual, 'LensPosition': 100 / DISTANCE_TO_SURFACE_CM})

        return picam2
    except Exception as e:
        print(f"Picamera2 init failed: {e}")
        sys.exit(1)

# --- INPUT HANDLING ---
async def handle_events(device):
    async for event in device.async_read_loop():
        if event.type == evdev.ecodes.EV_KEY and event.value == 0:
            code = event.code
            modifiers = device.active_keys()
            #is_shift = evdev.ecodes.KEY_LEFTSHIFT in modifiers or evdev.ecodes.KEY_RIGHTSHIFT in modifiers

            #if code == evdev.ecodes.BTN_MOUSE: next_factor()
            #elif code == evdev.ecodes.BTN_RIGHT: color_mode()
            #elif code == evdev.ecodes.BTN_MIDDLE: readout()
            
            elif code == evdev.ecodes.KEY_Q or code == evdev.ecodes.KEY_ESC: quit()
            elif code == evdev.ecodes.KEY_ENTER or code == evdev.ecodes.KEY_KPENTER: next_factor()
            elif code == evdev.ecodes.KEY_SLASH or code == evdev.ecodes.KEY_KPASTERISK: color_mode()
            
            # --- SHUTDOWN KEY ---
            # Mom's dedicated power off button (NumLock)
            elif code == evdev.ecodes.KEY_NUMLOCK: power_off()
            
            elif code == evdev.ecodes.KEY_KPASTERISK: color_mode()
            elif code == evdev.ecodes.KEY_KP5: readout()
            elif code == evdev.ecodes.KEY_KPPLUS: zoom(0.2)
            elif code == evdev.ecodes.KEY_KPMINUS: zoom(-0.2)
            elif code == evdev.ecodes.KEY_7: brightness(-0.1)
            elif code == evdev.ecodes.KEY_9: brightness(0.1)
            elif code == evdev.ecodes.KEY_1: contrast_ctrl(0.5)
            elif code == evdev.ecodes.KEY_3: contrast_ctrl(2)

# --- MAIN RUNNER ---
screen = screen_resolution_fbset()
camera = init_camera(screen[0], screen[1])
time.sleep(1) # Stabilize

try:
    scale(factor)
except Exception as e:
    pass

devices = []
loop = None

try:
    devices = [evdev.InputDevice(fn) for fn in evdev.list_devices()]
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for device in devices:
        device.grab()
        loop.create_task(handle_events(device))
    loop.run_forever()
except Exception as e:
    print(f"Error: {e}")
finally:
    if loop and loop.is_running():
        loop.stop()
    if bg_process != None and bg_process.poll() == None:
        os.killpg(os.getpgid(bg_process.pid), signal.SIGTERM)