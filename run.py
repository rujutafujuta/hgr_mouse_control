import argparse
import time
import tkinter as tk
from PIL import Image, ImageTk
import threading
import cv2
import numpy as np
import sys
import os
import pyautogui
from collections import deque 
import math
import logging 

import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Global Flags ---
running = True # Global flag to signal threads to stop

# --- Constants (Adjust as needed) ---
# MOUSE_SENSITIVITY_X = 1.5 # No longer needed for direct mapping
# MOUSE_SENSITIVITY_Y = 1.5 # No longer needed for direct mapping
# MOUSE_DEADZONE = 0.03    # No longer needed for direct mapping
MAX_CAMERA_INDEX = 3
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
# Define a small margin for clamping screen coordinates to avoid fail-safe corners
SCREEN_CLAMP_MARGIN = 5 # pixels

# --- Enums for Poses and Events ---
from enum import Enum, auto

class HandPosition(Enum):
    UNKNOWN = auto()
    FIST = auto()
    PINCH = auto()
    OPEN_HAND = auto()
    INDEX_FINGER_EXTENDED = auto()
    TWO_FINGERS_EXTENDED = auto()
    THUMB_UP = auto()

class Event(Enum):
    UNKNOWN = auto()
    CURSOR_MOVE = auto()
    LEFT_CLICK = auto()
    DOUBLE_CLICK = auto()
    RIGHT_CLICK = auto()
    DRAG_START = auto()
    DRAG_MOVE = auto()
    DROP_END = auto()
    SCROLL_UP = auto()
    SCROLL_DOWN = auto()
    SCROLL_MODE_START = auto()
    SCROLL_MODE_END = auto()
    ZOOM_IN = auto()
    ZOOM_OUT = auto()
    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()
    SWIPE_UP = auto()
    SWIPE_DOWN = auto()

# --- Action Controller ---
class ActionController:
    """
    Analyzes hand pose history and movement to detect and trigger events.
    """
    def __init__(self):
        self.hand_history = {} # {hand_id: deque([(timestamp, pose, norm_center, landmarks), ...])}
        self.active_states = {} # {hand_id: {'dragging': False, 'scrolling': False, 'last_tap_time': 0, ...}}
        self.HISTORY_LEN = 20
        self.TAP_MAX_MOVEMENT = 0.03
        self.TAP_MAX_DURATION_FRAMES = 8
        self.DOUBLE_TAP_MAX_INTERVAL_SEC = 0.5
        self.SWIPE_MIN_DIST = 0.15
        self.SWIPE_MAX_DURATION_FRAMES = 15
        self.DRAG_CONFIRM_FRAMES = 5
        self.SCROLL_CONFIRM_FRAMES = 5
        self.SCROLL_MOVEMENT_THRESHOLD = 0.04
        self.ZOOM_THRESHOLD = 0.03 # Normalized distance change threshold for zoom

    def _initialize_hand(self, hand_id):
        if hand_id not in self.hand_history:
            self.hand_history[hand_id] = deque(maxlen=self.HISTORY_LEN)
            self.active_states[hand_id] = {
                'dragging': False,
                'scrolling': False,
                'last_tap_time': 0,
                'last_action_time': 0,
                'scroll_start_y': None,
                'zoom_start_dist': None,
                'last_event': Event.UNKNOWN
            }
            # Changed initialization log level to DEBUG
            logging.debug(f"ActionController: Initialized tracking for hand {hand_id}")

    def update_hand_state(self, hand_id, pose, norm_center, landmarks, timestamp):
        self._initialize_hand(hand_id)
        self.hand_history[hand_id].append((timestamp, pose, norm_center, landmarks))

    def detect_action(self, hand_id):
        if hand_id not in self.hand_history or len(self.hand_history[hand_id]) < 2:
            return Event.UNKNOWN

        history = list(self.hand_history[hand_id])
        state = self.active_states[hand_id]
        now = history[-1][0]
        current_pose = history[-1][1]
        current_center = np.array(history[-1][2])
        # current_landmarks = history[-1][3] # Uncomment if needed

        # --- Handle Active States (Dragging/Scrolling) ---
        if state['dragging']:
            if current_pose == HandPosition.FIST:
                state['last_event'] = Event.DRAG_MOVE
                return Event.DRAG_MOVE
            else:
                logging.info(f"ActionController: Drag End detected for hand {hand_id} (Pose changed)")
                state['dragging'] = False
                state['last_action_time'] = now
                state['last_event'] = Event.DROP_END
                return Event.DROP_END

        if state['scrolling']:
            if current_pose == HandPosition.TWO_FINGERS_EXTENDED:
                if state['scroll_start_y'] is None:
                     state['scroll_start_y'] = current_center[1]
                     return Event.UNKNOWN

                delta_y = state['scroll_start_y'] - current_center[1]
                scroll_event = Event.UNKNOWN
                if delta_y > self.SCROLL_MOVEMENT_THRESHOLD:
                    logging.debug(f"ActionController: Scroll Up detected for hand {hand_id}")
                    scroll_event = Event.SCROLL_UP
                    state['scroll_start_y'] = current_center[1]
                    state['last_action_time'] = now
                elif delta_y < -self.SCROLL_MOVEMENT_THRESHOLD:
                    logging.debug(f"ActionController: Scroll Down detected for hand {hand_id}")
                    scroll_event = Event.SCROLL_DOWN
                    state['scroll_start_y'] = current_center[1]
                    state['last_action_time'] = now

                state['last_event'] = scroll_event
                return scroll_event
            else:
                logging.info(f"ActionController: Scroll Mode End detected for hand {hand_id} (Pose changed)")
                state['scrolling'] = False
                state['scroll_start_y'] = None
                state['last_action_time'] = now
                state['last_event'] = Event.SCROLL_MODE_END
                return Event.SCROLL_MODE_END

        # --- Detect Entering Modes / Discrete Actions ---
        time_since_last_action = now - state.get('last_action_time', 0)
        CLICK_COOLDOWN = 0.3 # Cooldown in seconds

        # Tap/Double Tap (PINCH)
        if current_pose == HandPosition.PINCH:
            pinch_frames = 0
            start_index = -1
            for i in range(len(history) - 1, -1, -1):
                if history[i][1] == HandPosition.PINCH: pinch_frames += 1
                else: start_index = i + 1; break
            else: start_index = 0

            if 0 < pinch_frames <= self.TAP_MAX_DURATION_FRAMES:
                 pinch_positions = [h[2] for h in history[start_index:]]
                 movement = np.linalg.norm(np.array(pinch_positions[-1]) - np.array(pinch_positions[0]))

                 if movement <= self.TAP_MAX_MOVEMENT:
                     last_tap = state['last_tap_time']
                     if (now - last_tap) < self.DOUBLE_TAP_MAX_INTERVAL_SEC and state['last_event'] == Event.LEFT_CLICK:
                         logging.info(f"ActionController: Double Click detected for hand {hand_id}")
                         state['last_tap_time'] = 0
                         state['last_action_time'] = now
                         state['last_event'] = Event.DOUBLE_CLICK
                         self.hand_history[hand_id].clear()
                         return Event.DOUBLE_CLICK
                     elif time_since_last_action > CLICK_COOLDOWN:
                         logging.info(f"ActionController: Left Click (Tap) detected for hand {hand_id}")
                         state['last_tap_time'] = now
                         state['last_action_time'] = now
                         state['last_event'] = Event.LEFT_CLICK
                         self.hand_history[hand_id].clear()
                         return Event.LEFT_CLICK

        # Drag Start (FIST)
        if current_pose == HandPosition.FIST:
            fist_frames = 0
            for i in range(len(history) - 1, -1, -1):
                if history[i][1] == HandPosition.FIST: fist_frames += 1
                else: break
            if fist_frames >= self.DRAG_CONFIRM_FRAMES:
                logging.info(f"ActionController: Drag Start detected for hand {hand_id}")
                state['dragging'] = True
                state['last_action_time'] = now
                state['last_event'] = Event.DRAG_START
                return Event.DRAG_START

        # Scroll Mode Start (TWO_FINGERS_EXTENDED)
        if current_pose == HandPosition.TWO_FINGERS_EXTENDED:
             scroll_pose_frames = 0
             for i in range(len(history) - 1, -1, -1):
                 if history[i][1] == HandPosition.TWO_FINGERS_EXTENDED: scroll_pose_frames += 1
                 else: break
             if scroll_pose_frames >= self.SCROLL_CONFIRM_FRAMES:
                 logging.info(f"ActionController: Scroll Mode Start detected for hand {hand_id}")
                 state['scrolling'] = True
                 state['scroll_start_y'] = current_center[1]
                 state['last_action_time'] = now
                 state['last_event'] = Event.SCROLL_MODE_START
                 return Event.SCROLL_MODE_START

        # Swipes (OPEN_HAND movement) - Check only if enough time passed since last action
        SWIPE_COOLDOWN = 0.5
        if current_pose == HandPosition.OPEN_HAND and len(history) > 2 and time_since_last_action > SWIPE_COOLDOWN:
            swipe_start_index = 0
            for i in range(len(history) - 2, -1, -1):
                 prev_pos = np.array(history[i][2])
                 start_pos = np.array(history[i+1][2])
                 if np.linalg.norm(start_pos - prev_pos) < 0.02: # Check for relatively static start point
                     swipe_start_index = i + 1
                     break

            if swipe_start_index < len(history) - 1:
                start_pos = history[swipe_start_index][2]
                end_pos = history[-1][2]
                duration_frames = len(history) - swipe_start_index
                delta = np.array(end_pos) - np.array(start_pos)
                distance = np.linalg.norm(delta)

                if distance > self.SWIPE_MIN_DIST and duration_frames <= self.SWIPE_MAX_DURATION_FRAMES:
                    angle = math.atan2(delta[1], delta[0])
                    swipe_event = Event.UNKNOWN
                    if -3*math.pi/4 < angle <= -math.pi/4: swipe_event = Event.SWIPE_UP
                    elif math.pi/4 < angle <= 3*math.pi/4: swipe_event = Event.SWIPE_DOWN
                    elif abs(angle) > 3*math.pi/4: swipe_event = Event.SWIPE_LEFT
                    elif abs(angle) <= math.pi/4: swipe_event = Event.SWIPE_RIGHT

                    if swipe_event != Event.UNKNOWN:
                        logging.info(f"AC: {swipe_event.name} detected for hand {hand_id}")
                        state['last_action_time'] = now
                        state['last_event'] = swipe_event
                        self.hand_history[hand_id].clear() # Clear history after swipe to prevent re-trigger
                        return swipe_event

        # Default: Cursor Movement (INDEX_FINGER_EXTENDED)
        if current_pose == HandPosition.INDEX_FINGER_EXTENDED:
             state['last_event'] = Event.CURSOR_MOVE
             return Event.CURSOR_MOVE

        # If no other event matched
        state['last_event'] = Event.UNKNOWN
        return Event.UNKNOWN

    def detect_two_hand_action(self, hand_id1, hand_id2):
        if hand_id1 not in self.hand_history or hand_id2 not in self.hand_history: return Event.UNKNOWN
        if len(self.hand_history[hand_id1]) < 2 or len(self.hand_history[hand_id2]) < 2: return Event.UNKNOWN

        history1 = list(self.hand_history[hand_id1])
        history2 = list(self.hand_history[hand_id2])
        state1 = self.active_states[hand_id1]
        state2 = self.active_states[hand_id2]
        now = max(history1[-1][0], history2[-1][0])

        pose1 = history1[-1][1]
        pose2 = history2[-1][1]

        if pose1 == HandPosition.OPEN_HAND and pose2 == HandPosition.OPEN_HAND:
            landmarks1 = history1[-1][3]
            landmarks2 = history2[-1][3]
            # Using wrist landmarks for zoom distance
            wrist1 = np.array([landmarks1.landmark[mp_hands.HandLandmark.WRIST].x, landmarks1.landmark[mp_hands.HandLandmark.WRIST].y])
            wrist2 = np.array([landmarks2.landmark[mp_hands.HandLandmark.WRIST].x, landmarks2.landmark[mp_hands.HandLandmark.WRIST].y])
            current_dist = np.linalg.norm(wrist1 - wrist2)

            prev_dist = state1.get('zoom_start_dist') # Check only one state, assume they are synced

            if prev_dist is None:
                # Initialize zoom tracking
                state1['zoom_start_dist'] = current_dist
                state2['zoom_start_dist'] = current_dist
                logging.debug(f"AC: Zoom tracking started for hands {hand_id1} & {hand_id2}. Initial distance: {current_dist:.4f}")
                return Event.UNKNOWN
            else:
                dist_change = current_dist - prev_dist
                zoom_event = Event.UNKNOWN

                # Using a percentage change or absolute threshold depending on need, absolute for now
                if dist_change > self.ZOOM_THRESHOLD:
                    zoom_event = Event.ZOOM_OUT
                elif dist_change < -self.ZOOM_THRESHOLD:
                    zoom_event = Event.ZOOM_IN

                if zoom_event != Event.UNKNOWN:
                     logging.info(f"AC: {zoom_event.name} detected between hands {hand_id1} & {hand_id2}. Change: {dist_change:.4f}")
                     state1['zoom_start_dist'] = current_dist # Update start dist to current dist after action
                     state2['zoom_start_dist'] = current_dist
                     state1['last_action_time'] = now
                     state2['last_action_time'] = now
                     state1['last_event'] = zoom_event
                     state2['last_event'] = zoom_event
                     # Consider clearing history for these hands if zoom is a discrete event
                     # self.hand_history[hand_id1].clear()
                     # self.hand_history[hand_id2].clear()
                     return zoom_event
                else:
                     # If no zoom event, just update the tracked distance for smoother tracking
                     state1['zoom_start_dist'] = current_dist
                     state2['zoom_start_dist'] = current_dist
                     return Event.UNKNOWN
        else:
            # Reset zoom tracking if not both hands are OPEN_HAND
            if state1.get('zoom_start_dist') is not None:
                 logging.debug(f"AC: Zoom tracking reset for hands {hand_id1} & {hand_id2}.")
                 state1['zoom_start_dist'] = None
                 state2['zoom_start_dist'] = None
            return Event.UNKNOWN


    def perform_action(self, event: Event, hand_id=None):
        # CURSOR_MOVE and DRAG_MOVE are handled directly in camera_processing
        if event in [Event.UNKNOWN, Event.CURSOR_MOVE, Event.DRAG_MOVE]:
            return

        # SCROLL_MODE_START/END are state changes, not direct actions here
        if event in [Event.SCROLL_MODE_START, Event.SCROLL_MODE_END]:
             if hand_id and hand_id in self.active_states:
                 if event == Event.SCROLL_MODE_START: self.active_states[hand_id]['scrolling'] = True
                 elif event == Event.SCROLL_MODE_END: self.active_states[hand_id]['scrolling'] = False
             return


        logging.info(f"AC: Performing action: {event.name}")
        try:
            if event == Event.SWIPE_UP: pyautogui.scroll(200)
            elif event == Event.SWIPE_DOWN: pyautogui.scroll(-200)
            elif event == Event.SWIPE_LEFT: pyautogui.hotkey('alt', 'left')
            elif event == Event.SWIPE_RIGHT: pyautogui.hotkey('alt', 'right')
            elif event == Event.LEFT_CLICK: pyautogui.click()
            elif event == Event.DOUBLE_CLICK: pyautogui.doubleClick()
            elif event == Event.RIGHT_CLICK: pyautogui.rightClick()
            elif event == Event.DRAG_START: pyautogui.mouseDown()
            elif event == Event.DROP_END: pyautogui.mouseUp()
            elif event == Event.SCROLL_UP: pyautogui.scroll(100) # Scroll amount can be adjusted
            elif event == Event.SCROLL_DOWN: pyautogui.scroll(-100) # Scroll amount can be adjusted
            elif event == Event.ZOOM_IN:
                pyautogui.keyDown('ctrl'); pyautogui.scroll(80); pyautogui.keyUp('ctrl') # Scroll amount for zoom can be adjusted
            elif event == Event.ZOOM_OUT:
                pyautogui.keyDown('ctrl'); pyautogui.scroll(-80); pyautogui.keyUp('ctrl') # Scroll amount for zoom can be adjusted
        except pyautogui.FailSafeException as e:
             # Log the fail-safe exception specifically, it's expected here if corners are hit
             logging.error(f"AC: PyAutoGUI FailSafeException during action {event.name}: {e}")
             # Attempt to release keys if they were pressed for the action
             pyautogui.keyUp('ctrl'); pyautogui.keyUp('alt'); pyautogui.mouseUp()
        except Exception as e:
            logging.error(f"AC: Error performing action {event.name}: {e}", exc_info=True)
            # Attempt to release keys/mouse on error
            pyautogui.keyUp('ctrl'); pyautogui.keyUp('alt'); pyautogui.mouseUp()


    def cleanup_hand(self, hand_id):
        if hand_id in self.hand_history:
            del self.hand_history[hand_id]
            # Reduced noise by changing cleanup log level to DEBUG
            logging.debug(f"AC: Cleaned up history for hand ID {hand_id}")
        if hand_id in self.active_states:
            if self.active_states[hand_id].get('dragging', False):
                try:
                    pyautogui.mouseUp()
                    logging.warning(f"AC: Released mouse button for lost dragging hand {hand_id}")
                except Exception as e:
                    logging.error(f"AC: Error releasing mouse on cleanup for hand {hand_id}: {e}")
            del self.active_states[hand_id]
            logging.debug(f"AC: Cleaned up state for hand ID {hand_id}")

# --- Drawer ---
class Drawer:
    """Handles drawing annotations on the frame."""
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # Increased font scale for better readability in larger overlay
        self.font_scale = 0.9
        self.text_color = (255, 255, 255) # White
        self.box_color_default = (0, 255, 0) # Green
        self.box_color_dragging = (0, 0, 255) # Red
        self.box_color_scrolling = (255, 0, 0) # Blue
        self.landmark_color = (0, 0, 255) # Red
        self.connection_color = (0, 255, 0) # Green
        self.thickness = 2 # Line thickness

    def draw_annotations(self, frame, active_hands_data):
        if not active_hands_data:
            return frame

        annotated_frame = frame.copy()
        frame_h, frame_w, _ = annotated_frame.shape

        for hand_id, data in active_hands_data.items():
            landmarks = data.get('hand_landmarks')
            pose = data.get('pose', HandPosition.UNKNOWN)
            bbox_pixel = data.get('bbox_pixel')
            handedness = data.get('handedness_label', 'N/A')
            is_dragging = data.get('is_dragging', False)
            is_scrolling = data.get('is_scrolling', False)

            # Draw Landmarks
            if landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # Draw Bounding Box and Info Text
            if bbox_pixel is not None and len(bbox_pixel) == 4:
                x1, y1, x2, y2 = bbox_pixel
                box_color = self.box_color_default
                if is_dragging: box_color = self.box_color_dragging
                elif is_scrolling: box_color = self.box_color_scrolling

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, self.thickness)

                # Info Text (ID, Handedness)
                info_text = f"ID: {hand_id} ({handedness})"
                # Increased font scale for better readability
                font_scale = 0.7
                (text_width, text_height), _ = cv2.getTextSize(info_text, self.font, font_scale, self.thickness)
                text_y_info = y1 - 30 # Position above the box
                if text_y_info < 15: text_y_info = y1 + 25 # Move below if too high
                # Adjust padding for text background
                cv2.rectangle(annotated_frame, (x1 - 5, text_y_info - text_height - 5), (x1 + text_width + 5, text_y_info + 5), (50, 50, 50), -1)
                cv2.putText(annotated_frame, info_text, (x1, text_y_info), self.font, font_scale, self.text_color, self.thickness)

                # Pose Text
                pose_text = f"Pose: {pose.name}"
                (pose_width, pose_height), _ = cv2.getTextSize(pose_text, self.font, font_scale, self.thickness)
                text_y_pose = y1 - 5 # Position just above the box or below info text
                 # Adjust y position based on info text height for better separation
                if text_y_pose < text_y_info + text_height + 5 : text_y_pose = y1 + text_height + 35 # Ensure separation
                if text_y_pose < 15: text_y_pose = y1 + 50 # Move below if too high
                 # Adjust padding for text background
                cv2.rectangle(annotated_frame, (x1 - 5, text_y_pose - pose_height - 5), (x1 + pose_width + 5, text_y_pose + 5), (50, 50, 50), -1)
                cv2.putText(annotated_frame, pose_text, (x1, text_y_pose), self.font, font_scale, self.text_color, self.thickness)


        return annotated_frame

# --- Helper Functions ---

def find_working_camera():
    logging.info("Attempting to find working camera...")
    for index in range(MAX_CAMERA_INDEX + 1):
        cap = None
        backend_used = "Default"
        try:
            if sys.platform == "win32":
                logging.debug(f"Trying camera index {index} with CAP_DSHOW backend...")
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                backend_used = "CAP_DSHOW"
            if cap is None or not cap.isOpened():
                logging.debug(f"Trying camera index {index} with default backend...")
                cap = cv2.VideoCapture(index)
                backend_used = "Default"

            if cap.isOpened():
                logging.info(f"Successfully opened camera handle at index {index} using backend: {backend_used}")
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    logging.info(f"Camera {index} provided a valid test frame.")
                    cap.release()
                    return index
                else:
                    logging.warning(f"Camera {index} opened, but could not read a valid test frame. Trying next...")
                    cap.release()
            else:
                 logging.debug(f"Camera index {index} could not be opened.")
        except Exception as e:
            logging.error(f"Error trying camera index {index}: {e}")
            if cap: cap.release()
    logging.error("No working cameras found!")
    return None

def get_finger_state(hand_landmarks):
    finger_state = {}
    if not hand_landmarks: return finger_state

    landmarks = hand_landmarks.landmark
    finger_tips = {'THUMB': 4, 'INDEX': 8, 'MIDDLE': 12, 'RING': 16, 'PINKY': 20}
    finger_pip = {'THUMB': 3, 'INDEX': 6, 'MIDDLE': 10, 'RING': 14, 'PINKY': 18}
    finger_mcp = {'THUMB': 2, 'INDEX': 5, 'MIDDLE': 9, 'RING': 13, 'PINKY': 17}
    wrist = landmarks[0]

    for finger, tip_idx in finger_tips.items():
        tip = landmarks[tip_idx]
        pip = landmarks[finger_pip[finger]]
        mcp = landmarks[finger_mcp[finger]]

        # Simple Vertical Check (Tip vs PIP/MCP) - Lower Y is higher up
        # Use distance checks for robustness if needed
        is_extended = tip.y < pip.y and pip.y < mcp.y # Basic check for fingers

        # Basic Thumb Check (more complex logic might be needed) - Keeping old_run.py logic
        if finger == 'THUMB':
             # Check if thumb tip is above thumb MCP and roughly above wrist Y
             index_mcp = landmarks[finger_mcp['INDEX']]
             is_extended = tip.y < mcp.y # Basic extension check
             is_thumb_up = is_extended and tip.y < index_mcp.y # Check if tip is above index base
             finger_state[finger] = is_extended
             finger_state['THUMB_UP'] = is_thumb_up # Keep the old thumb up state name
        else:
            finger_state[finger] = is_extended

    # Removed the new 'THUMB_UP_POSE' specific check from run.py

    return finger_state


def get_hand_pose_and_center(hand_landmarks, handedness_label):
    if not hand_landmarks:
        return HandPosition.UNKNOWN, None, None

    landmarks = hand_landmarks.landmark
    # Use the base of the palm (WRIST) for a relatively stable center point
    # You could also use the average of several palm landmarks if wrist is too jumpy
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    norm_center = (wrist.x, wrist.y) # Normalized coordinates (0.0 to 1.0)

    # Calculate normalized bounding box
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    padding_x = (x_max - x_min) * 0.1 # Add a small padding to the bbox
    padding_y = (y_max - y_min) * 0.1
    x_min = max(0.0, x_min - padding_x); y_min = max(0.0, y_min - padding_y)
    x_max = min(1.0, x_max + padding_x); y_max = min(1.0, y_max + padding_y)
    bbox_norm = (x_min, y_min, x_max, y_max)


    finger_state = get_finger_state(hand_landmarks)

    pose = HandPosition.UNKNOWN
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    # --- Pose Logic (Keeping old_run.py flow, especially for Fist) ---
    is_index_ext = finger_state.get('INDEX', False)
    is_middle_ext = finger_state.get('MIDDLE', False)
    is_ring_ext = finger_state.get('RING', False)
    is_pinky_ext = finger_state.get('PINKY', False)
    is_thumb_ext = finger_state.get('THUMB', False) # Using the old thumb state
    is_thumb_up = finger_state.get('THUMB_UP', False) # Using the old thumb up state name


    # Pinch (Thumb and Index tip close)
    pinch_dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    PINCH_THRESHOLD = 0.06 # Adjust based on camera distance/hand size
    if pinch_dist < PINCH_THRESHOLD and not is_middle_ext and not is_ring_ext and not is_pinky_ext:
        pose = HandPosition.PINCH
    # Index Finger Extended
    elif is_index_ext and not is_middle_ext and not is_ring_ext and not is_pinky_ext: # Removed the 'and not is_thumb_up_pose' check
        pose = HandPosition.INDEX_FINGER_EXTENDED
    # Two Fingers Extended
    elif is_index_ext and is_middle_ext and not is_ring_ext and not is_pinky_ext: # Removed the 'and not is_thumb_up_pose' check
        pose = HandPosition.TWO_FINGERS_EXTENDED
    # Open Hand (All fingers extended and spread)
    elif is_index_ext and is_middle_ext and is_ring_ext and is_pinky_ext:
         spread_dist = math.hypot(index_tip.x - pinky_tip.x, index_tip.y - pinky_tip.y)
         # Adjust spread threshold based on camera distance/hand size
         SPREAD_THRESHOLD = 0.15
         if spread_dist > SPREAD_THRESHOLD:
              pose = HandPosition.OPEN_HAND
         else: # Fingers extended but not spread might be UNKNOWN or another specific pose
              pose = HandPosition.UNKNOWN # Or a different pose if needed
    # Fist (All fingers not extended) - Keeping old_run.py logic and position in elif chain
    elif not is_index_ext and not is_middle_ext and not is_ring_ext and not is_pinky_ext:
         # Check if finger tips are close to the palm center
         # Using landmarks 5, 9, 13, 17 for palm center as in old_run.py
         palm_center_x = np.mean([landmarks[i].x for i in [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]])
         palm_center_y = np.mean([landmarks[i].y for i in [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]])
         avg_tip_palm_dist = np.mean([
             math.hypot(tip.x - palm_center_x, tip.y - palm_center_y)
             for tip in [index_tip, middle_tip, ring_tip, pinky_tip] # Exclude thumb for fist check
         ])
         FIST_DIST_THRESHOLD = 0.1 # Adjust based on camera distance/hand size
         if avg_tip_palm_dist < FIST_DIST_THRESHOLD:
              pose = HandPosition.FIST
         else: # Fingers not extended but not close to palm might be UNKNOWN
             pose = HandPosition.UNKNOWN # Or a different pose if needed
    # Thumb Up - Keeping old_run.py logic and position in elif chain
    elif is_thumb_up and not is_index_ext and not is_middle_ext and not is_ring_ext and not is_pinky_ext:
        pose = HandPosition.THUMB_UP
    # If none of the above, it's UNKNOWN
    else:
         pose = HandPosition.UNKNOWN


    return pose, norm_center, bbox_norm

class CameraOverlay(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hand Control Preview")
        self.configure(bg='black')
        self.attributes('-alpha', 0.85)
        self.attributes('-topmost', True)
        self.overrideredirect(True)

        # Increased overlay size (from run.py)
        self.win_w, self.win_h = 620, 440 # Larger preview window
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        pos_x = screen_w - self.win_w - 20
        pos_y = screen_h - self.win_h - 40
        self.geometry(f'{self.win_w}x{self.win_h}+{pos_x}+{pos_y}')

        self.canvas = tk.Canvas(self, width=self.win_w, height=self.win_h, bg='black', highlightthickness=0)
        self.canvas.pack()
        self.photo_image = None
        self.protocol("WM_DELETE_WINDOW", self.safe_destroy)

        # Removed key binding for 'q' to quit

        if sys.platform == "win32":
            try:
                from ctypes import windll
                # Get the handle to the parent window (the Tkinter frame)
                hwnd = windll.user32.GetParent(self.winfo_id())
                # Get current window style
                style = windll.user32.GetWindowLongW(hwnd, -20)
                # Add layered window style (0x80000) and WS_EX_TRANSPARENT (0x20)
                style = style | 0x80000 | 0x20
                # Set the new window style
                windll.user32.SetWindowLongW(hwnd, -20, style)
                # Set transparency key (0) and alpha (int(255 * 0.85))
                windll.user32.SetLayeredWindowAttributes(hwnd, 0, int(255 * 0.85), 0x2)
                logging.info("Overlay: Click-through enabled (Windows).")
            except Exception as e:
                logging.warning(f"Overlay: Failed to set click-through: {e}")

    def update_preview(self, frame):
        try:
            if frame is None or frame.size == 0: return
            if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4: frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if frame.shape[2] != 3: return

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Use current canvas size for resize
            img_pil = Image.fromarray(img).resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(image=img_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
            self.update_idletasks()
        except Exception as e:
            logging.error(f"Overlay update error: {e}", exc_info=False)
            self.photo_image = None

    def safe_destroy(self):
        global running
        if running:
            running = False
            logging.info("Overlay: Window close requested. Signaling exit.")
            # Use self.quit() to stop the Tkinter main loop (from run.py)
            self.quit()
            # self.destroy() # No longer need destroy here after self.quit()

    # Removed on_q_press method


# --- Main Processing Logic ---
def run_gesture_control(args):
    # Declare running as global here as it's modified in this scope
    global running

    action_controller = ActionController()
    drawer = Drawer()
    overlay = None
    try:
        overlay = CameraOverlay()
    except Exception as e:
        logging.error(f"Failed to create CameraOverlay: {e}. Exiting.")
        running = False; return

    active_hands_data = {}
    data_lock = threading.Lock()
    latest_annotated_frame = None

    def camera_processing():
        nonlocal active_hands_data, latest_annotated_frame
        global running # Declare running as global here

        camera_index = find_working_camera()
        if camera_index is None:
            logging.error("Camera processing thread: No working camera found.")
            running = False
            return

        cap = None
        try:
            if sys.platform == "win32": cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if cap is None or not cap.isOpened(): cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened(): raise IOError(f"Cannot open camera index {camera_index}")
            logging.info(f"Camera processing thread: Camera {camera_index} opened successfully.")
            # Set camera properties - order matters, set width/height before FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            logging.info(f"Camera properties: {actual_w}x{actual_h} @ {actual_fps:.2f} FPS")
        except Exception as e:
            logging.error(f"Camera processing thread: Error opening camera {camera_index}: {e}"); running = False; return

        logging.info("Camera processing thread: Initializing MediaPipe Hands...")
        # Keeping model complexity = 0 from old_run.py
        with mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands_processor:
            logging.info("Camera processing thread: MediaPipe Hands initialized.")
            last_frame_time = time.time()

            while running and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    # Keeping shorter sleep duration from run.py
                    if not ret or frame is None: time.sleep(0.005); continue

                    current_time = time.time()
                    # Prevent division by zero if time difference is too small
                    fps = 1.0 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
                    last_frame_time = current_time

                    frame = cv2.flip(frame, 1) # Flip horizontally for mirror effect
                    frame_h, frame_w, _ = frame.shape
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = hands_processor.process(image)
                    image.flags.writeable = True

                    current_frame_hands_temp = {}
                    detected_hand_ids = set()

                    if results.multi_hand_landmarks:
                        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                            handedness_label = "Unknown"; handedness_score = 0
                            if results.multi_handedness and hand_idx < len(results.multi_handedness):
                                handedness = results.multi_handedness[hand_idx].classification[0]
                                handedness_label = handedness.label; handedness_score = handedness.score

                            # Use a more stable hand ID if possible, e.g., based on first detection position or persistent tracking ID
                            # For simplicity now, using handedness and index
                            hand_id = f"{handedness_label}_{hand_idx}"
                            detected_hand_ids.add(hand_id)
                            pose, norm_center, bbox_norm = get_hand_pose_and_center(hand_landmarks, handedness_label)

                            if norm_center:
                                action_controller.update_hand_state(hand_id, pose, norm_center, hand_landmarks, current_time)
                                current_frame_hands_temp[hand_id] = {
                                    'pose': pose, 'norm_center': norm_center, 'bbox_norm': bbox_norm,
                                    'hand_landmarks': hand_landmarks, 'handedness_label': handedness_label,
                                    'is_dragging': action_controller.active_states.get(hand_id, {}).get('dragging', False),
                                    'is_scrolling': action_controller.active_states.get(hand_id, {}).get('scrolling', False)
                                }

                    # Determine primary hand for cursor control and detect actions
                    primary_hand_id = None
                    detected_event = Event.UNKNOWN
                    two_hand_event = Event.UNKNOWN

                    hand_ids_list = list(detected_hand_ids)
                    if len(hand_ids_list) == 2:
                        two_hand_event = action_controller.detect_two_hand_action(hand_ids_list[0], hand_ids_list[1])
                        if two_hand_event != Event.UNKNOWN:
                             detected_event = two_hand_event # Two-hand events take precedence

                    # Process single-hand events if no two-hand event detected
                    if detected_event == Event.UNKNOWN:
                         for hand_id in detected_hand_ids:
                              event = action_controller.detect_action(hand_id)
                              if event != Event.UNKNOWN:
                                   # Prioritize non-movement events
                                   if event not in [Event.CURSOR_MOVE, Event.DRAG_MOVE]:
                                        detected_event = event
                                        primary_hand_id = hand_id # Assign hand ID that triggered the action
                                        break # Process one discrete action at a time
                              # If no discrete event, check for cursor movement
                              elif detected_event == Event.UNKNOWN and current_frame_hands_temp.get(hand_id, {}).get('pose') == HandPosition.INDEX_FINGER_EXTENDED:
                                   detected_event = Event.CURSOR_MOVE
                                   primary_hand_id = hand_id # Assign primary hand for cursor move


                    # Perform detected discrete action
                    action_controller.perform_action(detected_event, primary_hand_id)

                    # Handle mouse movement (CursorMove or DragMove)
                    # Determine which hand controls the mouse based on pose or active state
                    mouse_control_hand_id = None
                    for hid in detected_hand_ids:
                        state = action_controller.active_states.get(hid, {})
                        if state.get('dragging', False): # Dragging hand controls mouse
                            mouse_control_hand_id = hid
                            break
                        elif current_frame_hands_temp.get(hid, {}).get('pose') == HandPosition.INDEX_FINGER_EXTENDED: # Index finger controls mouse if not dragging
                             mouse_control_hand_id = hid
                             break # Found the cursor hand

                    if mouse_control_hand_id and mouse_control_hand_id in current_frame_hands_temp:
                         hand_data = current_frame_hands_temp[mouse_control_hand_id]
                         norm_x, norm_y = hand_data['norm_center']
                         is_scrolling = action_controller.active_states.get(mouse_control_hand_id, {}).get('scrolling', False)
                         is_dragging = action_controller.active_states.get(mouse_control_hand_id, {}).get('dragging', False)


                         # Only move cursor if in cursor move mode or dragging
                         if (hand_data['pose'] == HandPosition.INDEX_FINGER_EXTENDED or is_dragging) and not is_scrolling:
                             # --- Direct Mapping from Normalized Hand Coords to Screen Coords (from run.py) ---
                             # norm_x and norm_y are typically from 0.0 to 1.0 relative to the camera frame
                             # We map these directly to screen coordinates
                             screen_x = int(norm_x * SCREEN_WIDTH)
                             screen_y = int(norm_y * SCREEN_HEIGHT)
                             # --- End Direct Mapping ---

                             # --- Clamping with Margin to Avoid Fail-Safe Corners (from run.py) ---
                             screen_x = int(max(SCREEN_CLAMP_MARGIN, min(SCREEN_WIDTH - 1 - SCREEN_CLAMP_MARGIN, screen_x)))
                             screen_y = int(max(SCREEN_CLAMP_MARGIN, min(SCREEN_HEIGHT - 1 - SCREEN_CLAMP_MARGIN, screen_y)))
                             # --- End Clamping ---

                             try:
                                pyautogui.moveTo(screen_x, screen_y, duration=0)
                             except pyautogui.FailSafeException as e:
                                 # Log the fail-safe exception specifically during move (from run.py)
                                 logging.error(f"Camera processing thread: PyAutoGUI FailSafeException during moveTo: {e}")
                             except Exception as e:
                                 logging.error(f"Camera processing thread: Error during pyautogui.moveTo: {e}")


                    # Clean up data for hands that are no longer detected
                    with data_lock:
                        current_active_ids = set(active_hands_data.keys())
                        lost_ids = current_active_ids - detected_hand_ids
                        for hand_id in lost_ids:
                            action_controller.cleanup_hand(hand_id)
                            if hand_id in active_hands_data:
                                del active_hands_data[hand_id] # Check existence before del
                        # Update active_hands_data with current frame data
                        active_hands_data.update(current_frame_hands_temp)
                        # Ensure dragging/scrolling states in active_hands_data are up-to-date from action_controller
                        for hand_id in active_hands_data:
                             if hand_id in action_controller.active_states:
                                 active_hands_data[hand_id]['is_dragging'] = action_controller.active_states[hand_id].get('dragging', False)
                                 active_hands_data[hand_id]['is_scrolling'] = action_controller.active_states[hand_id].get('scrolling', False)


                    # Prepare frame for drawing
                    draw_data = {}
                    with data_lock: draw_data = active_hands_data.copy()
                    for hand_id, data in draw_data.items():
                         if data.get('bbox_norm'): # Check if bbox_norm exists
                              x1n, y1n, x2n, y2n = data['bbox_norm']
                              data['bbox_pixel'] = (int(x1n * frame_w), int(y1n * frame_h), int(x2n * frame_w), int(y2n * frame_h))
                         else: data['bbox_pixel'] = None

                    annotated_frame = drawer.draw_annotations(frame, draw_data)
                    # Draw FPS and other info
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), drawer.font, 0.8, (0, 200, 0), 2)
                    if detected_event != Event.UNKNOWN:
                         event_text = f"Event: {detected_event.name}"
                         cv2.putText(annotated_frame, event_text, (10, 60), drawer.font, 0.8, (255, 255, 0), 2) # Cyan color

                    with data_lock: latest_annotated_frame = annotated_frame.copy()

                    # Show debug window if enabled
                    if args.debug:
                        cv2.imshow("Gesture Control Debug", annotated_frame)
                        # Keep waitKey here for the debug window's 'q' quit
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logging.info("Camera processing thread: 'q' pressed in debug window. Signaling exit.")
                            running = False; break

                except Exception as e:
                    logging.error(f"Camera processing thread: Error in loop: {e}", exc_info=True)
                    # Keeping shorter sleep duration from run.py
                    time.sleep(0.005)

            logging.info("Camera processing thread: Capture loop ended.")
        if cap: cap.release()
        # Destroy debug window if it exists
        if args.debug:
             try:
                 cv2.destroyWindow("Gesture Control Debug")
                 logging.info("Camera processing thread: Destroyed debug window.")
             except Exception as e:
                 logging.warning(f"Camera processing thread: Could not destroy debug window: {e}")

        logging.info("Camera processing thread: Resources released.")


    logging.info("Main thread: Starting camera processing thread.")
    processing_thread = threading.Thread(target=camera_processing, name="CamProcThread", daemon=True)
    processing_thread.start()

    def update_overlay():
        nonlocal latest_annotated_frame
        global running # Declare running as global here

        # Check if both Tkinter window and the processing thread are running
        if not running:
             if overlay and overlay.winfo_exists():
                 try:
                      # Use overlay.quit() for graceful Tkinter exit
                      overlay.quit()
                 except Exception as e:
                      logging.warning(f"Main thread: Error calling overlay.quit(): {e}")
             return # Stop updating if not running

        try:
            if overlay and overlay.winfo_exists():
                 with data_lock: frame_to_show = latest_annotated_frame
                 if frame_to_show is not None: overlay.update_preview(frame_to_show)
                 # Keeping overlay update rate from run.py
                 overlay.after(30, update_overlay) # Aim for ~33ms delay (~30 FPS)
            else:
                 # If overlay is gone but running is still True, something went wrong
                 logging.warning("Main thread: Overlay window not available for update, but running flag is True.")
                 running = False # Force stop
        except Exception as e:
            logging.error(f"Main thread: Error updating overlay: {e}", exc_info=True)
            running = False # Force stop on error


    if overlay:
        # Start the overlay update loop
        overlay.after(100, update_overlay)
        logging.info("Main thread: Starting Tkinter main loop.")
        try:
            # This call blocks until the Tkinter window is closed
            overlay.mainloop()
            logging.info("Main thread: Tkinter main loop finished.")
        except Exception as e:
            logging.error(f"Main thread: Error in Tkinter main loop: {e}", exc_info=True)
            running = False # Ensure running is False if Tkinter loop exits unexpectedly


    # After Tkinter main loop finishes, ensure processing thread stops
    logging.info("Main thread: Ensuring camera processing thread is stopped.")
    running = False # Explicitly set running to False

    if processing_thread.is_alive():
        logging.info("Main thread: Waiting for processing thread to join...")
        # Keeping increased timeout from run.py
        processing_thread.join(timeout=5)
        if processing_thread.is_alive():
            logging.warning("Main thread: Processing thread did not join cleanly. It might be blocked.")
        else:
            logging.info("Main thread: Processing thread joined successfully.")
    else:
        logging.info("Main thread: Processing thread already finished.")

    logging.info("Application exit sequence complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe Gesture Control System")
    parser.add_argument("--debug", action="store_true", help="Show debug visualization window.")
    args = parser.parse_args()

    # Set up logging with thread name (from run.py)
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

    print("Starting Hand Gesture Control Application...")
    # Disable pyautogui fail-safe initially for development/testing purposes
    # Consider enabling it or using a different fail-safe mechanism in a production environment
    # pyautogui.FAILSAFE = False # Keep this commented out as recommended

    try:
        run_gesture_control(args)
    except Exception as e:
         logging.critical(f"An unhandled exception occurred in the main execution: {e}", exc_info=True)

    print("Application finished.")
