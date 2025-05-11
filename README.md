## How to run: 
1. pip install -r requirements.txt 
2. python run.py

# gestures :
| Gesture                     | Event Detected       | Action Performed               |
| --------------------------- | -------------------- | ------------------------------ |
| Index finger up             | CURSOR\_MOVE         | Move mouse cursor              |
| Pinch                       | LEFT\_CLICK          | Left click                     |
| Double pinch (quick repeat) | DOUBLE\_CLICK        | Double click                   |
| Fist (hold)                 | DRAG\_START / MOVE   | Click and drag                 |
| Release fist                | DROP\_END            | Drop (release drag)            |
| Two fingers up              | SCROLL\_MODE\_START  | Enter scroll mode              |
| Hand move in scroll mode    | SCROLL\_UP / DOWN    | Vertical scroll                |
| Both hands spread/pinch     | ZOOM\_IN / ZOOM\_OUT | Zoom in/out (Ctrl + scroll)    |
| Fast swipe (open hand)      | SWIPE\_LEFT, etc.    | Arrow key presses / navigation |
