"""CAPTCHA OCR for GST portal using ddddocr and OpenCV.

The GST portal CAPTCHA is exactly 6 numerical digits, often with a 
red wavy line and a black grid on a white background.

We use OpenCV to extract the Red channel (which magically erases the red line),
apply a median blur to erase the thin grid, and then pass the clean image to ddddocr.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("gstr2b.captcha")

_VALID = re.compile(r"^[0-9]{6}$")
_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        import ddddocr  # type: ignore
        log.info("Initialising ddddocr reader...")
        _ocr = ddddocr.DdddOcr(show_ad=False)
        log.info("ddddocr ready.")
    return _ocr


def preprocess_image(image_bytes: bytes) -> bytes:
    """Erase the red line and grid using OpenCV."""
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # 1. Split into B, G, R channels
    # The red line is purely red (high R, low G, low B). 
    # White background is high R, G, B. Black text is low R, G, B.
    # If we only look at the Red channel, the red line and white background both appear as white!
    b, g, r = cv2.split(img)
    
    # 2. Threshold the Red channel
    _, binary = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Median Blur to remove thin grid lines
    # The grid lines are typically 1-2 pixels wide, while numbers are thick.
    cleaned = cv2.medianBlur(binary, 3)
    
    # Encode back to PNG bytes
    _, buf = cv2.imencode('.png', cleaned)
    return buf.tobytes()


def force_digits(text: str) -> str:
    """Map common letter misreadings into their corresponding digits."""
    mapping = {
        'O': '0', 'o': '0', 'Q': '0', 'D': '0', 'U': '0', 'u': '0',
        'I': '1', 'i': '1', 'l': '1', 'L': '1', 't': '1', 'T': '1',
        'Z': '2', 'z': '2',
        'S': '5', 's': '5',
        'G': '6', 'b': '6',
        'B': '8',
        'g': '9', 'q': '9'
    }
    res = ""
    for char in text:
        if char in mapping:
            res += mapping[char]
        elif char.isdigit():
            res += char
    return res


def solve_captcha(image_bytes: bytes) -> Optional[str]:
    """Best-effort solve using OpenCV preprocessing and ddddocr."""
    try:
        ocr = _get_ocr()
    except Exception as exc:  # noqa: BLE001
        log.error("ddddocr unavailable: %s", exc)
        return None

    try:
        # 1. Preprocess the image to remove red line and grid
        clean_bytes = preprocess_image(image_bytes)
        
        # 2. Run OCR
        raw_result = ocr.classification(clean_bytes)
        
        # 3. Force output to be numerical
        digits = force_digits(raw_result)
        
        log.debug("CAPTCHA ddddocr raw: '%s' -> forced digits: '%s'", raw_result, digits)
        
        # If exactly 6 digits, we have a match
        if _VALID.match(digits):
            return digits

        # Fallback: if it's longer than 6 digits, take the first 6
        if len(digits) >= 6:
            return digits[:6]

    except Exception as exc:  # noqa: BLE001
        log.warning("OCR pass failed: %s", exc)

    return None
