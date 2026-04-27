"""CAPTCHA OCR for GST portal using ddddocr and OpenCV.

The GST portal CAPTCHA is exactly 6 numerical digits, often with a 
red wavy line and a black grid on a white background.

We use OpenCV to extract the Red channel (which magically erases the red line),
and then pass the clean image to ddddocr. ddddocr is configured to ONLY
recognize numerical digits (0-9) via set_ranges(0), preventing any dropped characters.
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
        # Force ddddocr to only recognize numbers (0-9)
        _ocr.set_ranges(0)
        log.info("ddddocr ready (numerical only mode).")
    return _ocr


def preprocess_image(image_bytes: bytes) -> bytes:
    """Erase the red line using OpenCV."""
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # 1. Split into B, G, R channels
    # The red line is purely red. White background is high R, G, B. 
    # Black text/grid is low R, G, B.
    # By extracting the Red channel, the red line completely vanishes into the white background.
    b, g, r = cv2.split(img)
    
    # 2. Threshold the Red channel
    _, binary = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # We no longer blur the image because it was erasing the thin text numbers.
    # ddddocr is trained to handle noisy grids automatically!
    
    # Encode back to PNG bytes
    _, buf = cv2.imencode('.png', binary)
    return buf.tobytes()


def solve_captcha(image_bytes: bytes) -> Optional[str]:
    """Best-effort solve using OpenCV preprocessing and ddddocr."""
    try:
        ocr = _get_ocr()
    except Exception as exc:  # noqa: BLE001
        log.error("ddddocr unavailable: %s", exc)
        return None

    try:
        # 1. Preprocess the image to remove the red line
        clean_bytes = preprocess_image(image_bytes)
        
        # 2. Run OCR (it is locked to numbers only, so no letters will ever appear)
        digits = ocr.classification(clean_bytes)
        
        log.debug("CAPTCHA ddddocr (numbers only mode): '%s'", digits)
        
        # If exactly 6 digits, we have a match
        if _VALID.match(digits):
            return digits

        # Fallback: if it's longer than 6 digits, take the first 6
        if len(digits) >= 6:
            return digits[:6]

    except Exception as exc:  # noqa: BLE001
        log.warning("OCR pass failed: %s", exc)

    return None
