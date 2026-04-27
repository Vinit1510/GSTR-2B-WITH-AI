"""CAPTCHA OCR for GST portal using ddddocr.

We are using the ddddocr Beta model (which is much smarter)
and passing the RAW image directly to it, because altering the image
with OpenCV was actually destroying the letters and confusing the AI.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger("gstr2b.captcha")

# The captcha can contain letters and numbers! (e.g. 9S7B73)
_VALID = re.compile(r"^[A-Za-z0-9]{6}$")
_ocr = None

def _get_ocr():
    global _ocr
    if _ocr is None:
        import ddddocr  # type: ignore
        log.info("Initialising ddddocr reader (BETA MODEL)...")
        # beta=True uses a much more advanced neural network
        _ocr = ddddocr.DdddOcr(show_ad=False, beta=True)
        log.info("ddddocr BETA ready.")
    return _ocr

def solve_captcha(image_bytes: bytes) -> Optional[str]:
    """Best-effort solve using ddddocr Beta."""
    try:
        ocr = _get_ocr()
    except Exception as exc:  # noqa: BLE001
        log.error("ddddocr unavailable: %s", exc)
        return None

    try:
        # Run OCR directly on the RAW image (no OpenCV tricks)
        digits = ocr.classification(image_bytes)
        
        log.info("CAPTCHA AI Guessed: '%s'", digits)
        
        # If exactly 6 alphanumeric characters, we have a match
        if _VALID.match(digits):
            return digits

        # Fallback: if it's longer than 6, take the first 6
        if len(digits) >= 6:
            return digits[:6]
            
        # --- DEBUGGING: If it fails, save the image to the Desktop! ---
        import os
        desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')
        try:
            with open(os.path.join(desktop, "failed_captcha.png"), "wb") as f:
                f.write(image_bytes)
            log.info("Saved failed image to Desktop!")
        except Exception as e:
            log.warning("Could not save debug image: %s", e)

    except Exception as exc:  # noqa: BLE001
        log.warning("OCR pass failed: %s", exc)

    return None
