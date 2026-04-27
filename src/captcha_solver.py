"""CAPTCHA OCR for GST portal using ddddocr.

The GST portal CAPTCHA is exactly 6 alphanumeric characters.
ddddocr is highly optimized for these types of captchas and 
provides significantly better accuracy than EasyOCR without 
needing complex OpenCV preprocessing.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger("gstr2b.captcha")

_VALID = re.compile(r"^[A-Za-z0-9]{6}$")
_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        import ddddocr  # type: ignore
        log.info("Initialising ddddocr reader...")
        _ocr = ddddocr.DdddOcr(show_ad=False)
        log.info("ddddocr ready.")
    return _ocr


def _clean(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text or "")


def solve_captcha(image_bytes: bytes) -> Optional[str]:
    """Best-effort solve. Returns the 6-char string or None."""
    try:
        ocr = _get_ocr()
    except Exception as exc:  # noqa: BLE001
        log.error("ddddocr unavailable: %s", exc)
        return None

    try:
        # ddddocr takes bytes directly and is very fast
        raw_result = ocr.classification(image_bytes)
        cand = _clean(raw_result)
        
        log.debug("CAPTCHA ddddocr candidate: %s", cand)
        
        if _VALID.match(cand):
            return cand

        # Fallback: if it's longer than 6 chars, take the first 6
        if len(cand) >= 6:
            return cand[:6]

    except Exception as exc:  # noqa: BLE001
        log.warning("OCR pass failed: %s", exc)

    return None
