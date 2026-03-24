"""
app/utils/cleanup.py — Temporary file helpers.

Currently used for future file-upload support (e.g. when a client POSTs a
video file instead of a stream URL).  The functions here handle writing
multipart uploads to a temp path and deleting them after processing.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def write_upload_to_temp(data: bytes, suffix: str = ".mp4") -> str:
    """
    Write raw upload bytes to a named temporary file and return its path.

    The caller is responsible for deleting the file after use (call
    delete_temp_file()).  The file is NOT deleted automatically so that
    OpenCV can open it by path after the UploadFile stream is closed.

    Parameters
    ----------
    data   : Raw bytes from an UploadFile.read() call.
    suffix : File extension hint for OpenCV / FFmpeg.

    Returns
    -------
    Absolute path to the temporary file.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    logger.debug("Temp file created: %s (%d bytes)", path, len(data))
    return path


def delete_temp_file(path: str) -> None:
    """
    Safely delete a temporary file, logging but not raising on failure.

    Parameters
    ----------
    path : Absolute path returned by write_upload_to_temp().
    """
    try:
        Path(path).unlink(missing_ok=True)
        logger.debug("Temp file deleted: %s", path)
    except OSError as exc:
        logger.warning("Could not delete temp file %s: %s", path, exc)
