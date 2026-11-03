"""
modality_router.py -- File-based Modality Router
=================================================
Routes files to the correct modality pipeline based on file extension
or MIME type. No ML classifier needed -- pure rule-based routing.

Usage:
    from phase3_router.modality_router import ModalityRouter
    router = ModalityRouter()
    modality = router.route("photo.jpg")    # -> "image"
    sorted_files = router.route_directory("path/to/mixed/folder")
"""

import os
import mimetypes
from typing import Dict, List, Optional


class ModalityRouter:
    """Routes files to the correct modality pipeline based on file type."""

    # Extension -> modality mapping
    EXTENSION_MAP = {
        # Image
        ".jpg": "image", ".jpeg": "image", ".png": "image",
        ".bmp": "image", ".gif": "image", ".tiff": "image",
        ".tif": "image", ".webp": "image", ".ico": "image",
        # Text
        ".txt": "text", ".csv": "text", ".json": "text",
        ".xml": "text", ".md": "text", ".log": "text",
        ".html": "text", ".htm": "text", ".yaml": "text",
        ".yml": "text", ".tsv": "text", ".ini": "text",
        # Audio
        ".wav": "audio", ".mp3": "audio", ".flac": "audio",
        ".ogg": "audio", ".aac": "audio", ".wma": "audio",
        ".m4a": "audio", ".aiff": "audio", ".opus": "audio",
        # Video
        ".mp4": "video", ".avi": "video", ".mov": "video",
        ".mkv": "video", ".wmv": "video", ".flv": "video",
        ".webm": "video", ".m4v": "video", ".mpeg": "video",
        ".mpg": "video", ".3gp": "video",
    }

    # MIME type prefix -> modality mapping (fallback)
    MIME_MAP = {
        "image": "image",
        "text": "text",
        "audio": "audio",
        "video": "video",
    }

    def route(self, filepath: str) -> Optional[str]:
        """
        Determine the modality for a single file.

        Args:
            filepath: Path or filename to classify.

        Returns:
            One of "image", "text", "audio", "video", or None if unrecognized.
        """
        # Try extension first (fastest)
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        if ext in self.EXTENSION_MAP:
            return self.EXTENSION_MAP[ext]

        # Fallback: MIME type detection
        mime_type, _ = mimetypes.guess_type(filepath)
        if mime_type:
            prefix = mime_type.split("/")[0]
            if prefix in self.MIME_MAP:
                return self.MIME_MAP[prefix]

        return None

    def route_directory(self, dir_path: str) -> Dict[str, List[str]]:
        """
        Sort all files in a directory by modality.

        Args:
            dir_path: Path to directory containing mixed-modality files.

        Returns:
            Dict mapping modality -> list of file paths.
            Unrecognized files are under the key "unknown".
        """
        result: Dict[str, List[str]] = {
            "image": [], "text": [], "audio": [], "video": [], "unknown": []
        }

        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        for filename in os.listdir(dir_path):
            filepath = os.path.join(dir_path, filename)
            if os.path.isfile(filepath):
                modality = self.route(filename)
                if modality:
                    result[modality].append(filepath)
                else:
                    result["unknown"].append(filepath)

        return result

    def get_supported_extensions(self, modality: str) -> List[str]:
        """Get all supported file extensions for a given modality."""
        return [ext for ext, mod in self.EXTENSION_MAP.items() if mod == modality]

    def summary(self) -> str:
        """Return a summary of supported extensions per modality."""
        summary = {}
        for ext, mod in self.EXTENSION_MAP.items():
            summary.setdefault(mod, []).append(ext)
        lines = ["Modality Router - Supported Extensions:"]
        for mod in ["image", "text", "audio", "video"]:
            exts = sorted(summary.get(mod, []))
            lines.append(f"  {mod:>6}: {', '.join(exts)}")
        return "\n".join(lines)


if __name__ == "__main__":
    router = ModalityRouter()
    print(router.summary())

    # Quick test
    test_files = ["photo.jpg", "notes.txt", "song.wav", "clip.mp4",
                  "data.csv", "image.png", "record.flac", "movie.mkv"]
    print("\nRouting test:")
    for f in test_files:
        print(f"  {f:>15} -> {router.route(f)}")
