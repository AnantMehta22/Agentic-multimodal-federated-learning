"""
test_router.py -- Modality Router Test
=======================================
Creates 100 dummy mixed-modality files and verifies 100% correct routing.

Usage:
    cd <project_root>
    python -m phase3_router.test_router
"""

import os
import sys
import tempfile
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from phase3_router.modality_router import ModalityRouter


def test_routing():
    """Create 100 dummy files and test routing accuracy."""
    print(f"\n{'='*60}")
    print(f"  PHASE 3 -- MODALITY ROUTER TEST")
    print(f"{'='*60}")

    router = ModalityRouter()

    # Define test files: 25 per modality
    test_files = {
        "image": [f"img_{i}.{ext}" for i, ext in enumerate(
            ["jpg"]*6 + ["png"]*6 + ["bmp"]*4 + ["gif"]*3 + ["jpeg"]*3 + ["webp"]*3
        )],
        "text": [f"doc_{i}.{ext}" for i, ext in enumerate(
            ["txt"]*7 + ["csv"]*6 + ["json"]*4 + ["md"]*4 + ["xml"]*4
        )],
        "audio": [f"sound_{i}.{ext}" for i, ext in enumerate(
            ["wav"]*7 + ["mp3"]*6 + ["flac"]*4 + ["ogg"]*4 + ["aac"]*4
        )],
        "video": [f"clip_{i}.{ext}" for i, ext in enumerate(
            ["mp4"]*7 + ["avi"]*6 + ["mov"]*4 + ["mkv"]*4 + ["webm"]*4
        )],
    }

    # Create temp directory with dummy files
    temp_dir = tempfile.mkdtemp(prefix="router_test_")
    try:
        # Create all files
        expected = {}
        for modality, filenames in test_files.items():
            for fname in filenames:
                filepath = os.path.join(temp_dir, fname)
                with open(filepath, "w") as f:
                    f.write(f"dummy {modality} content")
                expected[fname] = modality

        total_files = sum(len(v) for v in test_files.values())
        print(f"\n  Created {total_files} test files in temp directory")
        print(f"  Files per modality: {', '.join(f'{k}={len(v)}' for k, v in test_files.items())}")

        # Test individual routing
        print(f"\n  Testing individual file routing...")
        correct = 0
        errors = []
        for fname, expected_mod in expected.items():
            result = router.route(fname)
            if result == expected_mod:
                correct += 1
            else:
                errors.append((fname, expected_mod, result))

        accuracy = correct / total_files * 100
        print(f"  Individual routing: {correct}/{total_files} correct ({accuracy:.1f}%)")

        if errors:
            print(f"\n  ERRORS:")
            for fname, exp, got in errors:
                print(f"    {fname}: expected={exp}, got={got}")

        # Test directory routing
        print(f"\n  Testing directory routing...")
        dir_result = router.route_directory(temp_dir)
        dir_correct = 0
        for modality in ["image", "text", "audio", "video"]:
            n_expected = len(test_files[modality])
            n_got = len(dir_result[modality])
            status = "OK" if n_got == n_expected else f"MISMATCH (expected {n_expected})"
            print(f"    {modality:>6}: {n_got} files -- {status}")
            dir_correct += min(n_got, n_expected)

        unknown = len(dir_result.get("unknown", []))
        print(f"    unknown: {unknown} files")

        # Final verdict
        print(f"\n  {'='*40}")
        if accuracy == 100.0 and unknown == 0:
            print(f"  PASSED -- 100% routing accuracy!")
        else:
            print(f"  FAILED -- {accuracy:.1f}% accuracy, {unknown} unknown")
        print(f"  {'='*40}")

    finally:
        shutil.rmtree(temp_dir)
        print(f"  Cleaned up temp directory\n")


if __name__ == "__main__":
    test_routing()
