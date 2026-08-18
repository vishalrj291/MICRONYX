"""
MICRONYX PS02
FastAPI inference backend

This API wraps the existing canonical MICRONYX localization
pipeline. It does NOT implement a second ML algorithm.

Flow:

    Upload search + reference
              |
              v
       Image validation
              |
              v
      micronyx_localize.py
              |
              v
      Existing XGBoost ranker
              |
              v
       Localization result
"""

from pathlib import Path
import json
import re
import subprocess
import sys
import tempfile
import time

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCALIZER = (
    PROJECT_ROOT
    / "scripts"
    / "micronyx_localize.py"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "validation"
    / "v02"
    / "learned_ranker"
    / "xgboost_ranker.json"
)


# ============================================================================
# APPLICATION
# ============================================================================

app = FastAPI(
    title="MICRONYX PS02 API",
    description=(
        "Physics-aware semiconductor target localization "
        "using canonical multi-resolution acquisition and "
        "learned candidate ranking."
    ),
    version="1.0.0",
)


# ============================================================================
# CORS
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HELPERS
# ============================================================================

def validate_image_file(
    path: Path,
    expected_shape=(1000, 1000),
):
    """
    Validate an uploaded grayscale image.
    """

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {path.name}",
        )

    if image.shape != expected_shape:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{path.name} must be "
                f"{expected_shape[1]}x{expected_shape[0]}; "
                f"got {image.shape[1]}x{image.shape[0]}"
            ),
        )

    return image


async def save_upload(
    upload: UploadFile,
    path: Path,
):
    """
    Save uploaded file to disk.
    """

    data = await upload.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail=f"Empty upload: {upload.filename}",
        )

    path.write_bytes(data)


def parse_localizer_output(
    stdout: str,
):
    """
    Parse the existing micronyx_localize.py CLI output.

    Expected output includes:

        Predicted center: (376.000, 569.000) px
        Model confidence: 0.999962
        Candidates: 250
        Localization runtime: 146.49 ms
    """

    result = {
        "predicted_x": None,
        "predicted_y": None,
        "confidence": None,
        "candidates": None,
        "runtime_ms": None,
        "top_candidates": [],
    }

    match = re.search(
        r"Predicted center:\s*"
        r"\(([-+0-9.eE]+),\s*([-+0-9.eE]+)\)",
        stdout,
    )

    if match:
        result["predicted_x"] = float(
            match.group(1)
        )
        result["predicted_y"] = float(
            match.group(2)
        )

    match = re.search(
        r"Model confidence:\s*"
        r"([-+0-9.eE]+)",
        stdout,
    )

    if match:
        result["confidence"] = float(
            match.group(1)
        )

    match = re.search(
        r"Candidates:\s*(\d+)",
        stdout,
    )

    if match:
        result["candidates"] = int(
            match.group(1)
        )

    match = re.search(
        r"Localization runtime:\s*"
        r"([-+0-9.eE]+)\s*ms",
        stdout,
    )

    if match:
        result["runtime_ms"] = float(
            match.group(1)
        )

    # ----------------------------------------------------------------
    # Parse TOP 10 CANDIDATES
    #
    # Example:
    # 1 x= 376.000 y= 569.000 prob=0.999962 DOG=0.857543
    # ----------------------------------------------------------------

    candidate_pattern = re.compile(
        r"^\s*(\d+)\s+"
        r"x=\s*([-+0-9.eE]+)\s+"
        r"y=\s*([-+0-9.eE]+)\s+"
        r"prob=\s*([-+0-9.eE]+)\s+"
        r"DOG=\s*([-+0-9.eE]+)",
        re.MULTILINE,
    )

    for match in candidate_pattern.finditer(
        stdout
    ):
        result["top_candidates"].append(
            {
                "rank": int(
                    match.group(1)
                ),
                "x": float(
                    match.group(2)
                ),
                "y": float(
                    match.group(3)
                ),
                "probability": float(
                    match.group(4)
                ),
                "dog_score": float(
                    match.group(5)
                ),
            }
        )

    return result


def run_localizer(
    search_path: Path,
    reference_path: Path,
):
    """
    Execute the existing production localization CLI.
    """

    if not LOCALIZER.exists():
        raise RuntimeError(
            f"Localization engine not found: {LOCALIZER}"
        )

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Trained XGBoost model not found: {MODEL_PATH}"
        )

    command = [
        sys.executable,
        str(LOCALIZER),
        "--search",
        str(search_path),
        "--reference",
        str(reference_path),
    ]

    started = time.perf_counter()

    process = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )

    wall_time_ms = (
        time.perf_counter()
        - started
    ) * 1000.0

    if process.returncode != 0:
        raise RuntimeError(
            "Localization engine failed.\n\n"
            f"STDOUT:\n{process.stdout}\n\n"
            f"STDERR:\n{process.stderr}"
        )

    result = parse_localizer_output(
        process.stdout
    )

    result["api_wall_time_ms"] = round(
        wall_time_ms,
        3,
    )

    result["engine"] = (
        "MICRONYX canonical localization "
        "+ XGBoost candidate ranking"
    )

    result["acquisition"] = {
        "search_width": 1000,
        "search_height": 1000,
        "reference_width": 1000,
        "reference_height": 1000,
        "search_pixels_per_unit": 5.0,
        "reference_pixels_per_unit": 50.0,
        "sampling_ratio": 10.0,
    }

    return result


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
def root():
    return {
        "name": "MICRONYX PS02",
        "status": "online",
        "version": "1.0.0",
        "model": "XGBoost",
        "acquisition": {
            "search": "1000x1000",
            "reference": "1000x1000",
            "sampling_ratio": "10x",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "localizer_available": LOCALIZER.exists(),
        "model_available": MODEL_PATH.exists(),
    }


@app.post("/api/localize")
async def localize(
    search: UploadFile = File(...),
    reference: UploadFile = File(...),
):
    """
    Localize the target using a search/reference image pair.
    """

    search_name = (
        search.filename
        or "search.png"
    )

    reference_name = (
        reference.filename
        or "reference.png"
    )

    search_suffix = (
        Path(search_name).suffix
        or ".png"
    )

    reference_suffix = (
        Path(reference_name).suffix
        or ".png"
    )

    with tempfile.TemporaryDirectory(
        prefix="micronyx_"
    ) as temp_dir:

        temp = Path(temp_dir)

        search_path = (
            temp
            / f"search{search_suffix}"
        )

        reference_path = (
            temp
            / f"reference{reference_suffix}"
        )

        await save_upload(
            search,
            search_path,
        )

        await save_upload(
            reference,
            reference_path,
        )

        # ------------------------------------------------------------
        # Validate dimensions BEFORE inference.
        # ------------------------------------------------------------

        validate_image_file(
            search_path
        )

        validate_image_file(
            reference_path
        )

        # ------------------------------------------------------------
        # Run existing localization pipeline.
        # ------------------------------------------------------------

        try:
            result = run_localizer(
                search_path,
                reference_path,
            )

        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail=(
                    "Localization timed out."
                ),
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=str(exc),
            )

    if (
        result["predicted_x"] is None
        or result["predicted_y"] is None
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "Localization completed but "
                "prediction could not be parsed."
            ),
        )

    return {
        "success": True,
        **result,
    }


# ============================================================================
# DEV ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )