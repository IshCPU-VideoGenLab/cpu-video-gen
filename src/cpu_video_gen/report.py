"""Paper-quality report generation for cpu-video-gen."""
import json, logging, os
from typing import Any
logger = logging.getLogger(__name__)

def save_json(data: Any, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved: %s", path)
    return path
