from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def load_dotenv(dotenv_path: str = ".env") -> None:
    """
    Minimal .env loader (no external deps).
    Lines like KEY=VALUE. Ignores comments and empty lines.
    Does NOT overwrite existing environment variables.
    """
    p = Path(dotenv_path)
    if not p.exists():
        return

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    google_cx: str
    cache_dir: Path
    user_agent: str = "lead-discovery-agent/1.0"

    # Default search localization (tweak as needed)
    hl: str = "en"         # interface language
    gl: str = "sa"         # geolocation
    cr: str = "countrySA"  # restrict to country (optional)
    safe: str = "off"      # safe search (off/on/active)

    @staticmethod
    def from_env(dotenv_path: Optional[str] = ".env") -> "Settings":
        if dotenv_path:
            load_dotenv(dotenv_path)

        api_key = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
        cx = os.getenv("GOOGLE_CSE_CX", "").strip()
        cache_dir = os.getenv("GOOGLE_CSE_CACHE_DIR", ".cache/google_cse").strip()
        user_agent = os.getenv("GOOGLE_CSE_USER_AGENT", "lead-discovery-agent/1.0").strip()

        if not api_key:
            raise RuntimeError("Missing GOOGLE_CSE_API_KEY in environment / .env")
        if not cx:
            raise RuntimeError("Missing GOOGLE_CSE_CX in environment / .env")

        return Settings(
            google_api_key=api_key,
            google_cx=cx,
            cache_dir=Path(cache_dir),
            user_agent=user_agent,
        )
