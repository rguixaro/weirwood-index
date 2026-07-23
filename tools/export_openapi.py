from __future__ import annotations

import json
from pathlib import Path

from weirwood_api.main import create_app


def main() -> None:
    destination = Path("openapi.json")
    destination.write_text(
        json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(destination.resolve())


if __name__ == "__main__":
    main()
