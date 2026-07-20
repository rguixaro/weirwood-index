from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run(
        "weirwood_api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
    )


if __name__ == "__main__":
    main()
