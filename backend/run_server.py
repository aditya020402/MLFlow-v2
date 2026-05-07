"""Start the FastAPI server with watchfiles restricted to source directories only.

Run with:  python run_server.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["agents", "graph", "api", "tools"],
    )
