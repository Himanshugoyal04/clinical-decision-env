"""
Server entry point for Clinical Decision Support Environment.
This module provides the main() function for the OpenEnv server entry point.
"""

import uvicorn
from app.main import app


def main():
    """Main entry point for the server."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )


if __name__ == "__main__":
    main()
