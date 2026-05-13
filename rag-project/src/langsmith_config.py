"""LangSmith configuration helpers.

This project supports both the newer LangSmith environment variables and the
older LangChain aliases that many existing `.env` files still use.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


def configure_langsmith_env() -> None:
    """Load local env values and normalize LangSmith-related aliases."""
    load_dotenv()

    aliases = {
        "LANGSMITH_API_KEY": "LANGCHAIN_API_KEY",
        "LANGSMITH_PROJECT": "LANGCHAIN_PROJECT",
        "LANGSMITH_ENDPOINT": "LANGCHAIN_ENDPOINT",
        "LANGSMITH_WORKSPACE_ID": "LANGCHAIN_WORKSPACE_ID",
    }

    for target, source in aliases.items():
        if not os.getenv(target) and os.getenv(source):
            os.environ[target] = os.environ[source]

    if not os.getenv("LANGSMITH_TRACING") and os.getenv("LANGCHAIN_TRACING_V2"):
        os.environ["LANGSMITH_TRACING"] = os.environ["LANGCHAIN_TRACING_V2"]

    # Keep LangSmith traffic off the local dummy proxy used in this workspace.
    langsmith_hosts = "api.smith.langchain.com,eu.api.smith.langchain.com,aws.api.smith.langchain.com"
    for key in ("NO_PROXY", "no_proxy"):
        current = os.getenv(key, "")
        parts = [part.strip() for part in current.split(",") if part.strip()]
        if langsmith_hosts not in current:
            parts.extend(langsmith_hosts.split(","))
            deduped = []
            for part in parts:
                if part not in deduped:
                    deduped.append(part)
            os.environ[key] = ",".join(deduped)
