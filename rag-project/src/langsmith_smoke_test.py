"""Small LangSmith tracing smoke test.

Run this after setting your LangSmith credentials to confirm that traces are
arriving in your LangSmith project.
"""

from __future__ import annotations

import os
from typing import Dict

from langsmith import traceable

from src.langsmith_config import configure_langsmith_env


configure_langsmith_env()


@traceable(name="langsmith_smoke_test", run_type="chain")
def run_smoke_test(message: str) -> Dict[str, str]:
    """Return a tiny payload so LangSmith records a simple trace."""
    return {
        "input": message,
        "output": message.upper(),
    }


def main() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit(
            "LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) is not set. "
            "Add it to your .env file or shell first."
        )

    if os.getenv("LANGSMITH_TRACING", "").lower() not in {"true", "1", "yes"}:
        raise SystemExit(
            "Tracing is disabled. Set LANGSMITH_TRACING=true "
            "(or LANGCHAIN_TRACING_V2=true) and run again."
        )

    if not os.getenv("LANGSMITH_WORKSPACE_ID") and not os.getenv("LANGCHAIN_WORKSPACE_ID"):
        print(
            "Hint: if your LangSmith key is workspace-scoped, set LANGSMITH_WORKSPACE_ID "
            "(or LANGCHAIN_WORKSPACE_ID) before running this test."
        )

    result = run_smoke_test("LangSmith smoke test from rag-project")
    print("LangSmith smoke test completed.")
    print(result["output"])


if __name__ == "__main__":
    main()
