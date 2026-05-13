"""
llm.py — Groq-powered LLM for RAG answer generation.

Model: llama-3.3-70b-versatile
  - Fast inference via Groq's LPU hardware.
  - Strong instruction-following and long-context reasoning.
  - Handles our ~3 000-token prompts comfortably within its 128K context window.

Prompt design:
  - System role sets the assistant's persona and grounding rules.
  - User turn injects the retrieved context blocks + original question.
  - Temperature 0.2 keeps answers factual; a slight non-zero value allows
    natural phrasing variation.
"""

import os
import re
from typing import Optional

from groq import Groq
from langsmith import traceable

from src.langsmith_config import configure_langsmith_env

SYSTEM_PROMPT = """You are a precise, knowledgeable research assistant.
Your job is to answer questions **strictly based on the context provided**.

Rules:
1. Base your answer ONLY on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question,
   clearly say: "I could not find a clear answer in the provided documents."
3. Be concise but complete. Use bullet points or numbered steps when helpful.
4. When citing specific information, mention the source (e.g., "According to [1]…").
5. Never make up facts, numbers, or references.
"""
configure_langsmith_env()


def normalize_api_key(api_key: Optional[str]) -> Optional[str]:
    """Strip whitespace and surrounding quotes from a key value."""
    if api_key is None:
        return None

    cleaned = api_key.strip().strip('"').strip("'")
    match = re.search(r"(gsk_[A-Za-z0-9_-]+)", cleaned)
    if match:
        cleaned = match.group(1)
    return cleaned or None


def build_user_prompt(query: str, context: str) -> str:
    """Compose the user-turn message with retrieved context and the question."""
    return (
        f"### Relevant Context\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"### Question\n\n"
        f"{query}\n\n"
        f"### Answer"
    )


def get_groq_client(api_key: Optional[str] = None) -> Groq:
    """
    Instantiate the Groq client.

    Looks for the API key in:
      1. The `api_key` argument (highest priority).
      2. The GROQ_API_KEY environment variable.

    Raises:
        ValueError: If no API key is found.
    """
    key = normalize_api_key(api_key or os.getenv("GROQ_API_KEY"))
    if not key:
        raise ValueError(
            "Groq API key not found. Set the GROQ_API_KEY environment variable "
            "or pass it directly to get_groq_client()."
        )
    return Groq(api_key=key)


@traceable(name="generate_answer", run_type="llm")
def generate_answer(
    query: str,
    context: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """
    Send the augmented prompt to Groq and return the generated answer.

    Args:
        query:       The user's original question.
        context:     Retrieved and formatted document chunks.
        api_key:     Groq API key (falls back to env var).
        model:       Groq model identifier (falls back to DEFAULT_MODEL).
        temperature: Sampling temperature (0 = deterministic, 1 = creative).
        max_tokens:  Maximum tokens in the response.

    Returns:
        The LLM's answer as a plain string.

    Raises:
        RuntimeError: If the Groq API call fails.
    """
    client = get_groq_client(api_key)
    user_prompt = build_user_prompt(query, context)
    selected_model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        raise RuntimeError(f"Groq API error: {exc}") from exc
