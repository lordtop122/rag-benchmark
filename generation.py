"""
The "generation" half of RAG: takes retrieved chunks and a query, builds a
prompt that grounds the model's answer in those chunks, and calls the
Claude API to generate a final answer.

Requires an Anthropic API key set as an environment variable:
    export ANTHROPIC_API_KEY="your-key-here"

Get a key at https://console.anthropic.com/settings/keys
"""

import os
from anthropic import Anthropic

# Claude Sonnet 5 - good balance of quality and cost for this kind of task.
MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are a question-answering assistant. Answer the user's \
question using ONLY the information in the provided context chunks. \
If the context does not contain enough information to answer, say so \
explicitly rather than guessing or using outside knowledge. \
Cite which chunk(s) you used by their chunk_id."""


def build_prompt(query, retrieved_chunks):
    context_block = "\n\n".join(
        f"[{c['chunk_id']}]\n{c['text']}" for c in retrieved_chunks
    )
    return f"""Context chunks:
{context_block}

Question: {query}

Answer the question using only the context above."""


def generate_answer(query, retrieved_chunks, client=None):
    """Call the Claude API to generate an answer grounded in retrieved chunks.

    Returns the generated answer text.
    """
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Get a key from https://console.anthropic.com/settings/keys "
                "and run: export ANTHROPIC_API_KEY='your-key-here'"
            )
        client = Anthropic(api_key=api_key)

    prompt = build_prompt(query, retrieved_chunks)

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text
