"""CLI runner for the HR Assistant demo.

Usage:
    python demo_cli.py "What are the rules for parental leave?"
"""
from __future__ import annotations

import asyncio
import sys

from langgraph.types import Command

from hr_assistant.graph import build_graph, default_input
from hr_assistant.logging import configure_logging


async def _run_stream(graph, input_value, cfg) -> None:
    """Drive the graph with `astream(stream_mode=['custom'])` so writer tokens
    print as they arrive from `llm_answer_node`."""
    async for mode, data in graph.astream(
        input_value, config=cfg, stream_mode=["custom"]
    ):
        if mode == "custom" and isinstance(data, dict):
            tok = data.get("token")
            if tok:
                sys.stdout.write(tok)
                sys.stdout.flush()


async def run(message: str) -> None:
    configure_logging()
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "cli-session"}}
    state = default_input(message)

    print(f"\n>>> {message}\n")
    await _run_stream(graph, state, cfg)

    # After the stream, check for a pending interrupt.
    snapshot = await graph.aget_state(cfg)
    interrupts = getattr(snapshot, "interrupts", None) or ()
    values = dict(snapshot.values)

    if interrupts:
        print("\n[!] Graph paused — sensitivity elevated.")
        for intr in interrupts:
            payload = getattr(intr, "value", intr)
            print(f"    Details: {payload}")
        answer = input("    Approve? [y/N]: ").strip().lower()
        approved = answer == "y"
        # Stream the resumed half so we see the rest of the tokens too.
        print()
        await _run_stream(graph, Command(resume={"approved": approved}), cfg)
        snapshot = await graph.aget_state(cfg)
        values = dict(snapshot.values)
        if not approved:
            # Denial short-circuits to END without running llm_answer, so
            # nothing streamed — print the canned denial message explicitly.
            print(values.get("response_text", "Denied."))

    print("\n\n--- Audit trail ---")
    for step in values.get("routing_path", []):
        print(f"  \u00b7 {step}")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python demo_cli.py "your question"')
        sys.exit(2)
    asyncio.run(run(sys.argv[1]))


if __name__ == "__main__":
    main()
