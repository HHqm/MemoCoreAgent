"""MCP server: lets an agent such as Claude Code keep long-term memory in a MemoryStore.

Four tools: remember a message verbatim, recall the turns relevant to a question, read the turns
behind cited ids, and forget given turns. Wiping a whole user is left to the library and CLI on
purpose, so an agent cannot erase everything with one tool call.
"""

from __future__ import annotations

try:  # mcp 2.x renamed FastMCP to MCPServer; the parts used here work the same in both
    from mcp.server.mcpserver import MCPServer
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

from memocore.store import Memory, MemoryStore

INSTRUCTIONS = """Long-term memory of past conversations, stored verbatim.
- Call `remember` for each message worth keeping (facts about the user, plans, preferences, events).
- Before answering a question that may depend on the past, call `recall` with the question.
- Each recalled line starts with an id such as [D3:12]. Cite the ids your answer relies on, and do not
  state anything the recalled lines do not support. `read_memories` returns the exact lines behind ids.
- Relative times were resolved when stored and appear as notes such as [time: yesterday = 7 May 2023]."""


def _line(memory: Memory) -> str:
    line = f"[{memory.id}] ({memory.session_time}) {memory.speaker}: {memory.text}"
    if memory.image:
        line += f" [shares an image: {memory.image}]"
    if memory.time_note:
        line += f" [{memory.time_note}]"
    return line


def build_server(store: MemoryStore, user: str) -> MCPServer:
    server = MCPServer("memocore", instructions=INSTRUCTIONS, log_level="WARNING")

    @server.tool()
    def remember(text: str, speaker: str = "user", session: str = "") -> str:
        """Store one message verbatim. `speaker` is who said it; `session` groups messages of one
        conversation (leave empty to group by time). Returns the id the message can be cited by."""
        memory = store.add(user, text, speaker=speaker, session=session or None)
        return f"Stored as [{memory.id}]" + (f" with [{memory.time_note}]" if memory.time_note else "")

    @server.tool()
    def recall(query: str, budget: int = 10) -> str:
        """Find the stored messages most relevant to `query` (up to `budget` of them), returned in
        conversation order with their ids and dates."""
        result = store.search(user, query, budget=max(1, min(budget, 50)))
        return result.context or "No memories stored yet."

    @server.tool()
    def read_memories(ids: list[str]) -> str:
        """Return the exact stored messages behind ids such as ["D3:12", "D5:2"], to check a citation."""
        found = store.get(user, ids)
        missing = sorted(set(ids) - {memory.id for memory in found})
        lines = [_line(memory) for memory in found]
        if missing:
            lines.append(f"Not found: {', '.join(missing)}")
        return "\n".join(lines)

    @server.tool()
    def forget(ids: list[str]) -> str:
        """Permanently delete the stored messages with these ids, together with everything derived from them."""
        return f"Deleted {store.forget(user, ids)} of {len(ids)} messages."

    return server
