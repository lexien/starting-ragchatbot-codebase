"""
Tests for AIGenerator using a mocked Anthropic client.

What each test catches
----------------------
test_end_turn_returns_text_directly      — AttributeError on ToolUseBlock.text if fallthrough broken
test_tool_use_calls_tool_manager         — tool_manager.execute_tool() never called → no content
test_tool_results_included_in_second_call — tool results not passed to second API call
test_no_tools_skips_tool_choice          — API validation error: tool_choice sent without tools
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure backend/ is on sys.path (conftest.py does this, but be explicit)
BACKEND_DIR = Path(__file__).parent.parent.resolve()
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ai_generator import AIGenerator


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_text_response(text: str) -> MagicMock:
    """Build a mock Anthropic response that ends with stop_reason='end_turn'."""
    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [content_block]
    return response


def _make_tool_use_response(tool_name: str, tool_id: str, tool_input: dict) -> MagicMock:
    """Build a mock Anthropic response that ends with stop_reason='tool_use'."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.id = tool_id
    tool_block.input = tool_input

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]
    return response


def _make_generator() -> tuple[AIGenerator, MagicMock]:
    """Return (AIGenerator, mock_messages_client) with Anthropic client patched."""
    mock_client = MagicMock()
    with patch("ai_generator.anthropic.Anthropic", return_value=mock_client):
        gen = AIGenerator(api_key="test-key", model="test-model")
    return gen, mock_client


# ── tests ─────────────────────────────────────────────────────────────────────

def test_end_turn_returns_text_directly():
    """
    When stop_reason is 'end_turn', generate_response() must return response.content[0].text.

    Catches: AttributeError if code accidentally tries .text on a ToolUseBlock,
    or if the fallthrough path is broken entirely.
    """
    gen, mock_client = _make_generator()
    expected = "Here is a direct answer."
    mock_client.messages.create.return_value = _make_text_response(expected)

    result = gen.generate_response(query="What is 2+2?")

    assert result == expected


def test_tool_use_calls_tool_manager():
    """
    When stop_reason is 'tool_use', _handle_tool_execution() must call
    tool_manager.execute_tool() with the tool name and inputs from the response.

    Catches: tool_manager.execute_tool() never invoked → synthesized answer has no content.
    """
    gen, mock_client = _make_generator()

    first_response = _make_tool_use_response(
        tool_name="search_course_content",
        tool_id="toolu_001",
        tool_input={"query": "MCP concepts"}
    )
    second_response = _make_text_response("MCP stands for Model Context Protocol.")
    mock_client.messages.create.side_effect = [first_response, second_response]

    mock_tool_manager = MagicMock()
    mock_tool_manager.execute_tool.return_value = "Chunk 1: MCP is a protocol..."

    tools = [{"name": "search_course_content", "description": "search", "input_schema": {}}]
    result = gen.generate_response(
        query="What is MCP?",
        tools=tools,
        tool_manager=mock_tool_manager
    )

    mock_tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="MCP concepts"
    )
    assert result == "MCP stands for Model Context Protocol."


def test_tool_results_included_in_second_api_call():
    """
    The tool result must appear in the messages list sent to the second API call.

    Catches: tool results built but never passed to the follow-up messages.create() call.
    """
    gen, mock_client = _make_generator()

    tool_result_text = "Chunk 1: RAG retrieval augments LLMs."
    first_response = _make_tool_use_response(
        tool_name="search_course_content",
        tool_id="toolu_002",
        tool_input={"query": "RAG definition"}
    )
    second_response = _make_text_response("RAG stands for Retrieval-Augmented Generation.")
    mock_client.messages.create.side_effect = [first_response, second_response]

    mock_tool_manager = MagicMock()
    mock_tool_manager.execute_tool.return_value = tool_result_text

    tools = [{"name": "search_course_content", "description": "search", "input_schema": {}}]
    gen.generate_response(query="What is RAG?", tools=tools, tool_manager=mock_tool_manager)

    # Second call's messages must contain a tool_result entry
    second_call_args = mock_client.messages.create.call_args_list[1]
    messages_sent = second_call_args.kwargs.get("messages") or second_call_args.args[0].get("messages", [])

    tool_result_messages = [
        m for m in messages_sent
        if isinstance(m.get("content"), list)
        and any(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert tool_result_messages, (
        "No tool_result message found in second API call — "
        "tool output was not passed back to Claude for synthesis"
    )

    # Verify our tool output is in there
    tool_result_content = tool_result_messages[0]["content"][0]["content"]
    assert tool_result_content == tool_result_text


def test_no_tools_skips_tool_choice():
    """
    When tools=None (or tools=[]), the API call must NOT include 'tool_choice'.

    Catches: Anthropic API validation error — tool_choice is invalid without tools.
    """
    gen, mock_client = _make_generator()
    mock_client.messages.create.return_value = _make_text_response("General answer.")

    gen.generate_response(query="What is Python?", tools=None, tool_manager=None)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    # Merge positional dict arg if present (handles both calling conventions)
    if not call_kwargs and mock_client.messages.create.call_args.args:
        call_kwargs = mock_client.messages.create.call_args.args[0]

    assert "tool_choice" not in call_kwargs, (
        "tool_choice was included in the API call even though no tools were provided"
    )
    assert "tools" not in call_kwargs, (
        "tools key was included in the API call even though tools=None"
    )


def test_two_sequential_tool_calls_returns_final_synthesis():
    """
    When Claude makes two tool calls in sequence, generate_response() runs both,
    forces a no-tools synthesis call third, and returns its text.
    """
    gen, mock_client = _make_generator()

    tool_use_r1 = _make_tool_use_response("search_course_content", "toolu_r1", {"query": "A"})
    tool_use_r2 = _make_tool_use_response("search_course_content", "toolu_r2", {"query": "B"})
    text_final = _make_text_response("Final synthesized answer.")
    mock_client.messages.create.side_effect = [tool_use_r1, tool_use_r2, text_final]

    mock_tool_manager = MagicMock()
    mock_tool_manager.execute_tool.side_effect = ["Result A", "Result B"]

    tools = [{"name": "search_course_content", "description": "search", "input_schema": {}}]
    result = gen.generate_response(query="Tell me about A and B", tools=tools, tool_manager=mock_tool_manager)

    assert result == text_final.content[0].text
    assert mock_client.messages.create.call_count == 3
    assert mock_tool_manager.execute_tool.call_count == 2

    # 2nd call (round 1) must include tools — rounds_remaining was 1
    second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
    assert "tools" in second_call_kwargs

    # 3rd call (round 2 / forced synthesis) must NOT include tools
    third_call_kwargs = mock_client.messages.create.call_args_list[2].kwargs
    assert "tools" not in third_call_kwargs
    assert "tool_choice" not in third_call_kwargs


def test_stops_after_max_two_rounds():
    """
    Even if round 2's response has stop_reason='tool_use', the loop must not make a 4th call.
    """
    gen, mock_client = _make_generator()

    tool_use_r1 = _make_tool_use_response("search_course_content", "toolu_r1", {"query": "A"})
    tool_use_r2 = _make_tool_use_response("search_course_content", "toolu_r2", {"query": "B"})
    text_final = _make_text_response("Done after cap.")
    mock_client.messages.create.side_effect = [tool_use_r1, tool_use_r2, text_final]

    mock_tool_manager = MagicMock()
    mock_tool_manager.execute_tool.side_effect = ["Result A", "Result B"]

    tools = [{"name": "search_course_content", "description": "search", "input_schema": {}}]
    gen.generate_response(query="Multi search", tools=tools, tool_manager=mock_tool_manager)

    assert mock_client.messages.create.call_count == 3
    assert mock_tool_manager.execute_tool.call_count == 2


def test_single_round_sufficient_no_extra_calls():
    """
    When Claude returns end_turn after seeing round-1 results, only 2 API calls are made
    and the 2nd call includes tools (rounds_remaining was 1).
    """
    gen, mock_client = _make_generator()

    tool_use_r1 = _make_tool_use_response("search_course_content", "toolu_r1", {"query": "X"})
    text_r2 = _make_text_response("One round was enough.")
    mock_client.messages.create.side_effect = [tool_use_r1, text_r2]

    mock_tool_manager = MagicMock()
    mock_tool_manager.execute_tool.return_value = "tool result"

    tools = [{"name": "search_course_content", "description": "search", "input_schema": {}}]
    result = gen.generate_response(query="Single search query", tools=tools, tool_manager=mock_tool_manager)

    assert result == text_r2.content[0].text
    assert mock_client.messages.create.call_count == 2
    assert mock_tool_manager.execute_tool.call_count == 1

    # 2nd call must include tools (Claude chose end_turn, not forced by cap)
    second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
    assert "tools" in second_call_kwargs


def test_tool_execution_error_returns_graceful_response():
    """
    When execute_tool raises, the error is passed to Claude as a tool_result with is_error=True
    and generate_response() still returns a string without raising.
    """
    gen, mock_client = _make_generator()

    tool_use_r1 = _make_tool_use_response("search_course_content", "toolu_err", {"query": "crash"})
    synthesis_response = _make_text_response("I couldn't retrieve that information.")
    mock_client.messages.create.side_effect = [tool_use_r1, synthesis_response]

    mock_tool_manager = MagicMock()
    mock_tool_manager.execute_tool.side_effect = RuntimeError("ChromaDB unavailable")

    tools = [{"name": "search_course_content", "description": "search", "input_schema": {}}]
    result = gen.generate_response(query="Search that crashes", tools=tools, tool_manager=mock_tool_manager)

    assert result == synthesis_response.content[0].text
    assert mock_client.messages.create.call_count == 2

    # The 2nd call's messages must contain a tool_result with the error text
    second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
    messages_sent = second_call_kwargs.get("messages", [])
    tool_result_blocks = [
        b
        for m in messages_sent
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert tool_result_blocks, "No tool_result block found in 2nd API call messages"
    error_content = tool_result_blocks[0]["content"]
    assert "Tool error:" in error_content or "ChromaDB unavailable" in error_content
