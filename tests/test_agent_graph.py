"""Tests for final-answer extraction in the LangGraph agent."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from forge.api.agent.graph import extract_final_answer


class TestExtractFinalAnswer:
    """Tests for extract_final_answer."""

    def test_plain_string_content(self) -> None:
        messages = [HumanMessage(content="q"), AIMessage(content="The answer.")]
        assert extract_final_answer(messages) == "The answer."

    def test_content_blocks_with_thinking_return_only_text(self) -> None:
        final = AIMessage(
            content=[
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "XGBoost wins "},
                {"type": "text", "text": "on accuracy per ms."},
            ]
        )
        answer = extract_final_answer([HumanMessage(content="q"), final])
        assert answer == "XGBoost wins on accuracy per ms."
        assert isinstance(answer, str)

    def test_skips_messages_with_tool_calls(self) -> None:
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "search_runs", "args": {}, "id": "call_1"}],
        )
        messages = [
            HumanMessage(content="q"),
            tool_call_msg,
            ToolMessage(content="[]", tool_call_id="call_1"),
            AIMessage(content="Done."),
        ]
        assert extract_final_answer(messages) == "Done."

    def test_no_final_ai_message_returns_empty_string(self) -> None:
        assert extract_final_answer([HumanMessage(content="q")]) == ""
