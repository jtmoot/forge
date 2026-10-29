"""LangGraph agent for natural language analysis of experiment and ops data.

Implements a ReAct-style agent that routes user questions to the
appropriate tools (query experiments, compare runs, semantic search,
ops summary, efficiency frontier) and synthesizes a final response.
"""

import logging
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from forge.api.agent.tools import build_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Forge Analyst, an AI assistant for analyzing ML experiments and operational data on the Forge platform.

You have access to 7 tools:

1. query_experiments — Search and filter experiment runs by dataset, model type, or metric thresholds. Use this for questions like "which model had the best accuracy" or "show me all LSTM runs" or any question about finding or filtering runs.

2. compare_runs — Side-by-side comparison of exactly two runs. Use this when the user names two specific runs or asks to compare two models directly.

3. search_similar — Semantic search over experiment history using embeddings. Use this when the question is vague or exploratory, like "find experiments similar to my SPY volatility work" or when exact filters won't capture the intent.

4. get_ops_summary — Summarize operational logs and agent activity for a time period. Use this for questions about costs, errors, agent behavior, or system health. Example: "show me ops anomalies from the last 24 hours."

5. compute_efficiency_frontier — Find Pareto-optimal runs that represent the best accuracy-to-latency tradeoffs. Use this when the user asks about efficiency, deployment tradeoffs, or "which model is best for production."

6. query_model_registry — Query the model registry for registered models, their versions, and deployment stages. Use this for questions like "what's in production?", "compare model versions", "what models are ready for promotion?", or "show me model history." Pass the model name, 'all', 'production', or 'staging' as the query.

7. check_drift — Check for data or model drift. Use for questions like "is SPY data drifting?", "which features are most unstable?", or "should I retrain the production model?". Pass a dataset name, 'summary', or 'all' as the query.

Rules:
- Always use a tool before answering questions about data. Never guess or make up metrics.
- If a tool returns empty results, tell the user what you searched for and suggest alternative queries.
- Format accuracy as percentages with 1 decimal place (e.g., 63.2%).
- Format latency in milliseconds with 1 decimal place (e.g., 0.9ms).
- Format memory in MB with 1 decimal place.
- When comparing models, always mention both accuracy AND compute cost — never just one.
- If the user's question could use multiple tools, use the most specific one first.
"""

class AgentState(TypedDict):
    """State passed through the LangGraph agent graph."""

    messages: list
    tools_used: list[str]
    intermediate_results: list[dict]


def _should_continue(state: AgentState) -> str:
    """Router: if the last AI message has tool calls, go to tools; else end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def _call_model(state: AgentState, llm_with_tools) -> dict:
    """Invoke the LLM with the current message history."""
    response = llm_with_tools.invoke(state["messages"])

    # Track which tools are being called
    tools_used = list(state.get("tools_used", []))
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            tools_used.append(tc["name"])

    return {"messages": state["messages"] + [response], "tools_used": tools_used}


def _call_tools(state: AgentState, tool_node: ToolNode) -> dict:
    """Execute the tool calls from the last AI message."""
    result = tool_node.invoke(state)

    # Capture intermediate results from tool responses
    intermediate = list(state.get("intermediate_results", []))
    new_messages = result.get("messages", [])
    for msg in new_messages:
        if isinstance(msg, ToolMessage):
            intermediate.append({
                "tool": msg.name,
                "result_preview": msg.content[:500] if len(msg.content) > 500 else msg.content,
            })

    return {
        "messages": state["messages"] + new_messages,
        "tools_used": state.get("tools_used", []),
        "intermediate_results": intermediate,
    }


def build_agent_graph(tools: list, model_name: str = "claude-sonnet-5-5") -> StateGraph:
    """Construct the LangGraph agent with the given tools.

    Args:
        tools: List of LangChain tool instances (already bound to a DB session).
        model_name: Anthropic model to use for reasoning.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    llm = ChatAnthropic(
        model=model_name,
        temperature=0,
        api_key=anthropic_api_key,
    )
    llm_with_tools = llm.bind_tools(tools)

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", lambda state: _call_model(state, llm_with_tools))
    graph.add_node("tools", lambda state: _call_tools(state, tool_node))

    # Set entry point
    graph.set_entry_point("agent")

    # Add edges
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_agent_query(question: str, db) -> dict[str, Any]:
    """Run a natural language query through the agent and return structured results.

    Args:
        question: The user's natural language question.
        db: Active SQLAlchemy database session.

    Returns:
        Dict with keys: answer, tools_used, intermediate_results.
    """
    tools = build_tools(db)
    agent = build_agent_graph(tools)

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ],
        "tools_used": [],
        "intermediate_results": [],
    }

    result = agent.invoke(initial_state)

    # Extract the final answer from the last AI message
    answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            answer = msg.content
            break

    return {
        "answer": answer,
        "tools_used": list(set(result.get("tools_used", []))),
        "intermediate_results": result.get("intermediate_results", []),
    }
