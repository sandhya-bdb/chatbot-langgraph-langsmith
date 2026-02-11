from __future__ import annotations

from typing import Annotated
import re
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from langgraph_cb.config import load_env
from langgraph_cb.tools.stocks import get_stock_price, prepare_buy


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph():
    load_env()

    tools = [get_stock_price, prepare_buy]
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    def _parse_buy_intent(text: str) -> tuple[int, str] | None:
        match = re.search(r"\bbuy\s+(\d+)\s+([a-zA-Z]{1,10})\b", text, re.IGNORECASE)
        if not match:
            return None
        quantity = int(match.group(1))
        symbol = match.group(2).upper()
        return quantity, symbol

    def chatbot_node(state: State):
        last = state["messages"][-1]
        if isinstance(last, HumanMessage):
            parsed = _parse_buy_intent(last.content)
            if parsed:
                quantity, symbol = parsed
                price = get_stock_price.invoke(symbol)
                total_price = price * quantity
                tool_content = prepare_buy.invoke(
                    {
                        "symbol": symbol,
                        "quantity": quantity,
                        "total_price": total_price,
                    }
                )
                return {
                    "messages": [
                        {
                            "role": "tool",
                            "content": tool_content,
                            "tool_call_id": "manual_prepare_buy",
                        }
                    ]
                }

        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def _is_tool_message(msg) -> bool:
        if isinstance(msg, ToolMessage) or getattr(msg, "type", None) == "tool":
            return True
        if isinstance(msg, dict):
            return msg.get("role") == "tool"
        return False

    def _get_content(msg) -> str:
        if isinstance(msg, dict):
            return str(msg.get("content", ""))
        return str(getattr(msg, "content", ""))

    def approval_node(state: State):
        last_message = state["messages"][-1]
        content = _get_content(last_message)

        if _is_tool_message(last_message) and content.startswith("REQUEST_BUY::"):
            _, symbol, quantity, total_price = content.split("::")

            decision = interrupt(
                f"Approve buying {quantity} {symbol} stocks for ${float(total_price):.2f}?"
            )

            if decision == "yes":
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                f"Approved: Bought {quantity} shares of {symbol} "
                                f"for ${total_price}"
                            )
                        )
                    ]
                }

            return {"messages": [AIMessage(content="Trade declined by human.")]}

        return {}

    memory = MemorySaver()
    builder = StateGraph(State)

    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("approval", approval_node)

    builder.add_edge(START, "chatbot")
    def route_from_chatbot(state: State):
        last = state["messages"][-1]
        content = _get_content(last)
        if _is_tool_message(last) and content.startswith("REQUEST_BUY::"):
            return "approval"
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    builder.add_conditional_edges("chatbot", route_from_chatbot)
    builder.add_edge("tools", "approval")
    builder.add_edge("approval", END)

    return builder.compile(checkpointer=memory)


def run_demo() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "buy_thread"}}

    graph.invoke(
        {"messages": [HumanMessage(content="What is the current price of 10 MSFT stocks?")]},
        config=config,
    )

    graph.invoke(
        {"messages": [HumanMessage(content="Buy 10 MSFT stocks at current price.")]},
        config=config,
    )

    decision = input("Approve (yes/no): ").strip().lower()
    state = graph.invoke(Command(resume=decision), config=config)

    print(state["messages"][-1].content)
