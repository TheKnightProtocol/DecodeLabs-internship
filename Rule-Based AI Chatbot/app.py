"""Streamlit Rule-Based AI Chatbot.

This application is a white-box, deterministic chatbot designed for industrial
AI training. It demonstrates the IPO model end to end:

Input: collect user text from a chat interface and sanitize it.
Process: apply explicit if-elif-else logic to select a response.
Output: display a traceable, labeled reply with audit logging.

The app intentionally avoids LLMs, ML models, neural networks, and external API
calls. In modern AI systems, this pattern resembles a control layer or guardrail
that can sit above a probabilistic model to enforce policy, compliance, and
safe behavior. The comments reference real guardrail concepts such as NVIDIA
NeMo Guardrails and Meta Llama Guard to anchor the training discussion in real
industry terminology.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import streamlit as st


APP_TITLE = "Rule-Based AI Chatbot"
APP_SUBTITLE = "Deterministic, white-box, IPO-driven chatbot for industrial training"
MODEL_TYPE = "Rule-Based / Deterministic"
ARCHITECTURE_LABEL = "White-Box Control Layer"
AUDIT_LOG_PATH = Path(__file__).with_name("app_audit.log")
LOGGER_NAME = "rule_based_chatbot"


def configure_logging() -> None:
    """Configure file-based audit logging for chatbot interactions.

    Audit logs are a common control-layer practice in regulated environments.
    A deterministic system is easier to review than a black-box model because
    each input-to-output decision can be traced after the fact.
    """

    logger = logging.getLogger(LOGGER_NAME)

    if logger.handlers:
        return

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    try:
        file_handler = logging.FileHandler(AUDIT_LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.propagate = False


def normalize_input(raw_text: str) -> str:
    """Normalize user input for deterministic rule evaluation.

    Sanitization is the Input stage of the IPO model. Lowercasing and trimming
    whitespace reduce ambiguity so the control layer can apply exact rules.
    This mirrors the front-end validation step used before guardrails such as
    NVIDIA NeMo Guardrails or Meta Llama Guard inspect content in larger AI
    systems.
    """

    return raw_text.lower().strip()


def process_input(normalized_text: str) -> dict[str, str]:
    """Map sanitized text to a deterministic response and intent label.

    The Process stage is a hard-coded decision tree. Each branch represents a
    traceable path from input to output, which is the key white-box advantage
    over probabilistic language models.
    """

    if normalized_text in {"hello", "hi", "hey"}:
        return {
            "intent": "greeting",
            "response": "Hello. I am a deterministic rule-based chatbot.",
            "path": "if normalized_text in {'hello', 'hi', 'hey'}",
        }
    elif normalized_text in {"quit", "exit", "bye"}:
        return {
            "intent": "exit",
            "response": "Goodbye. The browser stays open, and this deterministic session is now complete.",
            "path": "elif normalized_text in {'quit', 'exit', 'bye'}",
        }
    elif normalized_text == "help":
        return {
            "intent": "help",
            "response": (
                "Available commands: hello, hi, hey, help, time, about, ipo, guardrail, "
                "quit, exit, bye."
            ),
            "path": "elif normalized_text == 'help'",
        }
    elif normalized_text == "time":
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "intent": "system_time",
            "response": f"Local system time: {current_time}",
            "path": "elif normalized_text == 'time'",
        }
    elif normalized_text == "about":
        return {
            "intent": "about_project",
            "response": (
                "This project is a white-box chatbot built to demonstrate deterministic "
                "decision-making, traceability, and guardrail-style control logic."
            ),
            "path": "elif normalized_text == 'about'",
        }
    elif normalized_text in {"ipo", "ipo explanation", "explain ipo"}:
        return {
            "intent": "ipo_explanation",
            "response": (
                "IPO means Input, Process, Output: sanitize input, apply explicit logic, "
                "then return a predictable response."
            ),
            "path": "elif normalized_text in {'ipo', 'ipo explanation', 'explain ipo'}",
        }
    elif normalized_text in {"guardrail", "guardrails", "control layer"}:
        return {
            "intent": "guardrail_explanation",
            "response": (
                "A guardrail is deterministic control logic that validates or constrains "
                "behavior before a larger AI system responds."
            ),
            "path": "elif normalized_text in {'guardrail', 'guardrails', 'control layer'}",
        }
    else:
        return {
            "intent": "fallback",
            "response": (
                "Fallback response: this input is not mapped to a rule, so the system "
                "returns a safe deterministic default."
            ),
            "path": "else",
        }


def initialize_session_state() -> None:
    """Create Streamlit session keys for conversation history and status.

    Session state preserves the conversation across reruns, which is essential
    for a chat-style interface even when the underlying application remains fully
    deterministic.
    """

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "interaction_count" not in st.session_state:
        st.session_state.interaction_count = 0
    if "chat_closed" not in st.session_state:
        st.session_state.chat_closed = False


def log_interaction(raw_text: str, normalized_text: str, result: dict[str, str]) -> None:
    """Write a traceable audit record for each chatbot interaction.

    Audit logging is part of the control layer story in real deployments. It
    helps compliance teams reconstruct what happened, which rule fired, and why
    the response was issued.
    """

    logging.getLogger(LOGGER_NAME).info(
        "input=%s | normalized=%s | intent=%s | path=%s | response=%s",
        raw_text,
        normalized_text,
        result["intent"],
        result["path"],
        result["response"],
    )


def render_sidebar() -> None:
    """Render the architecture summary and guardrail status in the sidebar.

    The sidebar acts like a training dashboard, making the chatbot's control
    structure visible so the project remains easy to explain during a defense or
    portfolio review.
    """

    st.sidebar.title("Architecture")
    st.sidebar.markdown(
        """
        ```text
        Raw Input
            ↓
        normalize_input()
            ↓
        process_input()
            ↓
        Labeled Response
            ↓
        Audit Log
        ```
        """
    )

    st.sidebar.subheader("Decision Tree")
    st.sidebar.markdown(
        """
        ```text
        hello / hi / hey  -> greeting
        quit / exit / bye  -> exit
        help               -> command help
        time               -> local system time
        about              -> project summary
        ipo                -> IPO explanation
        guardrail          -> guardrail explanation
        anything else      -> fallback
        ```
        """
    )

    st.sidebar.subheader("Guardrail Status")
    st.sidebar.success("Active: deterministic control layer engaged")
    st.sidebar.caption(
        "This pattern is conceptually similar to front-end filtering used with systems "
        "such as NVIDIA NeMo Guardrails and Meta Llama Guard."
    )


def render_metrics() -> None:
    """Display project metrics that reinforce the deterministic architecture.

    Metrics help reviewers understand the system at a glance without implying any
    machine-learning behavior.
    """

    col1, col2, col3 = st.columns(3)
    col1.metric("Interactions", st.session_state.interaction_count)
    col2.metric("Model Type", "Rule-Based")
    col3.metric("Architecture", "White-Box")


def render_chat_history() -> None:
    """Replay the full conversation history with intent labels.

    Showing intent labels makes the deterministic path visible to the user and
    turns the UI into a teaching tool for the IPO model and control-layer
    concepts.
    """

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("intent"):
                st.caption(f"Intent: {message['intent']} | Path: {message.get('path', 'n/a')}")


def add_message(role: str, content: str, intent: str | None = None, path: str | None = None) -> None:
    """Store a chat message in session state.

    Persisting the conversation is useful for auditability and for demonstrating
    how the control layer processed each step of the interaction.
    """

    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "intent": intent,
            "path": path,
        }
    )


def main() -> None:
    """Run the Streamlit application.

    This function orchestrates Input, Process, and Output in a visible and
    deterministic order, mirroring the logic skeleton expected in industrial
    AI training portfolios.
    """

    configure_logging()
    initialize_session_state()

    st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    st.markdown(
        """
        This chatbot is a white-box system: every output comes from a traceable
        rule path, not from probabilistic text generation.
        """
    )

    render_sidebar()
    render_metrics()

    st.divider()
    render_chat_history()

    st.markdown("### Chat Input")
    user_text = st.chat_input("Type a command such as hello, help, ipo, guardrail, time, or exit")

    if user_text:
        normalized_text = normalize_input(user_text)
        result = process_input(normalized_text)

        add_message("user", user_text)
        add_message("assistant", result["response"], result["intent"], result["path"])

        st.session_state.interaction_count += 1
        st.session_state.chat_closed = result["intent"] == "exit"

        log_interaction(user_text, normalized_text, result)

        st.rerun()

    if st.session_state.chat_closed:
        st.info(
            "The exit rule has fired. The browser remains open, and the session can be "
            "reviewed without terminating the Streamlit application."
        )


if __name__ == "__main__":
    main()