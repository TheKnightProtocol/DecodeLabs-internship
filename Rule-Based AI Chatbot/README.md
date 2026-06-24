# Rule-Based AI Chatbot

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deterministic%20Chat%20UI-red.svg)](https://streamlit.io/)
[![Architecture](https://img.shields.io/badge/Architecture-White--Box%20Rule%20Engine-darkgreen.svg)]()

## Rule-Based AI Chatbot

This project is a production-ready, deterministic, white-box chatbot built for industrial AI training and portfolio presentation. It demonstrates the IPO model, explicit control flow, guardrail-oriented design, and a Streamlit web interface with traceable behavior.

No LLMs, no ML models, and no neural networks are used. Every response is produced by hard-coded program logic, making the system compliant, auditable, and easy to defend in a technical interview or training review.

### Architecture Diagram

```text
              +---------------------------+
User Input -->|  Input: normalize_input() |---+
              +---------------------------+   |
                                                 v
              +---------------------------+  +---------------------------+
              | Process: if-elif-else     |->| Output: labeled response  |
              | deterministic rule engine |  | + audit logging           |
              +---------------------------+  +---------------------------+
                         |                                |
                         v                                v
                   Guardrail / Control Layer        Conversation History
```

### IPO Model

#### Input

The application accepts user messages through a chat interface and sanitizes them with `lower()` and `strip()` so the rule engine receives stable input.

#### Process

The chatbot uses a fixed decision tree built with `if`, `elif`, and `else` statements. This is the control skeleton of the application and is intentionally transparent.

Supported intents include:

- greetings: `hello`, `hi`, `hey`
- exit: `quit`, `exit`, `bye`
- help: `help`
- time: `time`
- about: `about`
- IPO explanation: `ipo`, `ipo explanation`, `explain ipo`
- guardrail explanation: `guardrail`, `guardrails`, `control layer`
- fallback: everything else

#### Output

The result is displayed in the browser with intent labels, path labels, and audit logging so the full decision trail remains visible.

### White-Box vs Black-Box

| Aspect | White-Box Rule-Based Chatbot | Black-Box AI Model |
|---|---|---|
| Decision path | Explicit and traceable | Hidden inside model weights |
| Response style | Hard-coded and deterministic | Probabilistic and learned |
| Auditability | High | Lower |
| Compliance fit | Strong for regulated workflows | Requires extra guardrails |
| Debugging | Rule-by-rule inspection | Often indirect and statistical |

### Control Layer / Guardrail Concept

In modern AI applications, a deterministic layer can sit above a large language model or any probabilistic generator to enforce policy and safety. This project demonstrates that idea directly in a standalone form.

The app is intentionally similar in spirit to front-end guardrail systems such as NVIDIA NeMo Guardrails and Meta Llama Guard. The difference is that this project does not call those systems; it uses a local rule engine to model the same control-layer concept for training purposes.

### Streamlit Frontend

The interface provides:

- conversational chat display
- intent classification shown for every response
- sidebar architecture summary
- decision tree reference
- guardrail status indicator
- interaction metrics
- audit logging for traceability

### Installation

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

### Usage Guide

1. Type a message in the chat box.
2. The app sanitizes the input.
3. The rule engine maps the message to a deterministic intent.
4. The response appears with its intent label and processing path.
5. Type `exit`, `quit`, or `bye` to trigger the goodbye response. The browser stays open.

### Audit and Compliance Notes

- Every interaction is written to a local audit log file.
- The app avoids external API calls entirely.
- Responses are hard-coded and deterministic.
- The interface is suitable for demonstrations where traceability matters.

### Resume Talking Points

- Built a production-ready Streamlit chatbot using deterministic rule-based logic and the IPO model.
- Implemented a white-box control layer with explicit intent classification and audit logging.
- Designed a compliant, traceable system that demonstrates guardrail-style behavior without using LLMs.
- Delivered a browser-based conversational interface with transparent decision paths for every response.

### Keywords

deterministic, traceable, compliant, white-box, control layer, guardrail, IPO model, Streamlit, audit logging

### Repository Layout

```text
Rule-Based AI Chatbot/
├── app.py
├── requirements.txt
└── README.md
```

### License

Refer to the repository license file for usage terms.