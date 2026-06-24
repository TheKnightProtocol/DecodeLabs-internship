# DecodeLabs Internship

## Rule-Based AI Chatbot

This repository contains a deterministic, white-box Python chatbot built for industrial AI training. The implementation demonstrates the IPO model, explicit control flow, and a basic guardrail pattern that can be explained line by line.

The chatbot does not call any LLMs or external AI services. Every response is produced through hard-coded program logic, which makes the system traceable, predictable, and easy to audit.

### Project Goals

- Demonstrate the IPO model in a simple console application
- Show how a rule-based chatbot can act as a control layer
- Provide a white-box example where every output has a clear path from input to response
- Build a clean foundation for discussing AI guardrails in regulated environments

### Architecture Overview

#### Input

The chatbot runs in a continuous `while True` loop and collects user text from the console.

#### Process

User input is normalized with lowercase conversion and whitespace stripping using:

```python
raw_input.lower().strip()
```

After normalization, the chatbot evaluates the input with explicit `if`, `elif`, and `else` statements.

Supported rules:

- `hello` or `hi` return a predefined greeting
- `quit` or `exit` terminate the session
- any other input triggers a default fallback response

#### Output

The chatbot prints a clear response back to the console so the interaction remains easy to follow and verify.

### Why This Is a White-Box System

This chatbot is intentionally deterministic. Unlike neural network-based assistants, the decision path is visible and fixed:

- input is normalized
- rules are evaluated in order
- a known output is returned

That makes the code suitable for explaining transparency, compliance, and safety filtering concepts commonly discussed in modern AI systems.

### Guardrail / Control Layer Concept

In production AI systems, deterministic logic like this often sits above or around probabilistic models as a guardrail. Its role is to:

- filter or validate input before processing
- enforce allowed behaviors
- block undefined or unsafe paths
- provide predictable fallback handling

This project demonstrates that idea in its simplest form.

### Repository Structure

```text
DecodeLabs-internship/
├── README.md
├── rule_based_chatbot.py
└── Rule-Based AI Chatbot/
	└── rule_based_chatbot.py
```

### How To Run

Use Python 3 and run either copy of the chatbot script from the repository root:

```bash
python rule_based_chatbot.py
```

Or run the version stored inside the project folder:

```bash
python "Rule-Based AI Chatbot/rule_based_chatbot.py"
```

### Example Interaction

```text
Rule-Based AI Chatbot
Type 'hello' or 'hi' to test the greeting rule.
Type 'quit' or 'exit' to stop the program.

You: hello
Chatbot: Hello! I am a rule-based chatbot.

You: status
Chatbot: I did not understand that. Try 'hello', 'hi', 'quit', or 'exit'.

You: exit
Chatbot: Goodbye. Session ended deterministically.
```

### Validation

- Python syntax validation completed successfully
- The feature branch was pushed to GitHub

### Suggested Resume / Report Wording

> Built a deterministic rule-based AI chatbot in Python using the IPO model, explicit control flow, and white-box decision logic to demonstrate guardrail-style input filtering and traceable response generation.

### License

Refer to the repository license file for usage terms.