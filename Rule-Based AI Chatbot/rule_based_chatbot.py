"""Rule-Based AI Chatbot

This script demonstrates a white-box, deterministic chatbot built with the
IPO model:

Input: collect raw user text from the console.
Process: normalize the text and apply explicit if-elif-else rules.
Output: print a clearly traceable response.

There are no LLM calls here. Every response is produced by hard-coded control
flow, which makes this useful as a control layer or guardrail example in AI
systems.
"""


def normalize_text(raw_input):
    """Convert input to a stable comparison form."""
    return raw_input.lower().strip()


def generate_response(user_text):
    """Return a deterministic response for the provided normalized input."""
    if user_text in ("hello", "hi"):
        return "Hello! I am a rule-based chatbot."
    elif user_text in ("quit", "exit"):
        return "exit"
    else:
        # This fallback acts like a guardrail: unrecognized input is handled
        # by a safe, predictable default path instead of free-form generation.
        return "I did not understand that. Try 'hello', 'hi', 'quit', or 'exit'."


def main():
    """Run the chatbot interaction loop."""
    print("Rule-Based AI Chatbot")
    print("Type 'hello' or 'hi' to test the greeting rule.")
    print("Type 'quit' or 'exit' to stop the program.\n")

    while True:
        # Input phase: collect raw text from the user.
        raw_input_value = input("You: ")

        # Sanitization phase: normalize the text before rule evaluation.
        normalized_input = normalize_text(raw_input_value)

        # Process phase: deterministic control flow with explicit branches.
        response = generate_response(normalized_input)

        if response == "exit":
            print("Chatbot: Goodbye. Session ended deterministically.")
            break

        # Output phase: display the final response clearly in the console.
        print(f"Chatbot: {response}")


if __name__ == "__main__":
    main()