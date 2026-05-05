import os
from langfuse import observe, get_client
from langfuse.openai import openai

# =========================
# LANGFUSE CONFIG
# =========================
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-f947d407-5700-446f-8f53-a690fd63c2f5"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-588380b2-8cb9-49a7-acc6-9bf8b336af49"
os.environ["LANGFUSE_TIMEOUT"] = "120"

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=240,
)

langfuse = get_client()

chat_history = []


@observe(name="local-ollama-chat", as_type="generation")
def ask_phi3(prompt: str) -> str:
    temperature = 0
    max_tokens = 100
    model = "phi3:mini"

    chat_history.append({"role": "user", "content": prompt})

    langfuse.update_current_trace(
        metadata={
            "llm.prompt": prompt,
            "llm.provider": "ollama",
            "llm.model": model,
            "environment": "local",
        }
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=chat_history,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content

        chat_history.append({"role": "assistant", "content": answer})

        langfuse.update_current_generation(
            input=prompt,
            output=answer,
            model=model,
            metadata={
                "llm.temperature": temperature,
                "llm.max_tokens": max_tokens,
                "llm.response": answer,
                "llm.provider": "ollama",
                "status": "success",
            },
        )

        return answer

    except Exception as e:
        error_message = str(e)

        langfuse.update_current_generation(
            input=prompt,
            output=None,
            model=model,
            metadata={
                "llm.temperature": temperature,
                "llm.max_tokens": max_tokens,
                "llm.provider": "ollama",
                "status": "error",
                "error.type": type(e).__name__,
                "error.message": error_message,
            },
        )

        langfuse.update_current_trace(
            metadata={
                "status": "error",
                "error.type": type(e).__name__,
                "error.message": error_message,
            }
        )

        raise


if __name__ == "__main__":
    print("\nLocal AI Chat")
    print("Type your question and press Enter.")
    print("Type 'quit', 'exit', or 'q' when you are done.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q", "done"]:
            print("\nDone. Ending chat.")
            break

        if not user_input:
            print("Please type a question, or type 'quit' to exit.\n")
            continue

        try:
            result = ask_phi3(user_input)
            print(f"AI: {result}\n")
        except Exception as e:
            print("ERROR captured and sent to Langfuse:")
            print(type(e).__name__, str(e))
            print()

    langfuse.flush()