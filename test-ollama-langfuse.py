import os
from langfuse import observe, get_client
from langfuse.openai import openai

# =========================
# LANGFUSE CONFIG
# =========================
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-70e39cd5-c4bc-4101-ad75-0b7d1cf51db6"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-09d2efb2-a828-4443-94e1-046ce4509c3a"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
os.environ["LANGFUSE_TIMEOUT"] = "120"

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=240,
)

langfuse = get_client()


@observe(name="local-ollama-chat", as_type="generation")
def ask_phi3(prompt: str) -> str:
    temperature = 0
    max_tokens = 10
    model = "phi3:mini"

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
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content

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
    try:
        result = ask_phi3("Say hello")
        print(result)
    except Exception as e:
        print("ERROR captured and sent to Langfuse:")
        print(type(e).__name__, str(e))
    finally:
        langfuse.flush()