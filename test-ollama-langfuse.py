import os
from langfuse import observe, get_client
from langfuse.openai import openai

# Langfuse self-hosted instance
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-REPLACE_ME"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-REPLACE_ME"
os.environ["LANGFUSE_BASE_URL"] = "http://localhost:3000"

# Optional: route OpenTelemetry through your local collector
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

langfuse = get_client()


@observe(name="local-ollama-chat", as_type="generation")
def ask_phi3(prompt: str) -> str:
    response = client.chat.completions.create(
        name="phi3-mini-local-generation",
        model="phi3:mini",
        messages=[
            {
                "role": "system",
                "content": "You are a concise AI observability assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    langfuse.update_current_generation(
        input=prompt,
        output=answer,
        model="phi3:mini",
        metadata={
            "provider": "ollama",
            "environment": "local",
            "hardware": "cpu",
        },
        tags=["ollama", "phi3-mini", "local-test"],
    )

    return answer


if __name__ == "__main__":
    result = ask_phi3("Explain why OpenTelemetry matters for LLM applications.")
    print(result)

    # Ensures traces are sent before the script exits
    langfuse.flush()