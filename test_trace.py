import requests
from opentelemetry import trace

# Create OpenTelemetry tracer
tracer = trace.get_tracer("ollama-app")

def call_ollama(prompt: str):
    print("Calling Ollama...")

    with tracer.start_as_current_span("ollama-request") as span:
        span.set_attribute("llm.input", prompt)

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        output = response.json().get("response", "")

        span.set_attribute("llm.output", output)

    return output


if __name__ == "__main__":
    prompt = "Explain OpenTelemetry in simple terms"

    result = call_ollama(prompt)

    print("\nDone.\n")
    print("Response:\n")
    print(result)