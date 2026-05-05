from openai import OpenAI

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# =========================
# OTEL CONFIG
# =========================
SERVICE_NAME = "local-ollama-phi3"
OTEL_ENDPOINT = "http://localhost:4318/v1/traces"

resource = Resource.create({"service.name": SERVICE_NAME})
provider = TracerProvider(resource=resource)

exporter = OTLPSpanExporter(
    endpoint=OTEL_ENDPOINT,
    timeout=30,
)

processor = BatchSpanProcessor(
    exporter,
    schedule_delay_millis=500,
    max_export_batch_size=32,
    max_queue_size=256,
)

provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# =========================
# OLLAMA CONFIG
# =========================
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

chat_history = []


def ask_phi3(prompt: str) -> str:
    model = "phi3:mini"
    temperature = 0
    max_tokens = 100

    chat_history.append({"role": "user", "content": prompt})

    with tracer.start_as_current_span("ollama-chat-request") as span:
        span.set_attribute("llm.provider", "ollama")
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.prompt", prompt)
        span.set_attribute("llm.temperature", temperature)
        span.set_attribute("llm.max_tokens", max_tokens)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=chat_history,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = response.choices[0].message.content
            chat_history.append({"role": "assistant", "content": answer})

            span.set_attribute("llm.response", answer)
            span.set_attribute("status", "success")

            return answer

        except Exception as e:
            span.set_attribute("status", "error")
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            raise


if __name__ == "__main__":
    print("\nLocal AI Chat - Splunk OTEL Test")
    print("Type your question and press Enter.")
    print("Type 'quit', 'exit', or 'q' when you are done.\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q", "done"]:
                print("\nDone. Ending chat.")
                break

            if not user_input:
                print("Please type a question, or type 'quit' to exit.\n")
                continue

            result = ask_phi3(user_input)
            print(f"AI: {result}\n")

    finally:
        provider.force_flush(timeout_millis=30000)
        provider.shutdown()