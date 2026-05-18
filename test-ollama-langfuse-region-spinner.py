import os
import sys
import time
import threading

# =========================
# LANGFUSE CONFIG - must be BEFORE langfuse imports
# =========================
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-e78ef80d-57ab-4c7a-9fef-3a33ff566518"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-b69bdd5a-d3c6-4c9d-bf93-db9063453ec1"
os.environ["LANGFUSE_TIMEOUT"] = "120"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

# OTEL collector
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://localhost:4318/v1/traces"
os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
os.environ["OTEL_SERVICE_NAME"] = "local-ollama-phi3"

from langfuse import observe, get_client
from langfuse.openai import openai

# OTEL HERE

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


# =========================================================
# REGION / DATA PROTECTION GOVERNANCE
# =========================================================
REGION_LAWS = {
    "SA": {
        "name": "South Africa",
        "laws": "POPIA, PAIA, and where relevant National Credit Act / FICA",
        "default_retention_days": 30,
        "legal_basis": "consent",
    },
    "EU": {
        "name": "European Union",
        "laws": "GDPR, ePrivacy Directive, NIS2, and EU AI Act considerations",
        "default_retention_days": 30,
        "legal_basis": "consent_or_legitimate_interest",
    },
    "UK": {
        "name": "United Kingdom",
        "laws": "UK GDPR, Data Protection Act 2018, and Data (Use and Access) Act 2025 considerations",
        "default_retention_days": 30,
        "legal_basis": "consent_or_legitimate_interest",
    },
}


def choose_privacy_region() -> str:
    print("\n================================================")
    print("Data Protection Region Selection")
    print("================================================")
    print("As a firm, we respect data protection laws in the chosen region.")
    print("Please select the region whose privacy and governance rules should be applied:\n")
    print("1. SA - South Africa: POPIA / PAIA")
    print("2. EU - European Union: GDPR")
    print("3. UK - United Kingdom: UK GDPR / Data Protection Act 2018")
    choice = input("\nChoose region [SA/EU/UK] default SA: ").strip().upper()

    if choice in ["1", "SA", "SOUTH AFRICA"]:
        region = "SA"
    elif choice in ["2", "EU", "EUROPE", "EUROPEAN UNION"]:
        region = "EU"
    elif choice in ["3", "UK", "UNITED KINGDOM"]:
        region = "UK"
    else:
        region = "SA"

    selected = REGION_LAWS[region]
    print(f"\nSelected region: {selected['name']}")
    print(f"The following data protection laws/governance expectations will be applied: {selected['laws']}")
    print("The application will tag telemetry with privacy.region, retention metadata, and governance attributes.")
    print("================================================\n")
    return region


PRIVACY_REGION = choose_privacy_region()
PRIVACY_LAWS = REGION_LAWS[PRIVACY_REGION]["laws"]
PRIVACY_RETENTION_DAYS = REGION_LAWS[PRIVACY_REGION]["default_retention_days"]
PRIVACY_LEGAL_BASIS = REGION_LAWS[PRIVACY_REGION]["legal_basis"]


def privacy_attrs() -> dict:
    return {
        "privacy.region": PRIVACY_REGION,
        "privacy.laws": PRIVACY_LAWS,
        "privacy.retention_days": str(PRIVACY_RETENTION_DAYS),
        "processing.legal_basis": PRIVACY_LEGAL_BASIS,
        "privacy.notice": "As a firm we respect data protection laws in the chosen region.",
    }


def show_activity(message: str = "Searching / thinking"):
    stop_event = threading.Event()

    def spinner():
        symbols = ["|", "/", "-", "\\"]
        index = 0
        while not stop_event.is_set():
            sys.stdout.write(f"\r{message} {symbols[index % len(symbols)]}")
            sys.stdout.flush()
            index += 1
            time.sleep(0.15)
        sys.stdout.write("\r" + " " * 90 + "\r")
        sys.stdout.flush()

    thread = threading.Thread(target=spinner, daemon=True)
    thread.start()
    return stop_event


resource = Resource.create({
    "service.name": "local-ollama-phi3",
    "deployment.environment": "local",
    "privacy.region": PRIVACY_REGION,
    "privacy.laws": PRIVACY_LAWS,
    "privacy.retention_days": PRIVACY_RETENTION_DAYS,
    "processing.legal_basis": PRIVACY_LEGAL_BASIS,
})

provider = TracerProvider(resource=resource)

processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
)

provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)


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

    with tracer.start_as_current_span("ollama-chat-request") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.provider", "ollama")
        span.set_attribute("privacy.region", PRIVACY_REGION)
        span.set_attribute("privacy.laws", PRIVACY_LAWS)
        span.set_attribute("privacy.retention_days", PRIVACY_RETENTION_DAYS)
        span.set_attribute("processing.legal_basis", PRIVACY_LEGAL_BASIS)
        span.set_attribute("llm.prompt", prompt)

        langfuse.update_current_trace(
            metadata={
                "llm.prompt": prompt,
                "llm.provider": "ollama",
                "llm.model": model,
                "environment": "local",
            }
        )

        try:
            activity = show_activity(f"{model} is thinking")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=chat_history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            finally:
                activity.set()

            answer = response.choices[0].message.content

            chat_history.append({"role": "assistant", "content": answer})

            span.set_attribute("llm.response", answer)
            span.set_attribute("llm.temperature", temperature)
            span.set_attribute("llm.max_tokens", max_tokens)
            span.set_attribute("status", "success")

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

            span.set_attribute("status", "error")
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", error_message)

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
    print(f"Selected Privacy Region: {PRIVACY_REGION}")
    print(f"Applicable Laws: {PRIVACY_LAWS}")
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

    # langfuse.flush()
    provider.force_flush()
    provider.shutdown()
