from openai import OpenAI

from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
import sys
import time
import threading


# =========================
# APP / OTEL CONFIG
# =========================
SERVICE_NAME = "local-ollama-phi3"
APP_NAME = "test-ollama-splunk-only"
APP_TYPE = "interactive-chat"
APP_GROUP = "local-ai-observability"

MODEL = "phi3:mini"
PROVIDER = "ollama"
TENANT = "demo-tenant"
ENVIRONMENT = "local"

TRACE_ENDPOINT = "http://localhost:4318/v1/traces"
METRIC_ENDPOINT = "http://localhost:4318/v1/metrics"

# Local Ollama has no real API billing cost. These demo rates make dashboards non-zero.
INPUT_COST_PER_1M_TOKENS = 0.10
OUTPUT_COST_PER_1M_TOKENS = 0.20



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


# =========================
# OTEL RESOURCE
# =========================
resource = Resource.create(
    {
        "service.name": SERVICE_NAME,
        "deployment.environment": ENVIRONMENT,
        "tenant": TENANT,
        "app.group": APP_GROUP,
        "app.name": APP_NAME,
        "app.type": APP_TYPE,
        "llm.provider": PROVIDER,
        "llm.model": MODEL,
    }
)


# =========================
# OTEL TRACE SETUP
# =========================
trace_provider = TracerProvider(resource=resource)
trace_exporter = OTLPSpanExporter(endpoint=TRACE_ENDPOINT, timeout=30)
trace_processor = BatchSpanProcessor(
    trace_exporter,
    schedule_delay_millis=500,
    max_export_batch_size=32,
    max_queue_size=256,
)
trace_provider.add_span_processor(trace_processor)
trace.set_tracer_provider(trace_provider)
tracer = trace.get_tracer(__name__)


# =========================
# OTEL METRICS SETUP
# =========================
metric_exporter = OTLPMetricExporter(endpoint=METRIC_ENDPOINT, timeout=30)
metric_reader = PeriodicExportingMetricReader(
    metric_exporter,
    export_interval_millis=5000,
)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(__name__)


# =========================
# CUSTOM LLM METRICS
# =========================
llm_requests_counter = meter.create_counter("llm.requests", description="Number of LLM requests", unit="1")
prompt_tokens_counter = meter.create_counter("llm.prompt.tokens", description="Prompt/input tokens used", unit="tokens")
completion_tokens_counter = meter.create_counter("llm.completion.tokens", description="Completion/output tokens generated", unit="tokens")
total_tokens_counter = meter.create_counter("llm.total.tokens", description="Total LLM tokens used", unit="tokens")
prompt_cost_counter = meter.create_counter("llm.prompt.cost.usd", description="Estimated prompt token cost", unit="USD")
completion_cost_counter = meter.create_counter("llm.completion.cost.usd", description="Estimated completion token cost", unit="USD")
total_cost_counter = meter.create_counter("llm.total.cost.usd", description="Estimated total LLM cost", unit="USD")
llm_cost_counter = meter.create_counter("llm.cost.usd", description="Estimated total LLM cost grouped by tenant", unit="USD")


# =========================
# OLLAMA CONFIG
# =========================
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
chat_history = []


# =========================
# HELPERS
# =========================
def estimate_tokens(text: str) -> int:
    return max(1, len(text.split())) if text else 0


def get_usage(response, prompt_text: str, answer_text: str):
    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or prompt_tokens + completion_tokens
    else:
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(answer_text)
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


def calculate_cost(prompt_tokens: int, completion_tokens: int):
    prompt_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_1M_TOKENS
    completion_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_1M_TOKENS
    return prompt_cost, completion_cost, prompt_cost + completion_cost


def metric_attrs(status: str = "success", source: str = "interactive"):
    return {
        "tenant": TENANT,
        "llm.model": MODEL,
        "llm.provider": PROVIDER,
        "environment": ENVIRONMENT,
        "service.name": SERVICE_NAME,
        "app.group": APP_GROUP,
        "app.name": APP_NAME,
        "app.type": APP_TYPE,
        "status": status,
        "source": source,
    }


# =========================
# LLM CALL WITH TRACE + METRICS
# =========================
def ask_phi3(prompt: str) -> str:
    temperature = 0
    max_tokens = 100
    chat_history.append({"role": "user", "content": prompt})
    attrs = metric_attrs()

    with tracer.start_as_current_span("ollama-chat-request") as span:
        span.set_attribute("tenant", TENANT)
        span.set_attribute("app.group", APP_GROUP)
        span.set_attribute("app.name", APP_NAME)
        span.set_attribute("app.type", APP_TYPE)
        span.set_attribute("privacy.region", PRIVACY_REGION)
        span.set_attribute("privacy.laws", PRIVACY_LAWS)
        span.set_attribute("privacy.retention_days", PRIVACY_RETENTION_DAYS)
        span.set_attribute("processing.legal_basis", PRIVACY_LEGAL_BASIS)
        span.set_attribute("source", "interactive")
        span.set_attribute("llm.provider", PROVIDER)
        span.set_attribute("llm.model", MODEL)
        span.set_attribute("llm.prompt", prompt)
        span.set_attribute("llm.temperature", temperature)
        span.set_attribute("llm.max_tokens", max_tokens)

        try:
            activity = show_activity(f"{MODEL} is thinking")
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=chat_history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            finally:
                activity.set()
            answer = response.choices[0].message.content or ""
            chat_history.append({"role": "assistant", "content": answer})

            prompt_tokens, completion_tokens, total_tokens = get_usage(response, prompt, answer)
            prompt_cost, completion_cost, total_cost = calculate_cost(prompt_tokens, completion_tokens)

            span.set_attribute("llm.response", answer)
            span.set_attribute("llm.prompt.tokens", prompt_tokens)
            span.set_attribute("llm.completion.tokens", completion_tokens)
            span.set_attribute("llm.total.tokens", total_tokens)
            span.set_attribute("llm.prompt.cost.usd", prompt_cost)
            span.set_attribute("llm.completion.cost.usd", completion_cost)
            span.set_attribute("llm.total.cost.usd", total_cost)
            span.set_attribute("llm.cost.usd", total_cost)
            span.set_attribute("status", "success")

            llm_requests_counter.add(1, attrs)
            prompt_tokens_counter.add(prompt_tokens, attrs)
            completion_tokens_counter.add(completion_tokens, attrs)
            total_tokens_counter.add(total_tokens, attrs)
            prompt_cost_counter.add(prompt_cost, attrs)
            completion_cost_counter.add(completion_cost, attrs)
            total_cost_counter.add(total_cost, attrs)
            llm_cost_counter.add(total_cost, attrs)

            print("\nTelemetry emitted:")
            print(f"  service: {SERVICE_NAME}")
            print(f"  app.group: {APP_GROUP}")
            print(f"  model: {MODEL}")
            print(f"  total_tokens: {total_tokens}")
            print(f"  total_cost_usd: {total_cost:.8f}\n")
            return answer

        except Exception as e:
            span.set_attribute("status", "error")
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            llm_requests_counter.add(1, metric_attrs(status="error"))
            raise


if __name__ == "__main__":
    print("\nLocal AI Chat - Splunk OTEL Single Model")
    print(f"Service: {SERVICE_NAME}")
    print(f"Selected Privacy Region: {PRIVACY_REGION}")
    print(f"Applicable Laws: {PRIVACY_LAWS}")
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
        trace_provider.force_flush(timeout_millis=30000)
        meter_provider.force_flush(timeout_millis=30000)
        trace_provider.shutdown()
        meter_provider.shutdown()
        print("Telemetry flushed. Exiting.")
