from openai import OpenAI
from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter


SERVICE_NAME = "local-ollama-model-compare"
APP_NAME = "test-ollama-model-compare"
APP_TYPE = "model-comparison"
APP_GROUP = "local-ai-observability"
TENANT = "demo-tenant"
PROVIDER = "ollama"
ENVIRONMENT = "local"

MODELS = ["phi3:mini", "mistral"]

TRACE_ENDPOINT = "http://localhost:4318/v1/traces"
METRIC_ENDPOINT = "http://localhost:4318/v1/metrics"

# Demo cost rates only
INPUT_COST_PER_1M_TOKENS = 0.10
OUTPUT_COST_PER_1M_TOKENS = 0.20


resource = Resource.create({
    "service.name": SERVICE_NAME,
    "deployment.environment": ENVIRONMENT,
    "llm.provider": PROVIDER,
    "tenant": TENANT,
    "app.group": APP_GROUP,
    "app.name": APP_NAME,
    "app.type": APP_TYPE,
})

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

metric_exporter = OTLPMetricExporter(endpoint=METRIC_ENDPOINT, timeout=30)
metric_reader = PeriodicExportingMetricReader(
    metric_exporter,
    export_interval_millis=5000,
)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(__name__)

request_counter = meter.create_counter("llm.requests")
prompt_tokens_counter = meter.create_counter("llm.prompt.tokens")
completion_tokens_counter = meter.create_counter("llm.completion.tokens")
total_tokens_counter = meter.create_counter("llm.total.tokens")
cost_counter = meter.create_counter("llm.cost.usd")

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


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


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    input_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_1M_TOKENS
    output_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_1M_TOKENS
    return input_cost + output_cost


def ask_model(prompt: str, model: str) -> str:
    temperature = 0
    max_tokens = 100

    attrs = {
        "tenant": TENANT,
        "llm.model": model,
        "llm.provider": PROVIDER,
        "environment": ENVIRONMENT,
        "service.name": SERVICE_NAME,
        "app.group": APP_GROUP,
        "app.name": APP_NAME,
        "app.type": APP_TYPE,
    }

    with tracer.start_as_current_span("ollama-chat-request") as span:
        span.set_attribute("tenant", TENANT)
        span.set_attribute("app.group", APP_GROUP)
        span.set_attribute("app.name", APP_NAME)
        span.set_attribute("app.type", APP_TYPE)
        span.set_attribute("llm.provider", PROVIDER)
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.prompt", prompt)
        span.set_attribute("llm.temperature", temperature)
        span.set_attribute("llm.max_tokens", max_tokens)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content or ""

        prompt_tokens, completion_tokens, total_tokens = get_usage(
            response=response,
            prompt_text=prompt,
            answer_text=answer,
        )

        total_cost = calculate_cost(prompt_tokens, completion_tokens)

        span.set_attribute("llm.response", answer)
        span.set_attribute("llm.prompt.tokens", prompt_tokens)
        span.set_attribute("llm.completion.tokens", completion_tokens)
        span.set_attribute("llm.total.tokens", total_tokens)
        span.set_attribute("llm.cost.usd", total_cost)
        span.set_attribute("status", "success")

        request_counter.add(1, attrs)
        prompt_tokens_counter.add(prompt_tokens, attrs)
        completion_tokens_counter.add(completion_tokens, attrs)
        total_tokens_counter.add(total_tokens, attrs)
        cost_counter.add(total_cost, attrs)

        print(f"\nModel: {model}")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Completion tokens: {completion_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"Estimated cost USD: {total_cost:.8f}")

        return answer


def choose_model() -> str:
    print("\nAvailable models:")
    for i, model in enumerate(MODELS, start=1):
        print(f"{i}. {model}")

    print("3. compare-both")

    choice = input("\nChoose model [1/2/3]: ").strip()

    if choice == "1":
        return "phi3:mini"
    if choice == "2":
        return "mistral"
    if choice == "3":
        return "compare-both"

    print("Invalid choice. Defaulting to phi3:mini.")
    return "phi3:mini"


if __name__ == "__main__":
    print("\nLocal AI Chat - Model Comparison")
    print(f"Service: {SERVICE_NAME}")
    print(f"App Group: {APP_GROUP}")
    print("Type your question and press Enter.")
    print("Type 'q' to quit.\n")

    selected_model = choose_model()

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["q", "quit", "exit", "done"]:
                print("\nDone. Ending chat.")
                break

            if not user_input:
                continue

            if selected_model == "compare-both":
                with tracer.start_as_current_span("model-comparison") as parent_span:
                    parent_span.set_attribute("tenant", TENANT)
                    parent_span.set_attribute("app.group", APP_GROUP)
                    parent_span.set_attribute("app.name", APP_NAME)
                    parent_span.set_attribute("app.type", APP_TYPE)
                    parent_span.set_attribute("llm.prompt", user_input)
                    parent_span.set_attribute("comparison.models", ",".join(MODELS))

                    for model in MODELS:
                        answer = ask_model(user_input, model)
                        print(f"\nAI ({model}): {answer}\n")
            else:
                answer = ask_model(user_input, selected_model)
                print(f"\nAI ({selected_model}): {answer}\n")

    finally:
        print("Flushing telemetry...")
        trace_provider.force_flush(timeout_millis=30000)
        meter_provider.force_flush(timeout_millis=30000)
        trace_provider.shutdown()
        meter_provider.shutdown()
        print("Telemetry flushed.")