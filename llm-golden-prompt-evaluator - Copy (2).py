import time
from typing import Dict, List

from openai import OpenAI

from opentelemetry import metrics, trace
from opentelemetry.metrics import Observation

from opentelemetry.sdk.resources import Resource

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter


# =========================================================
# CONFIG
# =========================================================
SERVICE_NAME = "llm-golden-prompt-evaluator"
APP_NAME = "llm-golden-prompt-evaluator"
APP_TYPE = "evaluation"
APP_GROUP = "local-ai-observability"

MODELS = [
    "phi3:mini",
    "mistral",
]

TENANT = "demo-tenant"
PROVIDER = "ollama"
ENVIRONMENT = "local"

TRACE_ENDPOINT = "http://localhost:4318/v1/traces"
METRIC_ENDPOINT = "http://localhost:4318/v1/metrics"

EVAL_INTERVAL_SECONDS = 300

INPUT_COST_PER_1M_TOKENS = 0.10
OUTPUT_COST_PER_1M_TOKENS = 0.20


# =========================================================
# GOLDEN PROMPTS
# =========================================================
GOLDEN_PROMPTS: List[Dict[str, object]] = [
    {
        "prompt": "What is 2 + 2?",
        "expected_substrings": ["4"],
    },
    {
        "prompt": "What is the capital city of France?",
        "expected_substrings": ["paris"],
    },
    {
        "prompt": "What cloud provider has EC2?",
        "expected_substrings": ["aws", "amazon"],
    },
    {
        "prompt": "What does CPU stand for?",
        "expected_substrings": ["central", "processing", "unit"],
    },
    {
        "prompt": "What does HTTP stand for?",
        "expected_substrings": ["hypertext", "transfer", "protocol"],
    },
    {
        "prompt": "What is OpenTelemetry used for?",
        "expected_substrings": ["telemetry", "traces", "metrics"],
    },
    {
        "prompt": "What is Splunk Observability used for?",
        "expected_substrings": ["monitor", "observability"],
    },
    {
        "prompt": "What is Docker used for?",
        "expected_substrings": ["container"],
    },
    {
        "prompt": "What is Kubernetes used for?",
        "expected_substrings": ["orchestration", "containers"],
    },
    {
        "prompt": "What database language is used to query relational databases?",
        "expected_substrings": ["sql"],
    },
    {
        "prompt": "What is Redis commonly used for?",
        "expected_substrings": ["cache"],
    },
    {
        "prompt": "What is PostgreSQL?",
        "expected_substrings": ["database"],
    },
    {
        "prompt": "What is ClickHouse used for?",
        "expected_substrings": ["analytics", "database"],
    },
    {
        "prompt": "What is MinIO compatible with?",
        "expected_substrings": ["s3"],
    },
    {
        "prompt": "What is an API?",
        "expected_substrings": ["application", "programming", "interface"],
    },
    {
        "prompt": "What is latency?",
        "expected_substrings": ["delay", "time"],
    },
    {
        "prompt": "What is a trace in observability?",
        "expected_substrings": ["request", "span"],
    },
    {
        "prompt": "What is a metric in observability?",
        "expected_substrings": ["measurement", "value"],
    },
    {
        "prompt": "What is drift detection for AI models?",
        "expected_substrings": ["change", "performance"],
    },
    {
        "prompt": "What is a golden test set?",
        "expected_substrings": ["expected", "baseline"],
    },
]


# =========================================================
# OTEL RESOURCE
# =========================================================
resource = Resource.create(
    {
        "service.name": SERVICE_NAME,
        "deployment.environment": ENVIRONMENT,
        "tenant": TENANT,
        "app.group": APP_GROUP,
        "app.name": APP_NAME,
        "app.type": APP_TYPE,
    }
)


# =========================================================
# TRACE SETUP
# =========================================================
trace_provider = TracerProvider(resource=resource)

trace_exporter = OTLPSpanExporter(
    endpoint=TRACE_ENDPOINT,
    timeout=30,
)

trace_processor = BatchSpanProcessor(
    trace_exporter,
    schedule_delay_millis=500,
    max_export_batch_size=32,
    max_queue_size=256,
)

trace_provider.add_span_processor(trace_processor)
trace.set_tracer_provider(trace_provider)

tracer = trace.get_tracer(__name__)


# =========================================================
# METRICS SETUP
# =========================================================
metric_exporter = OTLPMetricExporter(
    endpoint=METRIC_ENDPOINT,
    timeout=30,
)

metric_reader = PeriodicExportingMetricReader(
    metric_exporter,
    export_interval_millis=5000,
)

meter_provider = MeterProvider(
    resource=resource,
    metric_readers=[metric_reader],
)

metrics.set_meter_provider(meter_provider)

meter = metrics.get_meter(__name__)


# =========================================================
# METRIC STATE
# =========================================================
latest_accuracy = {}

latest_passed = {}

latest_failed = {}

latest_total = {}


# =========================================================
# OBSERVABLE CALLBACKS
# =========================================================
def accuracy_callback(options):
    observations = []

    for model, value in latest_accuracy.items():
        observations.append(
            Observation(
                value,
                attributes={
                    "tenant": TENANT,
                    "llm.model": model,
                    "llm.provider": PROVIDER,
                    "environment": ENVIRONMENT,
                    "app.group": APP_GROUP,
                    "app.name": APP_NAME,
                    "app.type": APP_TYPE,
                },
            )
        )

    return observations


def passed_callback(options):
    observations = []

    for model, value in latest_passed.items():
        observations.append(
            Observation(
                value,
                attributes={
                    "tenant": TENANT,
                    "llm.model": model,
                    "llm.provider": PROVIDER,
                    "environment": ENVIRONMENT,
                    "app.group": APP_GROUP,
                    "app.name": APP_NAME,
                    "app.type": APP_TYPE,
                },
            )
        )

    return observations


def failed_callback(options):
    observations = []

    for model, value in latest_failed.items():
        observations.append(
            Observation(
                value,
                attributes={
                    "tenant": TENANT,
                    "llm.model": model,
                    "llm.provider": PROVIDER,
                    "environment": ENVIRONMENT,
                    "app.group": APP_GROUP,
                    "app.name": APP_NAME,
                    "app.type": APP_TYPE,
                },
            )
        )

    return observations


def total_callback(options):
    observations = []

    for model, value in latest_total.items():
        observations.append(
            Observation(
                value,
                attributes={
                    "tenant": TENANT,
                    "llm.model": model,
                    "llm.provider": PROVIDER,
                    "environment": ENVIRONMENT,
                    "app.group": APP_GROUP,
                    "app.name": APP_NAME,
                    "app.type": APP_TYPE,
                },
            )
        )

    return observations


# =========================================================
# OBSERVABLE METRICS
# =========================================================
meter.create_observable_gauge(
    "llm.eval.accuracy",
    callbacks=[accuracy_callback],
    unit="%",
    description="Golden prompt evaluation accuracy percentage",
)

meter.create_observable_gauge(
    "llm.eval.passed",
    callbacks=[passed_callback],
    unit="1",
    description="Golden prompt evaluation passed count",
)

meter.create_observable_gauge(
    "llm.eval.failed",
    callbacks=[failed_callback],
    unit="1",
    description="Golden prompt evaluation failed count",
)

meter.create_observable_gauge(
    "llm.eval.total",
    callbacks=[total_callback],
    unit="1",
    description="Golden prompt evaluation total count",
)


# =========================================================
# COUNTERS
# =========================================================
token_counter = meter.create_counter(
    "llm.total.tokens",
    description="Total tokens used",
)

cost_counter = meter.create_counter(
    "llm.cost.usd",
    description="Estimated LLM cost in USD",
)

latency_counter = meter.create_counter(
    "llm.eval.latency.ms",
    description="Evaluation latency in milliseconds",
)

request_counter = meter.create_counter(
    "llm.requests",
    description="Total LLM requests",
)


# =========================================================
# OLLAMA CLIENT
# =========================================================
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


# =========================================================
# HELPERS
# =========================================================
def estimate_tokens(text: str) -> int:
    if not text:
        return 0

    return max(1, len(text.split()))


def get_usage(response, prompt_text: str, answer_text: str):
    usage = getattr(response, "usage", None)

    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or (
            prompt_tokens + completion_tokens
        )
    else:
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(answer_text)
        total_tokens = prompt_tokens + completion_tokens

    return prompt_tokens, completion_tokens, total_tokens


def calculate_cost(prompt_tokens: int, completion_tokens: int):
    prompt_cost = (
        prompt_tokens / 1_000_000
    ) * INPUT_COST_PER_1M_TOKENS

    completion_cost = (
        completion_tokens / 1_000_000
    ) * OUTPUT_COST_PER_1M_TOKENS

    total_cost = prompt_cost + completion_cost

    return total_cost


def response_matches_expected(
    response_text: str,
    expected_substrings: List[str],
) -> bool:
    response_lower = response_text.lower()

    return all(
        expected.lower() in response_lower
        for expected in expected_substrings
    )


# =========================================================
# MODEL EVALUATION
# =========================================================
def evaluate_model(model: str):
    passed = 0
    failed = 0

    print(f"\n================================================")
    print(f"Evaluating model: {model}")
    print("================================================\n")

    with tracer.start_as_current_span("golden-prompt-evaluation") as parent_span:

        parent_span.set_attribute("llm.model", model)
        parent_span.set_attribute("tenant", TENANT)
        parent_span.set_attribute("app.group", APP_GROUP)
        parent_span.set_attribute("app.name", APP_NAME)
        parent_span.set_attribute("app.type", APP_TYPE)

        for index, item in enumerate(GOLDEN_PROMPTS, start=1):

            prompt = item["prompt"]
            expected_substrings = item["expected_substrings"]

            attrs = {
                "tenant": TENANT,
                "llm.model": model,
                "llm.provider": PROVIDER,
                "environment": ENVIRONMENT,
                "app.group": APP_GROUP,
                "app.name": APP_NAME,
                "app.type": APP_TYPE,
            }

            with tracer.start_as_current_span("golden-prompt-test") as span:

                span.set_attribute("llm.model", model)
                span.set_attribute("app.group", APP_GROUP)
                span.set_attribute("app.name", APP_NAME)
                span.set_attribute("app.type", APP_TYPE)
                span.set_attribute("llm.prompt", prompt)

                start_time = time.perf_counter()

                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        temperature=0,
                        max_tokens=120,
                    )

                    answer = response.choices[0].message.content or ""

                    latency_ms = (
                        time.perf_counter() - start_time
                    ) * 1000

                    prompt_tokens, completion_tokens, total_tokens = get_usage(
                        response=response,
                        prompt_text=prompt,
                        answer_text=answer,
                    )

                    total_cost = calculate_cost(
                        prompt_tokens,
                        completion_tokens,
                    )

                    is_pass = response_matches_expected(
                        answer,
                        expected_substrings,
                    )

                    if is_pass:
                        passed += 1
                        result = "PASS"
                    else:
                        failed += 1
                        result = "FAIL"

                    span.set_attribute("eval.result", result)
                    span.set_attribute("llm.total.tokens", total_tokens)
                    span.set_attribute("llm.cost.usd", total_cost)
                    span.set_attribute("llm.eval.latency.ms", latency_ms)

                    request_counter.add(1, attrs)
                    token_counter.add(total_tokens, attrs)
                    cost_counter.add(total_cost, attrs)
                    latency_counter.add(latency_ms, attrs)

                    print(
                        f"{index:02d}. "
                        f"{result} | "
                        f"{model} | "
                        f"{latency_ms:.2f} ms | "
                        f"tokens={total_tokens}"
                    )

                except Exception as exc:
                    failed += 1

                    span.set_attribute("status", "error")
                    span.set_attribute("error.type", type(exc).__name__)
                    span.set_attribute("error.message", str(exc))

                    print(
                        f"{index:02d}. ERROR | "
                        f"{model} | "
                        f"{type(exc).__name__}: {exc}"
                    )

        total = len(GOLDEN_PROMPTS)

        accuracy = (passed / total) * 100 if total else 0

        latest_accuracy[model] = accuracy
        latest_passed[model] = passed
        latest_failed[model] = failed
        latest_total[model] = total

        parent_span.set_attribute("llm.eval.accuracy", accuracy)

        print(f"\nModel: {model}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total: {total}")


# =========================================================
# MAIN LOOP
# =========================================================
if __name__ == "__main__":

    print("\n================================================")
    print("Multi-Model Golden Prompt Evaluation")
    print("================================================")
    print(f"Service: {SERVICE_NAME}")
    print(f"App Group: {APP_GROUP}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Interval: {EVAL_INTERVAL_SECONDS} seconds")
    print("Press Ctrl+C to stop.")
    print("================================================\n")

    try:
        while True:

            for model in MODELS:
                evaluate_model(model)

            print(
                f"\nSleeping for "
                f"{EVAL_INTERVAL_SECONDS} seconds...\n"
            )

            meter_provider.force_flush(timeout_millis=30000)
            trace_provider.force_flush(timeout_millis=30000)

            time.sleep(EVAL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nStopping evaluator...")

    finally:
        meter_provider.force_flush(timeout_millis=30000)
        trace_provider.force_flush(timeout_millis=30000)

        meter_provider.shutdown()
        trace_provider.shutdown()

        print("Telemetry flushed.")