import time
from typing import Dict, List

from openai import OpenAI

from opentelemetry import metrics
from opentelemetry.metrics import Observation
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter


# =========================
# CONFIG
# =========================
SERVICE_NAME = "local-ollama-phi3"
MODEL = "phi3:mini"
PROVIDER = "ollama"
ENVIRONMENT = "local"

OTEL_METRIC_ENDPOINT = "http://localhost:4318/v1/metrics"

EVAL_INTERVAL_SECONDS = 300  # 5 minutes
ACCURACY_THRESHOLD = 80.0


# =========================
# GOLDEN PROMPTS
# =========================
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
        "prompt": "What is drift detection for an AI model?",
        "expected_substrings": ["change", "performance"],
    },
    {
        "prompt": "What is a golden test set?",
        "expected_substrings": ["expected", "baseline"],
    },
]


# =========================
# OLLAMA CLIENT
# =========================
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


# =========================
# METRIC STATE
# =========================
last_eval = {
    "accuracy": 0.0,
    "passed": 0,
    "failed": 0,
    "total": len(GOLDEN_PROMPTS),
}


metric_attrs = {
    "service.name": SERVICE_NAME,
    "llm.model": MODEL,
    "llm.provider": PROVIDER,
    "environment": ENVIRONMENT,
    "eval.name": "golden_prompt_drift_eval",
}


# =========================
# OTEL METRICS SETUP
# =========================
resource = Resource.create({"service.name": SERVICE_NAME})

metric_exporter = OTLPMetricExporter(
    endpoint=OTEL_METRIC_ENDPOINT,
    timeout=30,
)

metric_reader = PeriodicExportingMetricReader(
    metric_exporter,
    export_interval_millis=60000,
)

meter_provider = MeterProvider(
    resource=resource,
    metric_readers=[metric_reader],
)

metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(__name__)


def accuracy_callback(options):
    return [
        Observation(
            last_eval["accuracy"],
            attributes=metric_attrs,
        )
    ]


def passed_callback(options):
    return [
        Observation(
            last_eval["passed"],
            attributes=metric_attrs,
        )
    ]


def failed_callback(options):
    return [
        Observation(
            last_eval["failed"],
            attributes=metric_attrs,
        )
    ]


def total_callback(options):
    return [
        Observation(
            last_eval["total"],
            attributes=metric_attrs,
        )
    ]


meter.create_observable_gauge(
    "llm.eval.accuracy",
    callbacks=[accuracy_callback],
    unit="%",
    description="Accuracy percentage for golden prompt drift evaluation",
)

meter.create_observable_gauge(
    "llm.eval.passed",
    callbacks=[passed_callback],
    unit="1",
    description="Number of golden prompts that passed",
)

meter.create_observable_gauge(
    "llm.eval.failed",
    callbacks=[failed_callback],
    unit="1",
    description="Number of golden prompts that failed",
)

meter.create_observable_gauge(
    "llm.eval.total",
    callbacks=[total_callback],
    unit="1",
    description="Total number of golden prompts evaluated",
)


# =========================
# EVALUATION LOGIC
# =========================
def ask_model(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=120,
    )

    return response.choices[0].message.content or ""


def response_matches_expected(response_text: str, expected_substrings: List[str]) -> bool:
    response_lower = response_text.lower()

    return all(
        expected.lower() in response_lower
        for expected in expected_substrings
    )


def run_eval_once() -> None:
    passed = 0
    failed = 0

    print("\nRunning golden prompt evaluation...")

    for index, item in enumerate(GOLDEN_PROMPTS, start=1):
        prompt = item["prompt"]
        expected_substrings = item["expected_substrings"]

        try:
            answer = ask_model(prompt)
            is_pass = response_matches_expected(answer, expected_substrings)

            if is_pass:
                passed += 1
                result = "PASS"
            else:
                failed += 1
                result = "FAIL"

            print(f"{index:02d}. {result} | Prompt: {prompt}")

        except Exception as exc:
            failed += 1
            print(f"{index:02d}. ERROR | Prompt: {prompt} | {type(exc).__name__}: {exc}")

    total = len(GOLDEN_PROMPTS)
    accuracy = (passed / total) * 100 if total else 0.0

    last_eval["accuracy"] = accuracy
    last_eval["passed"] = passed
    last_eval["failed"] = failed
    last_eval["total"] = total

    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {total}")

    meter_provider.force_flush(timeout_millis=30000)


if __name__ == "__main__":
    print("\nLLM Drift Evaluation Runner")
    print(f"Model: {MODEL}")
    print(f"Interval: {EVAL_INTERVAL_SECONDS} seconds")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            run_eval_once()
            print(f"\nSleeping for {EVAL_INTERVAL_SECONDS} seconds...\n")
            time.sleep(EVAL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nStopping drift evaluation runner...")

    finally:
        meter_provider.force_flush(timeout_millis=30000)
        meter_provider.shutdown()