import time
import threading
from typing import List, Dict

from openai import OpenAI

from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter


# =========================
# CONFIG
# =========================
SERVICE_NAME = "local-ollama-phi3"
TENANT = "demo-tenant"
MODEL = "phi3:mini"
PROVIDER = "ollama"
ENVIRONMENT = "local"

TRACE_ENDPOINT = "http://localhost:4318/v1/traces"
METRIC_ENDPOINT = "http://localhost:4318/v1/metrics"

EVAL_INTERVAL_SECONDS = 300

# Demo internal costs for local Ollama
INPUT_COST_PER_1M_TOKENS = 0.10
OUTPUT_COST_PER_1M_TOKENS = 0.20


# =========================
# GOLDEN PROMPTS FOR DRIFT
# =========================
GOLDEN_PROMPTS: List[Dict[str, object]] = [
    {"prompt": "What is 2 + 2?", "expected_substrings": ["4"]},
    {"prompt": "What is the capital city of France?", "expected_substrings": ["paris"]},
    {"prompt": "What cloud provider has EC2?", "expected_substrings": ["aws", "amazon"]},
    {"prompt": "What does CPU stand for?", "expected_substrings": ["central", "processing", "unit"]},
    {"prompt": "What does HTTP stand for?", "expected_substrings": ["hypertext", "transfer", "protocol"]},
    {"prompt": "What is OpenTelemetry used for?", "expected_substrings": ["telemetry"]},
    {"prompt": "What is Docker used for?", "expected_substrings": ["container"]},
    {"prompt": "What is Redis commonly used for?", "expected_substrings": ["cache"]},
    {"prompt": "What is PostgreSQL?", "expected_substrings": ["database"]},
    {"prompt": "What is latency?", "expected_substrings": ["delay", "time"]},
    {"prompt": "What is a trace in observability?", "expected_substrings": ["request", "span"]},
    {"prompt": "What is a metric in observability?", "expected_substrings": ["measurement", "value"]},
    {"prompt": "What is drift detection for an AI model?", "expected_substrings": ["change", "performance"]},
    {"prompt": "What is a golden test set?", "expected_substrings": ["expected", "baseline"]},
    {"prompt": "What is Kubernetes used for?", "expected_substrings": ["containers"]},
    {"prompt": "What is MinIO compatible with?", "expected_substrings": ["s3"]},
    {"prompt": "What is an API?", "expected_substrings": ["application", "programming", "interface"]},
    {"prompt": "What is Splunk Observability used for?", "expected_substrings": ["monitor"]},
    {"prompt": "What is ClickHouse used for?", "expected_substrings": ["analytics", "database"]},
    {"prompt": "What is SQL used for?", "expected_substrings": ["database", "query"]},
]


# =========================
# OTEL SETUP
# =========================
resource = Resource.create(
    {
        "service.name": SERVICE_NAME,
        "deployment.environment": ENVIRONMENT,
        "llm.provider": PROVIDER,
        "llm.model": MODEL,
        "tenant": TENANT,
    }
)

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


# =========================
# METRICS
# =========================
llm_requests_counter = meter.create_counter("llm.requests", unit="1")

prompt_tokens_counter = meter.create_counter("llm.prompt.tokens", unit="tokens")
completion_tokens_counter = meter.create_counter("llm.completion.tokens", unit="tokens")
total_tokens_counter = meter.create_counter("llm.total.tokens", unit="tokens")

prompt_cost_counter = meter.create_counter("llm.prompt.cost.usd", unit="USD")
completion_cost_counter = meter.create_counter("llm.completion.cost.usd", unit="USD")
total_cost_counter = meter.create_counter("llm.total.cost.usd", unit="USD")
llm_cost_counter = meter.create_counter("llm.cost.usd", unit="USD")

eval_accuracy_counter = meter.create_counter("llm.eval.accuracy", unit="%")
eval_passed_counter = meter.create_counter("llm.eval.passed", unit="1")
eval_failed_counter = meter.create_counter("llm.eval.failed", unit="1")
eval_total_counter = meter.create_counter("llm.eval.total", unit="1")


# =========================
# OLLAMA CLIENT
# =========================
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

chat_history = []
stop_eval = threading.Event()


# =========================
# HELPERS
# =========================
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def get_usage(response, prompt_text: str, answer_text: str):
    usage = getattr(response, "usage", None)

    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
    else:
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(answer_text)
        total_tokens = prompt_tokens + completion_tokens

    return prompt_tokens, completion_tokens, total_tokens


def calculate_cost(prompt_tokens: int, completion_tokens: int):
    prompt_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_1M_TOKENS
    completion_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_1M_TOKENS
    total_cost = prompt_cost + completion_cost

    return prompt_cost, completion_cost, total_cost


def base_metric_attrs(status: str = "success"):
    return {
        "tenant": TENANT,
        "llm.model": MODEL,
        "llm.provider": PROVIDER,
        "environment": ENVIRONMENT,
        "service.name": SERVICE_NAME,
        "status": status,
    }


def response_matches_expected(response_text: str, expected_substrings: List[str]) -> bool:
    response_lower = response_text.lower()
    return all(expected.lower() in response_lower for expected in expected_substrings)


# =========================
# LLM CALL WITH TRACE + COST
# =========================
def ask_phi3(prompt: str, source: str = "interactive") -> str:
    temperature = 0
    max_tokens = 100

    if source == "interactive":
        chat_history.append({"role": "user", "content": prompt})
        messages = chat_history
    else:
        messages = [{"role": "user", "content": prompt}]

    attrs = {
        **base_metric_attrs(),
        "source": source,
    }

    with tracer.start_as_current_span("ollama-chat-request") as span:
        span.set_attribute("tenant", TENANT)
        span.set_attribute("source", source)
        span.set_attribute("llm.provider", PROVIDER)
        span.set_attribute("llm.model", MODEL)
        span.set_attribute("llm.prompt", prompt)
        span.set_attribute("llm.temperature", temperature)
        span.set_attribute("llm.max_tokens", max_tokens)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = response.choices[0].message.content or ""

            if source == "interactive":
                chat_history.append({"role": "assistant", "content": answer})

            prompt_tokens, completion_tokens, total_tokens = get_usage(
                response=response,
                prompt_text=prompt,
                answer_text=answer,
            )

            prompt_cost, completion_cost, total_cost = calculate_cost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

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

            return answer

        except Exception as e:
            error_attrs = {**attrs, "status": "error"}

            span.set_attribute("status", "error")
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))

            llm_requests_counter.add(1, error_attrs)

            raise


# =========================
# DRIFT EVALUATION
# =========================
def run_drift_eval_once() -> None:
    passed = 0
    failed = 0

    eval_attrs = {
        **base_metric_attrs(),
        "eval.name": "golden_prompt_drift_eval",
    }

    with tracer.start_as_current_span("llm-drift-eval") as span:
        span.set_attribute("eval.name", "golden_prompt_drift_eval")
        span.set_attribute("eval.total", len(GOLDEN_PROMPTS))
        span.set_attribute("tenant", TENANT)

        print("\nRunning drift evaluation...")

        for index, item in enumerate(GOLDEN_PROMPTS, start=1):
            prompt = item["prompt"]
            expected = item["expected_substrings"]

            with tracer.start_as_current_span("llm-drift-eval-item") as item_span:
                item_span.set_attribute("eval.item", index)
                item_span.set_attribute("eval.prompt", prompt)
                item_span.set_attribute("eval.expected_substrings", ",".join(expected))

                try:
                    answer = ask_phi3(prompt, source="drift-eval")
                    is_pass = response_matches_expected(answer, expected)

                    if is_pass:
                        passed += 1
                        item_span.set_attribute("eval.result", "pass")
                    else:
                        failed += 1
                        item_span.set_attribute("eval.result", "fail")

                except Exception as e:
                    failed += 1
                    item_span.set_attribute("eval.result", "error")
                    item_span.set_attribute("error.type", type(e).__name__)
                    item_span.set_attribute("error.message", str(e))

        total = len(GOLDEN_PROMPTS)
        accuracy = (passed / total) * 100 if total else 0.0

        span.set_attribute("llm.eval.accuracy", accuracy)
        span.set_attribute("llm.eval.passed", passed)
        span.set_attribute("llm.eval.failed", failed)
        span.set_attribute("llm.eval.total", total)

        eval_accuracy_counter.add(accuracy, eval_attrs)
        eval_passed_counter.add(passed, eval_attrs)
        eval_failed_counter.add(failed, eval_attrs)
        eval_total_counter.add(total, eval_attrs)

        print(f"Drift eval accuracy: {accuracy:.2f}% | Passed: {passed} | Failed: {failed}")


def drift_eval_loop():
    while not stop_eval.is_set():
        try:
            run_drift_eval_once()
            meter_provider.force_flush(timeout_millis=30000)
            trace_provider.force_flush(timeout_millis=30000)
        except Exception as e:
            print(f"Drift eval error: {type(e).__name__}: {e}")

        stop_eval.wait(EVAL_INTERVAL_SECONDS)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("\nLocal AI Chat - Full Splunk Observability")
    print("Includes: traces + token cost + drift evaluation")
    print("Type your question and press Enter.")
    print("Type 'quit', 'exit', or 'q' when you are done.\n")

    eval_thread = threading.Thread(target=drift_eval_loop, daemon=True)
    eval_thread.start()

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q", "done"]:
                print("\nDone. Ending chat.")
                break

            if not user_input:
                print("Please type a question, or type 'quit' to exit.\n")
                continue

            result = ask_phi3(user_input, source="interactive")
            print(f"AI: {result}\n")

    finally:
        print("Stopping drift evaluator...")
        stop_eval.set()
        eval_thread.join(timeout=10)

        print("Flushing telemetry...")
        trace_provider.force_flush(timeout_millis=30000)
        meter_provider.force_flush(timeout_millis=30000)

        trace_provider.shutdown()
        meter_provider.shutdown()

        print("Telemetry flushed. Exiting.")