import hashlib
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

from openai import OpenAI

from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter


# =========================================================
# APP CONFIG
# =========================================================
SERVICE_NAME = "local-ollama-multi-model-governed"
APP_NAME = "test-ollama-full-observability-governed"
APP_TYPE = "full-observability-governed"
APP_GROUP = "local-ai-observability"

MODELS = ["phi3:mini", "mistral"]
PROVIDER = "ollama"
TENANT = "demo-tenant"
ENVIRONMENT = "local"

TRACE_ENDPOINT = "http://localhost:4318/v1/traces"
METRIC_ENDPOINT = "http://localhost:4318/v1/metrics"

INPUT_COST_PER_1M_TOKENS = 0.10
OUTPUT_COST_PER_1M_TOKENS = 0.20

# Drift detection now runs quietly in the background every 10 minutes.
# It only starts after consent is granted and after the first user question is submitted.
EVAL_INTERVAL_SECONDS = 600

RAW_USER_ID = "demo-user"
HASH_SALT = "local-demo-salt-change-in-production"
RETENTION_CLASS = "standard"
RETENTION_DAYS = 30
DATA_CLASSIFICATION = "internal-demo"
PROCESSING_PURPOSE = "ai_observability_demo"
LEGAL_BASIS = "consent"

BLOCK_ON_PII_DETECTED = False
SEND_SANITIZED_TEXT_TO_TRACE = True

PRIVACY_REGION = "SA"
CONSENT_AI_PROCESSING = False
CONSENT_TELEMETRY = False

REGION_LAWS = {
    "SA": ["POPIA", "PAIA where applicable", "FICA/NCA where financial data is involved"],
    "EU": ["GDPR", "ePrivacy Directive where applicable", "EU AI Act where applicable"],
    "UK": ["UK GDPR", "Data Protection Act 2018", "Data (Use and Access) Act 2025 where applicable"],
}

GOLDEN_PROMPTS: List[Dict[str, object]] = [
    {"prompt": "What is 2 + 2?", "expected_substrings": ["4"]},
    {"prompt": "What is the capital city of France?", "expected_substrings": ["paris"]},
    {"prompt": "What cloud provider has EC2?", "expected_substrings": ["aws"]},
    {"prompt": "What is Docker used for?", "expected_substrings": ["container"]},
    {"prompt": "What is Redis commonly used for?", "expected_substrings": ["cache"]},
]


# =========================================================
# RUNTIME GLOBALS INITIALIZED AFTER CONSENT
# =========================================================
trace_provider: Optional[TracerProvider] = None
meter_provider: Optional[MeterProvider] = None
tracer = None
meter = None

prompt_tokens_counter = None
completion_tokens_counter = None
total_tokens_counter = None
prompt_cost_counter = None
completion_cost_counter = None
total_cost_counter = None
llm_cost_counter = None
llm_requests_counter = None
eval_accuracy_counter = None
eval_passed_counter = None
eval_failed_counter = None
eval_total_counter = None
privacy_pii_detected_counter = None
privacy_prompt_blocked_counter = None
privacy_consent_denied_counter = None
governance_requests_counter = None

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
chat_history_by_model: Dict[str, List[Dict[str, str]]] = {model: [] for model in MODELS}
stop_eval = threading.Event()
drift_thread_started = False


# =========================================================
# USER GOVERNANCE FLOW
# =========================================================
def select_region() -> str:
    print("\nBefore using this AI system, please select your region.")
    print("As a firm, we respect data protection laws in your chosen region.")
    print("\nAvailable regions:")
    print("1. South Africa (SA)")
    print("2. European Union (EU)")
    print("3. United Kingdom (UK)")

    while True:
        choice = input("\nSelect region [1/2/3]: ").strip().lower()
        if choice in ["1", "sa", "south africa"]:
            return "SA"
        if choice in ["2", "eu", "european union"]:
            return "EU"
        if choice in ["3", "uk", "united kingdom"]:
            return "UK"
        print("Invalid selection. Please choose 1, 2, or 3.")


def request_consent(region: str) -> bool:
    laws = REGION_LAWS[region]
    print(f"\nYou selected: {region}")
    print("The following data protection laws/frameworks will be applied:")
    for law in laws:
        print(f"- {law}")

    print("\nTo use this model, your prompts, responses, telemetry, token usage,")
    print("cost metrics, privacy metadata, and governance events will be processed")
    print("according to the laws/frameworks of your selected region.")
    print("Possible personal information will be sanitized/redacted before telemetry is sent.")

    answer = input("\nDo you consent to continue? Type yes or no: ").strip().lower()
    return answer in ["yes", "y"]


# =========================================================
# OTEL INITIALIZATION AFTER CONSENT
# =========================================================
def initialize_otel() -> None:
    global trace_provider, meter_provider, tracer, meter
    global prompt_tokens_counter, completion_tokens_counter, total_tokens_counter
    global prompt_cost_counter, completion_cost_counter, total_cost_counter, llm_cost_counter
    global llm_requests_counter, eval_accuracy_counter, eval_passed_counter, eval_failed_counter, eval_total_counter
    global privacy_pii_detected_counter, privacy_prompt_blocked_counter, privacy_consent_denied_counter, governance_requests_counter

    resource = Resource.create(
        {
            "service.name": SERVICE_NAME,
            "deployment.environment": ENVIRONMENT,
            "tenant": TENANT,
            "llm.provider": PROVIDER,
            "app.group": APP_GROUP,
            "app.name": APP_NAME,
            "app.type": APP_TYPE,
            "privacy.region": PRIVACY_REGION,
            "privacy.retention_class": RETENTION_CLASS,
            "privacy.retention_days": RETENTION_DAYS,
            "data.classification": DATA_CLASSIFICATION,
            "processing.purpose": PROCESSING_PURPOSE,
            "processing.legal_basis": LEGAL_BASIS,
        }
    )

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
    metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=5000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    meter = metrics.get_meter(__name__)

    prompt_tokens_counter = meter.create_counter("llm.prompt.tokens", unit="tokens")
    completion_tokens_counter = meter.create_counter("llm.completion.tokens", unit="tokens")
    total_tokens_counter = meter.create_counter("llm.total.tokens", unit="tokens")
    prompt_cost_counter = meter.create_counter("llm.prompt.cost.usd", unit="USD")
    completion_cost_counter = meter.create_counter("llm.completion.cost.usd", unit="USD")
    total_cost_counter = meter.create_counter("llm.total.cost.usd", unit="USD")
    llm_cost_counter = meter.create_counter("llm.cost.usd", unit="USD")
    llm_requests_counter = meter.create_counter("llm.requests", unit="1")

    eval_accuracy_counter = meter.create_counter("llm.eval.accuracy", unit="%")
    eval_passed_counter = meter.create_counter("llm.eval.passed", unit="1")
    eval_failed_counter = meter.create_counter("llm.eval.failed", unit="1")
    eval_total_counter = meter.create_counter("llm.eval.total", unit="1")

    privacy_pii_detected_counter = meter.create_counter("privacy.pii.detected", unit="1")
    privacy_prompt_blocked_counter = meter.create_counter("privacy.prompt.blocked", unit="1")
    privacy_consent_denied_counter = meter.create_counter("privacy.consent.denied", unit="1")
    governance_requests_counter = meter.create_counter("governance.requests", unit="1")


# =========================================================
# TERMINAL ACTIVITY INDICATOR
# =========================================================
def show_activity(message: str = "Thinking"):
    stop_event = threading.Event()

    def spinner():
        symbols = ["|", "/", "-", "\\"]
        index = 0
        while not stop_event.is_set():
            sys.stdout.write(f"\r{message} {symbols[index % len(symbols)]}")
            sys.stdout.flush()
            index += 1
            time.sleep(0.15)
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    thread = threading.Thread(target=spinner, daemon=True)
    thread.start()
    return stop_event


# =========================================================
# GOVERNANCE HELPERS
# =========================================================
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "south_african_id": re.compile(r"\b\d{13}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "cvv": re.compile(r"(?i)\b(cvv|cvc|security code)[:= ]+\d{3,4}\b"),
    "phone": re.compile(r"\b(?:\+27|0)[0-9][0-9\s-]{7,12}\b"),
    "vehicle_vin": re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"),
}


def stable_hash(value: str) -> str:
    raw = f"{HASH_SALT}:{value}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sanitize_text(text: str) -> Tuple[str, List[str]]:
    if not text:
        return "", []

    sanitized = text
    detected = []
    replacements = {
        "email": "[EMAIL_REDACTED]",
        "south_african_id": "[SA_ID_REDACTED]",
        "credit_card": "[CARD_REDACTED]",
        "cvv": "[CVV_REDACTED]",
        "phone": "[PHONE_REDACTED]",
        "vehicle_vin": "[VIN_REDACTED]",
    }

    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(sanitized):
            detected.append(pii_type)
            sanitized = pattern.sub(replacements[pii_type], sanitized)

    return sanitized, detected


def validate_consent() -> bool:
    return bool(CONSENT_AI_PROCESSING and CONSENT_TELEMETRY)


def build_governance_attrs(model: str, status: str = "success", source: str = "interactive", pii_detected: bool = False, pii_types: Optional[List[str]] = None):
    pii_types = pii_types or []
    return {
        "tenant": TENANT,
        "llm.model": model,
        "llm.provider": PROVIDER,
        "environment": ENVIRONMENT,
        "service.name": SERVICE_NAME,
        "app.group": APP_GROUP,
        "app.name": APP_NAME,
        "app.type": APP_TYPE,
        "status": status,
        "source": source,
        "privacy.region": PRIVACY_REGION,
        "privacy.consent.ai": str(CONSENT_AI_PROCESSING).lower(),
        "privacy.consent.telemetry": str(CONSENT_TELEMETRY).lower(),
        "privacy.retention_class": RETENTION_CLASS,
        "privacy.retention_days": str(RETENTION_DAYS),
        "privacy.pii_detected": str(pii_detected).lower(),
        "privacy.pii_types": ",".join(pii_types) if pii_types else "none",
        "data.classification": DATA_CLASSIFICATION,
        "processing.purpose": PROCESSING_PURPOSE,
        "processing.legal_basis": LEGAL_BASIS,
    }


def attach_governance_span_attrs(span, model: str, source: str, prompt: str, pii_detected: bool, pii_types: List[str], session_id: str):
    sanitized_prompt, _ = sanitize_text(prompt)
    span.set_attribute("tenant", TENANT)
    span.set_attribute("app.group", APP_GROUP)
    span.set_attribute("app.name", APP_NAME)
    span.set_attribute("app.type", APP_TYPE)
    span.set_attribute("source", source)
    span.set_attribute("llm.provider", PROVIDER)
    span.set_attribute("llm.model", model)
    span.set_attribute("privacy.region", PRIVACY_REGION)
    span.set_attribute("privacy.consent.ai", CONSENT_AI_PROCESSING)
    span.set_attribute("privacy.consent.telemetry", CONSENT_TELEMETRY)
    span.set_attribute("privacy.retention_class", RETENTION_CLASS)
    span.set_attribute("privacy.retention_days", RETENTION_DAYS)
    span.set_attribute("privacy.pii_detected", pii_detected)
    span.set_attribute("privacy.pii_types", ",".join(pii_types) if pii_types else "none")
    span.set_attribute("data.classification", DATA_CLASSIFICATION)
    span.set_attribute("processing.purpose", PROCESSING_PURPOSE)
    span.set_attribute("processing.legal_basis", LEGAL_BASIS)
    span.set_attribute("user.hash", stable_hash(RAW_USER_ID))
    span.set_attribute("session.id", session_id)
    span.set_attribute("audit.event_time_utc", datetime.now(timezone.utc).isoformat())
    span.set_attribute("llm.prompt.hash", stable_hash(prompt))
    if SEND_SANITIZED_TEXT_TO_TRACE:
        span.set_attribute("llm.prompt.sanitized", sanitized_prompt[:4000])


# =========================================================
# GENERAL HELPERS
# =========================================================
def estimate_tokens(text: str) -> int:
    return max(1, len(text.split())) if text else 0


def get_usage(response, prompt_text: str = "", answer_text: str = ""):
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
    return prompt_cost, completion_cost, prompt_cost + completion_cost


def response_matches_expected(response_text: str, expected_substrings: List[str]) -> bool:
    response_lower = response_text.lower()
    return all(expected.lower() in response_lower for expected in expected_substrings)


# =========================================================
# LLM CALL WITH TRACE + COST + GOVERNANCE
# =========================================================
def ask_model(prompt: str, model: str, source: str = "interactive", show_spinner: bool = True) -> str:
    if not prompt.strip():
        return "Please type a question first."

    temperature = 0
    max_tokens = 100
    session_id = str(uuid.uuid4())

    sanitized_prompt, prompt_pii_types = sanitize_text(prompt)
    pii_detected = bool(prompt_pii_types)

    attrs = build_governance_attrs(model=model, source=source, pii_detected=pii_detected, pii_types=prompt_pii_types)

    with tracer.start_as_current_span("ollama-chat-request-governed") as span:
        attach_governance_span_attrs(span, model, source, prompt, pii_detected, prompt_pii_types, session_id)
        span.set_attribute("llm.temperature", temperature)
        span.set_attribute("llm.max_tokens", max_tokens)
        governance_requests_counter.add(1, attrs)

        if not validate_consent():
            deny_attrs = build_governance_attrs(model=model, status="blocked", source=source, pii_detected=pii_detected, pii_types=prompt_pii_types)
            span.set_attribute("status", "blocked")
            span.set_attribute("governance.block_reason", "missing_consent")
            privacy_consent_denied_counter.add(1, deny_attrs)
            llm_requests_counter.add(1, deny_attrs)
            return "Request blocked: required AI processing or telemetry consent is missing."

        if pii_detected:
            privacy_pii_detected_counter.add(1, attrs)
            span.set_attribute("governance.pii.action", "sanitized")
            if BLOCK_ON_PII_DETECTED:
                blocked_attrs = build_governance_attrs(model=model, status="blocked", source=source, pii_detected=True, pii_types=prompt_pii_types)
                span.set_attribute("status", "blocked")
                span.set_attribute("governance.block_reason", "pii_detected")
                privacy_prompt_blocked_counter.add(1, blocked_attrs)
                llm_requests_counter.add(1, blocked_attrs)
                return "Request blocked: possible personal information was detected. Please remove personal data and try again."

        if source == "interactive":
            chat_history_by_model[model].append({"role": "user", "content": sanitized_prompt})
            messages = chat_history_by_model[model]
        else:
            messages = [{"role": "user", "content": sanitized_prompt}]

        activity = None
        try:
            if show_spinner and source == "interactive":
                activity = show_activity(f"{model} is thinking")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if activity:
                activity.set()
                activity = None

            answer = response.choices[0].message.content or ""
            sanitized_answer, answer_pii_types = sanitize_text(answer)
            all_pii_types = sorted(set(prompt_pii_types + answer_pii_types))

            if answer_pii_types:
                privacy_pii_detected_counter.add(1, build_governance_attrs(model=model, source=source, pii_detected=True, pii_types=answer_pii_types))

            if source == "interactive":
                chat_history_by_model[model].append({"role": "assistant", "content": sanitized_answer})

            prompt_tokens, completion_tokens, total_tokens = get_usage(response, sanitized_prompt, sanitized_answer)
            prompt_cost, completion_cost, total_cost = calculate_cost(prompt_tokens, completion_tokens)

            span.set_attribute("llm.response.hash", stable_hash(answer))
            if SEND_SANITIZED_TEXT_TO_TRACE:
                span.set_attribute("llm.response.sanitized", sanitized_answer[:4000])
            span.set_attribute("llm.prompt.tokens", prompt_tokens)
            span.set_attribute("llm.completion.tokens", completion_tokens)
            span.set_attribute("llm.total.tokens", total_tokens)
            span.set_attribute("llm.prompt.cost.usd", prompt_cost)
            span.set_attribute("llm.completion.cost.usd", completion_cost)
            span.set_attribute("llm.total.cost.usd", total_cost)
            span.set_attribute("llm.cost.usd", total_cost)
            span.set_attribute("privacy.response_pii_detected", bool(answer_pii_types))
            span.set_attribute("privacy.all_pii_types", ",".join(all_pii_types) if all_pii_types else "none")
            span.set_attribute("status", "success")

            success_attrs = build_governance_attrs(model=model, status="success", source=source, pii_detected=bool(all_pii_types), pii_types=all_pii_types)
            llm_requests_counter.add(1, success_attrs)
            prompt_tokens_counter.add(prompt_tokens, success_attrs)
            completion_tokens_counter.add(completion_tokens, success_attrs)
            total_tokens_counter.add(total_tokens, success_attrs)
            prompt_cost_counter.add(prompt_cost, success_attrs)
            completion_cost_counter.add(completion_cost, success_attrs)
            total_cost_counter.add(total_cost, success_attrs)
            llm_cost_counter.add(total_cost, success_attrs)

            if source == "interactive":
                print("\nTelemetry emitted:")
                print(f"  region: {PRIVACY_REGION}")
                print(f"  model: {model}")
                print(f"  pii_detected: {bool(all_pii_types)}")
                print(f"  total_tokens: {total_tokens}")
                print(f"  total_cost_usd: {total_cost:.8f}\n")

            return sanitized_answer

        except Exception as e:
            if activity:
                activity.set()
            error_attrs = build_governance_attrs(model=model, status="error", source=source, pii_detected=pii_detected, pii_types=prompt_pii_types)
            span.set_attribute("status", "error")
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            llm_requests_counter.add(1, error_attrs)
            raise


# =========================================================
# QUIET DRIFT EVALUATION - BACKGROUND EVERY 10 MINUTES
# =========================================================
def run_drift_eval_once_quiet() -> None:
    for model in MODELS:
        passed = 0
        failed = 0
        total = len(GOLDEN_PROMPTS)
        eval_attrs = {
            "tenant": TENANT,
            "llm.model": model,
            "llm.provider": PROVIDER,
            "environment": ENVIRONMENT,
            "service.name": SERVICE_NAME,
            "eval.name": "golden_prompt_drift_eval",
            "app.group": APP_GROUP,
            "app.name": APP_NAME,
            "app.type": APP_TYPE,
            "privacy.region": PRIVACY_REGION,
            "privacy.retention_days": str(RETENTION_DAYS),
        }

        with tracer.start_as_current_span("llm-drift-eval-governed") as span:
            span.set_attribute("tenant", TENANT)
            span.set_attribute("llm.model", model)
            span.set_attribute("eval.name", "golden_prompt_drift_eval")
            span.set_attribute("privacy.region", PRIVACY_REGION)
            span.set_attribute("eval.total", total)

            for item in GOLDEN_PROMPTS:
                prompt = item["prompt"]
                expected = item["expected_substrings"]
                try:
                    answer = ask_model(prompt=prompt, model=model, source="drift-eval", show_spinner=False)
                    if response_matches_expected(answer, expected):
                        passed += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1

            accuracy = (passed / total) * 100 if total else 0.0
            span.set_attribute("llm.eval.accuracy", accuracy)
            span.set_attribute("llm.eval.passed", passed)
            span.set_attribute("llm.eval.failed", failed)
            span.set_attribute("llm.eval.total", total)

            # Send only summary metrics every 10 minutes.
            eval_accuracy_counter.add(accuracy, eval_attrs)
            eval_passed_counter.add(passed, eval_attrs)
            eval_failed_counter.add(failed, eval_attrs)
            eval_total_counter.add(total, eval_attrs)


def drift_eval_loop() -> None:
    while not stop_eval.is_set():
        stop_eval.wait(EVAL_INTERVAL_SECONDS)
        if stop_eval.is_set():
            break
        try:
            run_drift_eval_once_quiet()
            trace_provider.force_flush(timeout_millis=30000)
            meter_provider.force_flush(timeout_millis=30000)
        except Exception as e:
            print(f"\nBackground drift evaluation error: {type(e).__name__}: {e}\n")


def start_drift_thread_once() -> None:
    global drift_thread_started
    if drift_thread_started:
        return
    drift_thread_started = True
    eval_thread = threading.Thread(target=drift_eval_loop, daemon=True)
    eval_thread.start()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    try:
        PRIVACY_REGION = select_region()
        if not request_consent(PRIVACY_REGION):
            print("\nThank you. We will not proceed.")
            sys.exit(0)

        CONSENT_AI_PROCESSING = True
        CONSENT_TELEMETRY = True
        initialize_otel()

        print("\nThank you. Consent granted.")
        print(f"The laws/frameworks for {PRIVACY_REGION} will be applied.")
        print("You may now use the model. Type 'quit', 'exit', or 'q' to end.\n")

        first_question_received = False

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q", "done"]:
                print("\nDone. Ending chat.")
                break

            if not user_input:
                print("Please type a question, or type 'quit' to exit.\n")
                continue

            if not first_question_received:
                first_question_received = True
                start_drift_thread_once()

            with tracer.start_as_current_span("multi-model-interactive-chat-governed") as span:
                sanitized_input, pii_types = sanitize_text(user_input)
                span.set_attribute("tenant", TENANT)
                span.set_attribute("app.group", APP_GROUP)
                span.set_attribute("app.name", APP_NAME)
                span.set_attribute("app.type", APP_TYPE)
                span.set_attribute("source", "interactive")
                span.set_attribute("models", ",".join(MODELS))
                span.set_attribute("privacy.region", PRIVACY_REGION)
                span.set_attribute("privacy.consent.ai", CONSENT_AI_PROCESSING)
                span.set_attribute("privacy.consent.telemetry", CONSENT_TELEMETRY)
                span.set_attribute("privacy.pii_detected", bool(pii_types))
                span.set_attribute("privacy.pii_types", ",".join(pii_types) if pii_types else "none")
                span.set_attribute("llm.prompt.hash", stable_hash(user_input))
                if SEND_SANITIZED_TEXT_TO_TRACE:
                    span.set_attribute("llm.prompt.sanitized", sanitized_input[:4000])

                for model in MODELS:
                    print("\n==============================")
                    print(f"MODEL: {model}")
                    print("==============================")
                    result = ask_model(prompt=user_input, model=model, source="interactive", show_spinner=True)
                    print(f"AI ({model}): {result}\n")

    finally:
        stop_eval.set()
        if trace_provider and meter_provider:
            print("Flushing telemetry...")
            trace_provider.force_flush(timeout_millis=30000)
            meter_provider.force_flush(timeout_millis=30000)
            trace_provider.shutdown()
            meter_provider.shutdown()
            print("Telemetry flushed. Exiting.")
