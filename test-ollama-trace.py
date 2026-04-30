

import os
from openai import OpenAI

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"

provider = TracerProvider()
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("ollama-langfuse-splunk-test")

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

with tracer.start_as_current_span("ollama_phi3_test") as span:
    span.set_attribute("llm.system", "ollama")
    span.set_attribute("llm.model", "phi3:mini")
    span.set_attribute("service.name", "local-ollama-test")

    response = client.chat.completions.create(
        model="phi3:mini",
        messages=[
            {"role": "user", "content": "Explain why tracing is useful for local LLM apps."}
        ],
    )

    answer = response.choices[0].message.content
    span.set_attribute("llm.response", answer)

    print(answer)