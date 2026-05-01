import os
from openai import OpenAI

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

resource = Resource.create({
    "service.name": "local-ollama-test",
    "deployment.environment": "local",
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces"),
    schedule_delay_millis=500,
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("ollama-test")

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=30,
)

with tracer.start_as_current_span("ollama_phi3_test") as span:
    span.set_attribute("llm.system", "ollama")
    span.set_attribute("llm.model", "phi3:mini")

    response = client.chat.completions.create(
        model="phi3:mini",
        messages=[
            {"role": "user", "content": "Reply with exactly one word: hello"}
        ],
        max_tokens=10,
        temperature=0,
    )

    answer = response.choices[0].message.content
    span.set_attribute("llm.response", answer)
    print("Ollama response:", answer)

provider.force_flush()
provider.shutdown()