from openai import OpenAI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.openai import OpenAIInstrumentor

# 1. Identify the service in every trace
resource = Resource.create({
    "service.name": "ai-obs-demo",
    "service.version": "0.1.0",
    "deployment.environment": "local",
})

# 2. Wire OTel: provider → exporter → Collector on :4318
provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
))
trace.set_tracer_provider(provider)

# 3. Auto-instrument the OpenAI SDK — this is the magic line
OpenAIInstrumentor().instrument()

# 4. Call Ollama via the OpenAI SDK (Ollama speaks the dialect)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
response = client.chat.completions.create(
    model="phi3:mini",
    messages=[
        {"role": "system", "content": "You answer in one sentence."},
        {"role": "user", "content": "What is observability?"},
    ],
)

print(response.choices[0].message.content)

# 5. Force the exporter to flush before the process exits
provider.shutdown()