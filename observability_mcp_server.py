import os
import requests
from mcp.server.fastmcp import FastMCP

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


mcp = FastMCP("AI Observability MCP Server")

LANGFUSE_URL = os.getenv("LANGFUSE_URL", "http://localhost:3000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces")


resource = Resource.create({
    "service.name": "mcp-ai-observability-server",
    "deployment.environment": "local"
})

provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("mcp-ai-observability")


def check_url(name: str, url: str):
    try:
        response = requests.get(url, timeout=5)
        return {
            "name": name,
            "url": url,
            "status_code": response.status_code,
            "healthy": response.status_code < 500
        }
    except Exception as e:
        return {
            "name": name,
            "url": url,
            "healthy": False,
            "error": str(e)
        }


@mcp.tool()
def project_endpoints() -> dict:
    """Return the main local AI observability project endpoints."""
    return {
        "langfuse": LANGFUSE_URL,
        "ollama": OLLAMA_URL,
        "otel_collector_http": "http://localhost:4318",
        "otel_collector_grpc": "localhost:4317",
        "otel_traces_endpoint": OTEL_ENDPOINT,
        "splunk_observability_realm_hint": "eu0"
    }


@mcp.tool()
def check_observability_stack() -> dict:
    """Check whether Langfuse, Ollama, and OTel Collector endpoints are reachable."""
    return {
        "langfuse": check_url("Langfuse", LANGFUSE_URL),
        "ollama": check_url("Ollama", OLLAMA_URL),
        "otel_collector": check_url("OTel Collector", "http://localhost:4318")
    }


@mcp.tool()
def list_ollama_models() -> dict:
    """List locally available Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def send_test_trace(span_name: str = "mcp-test-span") -> dict:
    """Send a test trace to the local OpenTelemetry Collector."""
    try:
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("source", "mcp-server")
            span.set_attribute("project", "local-llm-observability")
            span.set_attribute("llm.provider", "ollama")
            span.set_attribute("llm.model", "phi3:mini")

        return {
            "sent": True,
            "span_name": span_name,
            "otel_endpoint": OTEL_ENDPOINT
        }
    except Exception as e:
        return {
            "sent": False,
            "error": str(e)
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")