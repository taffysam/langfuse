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
def get_langfuse_health() -> dict:
    """Check Langfuse availability and basic health."""

    try:
        response = requests.get(
            f"{LANGFUSE_URL}/api/public/health",
            timeout=10
        )

        return {
            "healthy": response.status_code == 200,
            "status_code": response.status_code,
            "url": LANGFUSE_URL
        }

    except requests.exceptions.ConnectionError as e:
        return {
            "healthy": False,
            "error_type": "connection_error",
            "error": str(e)
        }

    except requests.exceptions.Timeout as e:
        return {
            "healthy": False,
            "error_type": "timeout",
            "error": str(e)
        }

    except Exception as e:
        return {
            "healthy": False,
            "error_type": "unknown",
            "error": str(e)
        }



import subprocess


@mcp.tool()
def check_docker_containers() -> dict:
    """Check Docker container status."""

    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}|{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=15
        )

        if result.returncode != 0:
            return {
                "success": False,
                "stderr": result.stderr
            }

        containers = []

        for line in result.stdout.strip().splitlines():

            parts = line.split("|")

            if len(parts) == 2:
                containers.append({
                    "name": parts[0],
                    "status": parts[1]
                })

        return {
            "success": True,
            "count": len(containers),
            "containers": containers
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Docker command timed out"
        }

    except FileNotFoundError:
        return {
            "success": False,
            "error": "Docker executable not found"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
        
        

import time


@mcp.tool()
def benchmark_model(
    model_name: str,
    prompt: str
) -> dict:
    """Benchmark local Ollama model latency."""

    try:

        start = time.time()

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        end = time.time()

        response.raise_for_status()

        data = response.json()

        latency = round(end - start, 2)

        return {
            "success": True,
            "model": model_name,
            "latency_seconds": latency,
            "response": data.get("response"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count")
        }

    except requests.exceptions.ConnectionError as e:
        return {
            "success": False,
            "error_type": "connection_error",
            "error": str(e)
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error_type": "timeout",
            "error": "Model inference timed out"
        }

    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "error_type": "http_error",
            "status_code": response.status_code,
            "error": str(e)
        }

    except Exception as e:
        return {
            "success": False,
            "error_type": "unknown",
            "error": str(e)
        }
        
 
@mcp.tool()
def get_splunk_otel_health() -> dict:
    """Check local OpenTelemetry Collector health."""

    try:

        response = requests.get(
            "http://localhost:13133",
            timeout=5
        )

        return {
            "healthy": response.status_code == 200,
            "status_code": response.status_code
        }

    except Exception as e:
        return {
            "healthy": False,
            "error": str(e)
        }
        




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