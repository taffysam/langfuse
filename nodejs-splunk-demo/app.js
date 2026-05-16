const express = require("express");
const { trace } = require("@opentelemetry/api");

const app = express();
const tracer = trace.getTracer("custom-node-tracer");

const port = process.env.PORT || 4000;

app.get("/", (req, res) => {
  res.json({
    message: "Splunk O11y Node.js Demo",
    service: process.env.OTEL_SERVICE_NAME || "nodejs-splunk-demo"
  });
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok"
  });
});

app.get("/slow", async (req, res) => {
  await new Promise(resolve => setTimeout(resolve, 2000));

  res.json({
    message: "Slow endpoint completed"
  });
});

app.get("/database", async (req, res) => {
  const span = tracer.startSpan("database-query");

  try {
    await new Promise(resolve => setTimeout(resolve, 800));

    span.setAttribute("db.system", "postgresql");
    span.setAttribute("db.operation", "SELECT");
    span.setAttribute("db.statement", "SELECT * FROM users WHERE active = true");

    res.json({
      message: "Database query simulated"
    });
  } catch (err) {
    span.recordException(err);

    res.status(500).json({
      error: err.message
    });
  } finally {
    span.end();
  }
});

app.get("/error", (req, res) => {
  throw new Error("Splunk test error");
});

app.use((err, req, res, next) => {
  console.error("Application error:", err.message);

  res.status(500).json({
    error: err.message
  });
});

app.listen(port, () => {
  console.log(`App running on port ${port}`);
});