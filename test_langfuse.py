from langfuse import Langfuse

langfuse = Langfuse(
    public_key="YOUR_PUBLIC_KEY",
    secret_key="YOUR_SECRET_KEY",
    host="http://localhost:3000"
)

trace = langfuse.create_trace(
    name="test-trace",
    user_id="demo-user"
)

langfuse.create_generation(
    trace_id=trace.id,
    name="hello-model",
    model="phi3:mini",
    input="Hello world",
    output="Hi from Langfuse!"
)

langfuse.flush()