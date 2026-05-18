from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="phi3:mini",
    base_url="http://localhost:11434"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI observability assistant."),
    ("human", "{question}")
])

chain = prompt | llm

response = chain.invoke({
    "question": "Explain what my OpenTelemetry Collector does."
})

print(response.content)