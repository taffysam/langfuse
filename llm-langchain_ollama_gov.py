from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import sys
import time
import threading


# =========================================================
# REGION / DATA PROTECTION GOVERNANCE
# =========================================================
REGION_LAWS = {
    "SA": {
        "name": "South Africa",
        "laws": "POPIA, PAIA, and where relevant National Credit Act / FICA",
        "default_retention_days": 30,
        "legal_basis": "consent",
    },
    "EU": {
        "name": "European Union",
        "laws": "GDPR, ePrivacy Directive, NIS2, and EU AI Act considerations",
        "default_retention_days": 30,
        "legal_basis": "consent_or_legitimate_interest",
    },
    "UK": {
        "name": "United Kingdom",
        "laws": "UK GDPR, Data Protection Act 2018, and Data (Use and Access) Act 2025 considerations",
        "default_retention_days": 30,
        "legal_basis": "consent_or_legitimate_interest",
    },
}


def choose_privacy_region() -> str:
    print("\n================================================")
    print("Data Protection Region Selection")
    print("================================================")
    print("As a firm, we respect data protection laws in the chosen region.")
    print("Please select the region whose privacy and governance rules should be applied:\n")
    print("1. SA - South Africa: POPIA / PAIA")
    print("2. EU - European Union: GDPR")
    print("3. UK - United Kingdom: UK GDPR / Data Protection Act 2018")
    choice = input("\nChoose region [SA/EU/UK] default SA: ").strip().upper()

    if choice in ["1", "SA", "SOUTH AFRICA"]:
        region = "SA"
    elif choice in ["2", "EU", "EUROPE", "EUROPEAN UNION"]:
        region = "EU"
    elif choice in ["3", "UK", "UNITED KINGDOM"]:
        region = "UK"
    else:
        region = "SA"

    selected = REGION_LAWS[region]
    print(f"\nSelected region: {selected['name']}")
    print(f"The following data protection laws/governance expectations will be applied: {selected['laws']}")
    print("The application will tag telemetry with privacy.region, retention metadata, and governance attributes.")
    print("================================================\n")
    return region


PRIVACY_REGION = choose_privacy_region()
PRIVACY_LAWS = REGION_LAWS[PRIVACY_REGION]["laws"]
PRIVACY_RETENTION_DAYS = REGION_LAWS[PRIVACY_REGION]["default_retention_days"]
PRIVACY_LEGAL_BASIS = REGION_LAWS[PRIVACY_REGION]["legal_basis"]


def privacy_attrs() -> dict:
    return {
        "privacy.region": PRIVACY_REGION,
        "privacy.laws": PRIVACY_LAWS,
        "privacy.retention_days": str(PRIVACY_RETENTION_DAYS),
        "processing.legal_basis": PRIVACY_LEGAL_BASIS,
        "privacy.notice": "As a firm we respect data protection laws in the chosen region.",
    }


def show_activity(message: str = "Searching / thinking"):
    stop_event = threading.Event()

    def spinner():
        symbols = ["|", "/", "-", "\\"]
        index = 0
        while not stop_event.is_set():
            sys.stdout.write(f"\r{message} {symbols[index % len(symbols)]}")
            sys.stdout.flush()
            index += 1
            time.sleep(0.15)
        sys.stdout.write("\r" + " " * 90 + "\r")
        sys.stdout.flush()

    thread = threading.Thread(target=spinner, daemon=True)
    thread.start()
    return stop_event


llm = ChatOllama(
    model="phi3:mini",
    base_url="http://localhost:11434"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI observability assistant."),
    ("human", "{question}")
])

chain = prompt | llm

activity = show_activity("LangChain / Ollama is thinking")
try:
    response = chain.invoke({
        "question": "Explain what my OpenTelemetry Collector does."
    })
finally:
    activity.set()

print(response.content)
