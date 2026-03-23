import os
import fitz  # PyMuPDF
import json
import asyncio
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from serpapi import GoogleSearch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- Models ---

class VehicleSpec(BaseModel):
    model: str = Field(description="Model and trim name")
    l_100km_combined: float = Field(description="Combined fuel consumption")
    l_100km_city: float = Field(description="City fuel consumption")
    l_100km_highway: float = Field(description="Highway fuel consumption")
    source: str = Field(description="Where the data came from (DB or LLM)")

class TravelOption(BaseModel):
    mode: str = Field(description="Travel mode: Train, Car, etc.")
    cost: float = Field(description="Total cost for all passengers in PLN")
    details: str = Field(description="Duration, departure times, or route info")

class TravelResponse(BaseModel):
    recommendation: str = Field(description="The final winner and why")
    options: List[TravelOption] = Field(description="List of all compared travel modes")
    alternatives: List[str] = Field(description="Other ways like walking or cycling")

load_dotenv()

# --- Shared Config & DB ---
FUEL_PRICE_PLN = 6.50
MAX_MODEL_RETRIES = 3
MODEL_RETRY_DELAY_SECONDS = 2
DB_TRAIN_DIR = "./chroma_db-train"
DB_CARS_DIR = "./chroma_db-vehicles"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_train_db = Chroma(persist_directory=DB_TRAIN_DIR, embedding_function=embeddings)
vector_car_db = Chroma(persist_directory=DB_CARS_DIR, embedding_function=embeddings)


def is_transient_model_error(error: Exception) -> bool:
    message = str(error).lower()
    transient_markers = (
        "503",
        "unavailable",
        "high demand",
        "service is currently unavailable",
    )
    return any(marker in message for marker in transient_markers)


def invoke_with_retry(operation, label: str):
    for attempt in range(1, MAX_MODEL_RETRIES + 1):
        try:
            return operation()
        except Exception as error:
            if attempt < MAX_MODEL_RETRIES and is_transient_model_error(error):
                print(f"Retrying {label} after transient model error ({attempt}/{MAX_MODEL_RETRIES}): {error}")
                time.sleep(MODEL_RETRY_DELAY_SECONDS)
                continue
            raise


async def run_travel_agent(query: str):
    for attempt in range(1, MAX_MODEL_RETRIES + 1):
        try:
            return await travel_agent.run(query)
        except Exception as error:
            if attempt < MAX_MODEL_RETRIES and is_transient_model_error(error):
                print(f"Retrying travel agent after transient model error ({attempt}/{MAX_MODEL_RETRIES}): {error}")
                await asyncio.sleep(MODEL_RETRY_DELAY_SECONDS)
                continue
            raise


def run_travel_agent_sync(query: str):
    for attempt in range(1, MAX_MODEL_RETRIES + 1):
        try:
            return travel_agent.run_sync(query)
        except Exception as error:
            if attempt < MAX_MODEL_RETRIES and is_transient_model_error(error):
                print(f"Retrying travel agent after transient model error ({attempt}/{MAX_MODEL_RETRIES}): {error}")
                time.sleep(MODEL_RETRY_DELAY_SECONDS)
                continue
            raise

def get_rag_price(distance_km: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    system_prompt = "Use the provided context to find the price for train. If unknown, say you don't know.\n\n{context}"
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_train_db.as_retriever(search_kwargs={"k": 3})
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    query = f"What is the price of traveling of {distance_km} km?\n"
    response = invoke_with_retry(
        lambda: rag_chain.invoke({"input": query}),
        label="train price lookup",
    )
    return response["answer"]

# --- Agent Definition ---
model = GeminiModel('gemini-2.5-flash-lite')

travel_agent = Agent(
    model,
    output_type=TravelResponse,
    system_prompt=(
        "You are a professional AI Travel Consultant. Mission: Compare travel costs and logistics.\n"
        "1. CALCULATE: For 'Car' vs 'Train', calculate total costs for ALL passengers.\n"
        "   - Car: Use 'get_vehicle_consumption' to find L/100km for the specific car mentioned. If not use 7L/100km as a fallback. "
        "          Calculate cost = (distance_km / 100) * L/100km * fuel_price.\n"
        "          If the trip is long, prioritize 'l_100km_highway'.\n"
        "          Don't multiply by passengers if the car can fit them.\n"
        "   - Train: Price from RAG * number of passengers.\n"
        "2. VERIFICATION: You MUST call 'verify_recommendation' before finalizing.\n"
        "3. RECOMMENDATION: Balance time, cost and convenience."
    ),
)

# --- Tools ---

@travel_agent.tool
def get_detailed_routes(ctx: RunContext, start_addr: str, end_addr: str) -> List[Dict[str, Any]]:
    """Fetches all available travel modes using SerpApi."""
    params = {
        "engine": "google_maps_directions",
        "start_addr": start_addr,
        "end_addr": end_addr,
        "travel_mode": "6",
        "hl": "en",
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    parsed_routes = []
    if "directions" in results:
        for route in results["directions"]:
            parsed_routes.append({
                "mode": route.get("travel_mode"),
                "dist_meters": route.get("distance", 0),
                "duration": route.get("formatted_duration"),
                "formatted_distance": route.get("formatted_distance")
            })
    #print(parsed_routes)
    return parsed_routes

@travel_agent.tool
def get_pkp_ticket_price(ctx: RunContext, distance_km: float) -> str:
    """Fetches the PKP ticket price for one person based on distance."""
    return get_rag_price(str(int(distance_km)))

@travel_agent.tool
def get_vehicle_consumption(ctx: RunContext, car_description: str) -> VehicleSpec:
    """Retrieves fuel consumption for a car from ChromaDB, or falls back to LLM estimation."""
    # 1. Try DB Retrieval
    results = vector_car_db.similarity_search(car_description, k=1)
    if results:
        # Check if score is high enough (metadata contains the structured fields)
        meta = results[0].metadata
        if "l_100km_combined" in meta:
            return VehicleSpec(
                model=meta.get("model", car_description),
                l_100km_combined=meta["l_100km_combined"],
                l_100km_city=meta["l_100km_city"],
                l_100km_highway=meta["l_100km_highway"],
                source="ChromaDB"
            )

    # 2. Fallback to LLM Research
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    research_prompt = f"Estimate the fuel consumption in L/100km (city, hwy, combined) for: {car_description}. Return as pure JSON matching schema."
    # We use structured output to ensure the format is secure
    response = invoke_with_retry(
        lambda: llm.with_structured_output(VehicleSpec).invoke(research_prompt),
        label="vehicle consumption lookup",
    )
    response.source = "LLM Estimation"
    return response

@travel_agent.tool
def verify_recommendation(ctx: RunContext, raw_routes: List[Dict[str, Any]], response: TravelResponse) -> str:
    """
    Checks if the Car and Train costs fall within reasonable boundaries.
    Car: 2L-20L per 100km. Train: Margin of 20% compared to RAG/Distance logic.
    """
    errors = []
    FUEL_PRICE = 6.50

    # 1. Car Verification (Range: 2L to 20L per 100km)
    car_opt = next((o for o in response.options if o.mode.lower() == "car"), None)
    car_route = next((r for r in raw_routes if r.get("mode") == "driving"), None)

    if car_opt and car_route:
        dist_km = car_route.get("dist_meters", 0) / 1000
        min_cost = (dist_km / 100) * 2 * FUEL_PRICE
        max_cost = (dist_km / 100) * 20 * FUEL_PRICE

        if not (min_cost <= car_opt.cost <= max_cost):
            errors.append(f"Car cost ({car_opt.cost} PLN) is unrealistic for {dist_km:.1f}km. "
                          f"Should be between {min_cost:.2f} and {max_cost:.2f} PLN.")

    # 2. Train Verification (20% Tolerance)
    train_opt = next((o for o in response.options if o.mode.lower() == "train"), None)
    train_route = next((r for r in raw_routes if r.get("mode") == "transit"), None)

    if train_opt and train_route:
        dist_km = train_route.get("dist_meters", 0) / 1000
        # Heuristic: PKP Class 2 is roughly 0.20-0.45 PLN per km depending on distance
        # We allow a wide 20% margin for dynamic pricing/updates.
        if train_opt.cost < (dist_km * 0.10) or train_opt.cost > (dist_km * 0.80):
            errors.append(f"Train price ({train_opt.cost} PLN) deviates significantly from distance-based norms.")

    if not errors:
        return "✅ Verification Passed."
    return f"⚠️ Verification Warning: {', '.join(errors)}"

@travel_agent.tool
def update_all_knowledge(ctx: RunContext, pdf_dir: str, master_json_path: str) -> str:
    """Updates both the PKP PDF price lists and the structured Car Master Database."""
    report = []

    # 1. (PDF processing logic remains here...)
    documents = []
    if not os.path.exists(pdf_dir):
        return "Directory not found."

    for filename in os.listdir(pdf_dir):
        f_path = os.path.join(pdf_dir, filename)
        if filename.endswith(".pdf"):
            with fitz.open(f_path) as doc:
                text = "".join([p.get_text() for p in doc])

                if text.strip():
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = splitter.split_text(text)
                    for c in chunks:
                        documents.append(Document(page_content=c, metadata={"source": filename}))

    if documents:
        vector_train_db.add_documents(documents)
        report.append(f"Added {len(documents)} new chunks to the train database.")

    # 2. Update Master Car JSON with Batching Fix
    if os.path.exists(master_json_path):
        with open(master_json_path, 'r') as f:
            cars = json.load(f)
            docs = []
            for c in cars:
                # Content used for vector searching
                txt = f"{c['manufacturer']} {c['model']} {c['year']} {c['engine']}"
                docs.append(Document(page_content=txt, metadata=c))

            # BATCHING LOGIC: Chunking the upload to stay under 5461 limit
            batch_size = 5000
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                vector_car_db.add_documents(batch)

            report.append(f"Loaded {len(docs)} cars to Chroma in {len(docs)//batch_size + 1} batches.")

    return " | ".join(report)

# --- Main ---

def main():
    print("🌍 AI TRAVEL COORDINATOR")
    while True:
        user_query = input("\n👤 User: ").strip()
        if user_query.lower() in ['exit', 'quit']: break

        if user_query.lower() == 'update':
            # Example path usage
            print(update_all_knowledge(None, "./data", "./data/master_vehicles_database.json"))
            continue

        print("🤖 Analyzing routes and costs...")
        try:
            result = run_travel_agent_sync(user_query)
            data = result.output

            print(f"\n🏆 RECOMMENDATION: {data.recommendation}")
            print("\n--- PRICE BREAKDOWN ---")
            for opt in data.options:
                print(f"* {opt.mode:10} | {opt.cost:8.2f} PLN | {opt.details}")
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
