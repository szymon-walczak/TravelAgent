import os
import fitz  # PyMuPDF
from typing import Dict, List, Any
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

class TravelOption(BaseModel):
    mode: str = Field(description="Travel mode: Train, Car, etc.")
    cost: float = Field(description="Total cost for all passengers in PLN")
    details: str = Field(description="Duration, departure times, or route info")

class TravelResponse(BaseModel):
    recommendation: str = Field(description="The final winner and why")
    options: List[TravelOption] = Field(description="List of all compared travel modes")
    alternatives: List[str] = Field(description="Other ways like walking or cycling")

load_dotenv()

DB_DIR = "./chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def get_rag_price(distance_km: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    system_prompt = (
        "Use the provided context to find the price. If unknown, say you don't know.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    query = f"What is the price of traveling of {distance_km} km?"
    response = rag_chain.invoke({"input": query})
    return response["answer"]

model = GeminiModel('gemini-2.5-flash-lite')

travel_agent = Agent(
    model,
    output_type=TravelResponse,
    system_prompt=(
        "You are a professional AI Travel Consultant. Your mission is to compare travel costs and logistics. "
        "1. CALCULATE: For 'Car' vs 'Train', always calculate total costs for ALL passengers. "
        "   (Car: 7L/100km at 6.50 PLN/L. Train: Price from RAG * number of passengers). "
        "2. TRIP DETAILS: For the best options, provide specific details (Departure/Arrival times, Carriers, Route names). "
        "3. ALTERNATIVES: List other ways to get there (but don't show Walking and Cycling) just for context. "
        "4. RECOMMENDATION: Give a clear 'Winner' based on a balance of time and money."
    ),
)


@travel_agent.tool
def get_detailed_routes(ctx: RunContext, start_addr: str, end_addr: str) -> List[Dict[str, Any]]:
    """Fetches all available travel modes and specific trip details using SerpApi."""
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
            details = {
                "mode": route.get("travel_mode"),
                "distance": route.get("formatted_distance"),
                "duration": route.get("formatted_duration"),
                "dist_meters": route.get("distance", 0),
                "start_time": route.get("start_time", "N/A"),
                "end_time": route.get("end_time", "N/A"),
                "via": route.get("via", "N/A")
            }
            if "trips" in route:
                for trip in route["trips"]:
                    if "service_run_by" in trip:
                        details["carrier"] = trip["service_run_by"].get("name")
                        break
            parsed_routes.append(details)
    return parsed_routes

@travel_agent.tool
def get_pkp_ticket_price(ctx: RunContext, distance_km: float) -> str:
    """Fetches the PKP ticket price for one person based on distance."""
    return get_rag_price(str(int(distance_km)))

@travel_agent.tool
def update_price_knowledge_base(ctx: RunContext, directory_path: str) -> str:
    """Updates the vector database with new PDF price lists."""
    documents = []
    if not os.path.exists(directory_path): return "Path error."
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            with fitz.open(os.path.join(directory_path, filename)) as doc:
                text = "".join([p.get_text() for p in doc])
                if text.strip():
                    chunks = RecursiveCharacterTextSplitter(chunk_size=1000).split_text(text)
                    for c in chunks:
                        documents.append(Document(page_content=c, metadata={"source": filename}))
    if documents:
        vector_db.add_documents(documents)
        return f"Successfully added {len(documents)} chunks."
    return "No PDFs found."

def main():
    print("\n" + "="*60)
    print("🌍 WELCOME TO THE AI TRAVEL COORDINATOR")
    print("Commands: 'exit' to quit | 'update' to sync new price PDFs")
    print("="*60)

    while True:
        user_query = input("\n👤 User: ").strip()
        if user_query.lower() in ['exit', 'quit']: break

        if user_query.lower() == 'update':
            path = input("📂 Enter folder path: ")
            user_query = f"Please process the documents in this folder: {path}"

        print("🤖 Analyzing routes and costs...")
        try:
            result = travel_agent.run_sync(user_query)
            data = result.output

            print(f"\n🏆 RECOMMENDATION: {data.recommendation}")
            print("\n--- PRICE BREAKDOWN ---")
            for opt in data.options:
                print(f"* {opt.mode:10} | {opt.cost:8.2f} PLN | {opt.details}")

            for alt in data.alternatives:
                print(f"- {alt}")

        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()