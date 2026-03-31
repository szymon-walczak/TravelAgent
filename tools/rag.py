from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic_ai import RunContext

from config import GEMINI_MODEL_NAME, vector_car_db, vector_train_db
from models import VehicleSpec
from tools.retries import invoke_with_retry


def get_rag_price(distance_km: str) -> str:
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=0)
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


def get_pkp_ticket_price(ctx: RunContext, distance_km: float) -> str:
    """Fetches the PKP ticket price for one person based on distance."""
    return get_rag_price(str(int(distance_km)))


def get_vehicle_consumption(ctx: RunContext, car_description: str) -> VehicleSpec:
    """Retrieves fuel consumption for a car from ChromaDB, or falls back to LLM estimation."""
    results = vector_car_db.similarity_search(car_description, k=1)
    if results:
        meta = results[0].metadata
        if "l_100km_combined" in meta:
            return VehicleSpec(
                model=meta.get("model", car_description),
                l_100km_combined=meta["l_100km_combined"],
                l_100km_city=meta["l_100km_city"],
                l_100km_highway=meta["l_100km_highway"],
                source="ChromaDB",
            )

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME)
    research_prompt = (
        "Estimate the fuel consumption in L/100km (city, hwy, combined) for: "
        f"{car_description}. Return as pure JSON matching schema."
    )
    response = invoke_with_retry(
        lambda: llm.with_structured_output(VehicleSpec).invoke(research_prompt),
        label="vehicle consumption lookup",
    )
    response.source = "LLM Estimation"
    return response
