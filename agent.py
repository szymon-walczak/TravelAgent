from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

from config import GEMINI_MODEL_NAME
from models import TravelResponse
from tools import (
    get_detailed_routes,
    get_pkp_ticket_price,
    get_vehicle_consumption,
    verify_recommendation,
    update_all_knowledge,
)
from tools.retries import invoke_with_retry, invoke_with_retry_async

model = GeminiModel(GEMINI_MODEL_NAME)

travel_agent = Agent(
    model,
    output_type=TravelResponse,
    system_prompt=(
        "You are a professional AI Travel Consultant. Mission: Compare travel costs and logistics.\n"
        "1. CALCULATE: For 'Car' vs 'Train', calculate total costs for ALL passengers.\n"
        "   - Car: Use 'get_vehicle_consumption' to find L/100km for the specific car mentioned. If not use 7.5L/100km as a fallback. "
        "          Calculate cost = (distance_km / 100) * L/100km * fuel_price.\n"
        "          If the trip is long, prioritize 'l_100km_highway'.\n"
        "          Don't multiply by passengers if the car can fit them.\n"
        "   - Train: Price from RAG * number of passengers.\n"
        "2. VERIFICATION: You MUST call 'verify_recommendation' before finalizing.\n"
        "3. RECOMMENDATION: Balance time, cost and convenience."
    ),
)

travel_agent.tool(get_detailed_routes)
travel_agent.tool(get_pkp_ticket_price)
travel_agent.tool(get_vehicle_consumption)
travel_agent.tool(verify_recommendation)
travel_agent.tool(update_all_knowledge)


async def run_travel_agent(query: str):
    return await invoke_with_retry_async(
        lambda: travel_agent.run(query),
        label="travel agent",
    )


def run_travel_agent_sync(query: str):
    return invoke_with_retry(
        lambda: travel_agent.run_sync(query),
        label="travel agent",
    )
