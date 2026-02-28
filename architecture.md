# AI Travel Coordinator: Architecture & Setup
This document describes the technical architecture, technology stack, and operational flow of the AI Travel Coordinator, a multi-agent system designed to compare the logistics and costs of travel (specifically Train vs. Car) using RAG and real-time APIs.

## 🧭 System Overview
The system operates as an AI Agentic Workflow. Instead of a linear script, an AI Agent acts as a "reasoning engine" that decides which tools to call based on user intent.

## Core Components:
Orchestrator (Pydantic-AI): Manages the state, tools, and structured output.

Navigation Engine (SerpApi): Fetches real-time distance, duration, and transit schedules.

Financial Engine (RAG): Consults a local Vector Database (ChromaDB) containing PKP price lists to find ticket costs based on distance.

Schema Validator (Pydantic): Ensures the AI output matches a strict TravelResponse structure for reliable UI/CLI display.

## Operational FlowInput: 
````
User asks: "Is it cheaper for 4 people to go from Lublin to Warsaw?"
Reasoning: The Agent realizes it needs distance (SerpApi) and ticket prices (RAG).
Tool Execution:
Calls get_detailed_routes -> Returns ~170km.
Calls get_pkp_ticket_price(170) -> RAG finds "170km = 50 PLN".
Computation: The Agent calculates:
    Car: (170 / 100) x 7L x 6.50 PLN = approx 77.35 PLN 
    Train: $4 x 50 PLN = 200 PLN
Output: The Agent populates the TravelResponse schema with the winner (Car) and alternatives.
````

## How to run:
1. Set up Python environment
2. Install dependencies by poetry install
3. Set GOOGLE_API_KEY and SERPAPI_API_KEY environment variable (or create .env file)
4. Run the main script with poetry run python agent.py
5. Type update to load the PKP price list into the local ChromaDB vector database (only needed once or when prices change)
6. Input travel details and receive a comparison of Train vs. Car logistics and costs.