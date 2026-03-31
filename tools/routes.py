import os
from typing import Any, Dict, List

from pydantic_ai import RunContext
from serpapi import GoogleSearch


def get_detailed_routes(ctx: RunContext, start_addr: str, end_addr: str) -> List[Dict[str, Any]]:
    """Fetches all available travel modes using SerpApi."""
    params = {
        "engine": "google_maps_directions",
        "start_addr": start_addr,
        "end_addr": end_addr,
        "travel_mode": "6",
        "hl": "en",
        "api_key": os.getenv("SERPAPI_API_KEY"),
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    parsed_routes = []
    if "directions" in results:
        for route in results["directions"]:
            parsed_routes.append(
                {
                    "mode": route.get("travel_mode"),
                    "dist_meters": route.get("distance", 0),
                    "duration": route.get("formatted_duration"),
                    "formatted_distance": route.get("formatted_distance"),
                }
            )
    return parsed_routes
