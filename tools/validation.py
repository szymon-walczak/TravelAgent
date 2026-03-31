from typing import Any, Dict, List

from pydantic_ai import RunContext

from config import FUEL_PRICE_PLN
from models import TravelResponse


def verify_recommendation(ctx: RunContext, raw_routes: List[Dict[str, Any]], response: TravelResponse) -> str:
    """
    Checks if the Car and Train costs fall within reasonable boundaries.
    Car: 2L-20L per 100km. Train: Margin of 20% compared to RAG/Distance logic.
    """
    errors = []

    car_opt = next((o for o in response.options if o.mode.lower() == "car"), None)
    car_route = next((r for r in raw_routes if r.get("mode") == "driving"), None)

    if car_opt and car_route:
        dist_km = car_route.get("dist_meters", 0) / 1000
        min_cost = (dist_km / 100) * 2 * FUEL_PRICE_PLN
        max_cost = (dist_km / 100) * 20 * FUEL_PRICE_PLN

        if not (min_cost <= car_opt.cost <= max_cost):
            errors.append(
                f"Car cost ({car_opt.cost} PLN) is unrealistic for {dist_km:.1f}km. "
                f"Should be between {min_cost:.2f} and {max_cost:.2f} PLN."
            )

    train_opt = next((o for o in response.options if o.mode.lower() == "train"), None)
    train_route = next((r for r in raw_routes if r.get("mode") == "transit"), None)

    if train_opt and train_route:
        dist_km = train_route.get("dist_meters", 0) / 1000
        if train_opt.cost < (dist_km * 0.10) or train_opt.cost > (dist_km * 0.80):
            errors.append(
                f"Train price ({train_opt.cost} PLN) deviates significantly from distance-based norms."
            )

    if not errors:
        return "✅ Verification Passed."
    return f"⚠️ Verification Warning: {', '.join(errors)}"
