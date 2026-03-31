from typing import List

from pydantic import BaseModel, Field


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
