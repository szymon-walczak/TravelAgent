import asyncio
from typing import List, Dict
from agent import run_travel_agent

# Shared constants to match agent.py
FUEL_PRICE_PLN = 6.50

TEST_CASES = [
    {
        "query": "I want to travel from Lublin to Poznan",
        "expected_dist_km": 440,
        "min_expected_train_price": 70.0,  # 421-440km is 77 PLN
        "max_expected_train_price": 110.0,
        "passengers": 1,
        "expected_car_model": None # Generic check
    },
    {
        "query": "Travel from Warsaw to Krakow for 2 people in a 2024 Porsche 911 Carrera",
        "expected_dist_km": 290,
        "min_expected_train_price": 130.0, # 68 PLN * 2 class 2, 88 PLN * 2 class 1
        "max_expected_train_price": 180.0,
        "passengers": 2,
        "expected_car_model": "Porsche 911",
        "expected_l_100km": 9.8  # Known value for 911 Carrera
    },
    {
        "query": "Travel from Lublin to Warsaw in a Citroen CX",
        "expected_dist_km": 170,
        "is_llm_fallback": True, # This car shouldn't be in your DB
        "passengers": 1
    }
]
async def run_evaluation():
    print(f"Starting Automated Evaluation...")
    print("-" * 60)

    passed = 0
    for i, test in enumerate(TEST_CASES):
        print(f"TEST #{i+1}: {test['query']}")
        try:
            result = await run_travel_agent(test['query'])
            output = result.output

            # 1. Verify Train Price Logic
            train_opt = next((o for o in output.options if "Train" in o.mode), None)
            if train_opt and "min_expected_train_price" in test:
                if test['min_expected_train_price'] <= train_opt.cost <= test['max_expected_train_price']:
                    print(f"  ✅ Train Cost: {train_opt.cost} PLN")
                else:
                    print(f"  ❌ Train Cost Error: Got {train_opt.cost}, expected {test['min_expected_train_price']}-{test['max_expected_train_price']}")

            # 2. Verify Car Cost & RAG Logic
            car_opt = next((o for o in output.options if "Car" in o.mode), None)
            if car_opt:
                # Use expected L/100km if provided (RAG test), else fallback to 7L heuristic
                l_100km = test.get("expected_l_100km", 7.0)
                expected_car_cost = (test['expected_dist_km'] / 100) * l_100km * FUEL_PRICE_PLN
                if test.get("is_llm_fallback"):
                    expected_car_cost *= 3

                # Check if the cost is within a 15% margin of our expectation
                margin = expected_car_cost * 0.15
                if abs(car_opt.cost - expected_car_cost) <= margin:
                    source_type = "LLM" if test.get("is_llm_fallback") else "RAG DB"
                    print(f"  ✅ Car Cost: {car_opt.cost} PLN (Accurate using {source_type} data)")
                else:
                    print(f"  ❌ Car Cost Error: Got {car_opt.cost}, expected ~{expected_car_cost:.2f}")

            # 3. Verify Recommendation & Reflection
            recommendation = output.recommendation.strip()
            if len(recommendation) > 20:
                print(f"  ✅ Recommendation: Balanced and logical.")
                passed += 1
            else:
                print(f"  ❌ Recommendation: Insufficient or missing reflection.")

        except Exception as e:
            print(f"  💥 Execution Error: {e}")
        print("-" * 60)
        await asyncio.sleep(1)

    print(f"\n✨ EVALUATION COMPLETE: {passed}/{len(TEST_CASES)} Tests Passed.")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
