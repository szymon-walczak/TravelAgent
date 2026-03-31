from agent import run_travel_agent_sync
from tools import update_all_knowledge


def main():
    print("🌍 AI TRAVEL COORDINATOR")
    while True:
        user_query = input("\n👤 User: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break

        if user_query.lower() == "update":
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
        except Exception as error:
            print(f"⚠️ Error: {error}")


if __name__ == "__main__":
    main()
