# Agentic Travel Coordinator:
It loads price list for train tickets into a local vector database (ChromaDB) using RAG, and uses SerpApi to fetch real-time distance and duration for both car and train routes.
The AI Agent then compares the costs and logistics of both options to provide a recommendation.

````
User: I'd like to get the best way to travel between Gdansk and Cracov
````

````
🤖 Analyzing routes and costs...

🏆 RECOMMENDATION: The train is the recommended option as it is cheaper and more direct than driving, with a comparable travel time.

--- PRICE BREAKDOWN ---

Car | 420.50 PLN | The driving distance is 601 km and the estimated driving time is 5 hours and 35 minutes via the A1 highway.

Train | 180.00 PLN | The train journey is 690 km and takes 7 hours and 42 minutes, departing at 7:48 PM and arriving at 3:30 AM.

Flight
````