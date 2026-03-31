from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

FUEL_PRICE_PLN = 7.50
MAX_MODEL_RETRIES = 3
MODEL_RETRY_DELAY_SECONDS = 2
GEMINI_MODEL_NAME = "gemini-2.5-flash"
DB_TRAIN_DIR = "./chroma_db-train"
DB_CARS_DIR = "./chroma_db-vehicles"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_train_db = Chroma(persist_directory=DB_TRAIN_DIR, embedding_function=embeddings)
vector_car_db = Chroma(persist_directory=DB_CARS_DIR, embedding_function=embeddings)
