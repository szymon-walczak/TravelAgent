import json
import os

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic_ai import RunContext

from config import vector_car_db, vector_train_db


def update_all_knowledge(ctx: RunContext, pdf_dir: str, master_json_path: str) -> str:
    """Updates both the PKP PDF price lists and the structured Car Master Database."""
    report = []

    documents = []
    if not os.path.exists(pdf_dir):
        return "Directory not found."

    for filename in os.listdir(pdf_dir):
        f_path = os.path.join(pdf_dir, filename)
        if filename.endswith(".pdf"):
            with fitz.open(f_path) as doc:
                text = "".join([page.get_text() for page in doc])

                if text.strip():
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = splitter.split_text(text)
                    for chunk in chunks:
                        documents.append(Document(page_content=chunk, metadata={"source": filename}))

    if documents:
        vector_train_db.add_documents(documents)
        report.append(f"Added {len(documents)} new chunks to the train database.")

    if os.path.exists(master_json_path):
        with open(master_json_path, "r") as f:
            cars = json.load(f)
            docs = []
            for car in cars:
                txt = f"{car['manufacturer']} {car['model']} {car['year']} {car['engine']}"
                docs.append(Document(page_content=txt, metadata=car))

            batch_size = 5000
            for index in range(0, len(docs), batch_size):
                batch = docs[index:index + batch_size]
                vector_car_db.add_documents(batch)

            report.append(f"Loaded {len(docs)} cars to Chroma in {len(docs)//batch_size + 1} batches.")

    return " | ".join(report)
