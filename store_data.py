import pandas as pd
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import re
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# input data
fashion_data = [
    {"id": 1, "occasion": "wedding", "outfit_description": "When you attending a wedding, you should wear something light and elegant. Avoid all white as it is reserved for the bride and. Avoid all black if you are attending a Asian wedding as it is not an appropriate color for this occassion. An example outfit can be a sheer light pink dress with pearl earrings and nude heels."},
    {"id": 2, "occasion": "business meeting", "outfit_description": "For a business meeting, you should wear something polished and professional that conveys confidence and competence. Stick to neutral or dark colors for a refined look. Avoid overly casual or flashy attire. An example outfit can be a tailored navy blazer over a white blouse, paired with black dress pants or a pencil skirt, complemented by classic black pumps and minimal jewelry."},
    {"id": 3, "occasion": "family gathering", "outfit_description": "For a family gathering, you should wear something comfortable yet stylish, keeping in mind the setting and occasion. Choose soft, warm colors to create a welcoming feel. Avoid anything too formal or restrictive. An example outfit can be a cozy knit sweater with high-waisted jeans and ankle boots, accessorized with a delicate necklace and a crossbody bag."},
    {"id": 4, "occasion": "business intervew", "outfit_description": "For an interview, you should wear something professional and well-fitted to make a strong first impression. Stick to neutral or muted tones to keep the focus on your qualifications. Avoid loud patterns or overly casual attire. An example outfit can be a black structured blazer over a light blue button-up shirt, paired with tailored trousers and closed-toe loafers or pumps, with a sleek leather tote bag."},
    {"id": 5, "occasion": "music festival", "outfit_description": "For a music festival, you should wear something trendy and comfortable that allows for movement and reflects your personal style. Layering is key for changing weather conditions. Avoid heavy fabrics or restrictive clothing. An example outfit can be a flowy bohemian crop top with high-waisted denim shorts, paired with combat boots or comfortable sneakers, accessorized with layered necklaces, sunglasses, and a fringe crossbody bag."}
]

documents = [row["outfit_description"] for row in fashion_data]
metadatas = [{"outfit_id": str(row["id"]), "occasion": row["occasion"]} for row in fashion_data]
ids = [str(row["id"]) for row in fashion_data]

# initialize ChromaDB
client = chromadb.PersistentClient(path="fashion_vectordb")
# embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-3-small"
            )

collection = client.get_or_create_collection(name="fashion_collection", embedding_function=openai_ef)
# save
collection.add(documents=documents, metadatas=metadatas, ids=ids)
print("data saved ChromaDB！")

# load_collection = client.get_collection(name="fashion_collection")
# print("Total documents in collection:", load_collection.count())


# example
query = "summer wedding"
results = collection.query(query_texts=[query], n_results=1, include=['documents', 'distances', 'metadatas'])
print(results)
