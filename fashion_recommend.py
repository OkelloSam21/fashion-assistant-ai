from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import chromadb.utils.embedding_functions as embedding_functions
from langchain_openai import OpenAIEmbeddings
import chromadb
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document

# 1️⃣ API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2️⃣ connect ChromaDB
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
vector_store = Chroma(collection_name="fashion_collection", embedding_function=embedding_model, persist_directory="./fashion_vectordb")

# 3️⃣ initialize GPT-4
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

# System instruction template
custom_prompt = PromptTemplate(
    input_varaibles=["context", "question"],
    template=(
        "You are a helpful AI fashion assistant. Provide outfit recommendations based on the given context within 100 words. \
        You should include: \
            1. what needs to be avoided for the occasion \
            2. what colors are most appropriate \
            3. outfit according to season provided \
            4. outfit according to gender provided \
            5. outfit based on user preferred style"
        "Context:\n{context}\n\n"
        "User Question:\n{question}\n\n"
        "AI Fashion Assistant Response:"
    )
)

# 4️⃣ RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

def get_fashion_recommendation(gender, occasion, season, style, detail):
    query = f"I am a {gender}. What should I wear for a {occasion} in {season}. My common style is {style} and I want to let you know that {detail}"
    result = qa_chain({"query": query})
    print(result)
    response_text = result['result']
    response = f"**👗 Outfit Recommendation:**\n\n{response_text}\n\n"

    img_response = OpenAI().images.generate(
        model="dall-e-3",
        prompt="Generate a real image for this outfit: " + response_text,
        n=1,
        size="1024x1024"
    )
    return response,img_response.data[0].url
    return response

def refine_fashion_recommendation(detail, prev_prompt):
    query = f"Based on previous response {prev_prompt} and modify only this part : {detail}. Generate a refined recommendation."
    result = qa_chain({"query": query})
    response_text = result['result']
    response = f"**👗 Refined Outfit Recommendation:**\n\n{response_text}\n\n"

    img_response = OpenAI().images.generate(
        model="dall-e-3",
        prompt="Generate a real image for this outfit: " + response_text,
        n=1,
        size="1024x1024"
    )
    return response,img_response.data[0].url