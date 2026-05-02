import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
import json
import numpy as np
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.chat_models import ChatOllama

print(os.path.exists(r"C:\Users\vaish\AIML.pdf"))

DATA_PATH = "data/"
DB_PATH = "vectorstore/"
MEMORY_PATH = "memory.json"

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

def load_documents(pdf_paths=[], urls=[]):
    docs = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())

    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    clean_chunks = []
    for doc in chunks:
        text = doc.page_content.strip()

        # ❌ REMOVE BAD CHUNKS
        if (
            len(text) > 80 and
            not text.isdigit() and
            len(text.split()) > 5
        ):
            clean_chunks.append(doc)

    print(f"✅ Clean chunks kept: {len(clean_chunks)}")

    vectorstore = FAISS.from_documents(clean_chunks, embeddings)
    vectorstore.save_local(DB_PATH)

    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True  
    )

llm = ChatOllama(model="phi")

def load_memory():
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)

def update_memory(query, response):
    memory = load_memory()
    memory.append({
        "query": query,
        "response": response,
        "time": str(datetime.now())
    })
    save_memory(memory)


def classify_query(query):
    prompt = f"""
    Classify this query:
    {query}

    Categories:
    - factual
    - analytical
    - simple

    Only return category.
    """
    return llm.invoke(prompt).content.strip().lower()

def generate_queries(query):
    prompt = F"""
    Generate 3 alternative search queries for:
    {query}
    """
    result = llm.invoke(prompt).content
    
    return list(set(result.split("\n")))

def multi_query_retrieval(query, retriever):
    queries = generate_queries(query)
    docs = []

    for q in queries:
        docs.extend(retriever.invoke(q))

    # Deduplicate
    unique = {doc.page_content: doc for doc in docs}

    # 🔥 Filter junk content (VERY IMPORTANT)
    filtered_docs = [
        doc for doc in unique.values()
        if len(doc.page_content.strip()) > 50
    ]

    return filtered_docs

# Planner Agent

def planner_agent(query):
    return f"Plan: Break query '{query}' into steps."

# Retriever Agent

def retriever_agent(query, retriever):
    return multi_query_retrieval(query, retriever)

def analyst_agent(query, docs):
    context = "\n".join([d.page_content for d in docs[:5]])

    prompt = f"""
You are a strict AI assistant.

Answer ONLY from the given context.

RULES:
- If answer is NOT in context → say: "Answer not found in provided documents."
- Do NOT include hashtags
- Do NOT include extra questions
- Keep answer clean and structured

Context:
{context}

Question:
{query}

Answer:
"""

    return llm.invoke(prompt).content

def critic_agent(answer, docs):
    context = " ".join([d.page_content for d in docs[:3]])

    prompt = f"""
Check if the answer is supported by context.

Return ONLY:
- Valid
- Invalid

Answer:
{answer}

Context:
{context}
"""

    return llm.invoke(prompt).content.strip()

def evaluate(answer, docs):
    relevance = len(docs) / 10
    confidence = min(1.0, relevance)

    return {
        "relevance": relevance,
        "confidence": confidence
    }

def clean_output(text):
    text = text.replace("```", "")
    text = text.replace("python", "")

    # Remove unwanted trailing question
    if "Question:" in text:
        text = text.split("Question:")[0]

    # Remove hashtags
    text = text.replace("#", "")

    return text.strip()

def format_output(result):
    print("\n🧠 ANSWER:\n")
    print(result["answer"])

    print("\n📊 CONFIDENCE:", result["confidence"])
    print("✅ VALIDATION:", result["validation"])

    print("\n📚 SOURCES:")
    for i, doc in enumerate(result["sources"]):
        print(f"{i+1}. Page {doc.metadata.get('page')}")

def run_pipeline(query, pdf_path=None, urls=[]):

    # 🔥 Decide data sources dynamically
    pdf_paths = [pdf_path] if pdf_path else []

    # 🚨 Always rebuild DB when new data is provided
    if pdf_path or urls or not os.path.exists(os.path.join(DB_PATH, "index.faiss")):
        print("⚠️ Rebuilding Vector DB...")

        docs = load_documents(
            pdf_paths=pdf_paths,
            urls=urls
        )

        chunks = split_documents(docs)
        create_vectorstore(chunks)

        print("✅ Vector DB ready.")

    # ✅ Load vector DB
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 🔍 Retrieval
    docs = retriever_agent(query, retriever)

    # 🔍 Debug preview
    print("\n🔍 Retrieved Docs Preview:\n")
    for i, d in enumerate(docs[:3]):
        print(f"\n--- DOC {i+1} ---\n", d.page_content[:200])

    # 🧠 Agents
    answer = analyst_agent(query, docs)
    validation = critic_agent(answer, docs)
    metrics = evaluate(answer, docs)

    update_memory(query, answer)

    return {
        "answer": clean_output(answer),
        "sources": docs[:3],
        "validation": validation.strip(),
        "confidence": metrics["confidence"]
    }