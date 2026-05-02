🧠 Overview

This project is an Autonomous AI Research Assistant that can answer user queries using information from PDF documents and web URLs. It leverages Retrieval-Augmented Generation (RAG) with a multi-agent architecture to provide contextual, accurate, and validated responses.

The system combines semantic search, LLM reasoning, and evaluation layers to simulate how real-world AI assistants operate.

✨ Features
📄 PDF Ingestion – Upload and query documents
🌐 URL Support – Extract and query web content
🔍 Semantic Search – FAISS-based vector retrieval
🧠 LLM Reasoning – Powered by Ollama (LLaMA)
🔁 Multi-Query Retrieval – Improves answer accuracy
🤖 Multi-Agent System:
Retriever Agent
Analyst Agent
Critic Agent
📊 Validation & Confidence Scoring
💬 Interactive Chat UI (Streamlit)
⚡ Real-time Streaming Responses
🏗️ Architecture
User Query
   ↓
Query Expansion (Multi-query)
   ↓
Vector Search (FAISS)
   ↓
Retriever Agent
   ↓
Analyst Agent (LLM reasoning)
   ↓
Critic Agent (Validation)
   ↓
Final Answer + Confidence + Sources
🛠️ Tech Stack
Language: Python
LLM: Ollama (LLaMA3)
Frameworks: LangChain
Vector DB: FAISS
Embeddings: sentence-transformers (MiniLM)
Frontend: Streamlit
Document Loaders: PyPDFLoader, WebBaseLoader
📦 Project Structure
├── main.py          # Core pipeline (RAG + agents)
├── AIapp.py         # Streamlit UI
├── requirements.txt
├── data/            # Uploaded PDFs
└── README.md
⚙️ Setup Instructions
1. Clone the repository
git clone https://github.com/your-username/ai-research-assistant.git
cd ai-research-assistant
2. Install dependencies
pip install -r requirements.txt
3. Install & run Ollama
ollama run llama3
4. Run the app
streamlit run AIapp.py
📸 Demo

Add a screenshot of your Streamlit UI here (very important for impact)

🧪 Example Use Cases
📚 Ask questions from study notes (PDFs)
🌐 Extract knowledge from web pages
🧾 Resume analysis and job recommendations
🔍 Research assistance across multiple sources
🚧 Limitations
Depends on quality of input documents
Basic validation logic (can be improved)
No long-term conversational memory (yet)
🚀 Future Improvements
Conversation-aware memory
Better re-ranking for retrieval
Source highlighting in answers
Deployment (Cloud / Docker)
Improved UI/UX
🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

📬 Contact

If you have feedback or suggestions, feel free to connect!
