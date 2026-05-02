import streamlit as st
import time
from main import run_pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------- FIXED YELLOW THEME ----------------
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #fffbea;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #fff3cd;
}

/* TEXT COLORS */
h1, h2, h3, p, label {
    color: black !important;
}

/* USER MESSAGE */
.user-msg {
    background-color: #ffd54f;
    color: black;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 8px;
}

/* BOT MESSAGE */
.bot-msg {
    background-color: #ffffff;
    color: black;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 8px;
    border: 1px solid #ddd;
}

/* INPUT BAR */
div[data-testid="stChatInput"] {
    background-color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

if "urls" not in st.session_state:
    st.session_state.urls = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📂 Data Sources")

# PDF Upload
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    path = f"data/{uploaded_file.name}"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.pdf_path = path
    st.sidebar.success("✅ PDF uploaded")

# URL Input
url_input = st.sidebar.text_input("Enter URL")

if st.sidebar.button("Add URL"):
    if url_input:
        if url_input not in st.session_state.urls:
            st.session_state.urls.append(url_input)
            st.sidebar.success("✅ URL added")

# Active sources
st.sidebar.markdown("### 📌 Active Sources")
st.sidebar.write("PDF:", st.session_state.pdf_path if st.session_state.pdf_path else "None")
st.sidebar.write("URLs:", st.session_state.urls if st.session_state.urls else "[]")

# ---------------- HEADER ----------------
st.title("🤖 Autonomous AI Assistant")
st.caption("Chat + RAG + Multi-Agent System")

# ---------------- STREAM FUNCTION ----------------
def stream_text(text):
    placeholder = st.empty()
    output = ""

    for char in text:
        output += char
        placeholder.markdown(output)
        time.sleep(0.003)

    return output

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
query = st.chat_input("Ask anything...")

# ---------------- MAIN LOGIC ----------------
if query:
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    st.markdown(f"<div class='user-msg'>{query}</div>", unsafe_allow_html=True)

    with st.spinner("🤖 Thinking..."):
        try:
            result = run_pipeline(
                query,
                pdf_path=st.session_state.pdf_path,
                urls=st.session_state.urls
            )

            answer = result["answer"]

        except Exception as e:
            answer = f"❌ Error: {str(e)}"

    # Stream response
    st.markdown("<div class='bot-msg'>", unsafe_allow_html=True)
    streamed = stream_text(answer)
    st.markdown("</div>", unsafe_allow_html=True)

    # Save bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": streamed
    })

    # ---------------- DETAILS ----------------
    with st.expander("📊 Details"):
        st.write("Confidence:", result.get("confidence", 0))
        st.write("Validation:", result.get("validation", ""))

        st.subheader("📚 Sources")
        for i, doc in enumerate(result.get("sources", [])):
            st.write(f"Source {i+1} (Page {doc.metadata.get('page')})")
            st.write(doc.page_content[:300])