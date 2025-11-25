import streamlit as st
from main import load_pdfs, create_qa_chain, is_toxic, detect_and_store_name, query_router

# ==== Setup Halaman ====
st.set_page_config(page_title="LLM RAG Chatbot", layout="wide")

# ==== CSS Styling ====
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            background-color: #F0F0F0;
            padding: 20px;
        }
        .user-message {
            background-color: #D6EAF8;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 5px;
        }
        .bot-message {
            background-color: #FDEBD0;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .chat-title {
            color: #1F4E79;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ==== Inisialisasi State ====
for key in ["qa_chain", "chat_history", "uploaded_files", "selected_chat", "docs"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["chat_history", "uploaded_files", "docs"] else None

# ==== Layout Dua Kolom ====
col_sidebar, col_main = st.columns([1, 3], gap="large")

# ==== SIDEBAR ====
with col_sidebar:
    st.image("logo.png", width=300)
    st.markdown("## 🔎 <span class='chat-title'>Chat History</span>", unsafe_allow_html=True)
    st.markdown("Klik untuk melihat kembali isi percakapan sebelumnya.")
    st.markdown("Developed by Joseph Samuel Angelo · 2025")

    for i, chat in enumerate(st.session_state.chat_history[::-1]):
        label = f"{i+1}. {chat['user'][:30]}..."
        if st.button(label, key=f"btn_{i}", help=chat['user']):
            st.session_state.selected_chat = chat

    st.divider()
    uploaded_files = st.file_uploader("📎 Upload PDF (maks 5 file)", type="pdf", accept_multiple_files=True)

    if st.button("🔄 Reset Chat"):
        st.session_state.chat_history.clear()
        st.session_state.selected_chat = None
        if st.session_state.qa_chain and hasattr(st.session_state.qa_chain, "memory"):
            st.session_state.qa_chain.memory.clear()
        st.success("Riwayat telah direset.")

    if st.button("🗑️ Reset Semua"):
        for key in ["chat_history", "qa_chain", "uploaded_files", "selected_chat", "docs"]:
            st.session_state[key] = [] if isinstance(st.session_state.get(key), list) else None
        st.rerun()

# ==== PROSES PDF ====
if uploaded_files and len(uploaded_files) <= 5 and not st.session_state.qa_chain:
    with st.spinner("🔄 Memproses dokumen..."):
        docs = load_pdfs(uploaded_files)
        st.session_state.qa_chain, st.session_state.docs = create_qa_chain(docs)
        st.session_state.uploaded_files = uploaded_files
    st.success("✅ Dokumen siap digunakan!")

# ==== AREA TENGAH ====
with col_main:
    st.title("📄 LLM RAG Chatbot Berbasis Dokumen PDF")
    st.markdown("Ajukan pertanyaan berdasarkan dokumen yang telah kamu unggah.")

    if st.session_state.uploaded_files:
        filenames = ", ".join([f.name for f in st.session_state.uploaded_files])
        st.markdown(f"📂 **Dokumen aktif**: {filenames}")

    # Tombol untuk lihat chunks + vectors
    if st.session_state.docs:
        if st.button("🔍 Lihat Chunk & Vector Embedding"):
            from main import get_chunks_and_vectors, OllamaEmbeddings
            emb = OllamaEmbeddings(model="nomic-embed-text")
            data = get_chunks_and_vectors(st.session_state.docs, emb, n=5)

            for item in data:
                st.markdown(f"**Chunk {item['chunk_id']}**")
                st.write(f"📝 Text: {item['text']}")
                st.write(f"🔢 Vector (10 dimensi pertama): {item['vector']}")
                st.markdown("---")

    # Menampilkan Q&A dari riwayat
    if st.session_state.selected_chat:
        st.markdown("### 📌 Q&A dari Riwayat:")
        st.markdown(f"<div class='user-message'>🧑 Kamu: {st.session_state.selected_chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-message'>🤖 Bot: {st.session_state.selected_chat['bot']}</div>", unsafe_allow_html=True)
        st.markdown("---")

    # Menampilkan seluruh chat
    for chat in st.session_state.chat_history:
        st.markdown(f"<div class='user-message'>🧑 Kamu: {chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-message'>🤖 Bot: {chat['bot']}</div>", unsafe_allow_html=True)

# Chat input
if st.session_state.qa_chain:
    query = st.chat_input("Tanyakan sesuatu...")
    if query:
        detect_and_store_name(query)
        if is_toxic(query):
            st.warning("⚠️ Pertanyaan mengandung kata tidak pantas.")
        else:
            with st.spinner("🤖 Menjawab..."):
                response = query_router(query, st.session_state.qa_chain, st.session_state.docs, st.session_state.uploaded_files)
                if isinstance(response, str) and response.lower().startswith("maaf, slide"):
                    st.warning(response)
                else:
                    st.session_state.chat_history.append({"user": query, "bot": response})
                    st.session_state.selected_chat = None
                    st.rerun()
