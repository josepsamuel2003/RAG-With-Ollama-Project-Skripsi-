# main.py
import re
import requests
from tempfile import NamedTemporaryFile
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain.schema.embeddings import Embeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document

# ==================== TOXIC CHECK =====================
TOXIC_WORDS = ["anjing", "babi", "kontol", "goblok", "bangsat", "tolol", "bodoh"]
user_name = None

def is_toxic(text):
    return any(word in text.lower() for word in TOXIC_WORDS)

def detect_and_store_name(text):
    global user_name
    match = re.search(r"nama\s*(saya|aku)?\s*(adalah)?\s*:?[\s]*([A-Z][a-z]+)", text, re.IGNORECASE)
    if match:
        candidate = match.group(3)
        if candidate.lower() not in TOXIC_WORDS:
            user_name = candidate
    return user_name

# ==================== PROMPT ==========================
def get_personalized_prompt():
    greeting_line = f"Halo {user_name}," if user_name else "Halo,"
    prompt_template = f"""{greeting_line} kamu adalah asisten cerdas dan ramah.

Tugas kamu adalah membantu pengguna memahami isi dokumen PDF yang telah diunggah.

Berikut panduan menjawab:
1. Jika pengguna bertanya tentang slide tertentu (misalnya: "Slide ke-11"), carilah halaman dalam dokumen yang diawali dengan "Slide 11:" — atau bagian lain yang relevan secara konteks dan urutan isi.
2. Jika pengguna hanya bertanya pengertian suatu istilah (misalnya: "apa itu MoU", "jelaskan kontrak kerja") dan tidak menyebut nomor slide, berikan jawaban singkat berdasarkan isi dokumen tanpa perlu mencari slide tertentu.
3. Jika pengguna menyebut pengertian sekaligus nomor slide, tampilkan keduanya: pengertian secara umum terlebih dahulu, lalu lanjutkan dengan penjelasan isi dari slide terkait.

⚠️ Penting:
- Jangan membuat atau menyusun ulang pertanyaan.
- Jangan menambahkan pertanyaan baru.
- Jawab hanya berdasarkan pertanyaan asli dari pengguna.
- Jawaban harus merujuk isi dokumen, jangan mengarang.

---------------------
Dokumen Konteks:
{{context}}

Riwayat Percakapan:
{{chat_history}}

Pertanyaan:
{{question}}

Jawaban:
"""
    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=prompt_template
    )

# =========== LOADER: PER SLIDE (PER PAGE) ============
def load_pdfs(files):
    docs = []
    for uploaded_file in files:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()

        for i, page in enumerate(pages):
            slide_num = i + 1
            filename = uploaded_file.name.lower()

            page.page_content = (
                f"Slide ke-{slide_num}:\n"
                f"Slide ke-{slide_num} dari dokumen '{filename}':\n"
                f"{page.page_content}"
            )
            page.metadata["slide_number"] = slide_num
            page.metadata["filename"] = filename
            docs.append(page)

            print(f"✅ Loaded Slide ke-{slide_num} from '{filename}'")

    return docs

# ============== CARI SLIDE SECARA LANGSUNG ============
def search_slide_by_number(slide_num, filename, docs):
    filename = filename.lower()
    for doc in docs:
        if (
            doc.metadata.get("slide_number") == slide_num and
            filename in doc.metadata.get("filename", "")
        ):
            return doc.page_content
    return None

def find_slide_containing_keyword(keyword, filename, docs):
    keyword = keyword.lower()
    filename = filename.lower()
    for doc in docs:
        if (
            filename in doc.metadata.get("filename", "").lower()
            and keyword in doc.page_content.lower()
        ):
            slide_number = doc.metadata.get("slide_number", "?")
            return f"Keyword '{keyword}' ditemukan di slide ke-{slide_number} dari dokumen '{filename}'.\n\n{doc.page_content}"
    return f"Maaf, tidak ditemukan slide yang membahas '{keyword}' di dokumen '{filename}'."

def guess_relevant_filename(keyword, docs):
    keyword = keyword.lower()
    for doc in docs:
        if keyword in doc.page_content.lower():
            return doc.metadata.get("filename", "")
    return ""

# =============== OLLAMA EMBEDDING =====================
class OllamaEmbeddings(Embeddings):
    def __init__(self, model="nomic-embed-text"):
        self.model = model
        self.url = "http://localhost:11434/api/embeddings"

    def embed_documents(self, texts):
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text):
        return self._get_embedding(text)

    def _get_embedding(self, text):
        response = requests.post(self.url, json={"model": self.model, "prompt": text})
        response.raise_for_status()
        return response.json()["embedding"]

# ===================== QA CHAIN =======================
def create_qa_chain(docs):
    splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.from_documents(chunks, embeddings)

    llm = OllamaLLM(model="llama3.2")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = get_personalized_prompt()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return qa_chain, docs

# ==================== ROUTER FUNGSIONAL ====================
def query_router(query, qa_chain, docs, uploaded_files):
    slide_match = re.search(r"slide(?:\s*ke|-)?\s*(\d+)", query, re.IGNORECASE)
    doc_match = re.search(r"dokumen\s+(.+?)(\.pdf)?[\s\?]?", query, re.IGNORECASE)

    if slide_match:
        slide_num = int(slide_match.group(1))
        filename_guess = None

        if doc_match:
            filename_guess = doc_match.group(1).strip().lower()
        elif uploaded_files:
            filename_guess = uploaded_files[0].name.lower()

        if filename_guess:
            matched_docs = [
                doc for doc in docs
                if filename_guess in doc.metadata.get("filename", "").lower()
                and doc.metadata.get("slide_number") == slide_num
            ]
            if matched_docs:
                return matched_docs[0].page_content
            else:
                return f"Maaf, slide ke-{slide_num} dari dokumen '{filename_guess}' tidak ditemukan."

    keyword_slide_match = re.search(
        r"slide\s*ke\s*(berapa|berapa\s*ya)?\s*(?:yang\s*)?(?:membahas|berisi|tentang)?\s*(?:topik|materi)?\s*(mou|pakta integritas|kontrak kerja|aspek pengadaan|cism)\b",
        query.lower()
    )

    if keyword_slide_match:
        keyword = keyword_slide_match.group(2)
        filename_guess = guess_relevant_filename(keyword, docs)
        if not filename_guess:
            filename_guess = uploaded_files[0].name.lower() if uploaded_files else ""
        return find_slide_containing_keyword(keyword, filename_guess, docs)

    response = qa_chain.invoke({"question": query})
    return response.get("answer", "Maaf, tidak ada jawaban.")

# ==================== CHUNK & VECTOR VIEWER ====================
def get_chunks_and_vectors(docs, embeddings, n=5):
    """
    Mengambil n chunk teks beserta vektor embeddingnya.
    """
    texts = [doc.page_content for doc in docs[:n]]
    vectors = [embeddings.embed_query(t) for t in texts]

    results = []
    for i, (t, v) in enumerate(zip(texts, vectors)):
        results.append({
            "chunk_id": i + 1,
            "text": t[:300] + ("..." if len(t) > 300 else ""),
            "vector": v[:10]  # tampilkan hanya 10 dimensi pertama
        })
    return results
