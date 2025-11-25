from sentence_transformers import SentenceTransformer, util

# Load model sekali saja
model = SentenceTransformer("all-MiniLM-L6-v2")

def benchmark_single_stsb(question: str, answer: str, all_docs: list[str]) -> float:
    # Ambil paragraf dari dokumen yang paling mirip dengan pertanyaan
    sims = util.cos_sim(model.encode(question), model.encode(all_docs))[0]
    best_index = sims.argmax().item()
    ground_truth = all_docs[best_index]

    # Hitung similarity antara jawaban RAG dan isi asli
    similarity = util.cos_sim(model.encode(answer), model.encode(ground_truth))[0][0].item()
    return similarity
