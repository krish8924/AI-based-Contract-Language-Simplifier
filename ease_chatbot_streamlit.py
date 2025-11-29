# ease_chatbot_streamlit.py
import streamlit as st
from io import BytesIO
import docx
import PyPDF2
import re
import base64
import requests
import heapq
from langdetect import detect
import difflib
import textstat
import spacy
import json
import os
from datetime import datetime

OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "tinyllama"   
CHUNK_SIZE_WORDS = 900        
FALLBACK_SUMMARY_SENTENCES = 4

st.set_page_config(page_title="Clause Ease — Contract Simplifier", layout="wide")
st.title("📜 Ease — Contract Simplifier")


STOPWORDS = set([
    'a','an','the','and','or','in','on','at','to','is','are','was','were','be','for','of','with','as','by','that','this','it'
])

def simple_sent_tokenize(text):
    return re.split(r'(?<=[.!?])\s+', text)

def simple_word_tokenize(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

def fallback_extractive_summarize(text, max_sentences=FALLBACK_SUMMARY_SENTENCES):
    if not text or not text.strip():
        return ""
    sents = simple_sent_tokenize(text)
    if len(sents) <= max_sentences:
        return " ".join(sents)
    freq = {}
    for s in sents:
        for w in simple_word_tokenize(s):
            if w not in STOPWORDS:
                freq[w] = freq.get(w, 0) + 1
    scores = {}
    for i,s in enumerate(sents):
        tokens = simple_word_tokenize(s)
        if not tokens: continue
        scores[i] = sum(freq.get(t,0) for t in tokens) / len(tokens)
    top_idx = sorted(heapq.nlargest(max_sentences, scores, key=scores.get))
    return " ".join([sents[i] for i in top_idx])

def chunk_text(text, chunk_size_words=CHUNK_SIZE_WORDS):
    words = text.split()
    return [" ".join(words[i:i+chunk_size_words]) for i in range(0, len(words), chunk_size_words)]

def call_ollama(prompt, model=DEFAULT_MODEL, timeout=60):
    """Call ollama local HTTP API (/api/generate). Returns text or None on failure."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        r = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        # debug: st.write("Ollama call failed:", e)
        return None

def translate_chunk_to_english(chunk):
    prompt = (
        "Translate the following text into clear, natural English while preserving legal terms and meaning. "
        "If the text is already English, return it with minimal changes.\n\n"
        f"{chunk}\n\nResult:"
    )
    out = call_ollama(prompt)
    if out:
        return out.strip()
    # fallback: return chunk itself (still proceed)
    return chunk

def simplify_chunk(chunk):
    prompt = (
        "You are an assistant that simplifies legal/contract text into plain English. "
        "Produce a short, clear, bullet or paragraph style summary preserving meaning and important terms.\n\n"
        f"Text:\n{chunk}\n\nSimplified:"
    )
    out = call_ollama(prompt)
    if out:
        return out.strip()
    return fallback_extractive_summarize(chunk, max_sentences=6)

def process_document_text(full_text, chunk_size=CHUNK_SIZE_WORDS):
    """Translate & simplify document in chunks. Returns (translated_full, simplified_full)"""
    # detect language
    try:
        lang = detect(full_text)
    except:
        lang = "unknown"
    chunks = chunk_text(full_text, chunk_size_words=chunk_size)
    translated_chunks = []
    simplified_chunks = []
    for i,ch in enumerate(chunks):
        translated = translate_chunk_to_english(ch)
        translated_chunks.append(translated)
        simplified = simplify_chunk(translated)
        simplified_chunks.append(simplified)
    return "\n\n".join(translated_chunks), "\n\n".join(simplified_chunks)

def extract_clause_headings(text):
    """
    Very simple heading/ clause detector: looks for common clause keywords and
    capitalized lines (heuristic). Returns a list of (heading, sample_text).
    """
    headings = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    clause_keywords = [
        "Terminat", "Payment", "Confidential", "Governing Law", "Liabil", "Indemn",
        "Force Majeure", "Intellectual Property", "Dispute", "Notice", "Warranty",
        "Assignment", "Data Protection", "Privacy", "Breach", "Refund"
    ]
    for i,l in enumerate(lines[:400]):  # limit for speed
        # heuristic 1: uppercase short line
        if len(l) < 80 and sum(1 for c in l if c.isupper()) > (len(l)*0.3):
            headings.append((l, " ".join(lines[i+1:i+3])))
            continue
        # heuristic 2: contains keyword
        for k in clause_keywords:
            if k.lower() in l.lower():
                headings.append((l, " ".join(lines[i+1:i+3])))
                break
    # dedupe
    seen = set()
    out = []
    for h,s in headings:
        if h.lower() not in seen:
            seen.add(h.lower())
            out.append((h,s))
    return out[:40]

def extract_glossary_terms(text, top_n=20):
 
    # find candidate tokens: Capitalized words/phrases
    candidates = re.findall(r'\b[A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]{2,}){0,3}\b', text)
    freq = {}
    for c in candidates:
        freq[c] = freq.get(c,0) + 1
    sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    terms = [t for t,_ in sorted_terms[:top_n]]
    # ask Ollama to give simple definitions for terms (batch)
    if not terms:
        return {}
    prompt = "Provide a very short plain-English explanation (1-2 sentences) for each term below, using simple language:\n\n"
    for t in terms:
        prompt += f"- {t}\n"
    prompt += "\nReturn as JSON mapping term -> explanation."
    out = call_ollama(prompt)
    # try to parse JSON; otherwise produce simple fallback mapping
    glossary = {}
    if out:
        try:
            # Some models return pretty JSON; try to extract first JSON block
            js = re.search(r'\{[\s\S]*\}', out)
            if js:
                glossary = json.loads(js.group(0))
        except Exception:
            pass
    if not glossary:
        # fallback: brief autogenerated lines
        for t in terms:
            glossary[t] = "Short explanation: " + (call_ollama(f"Explain briefly what '{t}' means in a contract, in 1 sentence.") or "See clause.")
    return glossary

def compute_readability_metrics(text):
    # using textstat for quick metrics
    try:
        flesch = textstat.flesch_reading_ease(text)
        grade = textstat.text_standard(text, float_output=True)
    except Exception:
        flesch = None
        grade = None
    return {"flesch_reading_ease": flesch, "grade": grade}

def read_docx(file_bytes):
    doc = docx.Document(BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def read_pdf(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except:
            pages.append("")
    return "\n".join(pages)

def read_txt(file_bytes):
    return file_bytes.decode('utf-8', errors='ignore')


if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: {name, uploaded_at, original, translated, simplified, clauses, glossary, metrics}

if "nlp_loaded" not in st.session_state:
    st.session_state.nlp_loaded = False


if not st.session_state.nlp_loaded:
    try:
        nlp = spacy.load("en_core_web_sm")
        st.session_state.nlp_loaded = True
    except Exception:
        nlp = None
        st.session_state.nlp_loaded = False


st.sidebar.header("Upload / Sessions")
uploaded_file = st.sidebar.file_uploader("Upload contract (.txt, .docx, .pdf)", type=["txt","docx","pdf"])
if st.sidebar.button("New session / Clear view"):
    st.session_state.current_doc = None

# document list
st.sidebar.subheader("History")
for i,docmeta in enumerate(reversed(st.session_state.history[-10:])):
    idx = len(st.session_state.history) - 1 - i
    if st.sidebar.button(f"Open: {docmeta['name']} ({docmeta['uploaded_at']})", key=f"open_{idx}"):
        st.session_state.current_doc = st.session_state.history[idx]


if uploaded_file:
    file_bytes = uploaded_file.read()
    fname = uploaded_file.name
    try:
        if fname.lower().endswith(".docx"):
            full_text = read_docx(file_bytes)
        elif fname.lower().endswith(".pdf"):
            full_text = read_pdf(file_bytes)
        else:
            full_text = read_txt(file_bytes)
        st.sidebar.success(f"Uploaded: {fname}")

        
        with st.spinner("Processing (translate & simplify using TinyLlama via Ollama)... this may take a while"):
            translated, simplified = process_document_text(full_text)
            clauses = extract_clause_headings(full_text)
            glossary = extract_glossary_terms(full_text, top_n=12)
            metrics = compute_readability_metrics(simplified or translated or full_text)

        # save history entry
        entry = {
            "name": fname,
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "original": full_text,
            "translated": translated,
            "simplified": simplified,
            "clauses": clauses,
            "glossary": glossary,
            "metrics": metrics
        }
        st.session_state.history.append(entry)
        st.session_state.current_doc = entry
    except Exception as e:
        st.sidebar.error("Could not read file: " + str(e))


if "current_doc" in st.session_state and st.session_state.current_doc:
    doc = st.session_state.current_doc
    st.markdown(f"### Document: {doc['name']} — uploaded {doc['uploaded_at']}")
    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("Original (first 2000 chars)")
        st.text_area("Original Text", value=(doc['original'][:2000] + ("..." if len(doc['original'])>2000 else "")), height=300)
        st.markdown("**Detected clauses (heuristic)**")
        for h,s in doc['clauses'][:20]:
            st.markdown(f"- **{h}** — {s[:150]}")

    with col2:
        st.subheader("Simplified (English)")
        st.text_area("Simplified Result", value=doc['simplified'] or "—", height=300)
        st.markdown("**Glossary (auto)**")
        if doc['glossary']:
            for t,ex in doc['glossary'].items():
                st.markdown(f"- **{t}**: {ex}")
        else:
            st.markdown("_No glossary terms found._")
        st.markdown("**Readability**")
        st.write(doc['metrics'])

    st.markdown("---")
    # Q&A panel
    st.subheader("Ask questions about this document")
    q = st.text_input("Type a question (or select an example):", key="qa_input")
    if st.button("Get answer"):
        if not (doc.get('simplified') or doc.get('translated') or doc.get('original')):
            st.error("No processed document to answer from. Upload first.")
        else:
            # retrieval: match sentences from simplified + translated
            corpus = (doc.get('simplified','') + "\n\n" + doc.get('translated','') + "\n\n" + doc.get('original',''))
            sents = simple_sent_tokenize(corpus)
            # find best matches by token overlap
            q_tokens = set(simple_word_tokenize(q))
            scored = []
            for s in sents:
                s_tokens = set(simple_word_tokenize(s))
                if not s_tokens: continue
                score = len(q_tokens.intersection(s_tokens)) / max(1,len(q_tokens))
                scored.append((score,s))
            scored_sorted = sorted(scored, key=lambda x:x[0], reverse=True)
            top_text = " ".join([s for sc,s in scored_sorted[:6] if sc>0])
            if top_text:
                # give model prompt to produce concise answer based on retrieved text
                prompt = f"Answer the question concisely (1-3 sentences) using ONLY the context below. If uncertain, say 'Not mentioned'.\n\nContext:\n{top_text}\n\nQuestion: {q}\nAnswer:"
                ans = call_ollama(prompt)
                if not ans:
                    ans = top_text[:800] or "No answer found."
            else:
                # fallback ask full simplified doc
                prompt = f"Based on the simplified text below, answer briefly:\n\n{doc.get('simplified')}\n\nQuestion: {q}\nAnswer:"
                ans = call_ollama(prompt) or "No answer found."
            st.markdown("**Answer:**")
            st.write(ans)
            # save QA to history record
            doc.setdefault("qa", []).append({"q":q, "a":ans, "at": datetime.now().isoformat()})

    st.markdown("---")
    # download simplified
    if st.button("Download simplified as txt"):
        b = base64.b64encode((doc.get('simplified') or "").encode()).decode()
        href = f"data:file/txt;base64,{b}"
        st.markdown(f"[Download simplified file]({href})", unsafe_allow_html=True)

else:
    st.info("Upload a document from the sidebar or create a new session.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Settings**")
st.sidebar.write(f"Model: {DEFAULT_MODEL}")
st.sidebar.write(f"Ollama API: {OLLAMA_API_URL}")

st.sidebar.markdown("---")
st.sidebar.caption("Built with local TinyLlama via Ollama. Inspired by Clause_Ease project.")
