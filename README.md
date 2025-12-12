# Ease — Contract Simplifier Chatbot

**Ease** is an offline-capable contract simplifier and Q&A assistant.  
It accepts contracts (PDF / DOCX / TXT) in *any language*, translates them to **English**, chunk-processes the text, simplifies legal language using TinyLlama via Ollama, and supports Q&A over the simplified output.

---

## Key features
- Accepts PDF, DOCX, TXT (multilingual).  
- Language detection + chunked translation → English.  
- Chunk-based simplification (avoids oversized prompts).  
- Q&A: searches simplified text first, falls back to TinyLlama.  
- Runs locally via **Ollama** + **TinyLlama** model.  
- Streamlit UI with multi-chat sessions and download option.

---

## Short technical definitions
- **Sentence tokenization:** split text into sentences.  
- **Word tokenization:** split sentences into words/tokens.  
- **Extractive summarization:** pick important sentences from original text to form a summary.

---

## Requirements (tested on Windows/Linux)
- Python 3.10+  
- Ollama (local runtime) — Windows, macOS, or Linux. :contentReference[oaicite:0]{index=0}  
- Enough disk space for models (TinyLlama ≈ few hundred MB → GB depending on variant). :contentReference[oaicite:1]{index=1}

---

## Python dependencies
Create a file `requirements.txt` with (example):

```
streamlit
python-docx
PyPDF2
langdetect
requests
textstat
python-pptx
```

Install inside a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> If you see `ModuleNotFoundError: No module named 'textstat'`, run:  
> `pip install textstat`

---

## Install Ollama (from scratch)

### Windows (quick)
1. Download the Windows installer from Ollama: https://ollama.com/download/windows and run it. The installer starts a local Ollama server. :contentReference[oaicite:2]{index=2}

### Linux / macOS (quick)
From a terminal:
```bash
# one-line installer
curl -fsSL https://ollama.com/install.sh | sh
```
(see Ollama docs for manual options). :contentReference[oaicite:3]{index=3}

After install, confirm Ollama server is running:
```bash
ollama status
```

---

## Put models on a secondary drive / change model location

If your main (C:) drive is small you can store models elsewhere. Ollama supports an environment variable `OLLAMA_MODELS` (or `OLLAMA_HOME` for some versions) to point to a custom models directory. Steps (Windows example):

1. Create a folder on your larger drive (e.g., `D:\ollama_models`).
2. Set environment variable `OLLAMA_MODELS` to that path:
   - Windows → System → Advanced system settings → Environment Variables → New (User variable) → `OLLAMA_MODELS` = `D:\ollama_models`
3. Move existing model files from `C:\Users\<you>\.ollama\models` to `D:\ollama_models` (or let Ollama re-download).  
4. Restart your machine / log out & log in, then restart the Ollama service/app. :contentReference[oaicite:4]{index=4}

**Alternative (Linux):** use `bind` mount (mount --bind) or set systemd service env `OLLAMA_MODELS`. See docs and community discussion. :contentReference[oaicite:5]{index=5}

---

## Pull TinyLlama model (example)
Ollama provides model library entries. Example commands:

```bash
# common TinyLlama variant (1.1B)
ollama pull tinyllama:1.1b

# or a chat / quantized variant (if present)
ollama pull tinyllama:1.1b-chat-q4_0
```

This will download the model into the Ollama models directory (control with `OLLAMA_MODELS`). :contentReference[oaicite:6]{index=6}

---

## Run the Streamlit app
From project root (virtualenv active):

```bash
# if your Streamlit script is app.py or ease_chatbot.py
streamlit run ease_chatbot_ollama.py
```

The app will contact the local Ollama API at `http://localhost:11434/api/generate`. If Ollama is not running, the app will fall back to an extractive summarizer for chunks.

---

## How the pipeline (PDF → simplified English) works (deep)
1. **Upload** PDF/DOCX/TXT.  
2. **Extract** text via `PyPDF2` (or `python-docx` for docx).  
3. **Language detect** with `langdetect`.  
4. **Chunking:** split into chunks of ~700–1200 words to avoid LLM context overflow.  
5. For each chunk:
   - **Translate** chunk → English using TinyLlama (prompted).  
   - **Simplify** the translated chunk (plain English) using TinyLlama.  
6. **Merge** chunk outputs into final simplified document.  
7. **Q&A:** when user asks question:
   - Search simplified text (fast extractive match).
   - If inconclusive, ask TinyLlama with the simplified text as context for a short answer.

---

## UI (what's included)
- File uploader (PDF / DOCX / TXT).  
- Choice to run simplification or ask Q&A.  
- Multi-chat sessions saved in Streamlit session state.  
- Download last bot response.

---

## Troubleshooting & common errors

- **`Couldn't find '...\\.ollama\\id_ed25519'. Generating new private key`** then `mkdir ... Cannot create a file when that file already exists.`  
  This happens if `.ollama` path is a file (not a folder). Check if `C:\Users\<you>\.ollama` is incorrectly a file. Rename/delete that file (if safe) so Ollama can create the folder. Alternatively, set `OLLAMA_HOME`/`OLLAMA_MODELS` to a new directory before starting Ollama. (Make a backup before deleting.)  

- **Missing python module** (e.g., `textstat`): `pip install textstat`.  

- **Ollama won't use new `OLLAMA_HOME` / `OLLAMA_MODELS`:** reboot after setting env var and ensure Ollama has permissions. On Linux set variable in systemd service or use a bind-mount. :contentReference[oaicite:7]{index=7}

---

## Example prompts (recommended)
- Translate chunk:  
  `Translate the following text into English, preserving legal terms where possible:\n\n<CHUNK_TEXT>`
- Simplify chunk:  
  `Simplify the following contract text into plain English with short bullet points or short paragraphs:\n\n<TRANSLATED_TEXT>`
- Q&A polish:  
  `Use the context below to answer the question concisely.\n\nContext: <SIMPLIFIED_TEXT>\n\nQuestion: <QUESTION>\n\nAnswer in 1-3 sentences.`

---

## Development & contributions
- Keep your `requirements.txt` up to date.  
- Add OCR (e.g., `pytesseract`) for scanned PDFs as a future improvement.  
- Consider embedding-based Q&A for better accuracy (e.g., Faiss + sentence-transformers) as next step.

---

## References / useful links
- Ollama download & docs. :contentReference[oaicite:8]{index=8}  
- TinyLlama model info in Ollama library. :contentReference[oaicite:9]{index=9}  
- `OLLAMA_MODELS` env var & moving model storage. :contentReference[oaicite:10]{index=10}

---

## License
Include your preferred license (MIT recommended for student projects).

