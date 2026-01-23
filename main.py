import os
import json
import shutil
import logging
import uuid
import asyncio
from typing import List, Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from pypdf import PdfReader
import docx
import pandas as pd
from sqlalchemy import create_engine

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
STATIC_DIR = "static"
UPLOADS_DIR = "uploads"
CHROMA_DIR = "chroma_db"

for d in [STATIC_DIR, UPLOADS_DIR, CHROMA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# Storage for keys and settings
CONFIG_FILE = "config.json"

# Global status for long-running ingest tasks
ingest_status: Dict[str, dict] = {}

# Initialize Embeddings (Local)
logger.info("Loading Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
logger.info("Embedding Model Loaded.")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {
        "api_keys": {},
        "persona": "You are a helpful AI assistant.",
        "active_model": "llama-3.3-70b-versatile",
        "provider": "groq",
        "use_rag": True,
        "db": {
            "enabled": False,
            "type": "sqlite",
            "host": "",
            "port": "",
            "user": "",
            "password": "",
            "dbname": ""
        }
    }

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

def get_db_uri(db_config):
    db_type = db_config.get("type", "sqlite")
    if db_type == "sqlite":
        return "sqlite:///playground.db"
    elif db_type == "mysql":
        return f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    elif db_type == "postgresql":
        return f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    return None

@app.get("/")
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>")

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []

@app.post("/chat")
async def chat(request: ChatRequest):
    config = load_config()
    provider = config.get("provider", "groq")
    api_key = config.get("api_keys", {}).get(provider)
    model_name = config.get("active_model")
    persona = config.get("persona")
    use_rag = config.get("use_rag", True)
    db_config = config.get("db", {"enabled": False})

    if not api_key:
        return JSONResponse({"error": f"API Key for {provider} not found. Please set it in settings."}, status_code=400)

    try:
        if provider == "groq":
            llm = ChatGroq(groq_api_key=api_key, model_name=model_name, timeout=60)
        elif provider == "openai":
            llm = ChatOpenAI(api_key=api_key, model_name=model_name, timeout=60)
        elif provider == "gemini":
            llm = ChatGoogleGenerativeAI(google_api_key=api_key, model=model_name, timeout=60)
        else:
            return JSONResponse({"error": "Unsupported provider"}, status_code=400)

        # SQL Injection for context
        if db_config.get("enabled"):
            try:
                db_uri = get_db_uri(db_config)
                db = SQLDatabase.from_uri(db_uri)
                db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, return_sql=True)
                db_response = db_chain.invoke(request.message)
                return {"content": f"**[Database Query Result]**\n\n{db_response['result']}\n\n*SQL used: `{db_response['sql']}`*"}
            except Exception as dbe:
                logger.warning(f"DB Query failed: {str(dbe)}")

        context = ""
        if use_rag:
            docs = vectorstore.similarity_search(request.message, k=5)
            if docs:
                context = "\n\nKnowledge Base Context:\n" + "\n---\n".join([d.page_content for d in docs])

        chart_hint = """
        IMPORTANT DATA RULES:
        1. If user asks for a chart, provide ONLY the JSON block tagged with 'CHART_DATA'.
        2. NEVER mention hex codes, colors, or 'Chart Legend' in your text response.
        3. NEVER use the words 'code', 'JSON', or 'block' in your text.
        4. Focus ONLY on the business insight.
        5. For PIE or DOUGHNUT charts: Ensure you provide an ARRAY of distinct, high-contrast hex colors for the 'backgroundColor' in the dataset.
        6. For BAR or LINE charts: Use #e76f51 (Orange) and #2a9d8f (Teal) for datasets.
        
        Example JSON:
        ```CHART_DATA
        {
          "type": "pie",
          "data": {
            "labels": ["A", "B"],
            "datasets": [{"data": [10, 20], "backgroundColor": ["#e76f51", "#2a9d8f"]}]
          }
        }
        ```
        """

        sys_prompt = f"{persona}\n\n{chart_hint}"
        if context:
            sys_prompt += f"\n\n[CRITICAL DATA CONTEXT]\n{context}\n\nUse this data to answer accurately."

        messages = [SystemMessage(content=sys_prompt)]
        # Filter history to last 10 messages for better performance and context management
        for msg in request.history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=request.message))
        
        response = llm.invoke(messages)
        return {"content": response.content}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/config")
async def get_config():
    return load_config()

@app.post("/config")
async def update_config(config: dict):
    current = load_config()
    current.update(config)
    save_config(current)
    return current

@app.post("/test_db")
async def test_db(db_config: dict):
    uri = get_db_uri(db_config)
    if not uri:
        return JSONResponse({"error": "Invalid DB Type"}, status_code=400)
    try:
        engine = create_engine(uri, connect_args={'connect_timeout': 5} if db_config["type"] != "sqlite" else {})
        with engine.connect() as conn:
            return {"status": "success", "message": "Successfully connected to the database!"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

def extract_text(file_path, filename):
    text = ""
    if filename.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    elif filename.endswith(".docx"):
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
        text = df.to_string()
    elif filename.endswith(".csv"):
        df = pd.read_csv(file_path)
        text = df.to_string()
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return text

async def process_heavy_file(task_id: str, file_path: str, filename: str):
    try:
        ingest_status[task_id] = {"status": "processing", "progress": 0, "filename": filename}
        text = extract_text(file_path, filename)
        
        if not text.strip():
            ingest_status[task_id] = {"status": "error", "message": "No text extracted"}
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            vectorstore.add_texts(texts=batch, metadatas=[{"source": filename}] * len(batch))
            ingest_status[task_id]["progress"] = int((i / len(chunks)) * 100)
            
        ingest_status[task_id] = {"status": "completed", "filename": filename, "chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Async Ingest Error: {str(e)}")
        ingest_status[task_id] = {"status": "error", "message": str(e)}

@app.post("/ingest")
async def ingest_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOADS_DIR, f"{task_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    background_tasks.add_task(process_heavy_file, task_id, file_path, file.filename)
    return {"task_id": task_id, "status": "queued"}

@app.get("/ingest/status/{task_id}")
async def get_ingest_status(task_id: str):
    status = ingest_status.get(task_id)
    if not status:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return status

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Temp upload for current message context
    temp_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOADS_DIR, f"temp_{temp_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        text = extract_text(file_path, file.filename)
        # Trim text if too large for single message context
        if len(text) > 20000:
            text = text[:20000] + "... [Text Truncated]"
    except Exception as e:
        return JSONResponse({"error": f"Failed to extract text: {str(e)}"}, status_code=500)
    return {"text": text, "filename": file.filename}

@app.delete("/knowledge_base")
async def clear_kb():
    global vectorstore
    try:
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        os.makedirs(CHROMA_DIR)
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        return {"status": "Knowledge base cleared"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
