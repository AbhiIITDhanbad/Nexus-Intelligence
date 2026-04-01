import os
import shutil
import json
from typing import List
from pathlib import Path
import sys
from pathlib import Path 

import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

path_root = Path(__file__).resolve().parent.parent
sys.path.append(str(path_root))

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader

from orchestrator.main import Orchestrator
app = FastAPI(title="BI RAG API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH =  Path(__file__).parent.parent.parent / "config" / "agents.yaml"  # Adjust it according your config file location
orchestrator = Orchestrator(CONFIG_PATH)


UPLOAD_DIR = Path("Knowledge Base")
UPLOAD_DIR.mkdir(exist_ok=True)

class UploadResponse(BaseModel):
    message: str
    total_chunks: int


@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Ingest PDFs, extract text, and pass to Orchestrator to build the Vector DB.
    """
    all_loaded_docs = []

    for file in files:
        # 1. Save file locally temporarily
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Load the PDF using Langchain
        try:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            all_loaded_docs.extend(docs)
        except Exception as e:
            print(f"❌ Error loading {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process {file.filename}")

    if not all_loaded_docs:
        raise HTTPException(status_code=400, detail="No readable text found in uploaded files.")

    orchestrator.load_context(all_loaded_docs)

    return UploadResponse(
        message="Documents successfully ingested. Vector database primed and workflow ready.",
        total_chunks=len(all_loaded_docs)
    )

@app.websocket("/api/v1/chat/ws")
async def chat_websocket(websocket: WebSocket):
    """
    WebSocket endpoint that streams the LangGraph execution in real-time.
    """
    await websocket.accept()

    if not orchestrator.workflow:
        await websocket.send_json({
            "type": "error",
            "content": "No context loaded. Please upload documents first."
        })
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_text()
            query_data = json.loads(data)
            user_query = query_data.get("query")

            if not user_query:
                continue
            await websocket.send_json({
                "type": "agent_update",
                "agent": "System",
                "status": "Query received. Routing to agents..."
            })

            # 2. Stream the graph execution
            async for payload in orchestrator.stream_query(user_query, user_id="web_user"):
                await websocket.send_json(payload)

    except WebSocketDisconnect:
        print("🔌 Client disconnected from WebSocket.")
    except Exception as e:
        print(f"❌ WebSocket connection error: {e}")
        try:
            await websocket.send_json({"type": "error", "content": "Internal server error during stream."})
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting API Server...")
    print("Open http://127.0.0.1:8000/docs ")
    uvicorn.run(app, host="0.0.0.0")