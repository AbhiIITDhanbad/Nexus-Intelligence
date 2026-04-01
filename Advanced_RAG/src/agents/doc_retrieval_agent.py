# updated script
"""
Document Retrieval Agent: Retrieves relevant documents using enhanced hybrid search.
Combines vector similarity, BM25, query expansion, and reranking.
"""
import sys
import os
import asyncio
from typing import Dict, Any, List
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import ChatOllama  # Required for query expansion
from langchain_community.document_loaders import PyPDFLoader , TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM , OllamaEmbeddings, ChatOllama
from pydantic import BaseModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.base_agent import BaseAgent
from graph.state import AgentState, add_error_to_state
# FIX: Import the new enhanced retriever factory
from retrieval.enhanced_hybrid_search import create_enhanced_retriever

EMBEDDING_MODEL = "nomic-embed-text:latest"

# LLM_MODEL = "llama3.2:1b"
EMBEDDINGS = OllamaEmbeddings(model=EMBEDDING_MODEL)
LOWER_TH = 0.3
UPPER_TH = 0.7
class DocEvalScore(BaseModel):
    score: float
    reason: str

class DocRetrievalAgent(BaseAgent):
    """
    Document Retrieval Agent using Enhanced Hybrid Search.
    
    Responsibilities:
    1. Load and manage vector database
    2. Perform query expansion (via LLM)
    3. Hybrid search (BM25 + Semantic)
    4. Reranking and filtering
    5. Provide document context
    """
    
    def __init__(self, config , docs):
        """Initialize DocRetrievalAgent."""
        super().__init__(
            name="doc_retrieval_agent",
            description="Retrieves documents using enhanced hybrid search",
            timeout_seconds=300,
            max_retries=2
        )
        
        self.config = config or {}
        self.retriever = None
        self.docs = docs
        self._initialize_retriever()

    
    def _initialize_retriever(self):
        """Initialize the enhanced hybrid retriever."""
        try:
            # 1. Get configuration
            retrieval_config = self.config.get("retrieval", {})
            chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=50).split_documents(self.docs)
            # v_db= r"C:\Downloads\RAG\LocalDatabase"
            vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=EMBEDDINGS

                    )
            
            # 2. Initialize LLM for Query Expansion
            # We reuse the default model config for the expansion task
            print("vector store created")
            llm_config = self.config.get("llm", {})
            model_name = llm_config.get("default_model", "gpt-oss:120b-cloud")
            
            self.query_expansion_llm = ChatOllama(
                model=model_name,
                temperature=0.0
            )
            
            # 3. Enhanced Search Configuration
            hybrid_config = retrieval_config.get("hybrid_search", {})
            
            # Map config to enhanced retriever parameters
            enhanced_config = {
                "bm25_weight": hybrid_config.get("bm25_weight", 0.25),
                "semantic_weight": hybrid_config.get("semantic_weight", 0.65),
                "reranker_weight": 0.10,  # Default for enhanced
                "k": 15,                  # Fetch more for reranking (candidate pool)
                "final_k": retrieval_config.get("k", 5), # Final number to return
                "enable_query_expansion": True,
                "enable_reranking": True,
                "enable_metadata_boosting": True,
                "embedding_model": retrieval_config.get("embeddings", {}).get("model", "nomic-embed-text:latest")
            }
            
            # 4. Create Retriever
            self.retriever = create_enhanced_retriever(
                vector_store=vector_store,
                llm=self.query_expansion_llm,
                config=enhanced_config
            )
            print("retriever started")
            print(f"✅ DocRetrievalAgent initialized with Enhanced Hybrid Search")
            print(f"   Config: Weights(BM25={enhanced_config['bm25_weight']}, Sem={enhanced_config['semantic_weight']}), Final K={enhanced_config['final_k']}")
            
        except Exception as e:
            print(f"❌ Failed to initialize retriever: {e}")
            import traceback
            traceback.print_exc()
            self.retriever = None
    
    async def _execute_impl(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant documents using enhanced search.
        """
        print(f"🔍 DocRetrievalAgent searching for relevant documents...")
        
        if not self.retriever:
            error_msg = "Retriever not initialized"
            print(f"❌ {error_msg}")
            state = add_error_to_state(state, self.name, error_msg)
            return state
        
        try:
            # Get query from state
            query = state.get("query", "")
            
            if not query:
                print("⚠️  No query found in state")
                state["retrieved_documents"] = []
                return state
            
            documents = await self.retriever.retrieve(query)
            print("docs retrieved")
            state["docs"] = documents
            doc_eval_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a strict retrieval evaluator for RAG.\n"
                        "You will be given ONE retrieved chunk and a question.\n"
                        "Return a relevance score in [0.0, 1.0].\n"
                        "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
                        "- 0.0: chunk is irrelevant\n"
                        "Be conservative with high scores.\n"
                        "Also return a short reason.\n"
                        "Output JSON only",
                    ),
                    ("human", "Question: {question}\n\nChunk:\n{chunk}"),
                ]
            )           
            print("doc evaluation started")
            doc_eval_chain = doc_eval_prompt | self.query_expansion_llm.with_structured_output(DocEvalScore)
            # q = state["question"]
            scores: List[float] = []
            good: List[Document] = []
            state["retrieved_documents"] = documents
            for d in documents:
                out = doc_eval_chain.invoke({"question": query, "chunk": d.page_content})
                scores.append(out.score)

                if out.score > LOWER_TH:
                    good.append(d)
            print("Checking if good or bad docs")
            # CORRECT: at least one doc > UPPER_TH
            if any(s > UPPER_TH for s in scores):
                state["good_docs"] = good
                state["verdict"] = "CORRECT"
                state["reason"] = f"At least one retrieved chunk scored > {UPPER_TH}."

            # INCORRECT: all docs < LOWER_TH
            elif len(scores) > 0 and all(s < LOWER_TH for s in scores):
                state["good_docs"] = []
                state["verdict"]= "INCORRECT"
                state["reason"] = f"All retrieved chunks scored < {LOWER_TH}."

            # AMBIGUOUS: otherwise
            else:
                state["good_docs"]= good
                state["verdict"] = "AMBIGUOUS"
                state["reason"] = f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}."
            print("doc_eval_completed")
            # Update state
            print("Retreived_documents...")
            print(documents)
            print("*"*50)
            # Log retrieval
            print(f"✅ Retrieved {len(documents)} documents")
            
            # Show sample
            if documents:
                first_doc = documents[0]
                preview = first_doc.page_content[:100].replace('\n', ' ')
                print(f"   Top result: {preview}...")
                if first_doc.metadata:
                    print(f"   Metadata: {first_doc.metadata}")
            
            # Store comprehensive retrieval metadata
            if state.get("retrieval_metadata") is None:
                state["retrieval_metadata"] = {}
            
            stats = self.retriever.get_stats()
            
            state["retrieval_metadata"].update({
                "retrieval_method": "enhanced_hybrid",
                "total_documents_retrieved": len(documents),
                "expansion_rate": stats.get("expansion_rate", 0),
                "avg_score": stats.get("avg_final_score", 0),
                "retriever_stats": stats
            })
            
            return state
            
        except Exception as e:
            error_msg = f"Document retrieval failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            state = add_error_to_state(state, self.name, error_msg)
            state["retrieved_documents"] = []
            state["good_docs"]=[]
            return state

# Test function
async def test_doc_retrieval_agent():
    """Test the DocRetrievalAgent."""
    print("🧪 Testing DocRetrievalAgent (Enhanced)...")
    
    # Initialize agent
    # Mimic structure from agents.yaml
    config = {
        "llm": {"default_model": "gpt-oss:120b-cloud"},
        "retrieval": {
            "k": 3,
            "hybrid_search": {
                "bm25_weight": 0.25,
                "semantic_weight": 0.65
            }
        }
    }
    doc1 = Path(__file__).parent.parent.parent / "data" / "ai_infra_compute.txt"
    doc2 =  Path(__file__).parent.parent.parent / "data" /"emerging_ai_hardware.txt"
    doc3 =  Path(__file__).parent.parent.parent / "data" /"enterprise_ai.txt"
    doc4 =  Path(__file__).parent.parent.parent / "data" /"llm_training_alignment.txt"
    doc5 =  Path(__file__).parent.parent.parent / "data" /"rag_and_evaluation.txt"
    docs = (
    TextLoader(doc1,encoding='utf-8').load()
    + TextLoader(doc2,encoding='utf-8').load()
    + TextLoader(doc3,encoding='utf-8').load()
    + TextLoader(doc4,encoding = 'utf-8').load()
    + TextLoader(doc5,encoding='utf-8').load()
    ) 
    agent = DocRetrievalAgent(config,docs)
    
    if not agent.retriever:
        print("❌ Agent failed to initialize retriever. Check paths.")
        return

    state = {
        "query": "NVDIA partnership at 2023",
        "user_id": "test_user",
        "retrieved_documents": None,
        "errors": [],
        "agent_timestamps": {},
        "execution_path": [],
        "retrieval_metadata": None,
        "docs":None,
        "good_docs": None,
        "verdict":"",
        "reason":""
    }
    
    # Execute
    result = await agent.execute(state)
    
    print(result)

if __name__ == "__main__":
    asyncio.run(test_doc_retrieval_agent())
