# updated script
"""
Document Retrieval Agent: Retrieves relevant documents using enhanced hybrid search.
Combines vector similarity, BM25, query expansion, and reranking.
"""
import sys
import os
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
# from langchain_ollama import ChatOllama  # Required for query expansion
from langchain_community.document_loaders import PyPDFLoader , TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_chroma import Chroma
# from langchain_ollama import OllamaLLM , OllamaEmbeddings, ChatOllama
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint , HuggingFaceEndpointEmbeddings
from pydantic import BaseModel,Field
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.base_agent import BaseAgent
from graph.state import AgentState, add_error_to_state
# FIX: Import the new enhanced retriever factory
from retrieval.enhanced_hybrid_search import create_enhanced_retriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# EMBEDDING_MODEL = "nomic-embed-text:latest"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# LLM_MODEL = "llama3.2:1b"
EMBEDDINGS = HuggingFaceEndpointEmbeddings(repo_id=EMBEDDING_MODEL)
LOWER_TH = 0.5
UPPER_TH = 0.7
class DocEvalScore(BaseModel):
    '''
    Evaluate the Contexts provided to answer
    the query
    '''
    score: float = Field(description="Carefully Score the document retrieved contexts in order to answer the query",gt=-0.1,lt=1.1)
    reason: str = Field(description="Give a two-line key reasons behind your desicion")

class ExpandQuery(BaseModel):
    #"expanded_query": "original query term1 term2", "expansion_terms": ["term_1", "term_2",...,"term_n"]
    '''
    Query Expansion Schema
    '''
    expanded_query:str=Field(description="Return the enriched expanded query")
    expansion_terms:Optional[List[str]]=Field(default=[""],description="The expanded or newly added terms")


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
        self._initialize_split_store()

    
    def _initialize_split_store(self):
        """Initialize the enhanced hybrid retriever."""
        try:
            self.chunks = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=130).split_documents(self.docs)
            self.vector_store = Chroma.from_documents(
                    documents=self.chunks,
                    embedding=EMBEDDINGS
                )
            print("vector store created")
            model_name = "openai/gpt-oss-120b"
            self.query_expansion_llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
                repo_id=model_name,
                task="text-generation"
            ))
            print("llm created")       
        except Exception as e:
            print(f"❌ Failed to initialize retriever: {e}")
            import traceback
            traceback.print_exc()
            self.retriever = None
    def expand_query(self,query:str)->str:
            """
            Enrich the query
            """
            try:
                parser=PydanticOutputParser(pydantic_object=ExpandQuery)
                prompt=PromptTemplate(
                    template='''You are a query enrichment expert aiming high quality retrieval from Document.  
                                Don't OverComplicate the query while expanding and most importnatly don't miss the key intent of the user query.
                                Always target for more clear emphasized query after expansion. 
                                <Query>{query}</Query> \n {format_instruction}''',
                    input_variables=["query"],
                    partial_variables={'format_instruction':parser.get_format_instructions()}
                )
                chain= prompt | self.query_expansion_llm | parser
            
                result=chain.invoke({"query":query})
                return result.expanded_query
            except:
                print("Failed for query expansion")
                return query

    async def _execute_impl(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant documents using enhanced search.
        """
        print(f"🔍 DocRetrievalAgent searching for relevant documents...")
        
        try:
            # Get query from state
            query = state.get("query", "")
            print("Got the query")
            query=self.expand_query(query)
            state["query"]=query
            print("expanded the query")
            
            if not query:
                print("⚠️  No query found in state")
                state["retrieved_documents"] = []
                return state
            bm25_retriever = BM25Retriever.from_documents(documents=self.chunks)
            bm25_retriever.k =  6  # Retrieve top 2 results
            vector_store_retriever=self.vector_store.as_retriever(search_type="mmr",search_kwargs={"k": 6, "fetch_k": 20,"lambda_mult": 0.4})
            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_store_retriever],
                            weights=[0.3, 0.7])
            documents = ensemble_retriever.invoke(query)
            print("docs retrieved")
            state["docs"] = documents  
            parser=PydanticOutputParser(pydantic_object=DocEvalScore)         
            doc_eval_prompt=PromptTemplate(
                template='''
You are an expert Retrieval Quality Evaluator for a Corrective Retrieval-Augmented Generation (CRAG) system.
Your task is NOT to determine whether the chunk alone answers the question.
Instead, evaluate how useful this retrieved chunk is for constructing a complete, factually grounded answer.
Consider the following:

1. Relevance
- Does the chunk discuss concepts asked in the question?
2. Information Contribution
- Does the chunk contribute unique facts, evidence, definitions, examples, procedures or explanations?
3. Coverage Contribution
- Does the chunk cover one or more aspects required by the question?
- Multi-part questions often require multiple complementary chunks.
4. Redundancy
- Ignore whether other chunks may exist.
- Judge this chunk independently.
5. Noise
- Penalize boilerplate, unrelated text, advertisements, references or generic introductions.
Scoring:
1.0
Essential.
Contains critical information directly required for answering one or more important parts of the question.
0.8-0.9
Highly relevant.
Contains substantial supporting evidence or explanations.
0.6-0.7
Moderately useful.
Contributes partial information but is not central.
0.3-0.5
Marginally relevant.
Contains background information or weakly related concepts.
0.0-0.2
Irrelevant.
IMPORTANT:
Do NOT reduce the score simply because the chunk cannot answer the entire question.
A chunk covering one important aspect of a multi-hop question may still deserve a score close to 1.0.
Question:
{query}
Retrieved Chunks:
{chunk} \n {format_instruction}
''',
            input_variables=["query","chunk"],
            partial_variables={'format_instruction':parser.get_format_instructions()}
)
            print("doc evaluation started")
            doc_eval_chain = doc_eval_prompt | self.query_expansion_llm | parser
            # q = state["question"]
            scores: List[float] = []
            good: List[Dict] = [] #changed to Any
            state["retrieved_documents"] = documents
            batch_inputs = [
                {
                    "query": query,
                    "chunk": doc.page_content
                }
                for doc in documents
            ]
            results = doc_eval_chain.batch(batch_inputs)
            print("got the results")
            for doc, result in zip(documents, results):
                scores.append(result.score)

                if result.score >= LOWER_TH:
                    good.append({
                        "document": doc,
                        "score": result.score,
                        "reason": result.reason
                    })
            high_count = sum(score >= UPPER_TH for score in scores)
            medium_count = sum(LOWER_TH <= score < UPPER_TH for score in scores)
            low_count = sum(score < LOWER_TH for score in scores)

            avg_score = sum(scores) / len(scores) if scores else 0.0
            if high_count >= 2:
                verdict = "CORRECT"
                reason = f"{high_count} chunks scored above {UPPER_TH}."

            elif high_count == 0 and medium_count == 0:
                verdict = "INCORRECT"
                reason = f"All retrieved chunks scored below {LOWER_TH}."

            else:
                verdict = "AMBIGUOUS"
                reason = (
                    f"{high_count} high-quality, "
                    f"{medium_count} moderately relevant, "
                    f"{low_count} low-quality chunks."
                )

            # ---------- Update State ----------
            state["retrieved_documents"] = documents
            # state["good_docs"] = good_docs
            state["good_docs"] = [
                item["document"]
                for item in sorted(
                    good,
                    key=lambda x: x["score"],
                    reverse=True
                )
            ]
            state["verdict"] = verdict
            state["reason"] = reason
            print("doc_eval_completed")
            print(f"✅ Retrieved {len(documents)} documents")
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
    state = {
        "query": "Who was Swami Vivekananda ?",
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
    print("sending the state")
    # Execute
    result = await agent.execute(state)
    
    print(result.get("good_docs","Some error"))
    print("*"*50)
    print(result.get("verdict","Some error"))
if __name__ == "__main__":
    asyncio.run(test_doc_retrieval_agent())
