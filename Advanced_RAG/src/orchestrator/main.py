#!/usr/bin/env python3
"""
Main Orchestrator for Phase 1 Multi-Agent RAG System..
Executes the complete LangGraph workflow including:
1. Query Analyzer
2. Research Agent (Tavily)
3. Doc Retrieval Agent (Enhanced Hybrid)
4. Fact Verification Agent
5. Synthesis Agent
"""
import os
import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import time
import json
from langchain_community.document_loaders import TextLoader , PyPDFLoader
# Add src to path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph.workflow import MultiAgentGraph
from graph.state import get_state_summary
 

class Orchestrator:
    """
    Manages configuration, workflow initialization, and test execution.
    """
    def __init__(self, config_path ):
        self.config = self._load_config(config_path)
        self.docs = None
        # self.docs = None
        self.workflow = None
        # self._initialize_workflow()
    def load_context(self, docs):
        """
        Called by the FastAPI /upload endpoint. 
        Ingests the documents and builds the LangGraph and Vector DB.
        """
        print(f"\n📥 Ingesting {len(docs)} document chunks into the system...")
        self.docs = docs
        self._initialize_workflow()
        
        return {"status": "success", "message": "Context loaded and workflow initialized."}
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from YAML file with defaults."""
        default_config = {
            "llm": {
                "default_model": "gpt-oss:120b-cloud", 
                "research_model": "gpt-oss:120b-cloud",
                "synthesis_model": "gpt-oss:120b-cloud",
                "temperature": 0.0
            },
            "agents": {
                "timeout_seconds": 600,
                "max_retries": 2
            },
            "workflow": {
                "max_total_timeout": 600,
                "use_langgraph": True
            },
            "retrieval": {
                "k": 5,
                "hybrid_search": {
                    "bm25_weight": 0.25,
                    "semantic_weight": 0.75
                }
            },
            "fact_verification": {
                "contradiction_threshold": 0.7,
                "min_sources": 2
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "enable_streaming": True
            },
            "evaluation": {
                "golden_queries_path": "data/golden_queries.json",
                "metrics": ["faithfulness", "answer_relevance", "context_recall"]
            }
        }
            
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    
                    # Recursive update helper
                    def deep_update(target, source):
                        for key, value in source.items():
                            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                                deep_update(target[key], value)
                            else:
                                target[key] = value
                    
                    if loaded_config:
                        deep_update(default_config, loaded_config)
                        print(f"✅ Configuration loaded from {config_path}")
            except Exception as e:
                print(f"⚠️ Error loading config from {config_path}: {e}")
                print("Using default configuration.")
        else:
            print("ℹ️ No config file found. Using default configuration.")
        
        return default_config
    
    def _initialize_workflow(self):
        """Initialize the LangGraph workflow."""
        print("\n🔄 Initializing LangGraph workflow...")
        try:
            self.workflow = MultiAgentGraph(self.config,self.docs)
            print("✅ Workflow initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize workflow: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    async def process_query(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a query through the complete system."""
        print(f"\n🔍 Processing: {query[:80]}...")
        
        if not self.workflow:
            return {"error": "Workflow not initialized"}
        
        # Process using LangGraph
        result = await self.workflow.process_query(query, user_id)
        
        # Safe extraction helpers
        research_res = result.get("research_results") or {}
        docs = result.get("good_docs") or []
        citations = result.get("citations") or []
        errors = result.get("errors") or []
        
        # Extract Verification Data
        verified_facts = result.get("verified_facts") or []
        contradictions = result.get("contradiction_report") or []

        # Create response
        response = {
            "query": query,
            "answer": result.get("final_answer"),
            "confidence": result.get("confidence_score", 0),
            # "intent": result.get("intent"),
            "execution_path": result.get("execution_path", []),
            
            # Detailed Statistics
            "stats": {
                "research_findings": research_res.get("findings", []),
                "retrieved_documents": docs,
                "citations_count": len(citations),
                "verified_facts_count": len(verified_facts),
                "contradictions_found": len(contradictions)
            },
            
            "errors": errors,
            "state_summary": get_state_summary(result)
        }
        
        return response
    async def stream_query(self, query: str, user_id: Optional[str] = None):
            """
            Asynchronous generator that yields real-time state updates.
            """
            print(f"\n🌊 Starting stream for query: {query[:80]}...")
            
            if not self.workflow:
                yield {"type": "error", "content": "Workflow not initialized. Please load documents first."}
                return

            from graph.state import create_initial_state
            initial_state = create_initial_state(query, user_id)
            config = {"configurable": {"thread_id": f"stream_{time.time()}"}}

            try:
                async for output in self.workflow.graph.astream(initial_state, config):
                    
                    for node_name, state_update in output.items():
                        # 1. Base status update
                        payload = {
                            "type": "agent_update",
                            "agent": node_name,
                            "status": f"Agent '{node_name}' completed its task."
                        }
                        
                        # 2. Document Retrieval Intercept
                        if node_name == "doc_retrieval_agent":
                            payload["verdict"] = state_update.get("verdict")
                            payload["docs_found"] = len(state_update.get("good_docs", []))
                        
                        # 3. Verification Intercept
                        if node_name == "fact_verification_agent":
                            contradictions = state_update.get("contradiction_report", [])
                            if contradictions:
                                payload["alert"] = "Contradictions found in sources!"
                                payload["contradictions"] = contradictions

                        yield payload

                        # 4. Catch the Final Answer directly from the final node!
                        if node_name == "synthesis_agent":
                            yield {
                                "type": "final_answer",
                                "content": state_update.get("final_answer", "No final answer generated."),
                                "confidence": state_update.get("confidence_score", 0.0),
                                "citations": state_update.get("citations", []),
                                "execution_path": state_update.get("execution_path", [])
                            }

            except Exception as e:
                import traceback
                traceback.print_exc()
                yield {"type": "error", "content": f"Graph execution failed: {str(e)}"}
    async def run_golden_queries_test(self):
        """Run tests with golden queries."""
        # BASE_DIR = Path(__file__).parent.parent.parent
        GOLDEN_DATASET_PATH = Path(__file__).parent.parent.parent / "benchmarks" / "golden_dataset.json"
        with open(GOLDEN_DATASET_PATH, 'r', encoding='utf-8') as f:
            golden_data = json.load(f)
        print("\n🧪 RUNNING INTEGRATION TESTS")
        print("=" * 60)
        
        results = []
        answers = {}
        for i, test in enumerate(golden_data, 1):
            # if i==10 :
                start = time.perf_counter()
                print(f"\nTest {i}: {test['question']}...")
                print("-" * 40)
                
                result = await self.process_query(test["question"])
                stats = result.get("stats", {})
                
                # Evaluate result
                evaluation = {
                    "test_id": i,
                    "query": test["question"],
                    "has_answer": bool(result.get("answer")),
                    "confidence": result.get("confidence", 0),
                    "execution_success": len(result.get("errors", [])) == 0,
                    "facts_verified": stats.get("verified_facts_count", 0),
                    "contradictions": stats.get("contradictions_found", 0),
                    "path": result.get("execution_path", [])
                }
                end = time.perf_counter()
                results.append(evaluation)
                print(f"total time to execute the query is {end - start}")
                print(f"✅ Answer generated: {bool(result['answer'])}")
                print(f"📊 Confidence: {result['confidence']:.2f}")
                print(f"⚖️  Verification: {stats.get('verified_facts_count')} facts, {stats.get('contradictions_found')} contradictions")
                
                path_display = list(result['execution_path']) # Create a copy
                if result['answer'] and "synthesis_agent" not in path_display:
                    path_display.append("synthesis_agent (inferred)")
                    
                print(f"🚶 Path: {' → '.join(path_display)}")

                if result.get("errors"):
                    print(f"⚠️  Errors encountered: {len(result['errors'])}")
                    for err in result['errors']:
                        print(f"   - {err.get('agent', 'unknown')}: {err.get('error')}")
                
                if result["answer"]:
                    print("*"*50)
                    print(f"\n📝 answer: {result['answer']}.")
                    print("*"*50)
                    answers[test["question"]] = result['answer']
                else: print("no answer generated")

        return results
    
    async def invoke_query(self):
        result = await self.process_query(self.query,self.user_id)
        return result



    def print_system_status(self):
        """Print detailed system status."""
        print("\n" + "="*60)
        print("🤖 PHASE 1 MULTI-AGENT RAG SYSTEM - COMPLETE")
        print("="*60)
        if self.workflow:
            info = self.workflow.get_graph_info()
            print(f"Status: {info.get('status', 'Unknown')}")
            print(f"Active Agents: {', '.join(info.get('agents', []))}")
        else:
            print("Status: Workflow Not Initialized")
        print("="*60)

async def main():
    """Main entry point for Phase 1 complete system."""
    # Initialize orchestrator
    # Adjust path to where your config is actually stored

    config_path = Path(__file__).parent.parent.parent / "config" / "agents.yaml" # Route to your directory where config file is located
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
# Use your directory paths
    print("In main.py Orchestration class starting")
    orchestrator = Orchestrator(config_path,docs)

    
    # Print Status
    orchestrator.print_system_status()
    
    # Run integration tests
    await orchestrator.run_golden_queries_test()


    
    print("\n" + "=" * 60)
    print("✅ SYSTEM EXECUTION COMPLETE")
    print()

if __name__ == "__main__":
    asyncio.run(main())



    