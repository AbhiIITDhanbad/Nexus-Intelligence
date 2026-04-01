"""
LangGraph Workflow for Multi-Agent RAG System.
Implements the orchestration using LangGraph with conditional routing..
"""
import sys
import os
import asyncio
from typing import Dict, Any, Literal, Optional
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from graph.state import AgentState , create_initial_state
# from agents.query_analyzer import QueryAnalyzerAgent
from agents.research_agent import ResearchAgent
from agents.doc_retrieval_agent import DocRetrievalAgent
from agents.synthesis_agent import SynthesisAgent
from agents.fact_verification_agent import FactVerificationAgent
 

class MultiAgentGraph:
    """
    LangGraph implementation of the multi-agent workflow.
    """
    
    def __init__(self, config,docs):
        """Initialize the LangGraph workflow."""
        self.config = config or {}
        self.docs = docs
        self.graph = None
        self.checkpointer = None
        self.agents = {}
        
        self._initialize_agents()
        self._build_graph()
        
    def _initialize_agents(self):
        """Initialize all agents."""
        print("🤖 Initializing agents for LangGraph...")
        
        self.agents["doc_retrieval_agent"] = DocRetrievalAgent(self.config,self.docs)
        print("Node doc_retreival created")
        # Initialize ResearchAgent
        self.agents["research_agent"] = ResearchAgent(self.config)
        print("Node agent_retrieval created")
        # Initialize DocRetrievalAgent

        self.agents["fact_verification_agent"] = FactVerificationAgent(self.config)
        print("Node Fact_verification_agent created")
        # Initialize SynthesisAgent
        self.agents["synthesis_agent"] = SynthesisAgent(self.config)
        print("Node synthesis_agent created")
        print(f"✅ All {len(self.agents)} agents initialized")
    
    def _build_graph(self):
            """Build the LangGraph state machine."""
            print("🔄 Building LangGraph workflow...")
            
            # Create the graph
            workflow = StateGraph(AgentState)
            
            # Add nodes (agents)
            workflow.add_node("doc_retrieval_agent", self._run_doc_retrieval_agent)
            workflow.add_node("research_agent", self._run_research_agent)
            workflow.add_node("fact_verification_agent", self._run_fact_verification_agent)
            workflow.add_node("synthesis_agent", self._run_synthesis_agent)
            
            workflow.set_entry_point("doc_retrieval_agent")
            workflow.add_conditional_edges(
                "doc_retrieval_agent",
                self._route_after_analysis,
                {
                    "research_agent": "research_agent",
                    "synthesis_agent":"synthesis_agent"
                }
            )
            workflow.add_edge("research_agent", "fact_verification_agent")
            workflow.add_edge("fact_verification_agent", "synthesis_agent")
            workflow.add_edge("synthesis_agent", END)
            
            # Add memory for conversation
            memory = InMemorySaver()
            
            self.graph = workflow.compile(checkpointer=memory)
            
    async def _run_research_agent(self, state: AgentState) -> AgentState:
        """Run research agent."""
        print(f"\n🌐 [Graph] Running Research Agent...")
        result = await self.agents["research_agent"].execute(state)
        print(f"✅ [Graph] Research complete: {len(result.get('research_results', {}).get('findings', []))} findings")
        return result
    
    async def _run_doc_retrieval_agent(self, state: AgentState) -> AgentState:
        """Run document retrieval agent."""
        print(f"\n📚 [Graph] Running Document Retrieval Agent...")
        result = await self.agents["doc_retrieval_agent"].execute(state)
        docs_count = len(result.get("good_docs", [])) if result.get("good_docs") else 0
        print(f"✅ [Graph] Document retrieval complete: {docs_count} documents")
        return result
    
    async def _run_fact_verification_agent(self, state: AgentState) -> AgentState:
        """Run fact verification agent."""
        print(f"\n ⚖️ [Graph] Running Fact Verification Agent...")
        result = await self.agents["fact_verification_agent"].execute(state)
        
        # Log quick stats
        contradictions = result.get("contradiction_report", [])
        verified = result.get("verified_facts", [])
        print(f"✅ [Graph] Verification complete: {len(verified)} facts checked, {len(contradictions)} contradictions found")
        
        return result
    
    async def _run_synthesis_agent(self, state: AgentState) -> AgentState:
        """Run synthesis agent."""
        print(f"\n🧠 [Graph] Running Synthesis Agent...")
        result = await self.agents["synthesis_agent"].execute(state)
        confidence = result.get("confidence_score", 0)
        print(f"✅ [Graph] Synthesis complete: Confidence={confidence}")
        return result

    
    def _route_after_analysis(self, state: AgentState) -> Literal["synthesis_agent","research_agent"]:
        """Determine routing after query analysis."""
        # routing = state.get("routing_decision", "parallel")
        if state["verdict"] == "CORRECT":
            return "synthesis_agent"
        else:
            return "research_agent"
    
    async def process_query(self, query: str, user_id: Optional[str] = None) -> AgentState:
        """
        Process a query through the LangGraph workflow.
        """
        print(f"\n🚀 Processing query via LangGraph: {query[:80]}...")
        print("=" * 60)
        
        initial_state = create_initial_state(query,user_id)
        try:
            # Execute the graph
            config = {"configurable": {"thread_id": f"thread_{datetime.now().timestamp()}"}}
            result = await self.graph.ainvoke(initial_state, config)
            
            print(f"\n✅ Graph execution complete")
            print(f"📊 Final confidence: {result.get('confidence_score', 0)}")
            print(f"🚶 Execution path: {' → '.join(result.get('execution_path', []))}")
            
            return result
            
        except Exception as e:
            print(f"❌ Graph execution failed: {e}")
            import traceback
            traceback.print_exc()
            
            initial_state["errors"].append({
                "stage": "graph_execution",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            initial_state["final_answer"] = f"Graph execution failed: {str(e)[:100]}..."
            initial_state["confidence_score"] = 0.0
            
            return initial_state
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the graph structure."""
        if not self.graph:
            return {"status": "not_compiled"}
        
        nodes = list(self.graph.nodes)
        
        return {
            "status": "compiled",
            "nodes": nodes,
            "agents": list(self.agents.keys()),
            "has_checkpointer": self.graph.checkpointer is not None
        }


# Test function
def test_langgraph_workflow():
    """Test the LangGraph workflow."""
    print("🧪 Testing LangGraph Workflow")
    print("=" * 60)
    
    # Initialize workflow
    workflow = MultiAgentGraph()
    
    # Get graph info
    info = workflow.get_graph_info()
    print(f"📊 Graph Info: {info['status']}")
    print(f"   Nodes: {', '.join(info['nodes'])}")
    print(f"   Agents: {', '.join(info['agents'])}")
    
    # Test queries
    test_queries = [
        "Find the technical specification document for Tesla's Dojo D1 chip, identify its theoretical peak FP32 performance, and verify if this claim was benchmarked in an independent analysis by MLPerf.",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query[:80]}...")
        print("-"*60)
        
        result = workflow.process_query(query)
        
        print(f"\n📝 RESULT:")
        result = asyncio.run(result)
        # FIX: Safe access to final_answer to prevent TypeError
        final_answer = result.get('final_answer') or "No answer generated"
        print(f"Answer: {final_answer}...")
        
        print(f"Confidence: {result.get('confidence_score', 'N/A')}")
        print(f"Intent: {result.get('intent', 'N/A')}")
        
        # Show agent execution
        if result.get("execution_path"):
            print(f"Execution: {' → '.join(result['execution_path'])}")
        
        # Show errors if any
        if result.get("errors"):
            print(f"Errors: {len(result['errors'])}")
            for e in result['errors']:
                print(f"  - {e.get('agent', 'System')}: {e.get('error')}")
    
    print(f"\n{'='*60}")
    print("✅ LangGraph workflow test complete")


if __name__ == "__main__":
    test_langgraph_workflow()

