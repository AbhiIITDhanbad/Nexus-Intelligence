"""
Module: state.py
"""
"""
LangGraph State Definition for Multi-Agent RAG System.
This is the shared memory that flows through all agents.
"""
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class AgentState(TypedDict):
    """
    State definition for LangGraph workflow.
    All agents read from and write to this state.
    """
    # Core query information
    query: str  # Original user query
    user_id: Optional[str]  # For personalization (future)
    # question: str

    docs: List[Document] ###
    good_docs: List[Document] ###

    verdict: str ###
    reason: str ###
    
    # Agent outputs
    research_results: Optional[Dict[str, Any]]  # Research agent findings
    retrieved_documents: Optional[List[Document]]  # DocRetrieval output
    fact_check_results: Optional[List[Dict]]  # Contradictions found
    intermediate_answers: Optional[Dict[str, str]]  # Agent outputs before synthesis
    
    # Final results
    final_answer: Optional[str]  # Synthesis agent output
    confidence_score: Optional[float]  # 0.0 to 1.0
    citations: Optional[List[Dict]]  # Source citations
    
    # Metadata and tracking
    agent_timestamps: Optional[Dict[str, datetime]]  # When each agent ran
    retrieval_sources: List[str]  # Source IDs used
    errors: List[Dict]  # Any errors encountered
    execution_path: List[str]  # Order of agent execution
    
    # Performance metrics (for evaluation)
    latency_per_agent: Dict[str, float]  # Time taken by each agent
    token_usage: Dict[str, int]  # Tokens used per agent
    
    # Retrieval metadata
    retrieval_metadata: Optional[Dict[str, Any]]  # Additional retrieval info

    verified_facts: Optional[List[Dict]]  # Verified facts after validation
    contradiction_report: Optional[List[Dict]]  # Contradictions found
    verification_confidence: Optional[float]  # Overall confidence score
    source_credibility_adjustments: Optional[Dict]



# Helper function to create initial state
def create_initial_state(query: str, user_id: Optional[str] = None) -> AgentState:
    """Create initial state for a new query."""
    return {
        "query": query,
        "user_id": user_id,
        "research_results": None,
        "retrieved_documents": None,
        "docs" : [],
        "good_docs" : [],
        "verdict" : "",
        "reason" : "",
        "fact_check_results": None,
        "intermediate_answers": None,
        "final_answer": None,
        "confidence_score": None,
        "citations": None,
        "agent_timestamps": {},
        "retrieval_sources": [],
        "errors": [],
        "execution_path": [],
        "latency_per_agent": {},
        "token_usage": {},
        "retrieval_metadata": None,
        "verified_facts": None,
        "contradiction_report": None,
        "verification_confidence": None,
        "source_credibility_adjustments": None,
    }


def update_state_timestamp(state: AgentState, agent_name: str) -> AgentState:
    """Update timestamp for an agent's execution."""
    state["agent_timestamps"][agent_name] = datetime.now()
    state["execution_path"].append(agent_name)
    return state


def add_error_to_state(state: AgentState, agent_name: str, error: str, traceback: Optional[str] = None) -> AgentState:
    """Add error to state for tracking."""
    error_entry = {
        "agent": agent_name,
        "error": error,
        "timestamp": datetime.now().isoformat()
    }
    if traceback:
        error_entry["traceback"] = traceback
    
    if "errors" not in state:
        state["errors"] = []
    
    state["errors"].append(error_entry)
    return state


def validate_state(state: AgentState) -> bool:
    """Validate the state structure."""
    required_fields = ["query", "errors", "execution_path"]
    
    for field in required_fields:
        if field not in state:
            return False
    
    return True


def get_state_summary(state: AgentState) -> Dict[str, Any]:
    """Get a summary of the current state."""
    return {
        "query": state.get("query", "No query"),
        # "intent": state.get("intent", "Unknown"),
        "errors_count": len(state.get("errors", [])),
        "agents_executed": len(state.get("execution_path", [])),
        "has_research": state.get("research_results") is not None,
        "has_documents": bool(state.get("good_docs")),
        "has_answer": bool(state.get("final_answer")),
        "confidence": state.get("confidence_score", 0.0)
    }