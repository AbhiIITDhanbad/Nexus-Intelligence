"""
Synthesis Agent: Combines verified facts, research, and documents into final answer.
Fourth core agent in the multi-agent workflow.
"""
import sys
import os
import asyncio
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_ollama import OllamaLLM , ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser , JsonOutputParser
from langchain_core.runnables import RunnableLambda
from agents.base_agent import BaseAgent
from graph.state import AgentState, add_error_to_state


class SynthesisResult(BaseModel):
    answer: str = Field(description="The comprehensive, detailed synthesized answer")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Sources used")
    contradictions: List[str] = Field(default_factory=list, description="Contradictions addressed")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made")

class SynthesisAgent(BaseAgent):
    """
    Synthesizes information into final answer, prioritizing verified facts.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="synthesis_agent",
            description="Synthesizes information into final answer",
            timeout_seconds=300,
            max_retries=2
        )
        self.config = config or {}
        self._initialize_llm()
        self._initialize_prompts()
        
    def _initialize_llm(self):
        llm_config = self.config.get("llm", {})
        model = llm_config.get("synthesis_model", "gpt-oss:120b-cloud")
        
        temp = llm_config.get("temperature", 0.1)
        if isinstance(temp, dict): temp = temp.get("default", 0.1)
        
        self.llm = ChatOllama(model=model, temperature=float(temp))
        print(f"✅ SynthesisAgent LLM initialized: {model}")
    
    def _initialize_prompts(self):
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Lead Technical Analyst.
Synthesize a comprehensive answer based on the provided VERIFIED FACTS and SOURCES.

### INPUT CONTEXT:
- **User Query**: {query}


### SOURCE DATA:
1. **VERIFIED FACTS** (True Data):
{verified_facts_summary}

2. **Contradictions**:
{contradiction_report}

3. **Raw Research**:
{raw_data_summary}

### OUTPUT FORMAT:
Return a single JSON object. Do not include markdown formatting or explanations.
Follow this exact structure:
{{
    "answer": "Detailed answer text here...",
    "confidence": Overall confidence score between 0 to 1,
    "citations": [
        {{"source_id": "doc_1", "content_referenced": "...", "credibility": 0.9}}
    ],
    "contradictions": ["List any conflicts found"],
    "assumptions": ["List any assumptions made"]
}}

### REQUIREMENTS:
1. **Detail**: Provide a detailed, multi-paragraph analysis.
2. **Data**: Include specific numbers/metrics from the Verified Facts.
3. **Accuracy**: If facts conflict, acknowledge the discrepancy.
### CRITICAL RULES:
1. If the retrieved documents do not contain the specific metric, you MUST state "Information not found in context."
2. DO NOT estimate or hallucinate future projections unless explicitly stated in the source text.             
             
             """),


            ("human", "Synthesize the final answer now.")
        ])

    
    async def execute(self, state: AgentState) -> AgentState:
        """
        Diagnostic override to catch errors that BaseAgent might swallow.
        """
        try:
            return await self._execute_impl(state)
        except Exception as e:
            print(f"❌ CRITICAL ERROR in SynthesisAgent: {e}")
            traceback.print_exc()
            state["final_answer"] = f"Agent Crashed: {str(e)}"
            state["confidence_score"] = 0.0
            state["errors"].append({"agent": self.name, "error": str(e)})
            return state

    async def _execute_impl(self, state: AgentState) -> AgentState:
        print(f"🧠 SynthesisAgent synthesizing final answer...")
        
        try:
            inputs = self._prepare_synthesis_inputs(state)
            print("synthesis inputs got")
            
            if not self._has_sufficient_information(inputs):
                print("⚠️  Insufficient information")
                return await self._provide_minimal_answer(state, inputs)
            
            self.parser = PydanticOutputParser(pydantic_object=SynthesisResult)
            
            chain = (
                self.synthesis_prompt | self.llm.with_structured_output(SynthesisResult)
            )
            print("Chaining done")
            synthesis_result = await chain.ainvoke(inputs)
            print("got results")
            
            
            # Confidence Calculation
            up_conf = state.get("verification_confidence") or 0.5
            final_conf = (up_conf * 0.4) + (synthesis_result.confidence * 0.6)
            
            # Update State
            state["final_answer"] = synthesis_result.answer
            state["confidence_score"] = float(f"{final_conf:.2f}")
            state["citations"] = synthesis_result.citations
            
            if state.get("intermediate_answers") is None:
                state["intermediate_answers"] = {}
            state["intermediate_answers"]["synthesis"] = synthesis_result.model_dump()
            
            print(f"✅ Synthesis complete. Confidence: {state['confidence_score']}")
            return state
            
        except Exception as e:
            print(f"❌ Logic Error in _execute_impl: {e}")
            traceback.print_exc()
            return await self._provide_fallback_answer(state)
    

    def _prepare_synthesis_inputs(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
        
        # 1. Facts
        facts = state.get("verified_facts") or []
        fact_summary = ""
        if facts:
            fact_summary = f"Verified Facts ({len(facts)} items):\n"
            for i, f in enumerate(facts):
                status = f.get("verification_status", "unknown")
                content = f.get("content", "")
                fact_summary += f"{i+1}. [{status}] {content}\n"
        else:
            fact_summary = "No verified facts available."
            print(fact_summary)
        
        # 2. Contradictions
        reports = state.get("contradiction_report") or []
        con_summary = "None" if not reports else str(reports)
        
        # 3. Raw Data (Backup)
        findings = (state.get("research_results") or {}).get("findings", [])
        docs = state.get("good_docs") or []
        
        raw_summary = ""
        if docs:
            raw_summary += f"\nDocuments ({len(docs)}):\n"
            for d in docs:
                raw_summary += f"- {d.page_content}...\n"
        if findings:
            raw_summary += f"\nResearch Findings ({len(findings)}):\n"
            for f in findings:
                raw_summary += f"- {f.get('content', '')}...\n"
        
        if not raw_summary:
            raw_summary = "No raw data available."
        
        return {
            "query": query,
            # "query_intent": intent,
            # "query_complexity": complexity,
            "verified_facts_summary": fact_summary,
            "contradiction_report": con_summary,
            "raw_data_summary": raw_summary
        }
    
    def _has_sufficient_information(self, inputs: Dict[str, Any]) -> bool:
        has_facts = "No verified facts" not in inputs["verified_facts_summary"]
        has_raw = "No raw data" not in inputs["raw_data_summary"]
        return has_facts or has_raw

    async def _provide_minimal_answer(self, state: AgentState, inputs: Dict[str, Any]) -> AgentState:
        state["final_answer"] = f"Insufficient data for: {inputs['query']}"
        state["confidence_score"] = 0.1
        return state

    async def _provide_fallback_answer(self, state: AgentState) -> AgentState:
        state["final_answer"] = "Error generating answer. Please check logs."
        state["confidence_score"] = 0.0
        return state

async def test_synthesis_agent():
    print("🧪 Testing SynthesisAgent (Enhanced Mode)...")
    
    config = {
        "llm": {"synthesis_model": "gpt-oss:120b-cloud", "temperature": 0.1}
    }
    agent = SynthesisAgent(config)
    
    state = {
        "query": "Write about ServishNow NVDIA partnership in 2023 in brief",

        'verified_facts': [{'fact_id': 'group_ai_nvidia', 'content': 'In a follow‑up announcement (reported May 2024), ServiceNow and NVIDIA expanded the partnership by launching a new class of enterprise AI agents called “Apriel\u202fNemotron\u202f15B.” The agents are built on a 15‑billion‑parameter LLM and were unveiled at ServiceNow’s Knowledge 2025 conference in Las Vegas, attended by roughly 5,000 partners and customers.', 'sources': [{'type': 'industry_report', 'credibility': 0.65, 'url': 'https://channelbuzz.ca/2023/05/servicenow-nvidia-announce-major-partnership-around-generative-ai-41081/'}, {'type': 'news_article', 'credibility': 0.7, 'url': 'https://m.economictimes.com/tech/artificial-intelligence/servicenow-nvidia-expand-partnership-launch-new-ai-agent/articleshow/120999550.cms'}, {'type': 'blog', 'credibility': 0.6, 'url': 'https://www.servicenow.com/platform/infrastructure/nvidia.html'}], 'verification_status': 'verified', 'confidence': 0.95, 'contradictions_found': False, 'source_count': 3}],
        "research_results": {
             'findings': [
                 {'content': 'At the ServiceNow Knowledge 2023 event in Las Vegas (May 18, 2023), ServiceNow and NVIDIA announced a major partnership to build generative AI across enterprise IT. The collaboration will use NVIDIA’s software stack, AI services, and accelerated infrastructure to create custom large‑language models (LLMs) that run on ServiceNow’s Now Platform, with an emphasis on faster model training and inference.', 
                  'source_type': 'industry_report', 'recency': '2023', 'credibility': 0.65, 'key_metrics': ['Partnership announcement date: May 18, 2023', 'Event: ServiceNow Knowledge 2023'], 'source_title': 'ServiceNow, NVIDIA announce major partnership around generative AI', 'source_url': 'https://channelbuzz.ca/2023/05/servicenow-nvidia-announce-major-partnership-around-generative-ai-41081/'}, 
                  {'content': 'In a follow‑up announcement (reported May 2024), ServiceNow and NVIDIA expanded the partnership by launching a new class of enterprise AI agents called “Apriel\u202fNemotron\u202f15B.” The agents are built on a 15‑billion‑parameter LLM and were unveiled at ServiceNow’s Knowledge 2025 conference in Las Vegas, attended by roughly 5,000 partners and customers.', 
                   'source_type': 'news_article', 'recency': '2024', 'credibility': 0.7, 'key_metrics': ['Model size: 15\u202fB parameters', 'Attendees: ~5,000 partners/customers', 'Conference: Knowledge 2025 (May 6‑8)'], 'source_title': 'ServiceNow, Nvidia expand partnership, launch new AI agent', 'source_url': 'https://m.economictimes.com/tech/artificial-intelligence/servicenow-nvidia-expand-partnership-launch-new-ai-agent/articleshow/120999550.cms'}, 
                   {'content': 'ServiceNow’s NOW platform now accesses NVIDIA NIM inference micro‑services, enhancing its own “Now\u202fLLMs.” The integration promises faster inference, scalable deployment and reduced cost per query across a variety of workflow‑automation use cases.', 
                    'source_type': 'news_article', 'recency': '2024', 'credibility': 0.7, 'key_metrics': ['Technology: NVIDIA NIM inference micro‑services', 'Benefit: Faster, scalable, cost‑effective deployment'], 'source_title': 'ServiceNow (NOW) Enhances GenAI With NVIDIA Partnership', 'source_url': 'https://finance.yahoo.com/news/servicenow-now-enhances-genai-nvidia-175300070.html'}, 
                    {'content': 'NVIDIA issued a press release outlining forward‑looking plans to co‑develop telco‑specific GenAI solutions with ServiceNow, leveraging NVIDIA’s hardware (GH200, H100) and software (NeMo, AI Enterprise). The release emphasizes joint go‑to‑market strategies for telecom operators but does not disclose concrete rollout dates.', 
                     'source_type': 'technical_spec', 'recency': '2024', 'credibility': 0.7, 'key_metrics': ['Target industry: Telecommunications', 'Hardware referenced: GH200, H100 GPUs'], 'source_title': 'ServiceNow and NVIDIA Expand Partnership With Introduction of ...', 'source_url': 'https://nvidianews.nvidia.com/news/servicenow-nvidia-telco-specific-genai-solutions'}, 
                     {'content': 'ServiceNow’s official partnership page confirms the collaboration with NVIDIA and offers a “Talk to an expert” CTA, positioning the alliance as a way to accelerate AI‑driven workflow automation. No quantitative details are provided.', 
                      'source_type': 'blog', 'recency': 'older', 'credibility': 0.6, 'key_metrics': [], 'source_title': 'ServiceNow and NVIDIA Partnership', 'source_url': 'https://www.servicenow.com/platform/infrastructure/nvidia.html'}],
        },
        'good_docs': [type('Document',(),{'metadata':{'source': 'C:\\Downloads\\RAG\\data\\enterprise_ai.txt'}, 
                                                                'page_content':'Model management, deployment, and monitoring\n\nDelta Lake + Photon Engine\n\nPerformance claims of 2–8× improvement for analytical queries\n\nDatabricks positioned itself as “AI-first data infrastructure.”\n\n3.3 Strategic Divergence\nAspect\tSnowflake\tDatabricks\nCore Identity\tData-centric\tAI-centric\nStrength\tGovernance, sharing\tML workflows\nBuyer\tRegulated enterprises\tData science teams\n'
                                                                '\nNeither platform clearly dominates; choice depends on enterprise priorities.\n\n4. Market Reaction to Enterprise AI Announcements\n'
                                                                '4.1 Why Stock Reactions Matter\n\nStock price movements capture:\n\nInvestor expectations\n\nPerceived monetization potential\n\nCompetitive validation\n\nAI announcements often trigger short-term volatility but signal long-term strategic confidence.\n\n'
                                                                '4.2 ServiceNow–NVIDIA Partnership Case Study\n\nIn November 2023, ServiceNow announced a partnership with NVIDIA to build enterprise AI solutions.'})()],
        # "errors": [],
        "final_answer": None,
        "intermediate_answers": {},
        "agent_timestamps": {},
        'contradiction_report': [], 'verification_confidence': 0.9499999999999998, 'latency_per_agent': {'fact_verification_agent': 0.013104}
    }
    
    print("\n▶️  Executing Agent...")
    result = await agent.execute(state)

    print("results")
    
    print("\n🔍  RESULT ANALYSIS:")
    answer = result.get('final_answer')
    
    if answer:
        print(f"✅ Answer Generated:\n{'-'*60}\n{answer}\n{'-'*60}")
        print(f"📊 Confidence: {result.get('confidence_score')}")
    else:
        print("❌ Answer is NONE.")

if __name__ == "__main__":
    asyncio.run(test_synthesis_agent())
