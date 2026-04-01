import sys
import os
import asyncio
import json
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any
# LangChain & Ollama Imports
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import your existing backend Orchestrator
from src.orchestrator.main import Orchestrator
# # Import your existing backend Orchestrator
# from main import Orchestrator

# --- 1. DEFINE THE JUDGE SCHEMA ---
# This forces the 120B model to reply ONLY with these specific fields
class EvaluationScore(BaseModel):
    faithfulness: float = Field(description="Score 0.0 to 1.0. Are all claims supported by the context?")
    relevancy: float = Field(description="Score 0.0 to 1.0. Does it directly answer the user's question?")
    accuracy: float = Field(description="Score 0.0 to 1.0. Does it contain the facts from the ground truth?")
    reasoning: str = Field(description="1-2 sentences explaining exactly why points were deducted, if any.")

# --- 2. BUILD THE ROBUST HYBRID EVALUATION PROMPT ---
EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert, impartial AI judge evaluating an advanced Hybrid Multi-Agent System (Local RAG + Web Research).
    You will be provided with a User Question, the Expected Ground Truth, the Combined Knowledge Base (what the system found locally and on the web), and the System's Generated Answer.
    
    Score the Generated Answer strictly from 0.0 to 1.0 on three metrics:
    
    1. Faithfulness (Anti-Hallucination): Score 1.0 if all factual claims in the Generated Answer are supported by the Combined Knowledge Base. Deduct points ONLY for true hallucinations (facts that the LLM made up out of thin air that exist nowhere in the local docs or web data).
    2. Relevancy: Score 1.0 if the answer directly, clearly, and concisely addresses the User Question. Deduct points for rambling or irrelevant tangents.
    3. Accuracy: Score 1.0 if the Generated Answer successfully captures the core factual claims present in the Expected Ground Truth.
    Strictly return in json format with keys named faithfulness(float) , relevancy(float) , accuracy(float) and resaoning(string)
    """
     ),
    ("human", """
    ### Input Data ###
    User Question: {question}
    Expected Ground Truth: {ground_truth}
    
    Combined Knowledge Base Used by System:
    {context}
    
    System's Generated Answer:
    {answer}
    """)
])

async def run_native_evaluation():
    print("🚀 Starting Native LLM-as-a-Judge Evaluation Pipeline...")

    # 1. Initialize the Judge (Chaining the prompt, model, and strict output parser)
    print("⚖️ Booting up gpt-oss:120b-cloud as the Judge...")
    raw_llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.0)
    # parser = PydanticOutputParser(pydantic_object=EvaluationScore)
    judge_chain = EVAL_PROMPT | raw_llm | JsonOutputParser()

    # 2. Initialize your Multi-Agent System
    config_path = r"C:\Downloads\RAG\Advanced_RAG\config\agents.yaml"
    orchestrator = Orchestrator(config_path)
    
    # Load Context Documents
    print("📚 Chunking and embedding context documents...")
    try:
        data_dir = Path("data")
        docs = (
     TextLoader(r"C:\Downloads\RAG\data\ai_infra_compute.txt",encoding='utf-8').load()
    + TextLoader(r"C:\Downloads\RAG\data\emerging_ai_hardware.txt",encoding='utf-8').load()
    + TextLoader(r"C:\Downloads\RAG\data\enterprise_ai.txt",encoding='utf-8').load()
    + TextLoader(r"C:\Downloads\RAG\data\llm_training_alignment.txt",encoding = 'utf-8').load()
    + TextLoader(r"C:\Downloads\RAG\data\rag_and_evaluation.txt",encoding='utf-8').load()
        )
        orchestrator.load_context(docs)
        print("✅ Vector database primed.")
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return

    # 3. Load Golden Dataset
    dataset_path = r"C:\Downloads\RAG\Advanced_RAG\benchmarks\golden_dataset2.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)

    results_data = []

    print(f"\n📋 Processing and Grading {len(golden_data)} queries...\n")

    # 4. Run Execution and Grading Loop
    for i, item in enumerate(golden_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        # requires_web = item.get("requires_web", False) # Fallback to False if you forgot to add it
        
        print(f"[{i}/{len(golden_data)}] Executing: {question[:50]}...")
        
        # A. Run through your system
        result = await orchestrator.process_query(question, user_id="evaluator")
        generated_answer = result.get("answer", "No answer generated")
        print("*"*50)
        print(generated_answer)
        print("*"*50)
        execution_path = result.get("execution_path", [])
        stats = result.get("stats", {})
        state_summary = result.get("state_summary", {})
        
        # FIX: Combine Local Docs AND Web Research into a single Knowledge Base for the Judge
        retrieved_docs = stats.get("retrieved_documents", [])
        web_research = stats.get("research_findings", "") # Adjust this key if your state calls it something else
        
        context_text = "--- LOCAL DOCUMENT EXTRACTS ---\n"
        context_text += "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No local documents retrieved."
        context_text += "\n\n--- WEB RESEARCH RESULTS ---\n"
        context_text += str(web_research) if web_research else "No web research performed."

        # B. Agentic (Python) Metrics Calculation
        # used_web = "research_agent" in execution_path
        # routing_score = 1.0 if used_web == requires_web else 0.0
        contradictions_caught = stats.get("contradictions_found", 0)

        # C. Generative (LLM Judge) Metrics Calculation
        print(f"   -> Grading response with 120B model...")
        try:
            judge_scores = judge_chain.invoke({
                "question": question,
                "ground_truth": ground_truth,
                "context": context_text,
                "answer": generated_answer
            })
            print("_"*50)
            print("REPORT")
            print(judge_scores)
            print("_"*50)
            f_score = judge_scores["faithfulness"]
            r_score = judge_scores["relevancy"]
            a_score = judge_scores["accuracy"]
            reasoning = judge_scores["reasoning"]
            
        except Exception as e:
            print(f"   -> ⚠️ LLM Judge Failed to parse output: {e}")
            f_score, r_score, a_score, reasoning = 0.0, 0.0, 0.0, "Parsing Error"

        # D. Save the row
        results_data.append({
            "Question": question,
            "Faithfulness": f_score,
            "Relevancy": r_score,
            "Accuracy (vs Ground Truth)": a_score,
            # "Routing_Score": routing_score,
            # "Used_Web": used_web,
            "Contradictions_Caught": contradictions_caught,
            "Judge_Reasoning": reasoning,
            "Execution_Path": " -> ".join(execution_path),
            "Generated_Answer": generated_answer
        })

    # 5. Export and Summarize
    final_df = pd.DataFrame(results_data)
    output_file = "native_evaluation_report.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Evaluation Complete! Report saved to '{output_file}'")
    
    print("\n" + "="*40)
    print("📊 MULTI-AGENT SYSTEM SCORECARD")
    print("="*40)
    print(f"Avg Faithfulness:       {final_df['Faithfulness'].mean():.2f} / 1.00")
    print(f"Avg Answer Relevancy:   {final_df['Relevancy'].mean():.2f} / 1.00")
    print(f"Avg Ground Truth Match: {final_df['Accuracy (vs Ground Truth)'].mean():.2f} / 1.00")
    print(f"Routing Accuracy:       {(final_df['Routing_Score'].mean() * 100):.1f}%")
    print(f"Total Contradictions:   {final_df['Contradictions_Caught'].sum()}")
    print("="*40)

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(run_native_evaluation())