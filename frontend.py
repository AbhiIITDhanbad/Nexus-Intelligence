import streamlit as st
import requests
import json
import websocket  


API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/api/v1/chat/ws"

st.set_page_config(
    page_title="Business Intelligence RAG", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR: DOCUMENT INGESTION ---
with st.sidebar:
    st.header("📄 Knowledge Base")
    st.write("Upload competitive analysis, 10-Ks, or market reports.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF Reports", 
        accept_multiple_files=True, 
        type=['pdf']
    )
    
    if st.button("Prime Vector Database", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Chunking & embedding documents..."):
                # Prepare files for multipart/form-data upload
                files = [
                    ("files", (file.name, file.getvalue(), "application/pdf")) 
                    for file in uploaded_files
                ]
                
                try:
                    response = requests.post(f"{API_URL}/api/v1/upload", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ {data['message']}")
                        st.caption(f"Processed {data['total_chunks']} chunks.")
                    else:
                        st.error(f"Upload failed: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Backend server is not running. Please start FastAPI.")
        else:
            st.warning("Please upload at least one PDF.")

# --- MAIN CHAT INTERFACE ---
st.title("📊 Competitive Intelligence Chat")
st.markdown("Ask deep analytical questions about your uploaded business reports. The multi-agent system will retrieve, verify, and synthesize the data.")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- CHAT INPUT & WEBSOCKET HANDLING ---
if prompt := st.chat_input("E.g., What are the projected financial benefits for Q4?"):
    
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Display assistant response
    with st.chat_message("assistant"):
        # The st.status container acts as a dropdown for the agent "thoughts"
        status_container = st.status("🤖 Multi-Agent Workflow Triggered...", expanded=True)
        answer_container = st.container()
        
        try:
            # Connect to FastAPI WebSocket
            ws = websocket.create_connection(WS_URL)
            ws.send(json.dumps({"query": prompt}))
            
            final_answer = ""
            
            # Listen to the stream
            while True:
                result = ws.recv()
                data = json.loads(result)
                
                # Handle Agent Status Updates
                if data["type"] == "agent_update":
                    agent_name = data["agent"].replace("_", " ").title()
                    status_text = f"**{agent_name}**: {data['status']}"
                    
                    # Add BI-specific context
                    if "verdict" in data:
                        color = "green" if data["verdict"] == "CORRECT" else "orange"
                        status_text += f" • Context: :{color}[{data['verdict']}]"
                    if "alert" in data:
                        status_text += f"\n🚨 **Alert:** {data['alert']}"
                        for contradiction in data.get("contradictions", []):
                            status_text += f"\n  - {contradiction}"
                            
                    status_container.write(status_text)
                
                # Handle Final Synthesis
                elif data["type"] == "final_answer":
                    final_answer = data["content"]
                    confidence = data.get("confidence", 0) * 100
                    
                    # Write to the container (it will stack them normally now!)
                    answer_container.markdown(final_answer)
                    answer_container.caption(f"**Confidence Score:** {confidence:.1f}%")
                    
                    # Close up the status dropdown nicely
                    status_container.update(
                        label="✅ Analysis Complete", 
                        state="complete", 
                        expanded=False
                    )
                    break
                    
                # Handle Errors
                elif data["type"] == "error":
                    status_container.update(label="❌ Pipeline Error", state="error")
                    answer_container.error(data["content"])
                    break
                    
            ws.close()
            
            # Save the final answer to chat history
            if final_answer:
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
        except ConnectionRefusedError:
            status_container.update(label="Connection Failed", state="error")
            st.error("Could not connect to the backend. Ensure FastAPI is running on port 8000.")
        except Exception as e:
            status_container.update(label="Execution Failed", state="error")
            st.error(f"An unexpected error occurred: {str(e)}")