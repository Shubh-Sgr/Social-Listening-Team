import os
import warnings
import gradio as gr
import time
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

# --- 1. SYSTEM SETUP ---
warnings.filterwarnings("ignore")
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

# API Keys (Ensure these are set in Hugging Face Settings > Secrets)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- 2. AGENT LOGIC ---
# Using Llama 3.3 for high-speed reasoning
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
search_tool = TavilySearchResults(k=3)

class AgentState(TypedDict):
    topic: str
    raw_data: str
    final_post: str

def scout_node(state: AgentState):
    """The Scout: Searches the live web for trending data."""
    query = f"latest trending news and breakthroughs about {state['topic']} in the last 24 hours"
    results = search_tool.invoke({"query": query})
    return {"raw_data": str(results)}

def writer_node(state: AgentState):
    """The Writer: Transforms raw data into a social post."""
    prompt = f"""
    You are an expert Social Media Strategist. 
    Based on this data: {state['raw_data']}
    
    Draft a high-engagement LinkedIn post about {state['topic']}.
    Use a hook, bullet points, and hashtags. 
    Make it sound human and insightful.
    """
    response = llm.invoke(prompt)
    return {"final_post": response.content}

# Compile the Graph
workflow = StateGraph(AgentState)
workflow.add_node("scout", scout_node)
workflow.add_node("writer", writer_node)
workflow.set_entry_point("scout")
workflow.add_edge("scout", "writer")
workflow.add_edge("writer", END)
agent_app = workflow.compile()

# --- 3. UI STYLE (CSS) ---
CSS = """
.gradio-container { background-color: #0b0f19 !important; }
#title-area { text-align: center; padding: 30px 0; }
.hero-text { background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
.feature-card { background: #1f2937; padding: 20px; border-radius: 12px; border: 1px solid #374151; }
#search-log { font-family: monospace; color: #9ca3af; font-size: 0.85em; }
"""

# --- 4. THE INTERFACE ---
with gr.Blocks(css=CSS, title="TrendScout AI") as demo:
    
    # Landing Page Header
    with gr.Column(elem_id="title-area"):
        gr.HTML("""
            <div style="font-size: 40px;">📡</div>
            <h1 class="hero-text" style="font-size: 3em; margin: 0;">TrendScout AI</h1>
            <p style="color: #9ca3af; font-size: 1.1em;">Autonomous social listening & content generation.</p>
        """)

    # Explanation Grid
    with gr.Row():
        with gr.Column():
            gr.HTML("""<div class="feature-card"><h3>🔍 The Scout</h3><p>Real-time web search for the latest news.</p></div>""")
        with gr.Column():
            gr.HTML("""<div class="feature-card"><h3>✍️ The Writer</h3><p>Llama 3.3 powered social media drafting.</p></div>""")

    gr.HTML("<br>")

    # Chat UI
    with gr.Group():
        # Corrected: Removed 'type' and 'bubble_full_width' for Gradio 6.x compatibility
        chatbot = gr.Chatbot(height=450, show_label=False)
        with gr.Row():
            user_input = gr.Textbox(
                show_label=False, 
                placeholder="Enter a topic...",
                scale=4,
                container=False
            )
            submit_btn = gr.Button("Analyze 🚀", variant="primary", scale=1)

    # Log/Status Area
    with gr.Accordion("Agent Logs", open=False):
        status_log = gr.Markdown("System standing by...", elem_id="search-log")

    # --- 5. EXECUTION LOGIC ---
    def chat_process(message, history, progress=gr.Progress()):
        if not message.strip():
            return history, "Empty input."
            
        # Add User Message to History
        history.append({"role": "user", "content": message})
        yield history, "🔍 Agent 1 is scouting the web..."
        
        try:
            # Step 1: Scout
            progress(0.3, desc="Scouting...")
            result = agent_app.invoke({"topic": message})
            
            # Step 2: Writer
            progress(0.7, desc="Writing...")
            time.sleep(0.4) 
            
            # Final Output
            final_content = result.get("final_post", "Error generating post.")
            history.append({"role": "assistant", "content": final_content})
            
            yield history, f"✅ Done. Found {len(result.get('raw_data', ''))} characters of source data."
            
        except Exception as e:
            history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
            yield history, "Technical failure."

    # Bind interactions
    submit_btn.click(chat_process, [user_input, chatbot], [chatbot, status_log])
    user_input.submit(chat_process, [user_input, chatbot], [chatbot, status_log])

# --- 6. LAUNCH ---
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=int(os.environ.get("PORT", 7860)),
        theme="soft", # Apply theme here instead of constructor
        footer_links=["gradio"]
    )
