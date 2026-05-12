import os
import warnings
import gradio as gr
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# FIX: The most reliable way to import the Tavily tool in 2026
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. SILENCE WARNINGS & SETUP
warnings.filterwarnings("ignore")
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

# API Keys 
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# 2. DEFINE THE BRAIN & TOOLS
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# We use the community version but suppress the warning, 
# as it is currently the most stable across Python 3.13 environments.
search_tool = TavilySearchResults(k=3) 

# 3. DEFINE THE STATE
class AgentState(TypedDict):
    topic: str
    raw_data: str
    final_post: str

# 4. DEFINE THE NODES
def scout_node(state: AgentState):
    """The Scout: Searches for real-time news."""
    query = f"latest trending news about {state['topic']} last 24 hours"
    results = search_tool.invoke({"query": query})
    return {"raw_data": str(results)}

def writer_node(state: AgentState):
    """The Ghostwriter: Turns news into social content."""
    prompt = f"Write a viral LinkedIn post based on this news: {state['raw_data']}"
    response = llm.invoke(prompt)
    return {"final_post": response.content}

# 5. BUILD THE GRAPH
workflow = StateGraph(AgentState)
workflow.add_node("scout", scout_node)
workflow.add_node("writer", writer_node)
workflow.set_entry_point("scout")
workflow.add_edge("scout", "writer")
workflow.add_edge("writer", END)
app = workflow.compile()

# 6. UI LOGIC
def run_social_team(topic, history):
    if not os.getenv("TAVILY_API_KEY") or not os.getenv("GROQ_API_KEY"):
        return "❌ Missing API Keys in Secrets!"
    result = app.invoke({"topic": topic})
    return result.get("final_post", "Error generating post.")

demo = gr.ChatInterface(fn=run_social_team, title="📡 Social Listening AI")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, theme="soft")
