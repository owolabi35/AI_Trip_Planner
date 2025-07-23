from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agent.agentic_workflow import GraphBuilder
from utils.save_to_document import save_document
from starlette.responses import JSONResponse
import os
import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_travel_agent(query: QueryRequest):
    try:
        print(query)
        graph = GraphBuilder(model_provider="groq")
        react_app = graph()
        
        # Safely attempt to draw and save the graph if the method exists
        try:
            graph_obj = getattr(react_app, "get_graph", None)
            if callable(graph_obj):
                graph_instance = graph_obj()
                draw_png = getattr(graph_instance, "draw_mermaid_png", None)
                if callable(draw_png):
                    png_graph = draw_png()
                    with open("my_graph.png", "wb") as f:
                        f.write(png_graph)
                    print(f"Graph saved as 'my_graph.png' in {os.getcwd()}")
        except Exception as graph_err:
            print(f"Graph drawing error: {graph_err}")

        messages = {"messages": [query.question]}
        output = react_app.invoke(messages)

        # If result is dict with messages:
        if isinstance(output, dict) and "messages" in output and output["messages"]:
            last_message = output["messages"][-1]
            # Handle both dict and object with .content
            if isinstance(last_message, dict) and "content" in last_message:
                final_output = last_message["content"]
            elif hasattr(last_message, "content"):
                final_output = last_message.content
            else:
                final_output = str(last_message)
        else:
            final_output = str(output)
        
        return {"answer": final_output}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
