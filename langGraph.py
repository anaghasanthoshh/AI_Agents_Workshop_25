from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class State(TypedDict):
    messages : Annotated[list,add_messages]

graphbuilder=StateGraph(State)

def chat_bot(state:State):
    return {"messages":[model.invoke(state["messages"])]}

graphbuilder.add_node("chatbot",chat_bot)
graphbuilder.add_edge(START,"chatbot")
graphbuilder.add_edge("chatbot",END)

graph=graphbuilder.compile()






