import json
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

SYSTEM_RULES = """
You are InsightForge, an AI Business Intelligence Assistant.

You MUST:
- Use the provided context (retrieved KPI tables) to answer.
- If the context is insufficient, say what is missing and suggest what to check next.
- Provide numeric evidence when possible.
- Produce actionable recommendations (next steps).
- Keep answers concise but insightful.

Output format:
1) Key findings (bullets)
2) Supporting evidence (numbers from context)
3) Recommendations (bullets)
"""

def get_llm(provider: str, groq_api_key: str, groq_model: str, ollama_model: str):
    if provider == "ollama":
        return ChatOllama(model=ollama_model, temperature=0.2)
    return ChatGroq(api_key=groq_api_key, model=groq_model, temperature=0.2)

def format_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        title = d.metadata.get("id", "context")
        parts.append(f"### {title}\n{d.page_content}")
    return "\n\n".join(parts)

class JSONChatMemory:
    """
    Simple persistent memory stored as a JSON list of messages.
    """
    def __init__(self, path: str, max_turns: int = 8):
        self.path = Path(path)
        self.max_turns = max_turns
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[Dict[str, str]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, messages: List[Dict[str, str]]) -> None:
        self.path.write_text(json.dumps(messages, indent=2), encoding="utf-8")

    def append(self, role: str, content: str) -> None:
        messages = self.load()
        messages.append({"role": role, "content": content})
        max_msgs = self.max_turns * 2
        messages = messages[-max_msgs:]
        self.save(messages)

    def as_lc_messages(self):
        msgs = []
        for m in self.load():
            if m["role"] == "user":
                msgs.append(HumanMessage(content=m["content"]))
            else:
                msgs.append(AIMessage(content=m["content"]))
        return msgs

def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_RULES),
        ("system", "Conversation history:\n{history}"),
        ("system", "Retrieved context (KPI tables / summaries):\n{context}"),
        ("human", "{question}")
    ])

def run_rag(llm, prompt: ChatPromptTemplate, docs: List[Document], question: str, memory: JSONChatMemory) -> str:
    history_msgs = memory.as_lc_messages()
    history_text = "\n".join([f"{m.type.upper()}: {m.content}" for m in history_msgs]) if history_msgs else "(none)"

    ctx = format_context(docs) if docs else "(no retrieved context)"

    chain_input = {
        "history": history_text,
        "context": ctx,
        "question": question
    }

    resp = llm.invoke(prompt.format_messages(**chain_input))
    answer = resp.content if hasattr(resp, "content") else str(resp)

    memory.append("user", question)
    memory.append("assistant", answer)
    return answer
