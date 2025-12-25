from typing import List, Dict
from langchain.evaluation import QAEvalChain

def run_qa_eval(llm, examples: List[Dict[str, str]], predictions: List[Dict[str, str]]):
    """
    examples: [{"query": "...", "answer": "gold reference"}]
    predictions: [{"query": "...", "result": "model answer"}]
    """
    chain = QAEvalChain.from_llm(llm)
    graded = chain.evaluate(examples, predictions)
    return graded
