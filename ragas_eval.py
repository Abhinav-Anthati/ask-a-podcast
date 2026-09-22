"""Evaluation script for Ask-a-Podcast using RAGAS."""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from run_eval import results

answerable_flags = results.pop("answerable")

dataset = Dataset.from_dict(results)

judge_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-haiku-4-5-20251001"))
judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

score = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=judge_embeddings,
)

df = score.to_pandas()
df["answerable"] = answerable_flags

answerable_df = df[df["answerable"]]
unanswerable_df = df[~df["answerable"]]

print("=== ANSWERABLE QUESTIONS (real content in corpus) ===")
print(f"Count: {len(answerable_df)}")
print(f"Faithfulness:       {answerable_df['faithfulness'].mean():.3f}")
print(f"Answer Relevancy:   {answerable_df['answer_relevancy'].mean():.3f}")
print(f"Context Precision:  {answerable_df['context_precision'].mean():.3f}")
print(f"Context Recall:     {answerable_df['context_recall'].mean():.3f}")

print("\n=== UNANSWERABLE QUESTIONS (should refuse) ===")
print(f"Count: {len(unanswerable_df)}")
print(f"Faithfulness:       {unanswerable_df['faithfulness'].mean():.3f}")

print("\n=== PER-QUESTION BREAKDOWN ===")
pd_display = df[["user_input", "answerable", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
print(pd_display.to_string(index=False))