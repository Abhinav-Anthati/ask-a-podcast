from eval_dataset import test_questions
from rag_graph import app_graph
import anthropic, os
from hybrid_search import rebuild_bm25_index
import psycopg2
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()
rebuild_bm25_index(cur)
cur.close()
conn.close()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

results = {"question": [], "answer": [], "contexts": [], "reference": []}

for item in test_questions:
    graph_result = app_graph.invoke({
        "question": item["question"], "podcast_url": None,
        "sub_queries": [], "rows": [], "attempt": 0
    })
    contexts = [row[1] for row in graph_result["rows"]]

    context_string = "\n\n".join(contexts)
    prompt = f"Answer using only this context. Be concise.\n\nContext:\n{context_string}\n\nQuestion: {item['question']}\n\nAnswer:"
    response = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=300,
        messages=[{"role": "user", "content": prompt}])
    answer = response.content[0].text

    results["question"].append(item["question"])
    results["answer"].append(answer)
    results["contexts"].append(contexts)
    results["reference"].append(item["reference"])

print(results)