import os
import pandas as pd
import json
import time
import sys
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# ---- 1. Load API Keys from .env ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ---- 2. Load Data ----
df = pd.read_csv('tmdb_5000_movies.csv')
df = df.fillna("")

def row_to_sentence(row):
    genres = row.get('genres', '[]')
    try:
        genres = ", ".join([g['name'] for g in json.loads(genres.replace("'", '"'))])
    except:
        genres = genres
    return (
        f"Movie titled '{row.get('original_title', '')}' is a {genres} film "
        f"with a rating of {row.get('vote_average', 'N/A')} and popularity score {row.get('popularity', 'N/A')}. "
        f"Overview: {row.get('overview', '')}"
    )

sentences = df.apply(row_to_sentence, axis=1).tolist()

# ---- 3. Get Hybrid Embeddings from OpenAI ----
client = OpenAI(api_key=OPENAI_API_KEY)

print("Encoding movies with OpenAI embeddings...")
batch_size = 20
all_embeds = []
max_retries = 3
base_delay = 5

for i in range(0, len(sentences), batch_size):
    batch = sentences[i:i + batch_size]
    print(f"Processing batch {i // batch_size + 1}/{(len(sentences) + batch_size - 1) // batch_size}")

    retry_count = 0
    while retry_count <= max_retries:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=batch
            )
            for item in response.data:
                all_embeds.append({
                    "dense": item.embedding,
                    "sparse": item.sparse_embedding
                })

            if i + batch_size < len(sentences):
                print("Waiting 5 seconds before next batch...")
                time.sleep(base_delay)
            break

        except Exception as e:
            retry_count += 1
            print(f"Error: {e}")
            if retry_count == max_retries:
                user_input = input("Max retries reached. Continue with embeddings collected so far? (y/n): ")
                if user_input.lower() != 'y':
                    print("Aborting.")
                    sys.exit(1)
                else:
                    break
            else:
                wait_time = base_delay * (2 ** retry_count)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

embeds = all_embeds

# ---- 4. Pinecone Setup ----
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1-aws")
index_name = "hybrid-movies-index"

if pc.has_index(index_name):
    print("Deleting old index...")
    pc.delete_index(index_name)

print("Creating new hybrid index...")
pc.create_index(
    name=index_name,
    dimension=3072,  # text-embedding-3-large = 3072 dims
    metric="cosine",
    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
)
index = pc.Index(index_name)

# ---- 5. Prepare and Upsert Records ----
print("Preparing upsert records...")
upsert_records = []

for i, (row, emb) in enumerate(zip(df.to_dict(orient="records"), embeds)):
    dense_vector = emb["dense"]
    sparse_vector = emb.get("sparse")

    # Ensure all vector values are converted to float
    record = {
        "id": str(row["id"]),
        "values": [float(val) for val in dense_vector],
        "metadata": row
    }

    if sparse_vector is not None:
        record["sparse_values"] = sparse_vector

    upsert_records.append(record)

print("Upserting to Pinecone...")
batch_size = 100
for i in range(0, len(upsert_records), batch_size):
    batch = upsert_records[i:i + batch_size]
    index.upsert(batch)

print(f"All movies upserted to hybrid index '{index_name}' using OpenAI embeddings!")
