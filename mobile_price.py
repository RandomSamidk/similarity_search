from pinecone import Pinecone,ServerlessSpec
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from dotenv import load_env
import os

load_env()
api_key=os.getenv("PINECONE_API_KEY")

df=pd.read_csv('MobilePhonePrice.csv')
# Clean up column names (remove spaces, lower, etc.)
df.columns = [c.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '').lower() for c in df.columns]

# Some rows have units in the column value (e.g., "128GB", "64 GB", etc.). Let's clean these.
def clean_value(val):
    if pd.isnull(val):
        return ""
    # Remove common units and symbols, keep numbers and +, letters
    return re.sub(r'(\s?gb|\s?mp|\s?mah|\$|,|")', '', str(val), flags=re.IGNORECASE).strip()


def row_to_sentence(row):
    return (
        f"The {row.get('brand','')} {row.get('model','')} has {clean_value(row.get('storage',''))}GB storage, "
        f"{clean_value(row.get('ram',''))}GB RAM, {clean_value(row.get('screen_size_inches',''))} inch screen, "
        f"{clean_value(row.get('camera_mp',''))} MP camera(s), {clean_value(row.get('battery_capacity_mah',''))} mAh battery, "
        f"and costs ${clean_value(row.get('price_',''))}."
    )
print(df.columns)

sentences = df.apply(row_to_sentence, axis=1).tolist()

# Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences, show_progress_bar=True)

# Pinecone client
pc = Pinecone(api_key=api_key, environment="us-east-1-aws")
index_name = "phone-prices-index"

# Delete index if it exists, then create new one
if pc.has_index(index_name):
    pc.delete_index(index_name)

pc.create_index(
    name=index_name,
    dimension=384,  # for all-MiniLM-L6-v2
    metric="cosine",
    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
)

# Connect to the new index
index = pc.Index(index_name)



# Prepare upsert records
upsert_records = []
for i, (embedding, row) in enumerate(zip(embeddings, df.to_dict(orient="records"))):
    upsert_records.append({
        "id": f"{row['brand']}_{row['model']}_{i}",
        "values": embedding.tolist(),
        "metadata": row
    })

# Batch upsert for efficiency
batch_size = 100
for i in range(0, len(upsert_records), batch_size):
    batch = upsert_records[i:i+batch_size]
    index.upsert(batch)

print(f"All vectors upserted to index '{index_name}' with CSV row as metadata.")