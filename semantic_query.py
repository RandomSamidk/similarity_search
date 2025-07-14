from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os,json

load_dotenv()
api_key=os.getenv("PINECONE_API_KEY")

# Connect to Pinecone
pc = Pinecone(api_key=api_key, environment="us-east-1-aws")
index = pc.Index("updated-movies-index")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

while True:
    query = input("Enter your search (or 'exit'): ")
    if query.strip().lower() == 'exit':
        break
    query_embedding = model.encode([query])[0]
    response = index.query(
        vector=query_embedding.tolist(),
        top_k=5,
        include_metadata=True
    )

    print("\nTop Matches:")
    for match in response['matches']:
        md = match['metadata']
        # Safely parse genres from JSON-like string to list of names
        genres_raw = md.get('genres', '[]')
        try:
            genres_list = json.loads(genres_raw.replace("'", '"'))  # Some files use single quotes
            genres = ", ".join([g.get('name', '') for g in genres_list])
        except Exception:
            genres = str(genres_raw)
        print(f"Score: {match['score']:.4f}")
        print(f"Title: {md.get('original_title', 'N/A')}")
        print(f"Genres: {genres}")
        print(f"Rating: {md.get('vote_average', 'N/A')}")
        print(f"Popularity: {md.get('popularity', 'N/A')}")
        print('-' * 40)
