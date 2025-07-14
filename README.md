# Pinecone Vector DB

This repository demonstrates the use of Pinecone, an advanced vector database, for various applications including semantic queries, upserts, and managing mobile phone price data. The repository utilizes Pinecone's integration with machine learning models to efficiently query and store high-dimensional data.

## Requirements

- Python 3.x
- Required packages listed in `requirements.txt`

## Setup

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/pinecone_vector_db.git
    cd pinecone_vector_db
    ```

2. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file and add your Pinecone and OpenAI API keys:

    ```plaintext
    PINECONE_API_KEY=your_pinecone_api_key
    OPENAI_API_KEY=your_openai_api_key
    ```

## Scripts

### 1. Vector DB Script

- **File:** `vector_db.py`
- **Functionality:** Initializes a Pinecone client and creates a dense index using integrated embeddings. The script upserts various records into the index for querying.

### 2. Semantic Query

- **File:** `semantic_query.py`
- **Functionality:** Uses Sentence Transformers for query embedding and performs semantic search against a Pinecone index.

### 3. Upsert Movies

- **File:** `upsert_movies.py`
- **Functionality:** Loads a CSV file with movie data, computes hybrid embeddings using OpenAI API, and upserts these records to a Pinecone index.

### 4. Mobile Phone Prices

- **File:** `mobile_price.py`
- **Functionality:** Processes mobile phone price data, cleans it, generates embeddings, and upserts them into a Pinecone index.

## Data Files

- `MobilePhonePrice.csv`: Contains data about various mobile phones and their specifications.
- `tmdb_5000_movies.csv`: Dataset of movies used to demonstrate the movie upsert functionality.

## Usage

- Run each script to perform its respective task. Ensure the necessary API keys are available in your environment.

## License

This project is licensed under the MIT License.

---

Feel free to modify this template according to your project's specifics.

