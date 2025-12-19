import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
from typing import List, Dict
import time

class SemanticSearchEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        print("Initializing ChromaDB")
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="movies",
            metadata={"description": "Movie embeddings for semantic search"}
        )
        
    def prepare_documents(self, df: pd.DataFrame) -> List[Dict]:
        documents = []
        
        for _, row in df.iterrows():
            # Use pre-processed search_text if available
            if 'search_text' in row and pd.notna(row['search_text']):
                search_text = row['search_text']
            else:
                # Fallback: create basic search text
                genres_str = row.get('genres_str', '')
                search_text = f"{row['title']} {row['title']} {row['overview']} Genres: {genres_str}"
            
            # Build metadata
            metadata = {
                'title': row['title'],
                'overview': row['overview'],
                'year': int(row['year']) if pd.notna(row.get('year')) else 0,
                'rating': float(row['rating']) if pd.notna(row.get('rating')) else 0.0
            }
            
            # Add optional fields
            if 'genres_str' in row:
                metadata['genres'] = row['genres_str']
            if 'cast_str' in row and pd.notna(row['cast_str']):
                metadata['cast'] = row['cast_str']
            if 'director' in row and pd.notna(row['director']):
                metadata['director'] = row['director']
            if 'keywords_str' in row and pd.notna(row['keywords_str']):
                metadata['keywords'] = row['keywords_str']
            if 'popularity' in row and pd.notna(row['popularity']):
                metadata['popularity'] = float(row['popularity'])
            
            documents.append({
                'id': str(row['id']),
                'text': search_text,
                'metadata': metadata
            })
        
        return documents
    
    def index_movies(self, df: pd.DataFrame):
        print("\nPreparing documents")
        documents = self.prepare_documents(df)
        
        print(f"Generating embeddings for {len(documents)} movies")
        start_time = time.time()
        
        # Extract texts for embedding
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings in batch
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Store in ChromaDB
        self.collection.add(
            ids=[doc['id'] for doc in documents],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[doc['metadata'] for doc in documents]
        )
        
        elapsed = time.time() - start_time
        print(f"Indexed {len(documents)} movies in {elapsed:.2f}s")
        print(f"  Average: {elapsed/len(documents)*1000:.1f}ms per movie")
        
    def search(self, query: str, top_k: int = 5, filter_year: int = None, 
               filter_genre: str = None, min_rating: float = None) -> List[Dict]:
        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Build where clause for filtering
        where_clause = {}
        if filter_year:
            where_clause["year"] = {"$gte": filter_year}
        if min_rating:
            where_clause["rating"] = {"$gte": min_rating}
        
        # Increase results if we need to filter genres
        n_results = top_k * 3 if filter_genre else top_k
        
        # Search vector database
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        # Format results
        matches = []
        if results['ids'][0]:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                
                # Post-filter by genre if needed
                if filter_genre:
                    genres = metadata.get('genres', '').lower()
                    if filter_genre.lower() not in genres:
                        continue
                
                match = {
                    'id': results['ids'][0][i],
                    'title': metadata['title'],
                    'overview': metadata['overview'],
                    'year': metadata['year'],
                    'rating': metadata['rating'],
                    'similarity_score': 1 - results['distances'][0][i]
                }
                
                # Add optional fields
                if 'genres' in metadata:
                    match['genres'] = metadata['genres']
                if 'cast' in metadata:
                    match['cast'] = metadata['cast']
                if 'director' in metadata:
                    match['director'] = metadata['director']
                if 'keywords' in metadata:
                    match['keywords'] = metadata['keywords']
                if 'popularity' in metadata:
                    match['popularity'] = metadata['popularity']
                
                matches.append(match)
                
                # Stop when we have enough results
                if len(matches) >= top_k:
                    break
        
        return matches
    
    def get_stats(self):
        count = self.collection.count()
        return {
            'total_movies': count,
            'model': self.model.get_sentence_embedding_dimension()
        }

# Example usage and testing
if __name__ == "__main__":
    # Load processed TMDB data
    try:
        df = pd.read_csv('movies_processed.csv')
        print(f"Loaded {len(df)} movies from processed dataset")
    except FileNotFoundError:
        print("Error: movies_processed.csv not found!")
        exit(1)
    
    # Initialize search engine
    print("\nInitializing semantic search engine")
    engine = SemanticSearchEngine()
    
    # Index movies
    engine.index_movies(df)
    
    # Test queries
    test_queries = [
        "psychological thriller about memory and identity",
        "heartwarming movie about family and friendship",
        "sci-fi about artificial intelligence",
        "action movie with incredible visual effects",
        "romantic comedy set in Paris"
    ]

    for query in test_queries[:3]:  # Test first 3 queries
        print(f"\nQuery: '{query}'\n")
        results = engine.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']} ({result.get('year', 'N/A')})")
            print(f"   Similarity: {result['similarity_score']:.3f}")
            print(f"   Rating: {result['rating']}/10")
            if 'genres' in result:
                print(f"   Genres: {result['genres']}")
            if 'cast' in result and result['cast']:
                print(f"   Cast: {result['cast']}")
            print(f"   {result['overview'][:120]}")
    
    # Show stats
    stats = engine.get_stats()
    print(f"\n\nStats: {stats['total_movies']} movies indexed")
    print(f"Embedding dimension: {stats['model']}")