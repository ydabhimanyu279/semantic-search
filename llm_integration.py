import os
from typing import List, Dict
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

class RAGSystem:
    def __init__(self, search_engine):
        self.search_engine = search_engine
        
        # Initialize the new GenAI Client
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            print("Warning: No GEMINI_API_KEY found.")
            self.client = None
        else:
            # NEW: Client-based initialization
            self.client = genai.Client(api_key=api_key)

            self.persona= (
                "You are a helpful movie recommendation assistant. "
                "Use the provided context to recommend 1-3 movies. "
                "Explain WHY each recommendation matches the user's query. "
            )
            
            # Define the model and default configuration once
            self.model_id = "gemini-3-flash-preview"
            self.config = types.GenerateContentConfig(
                temperature=0.7,
                # NEW: System instructions are now a formal configuration parameter
                system_instruction=self.persona
            )
            

    def format_results_for_llm(self, results: List[Dict]) -> str:
        context = "Here are the most relevant movies found:\n\n"
        
        for i, movie in enumerate(results, 1):
            context += f"{i}. **{movie['title']}** ({movie.get('year', 'N/A')})\n"
            if 'genres' in movie:
                context += f"   Genres: {movie['genres']}\n"
            context += f"   Rating: {movie['rating']}/10\n"
            
            if 'cast' in movie and movie['cast']:
                context += f"   Cast: {movie['cast']}\n"
            
            if 'director' in movie and movie['director']:
                context += f"   Director: {movie['director']}\n"
            
            context += f"   Overview: {movie['overview']}\n"
            context += f"   Relevance Score: {movie['similarity_score']:.2%}\n\n"
        
        return context
    
    def generate_response(self, query: str, top_k: int = 5) -> Dict:
        # Retrieve
        results = self.search_engine.search(query, top_k=top_k)
        
        if not results:
            return {
                'results': [],
                'response': "I couldn't find any movies matching your query.",
                'context': ''
            }
        
        # Format Context
        context = self.format_results_for_llm(results)
        
        # Generate
        if not self.client:
            response_text = self._generate_fallback_response(query, results)
        else:
            response_text = self._generate_gemini_response(query, context)
        
        return {
            'results': results,
            'response': response_text,
            'context': context
        }
    
    def _generate_gemini_response(self, query: str, context: str) -> str:
        try:
            prompt = f"""User Query: "{query}\n\n{context}"

{context}
"""
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=self.config
            )
            
            return response.text
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "I encountered an error while analyzing the movies. Please try again."

    def _generate_fallback_response(self, query: str, results: List[Dict]) -> str:
        response = f"Top matches for '{query}':\n\n"
        for i, movie in enumerate(results[:3], 1):
            response += f"{i}. {movie['title']} ({movie.get('year', 'N/A')}) - "
            if 'genres' in movie:
                response += f"{movie['genres']}\n"
            response += f"   {movie['overview'][:150]}...\n"
            response += f"   Match score: {movie['similarity_score']:.1%}\n\n"
        return response

# Testing
if __name__ == "__main__":
    from embedding_system import SemanticSearchEngine
    import pandas as pd
    
    # Load data and initialize
    try:
        df = pd.read_csv('movies_processed.csv')
    except FileNotFoundError:
        print("Movies_processed.csv not found!")
        exit(1)
    
    print(f"Loaded {len(df)} movies")
    
    print("\nInitializing semantic search engine")
    search_engine = SemanticSearchEngine()
    search_engine.index_movies(df)
    
    # Initialize RAG
    print("\nInitializing RAG system with Gemini")
    rag = RAGSystem(search_engine)
    
    # Test query
    query = "psychological thriller about memory and identity"
    print(f"\nQuery: {query}\n")
    
    result = rag.generate_response(query, top_k=5)
    
    print("\nGEMINI-ENHANCED RESPONSE:\n")
    print(result['response'])
    
    print("\n\nRAW SEARCH RESULTS:\n")
    for i, movie in enumerate(result['results'][:3], 1):
        print(f"{i}. {movie['title']} ({movie.get('year', 'N/A')}) - Score: {movie['similarity_score']:.3f}")