import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from embedding_system import SemanticSearchEngine
from llm_integration import RAGSystem
import time

# Page config
st.set_page_config(
    page_title="Semantic Search for Movies",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .movie-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .score-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    try:
        df = pd.read_csv('movies_processed.csv')
    except FileNotFoundError:
        st.error("Error: movies_processed.csv not found!")
        st.stop()
    
    search_engine = SemanticSearchEngine()
    search_engine.index_movies(df)
    rag = RAGSystem(search_engine)
    
    return search_engine, rag, df

def display_movie_card(movie, rank):
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {rank}. {movie['title']} ({movie.get('year', 'N/A')})")
            
            if 'genres' in movie:
                st.markdown(f"**Genres:** {movie['genres']}")
            
            if 'cast' in movie and movie['cast']:
                st.markdown(f"**Cast:** {movie['cast']}")
            
            if 'director' in movie and movie['director']:
                st.markdown(f"**Director:** {movie['director']}")
            
            st.write(movie['overview'])
        
        with col2:
            st.metric("Rating", f"{movie['rating']}/10")
            st.markdown(
                f"<div class='score-badge'>Match: {movie['similarity_score']:.1%}</div>",
                unsafe_allow_html=True
            )
            
            if 'popularity' in movie:
                st.caption(f"Popularity: {movie['popularity']:.1f}")
        
        st.divider()

def create_similarity_chart(results):
    titles = [r['title'][:30] + '...' if len(r['title']) > 30 else r['title'] for r in results]
    scores = [r['similarity_score'] * 100 for r in results]
    
    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=titles,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale='Reds',
                showscale=False
            ),
            text=[f"{s:.1f}%" for s in scores],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Semantic Similarity Scores",
        xaxis_title="Match Percentage",
        yaxis_title="",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def main():
    # Header
    st.title("Semantic Search for Movies")
    st.markdown("""
    This project showcases semantic search using sentence embeddings 
    and LLM-powered result synthesis using Gemini for content discovery.
    """)
    
    # Load system
    with st.spinner("Loading search engine"):
        search_engine, rag, df = load_system()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        search_mode = st.radio(
            "Search Mode",
            ["RAG using Gemini", "Semantic Only"],
            help="RAG mode uses Gemini to synthesize results"
        )
        
        num_results = st.slider("Number of results", 3, 10, 5)
        
        year_filter = st.checkbox("Filter by year")
        if year_filter:
            min_year = st.number_input("Movies after year:", 1990, 2023, 2000)
        else:
            min_year = None
        
        st.divider()
        
        st.header("Dataset Info")
        st.metric("Total Movies", len(df))
        st.metric("Embedding Model", "all-MiniLM-L6-v2")
        st.metric("Vector Dimension", "384")
        st.metric("LLM", "Gemini 3 Flash Preview")
    
    # Main search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search for content:",
            placeholder="e.g., 'Psychological Thriller, Action, Adventure, Science Fiction'",
            help="Try natural language descriptions, not just keywords!"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Example queries
    st.markdown("Try these examples:")
    examples = [
        "Psychological thriller about memory",
        "Heartwarming movie about family",
        "Action movie with amazing visuals",
        "Romantic comedy"
    ]
    
    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with cols[i]:
            if st.button(f"'{example[:20]}...'", key=f"ex_{i}", use_container_width=True):
                query = example
                search_button = True
    
    # Search results
    if query and search_button:
        st.divider()
        
        with st.spinner("Searching..."):
            start_time = time.time()
            
            if search_mode == "RAG using Gemini":
                result = rag.generate_response(query, top_k=num_results)
                results = result['results']
                
                # LLM Response
                st.subheader("AI-Powered Recommendation")
                st.info(result['response'])
                
            else:
                results = search_engine.search(query, top_k=num_results, filter_year=min_year)
            
            search_time = time.time() - start_time
        
        # Results tabs
        tab1, tab2, tab3 = st.tabs(["Results", "Visualizations", "Technical"])
        
        with tab1:
            st.subheader(f"Found {len(results)} matches in {search_time:.2f}s")
            
            for i, movie in enumerate(results, 1):
                display_movie_card(movie, i)
        
        with tab2:
            if results:
                st.plotly_chart(create_similarity_chart(results), use_container_width=True)
                
                # Genre distribution
                genre_counts = {}
                for movie in results:
                    if 'genres' in movie and movie['genres']:
                        for genre in movie['genres'].split(', '):
                            genre = genre.strip()
                            if genre:
                                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                
                if genre_counts:
                    fig_pie = px.pie(
                        values=list(genre_counts.values()),
                        names=list(genre_counts.keys()),
                        title="Genre Distribution in Results"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab3:
            st.subheader("How it works")
            
            st.markdown("""
            Architecture:
            1. Embedding Generation: Movie descriptions > Vector embeddings
            2. Vector Search: Query embedding > ChromaDB > Similar movies
            3. LLM Synthesis: Results + Query > Gemini > Natural language response
            
            Why this is better than keyword search:
            1. Understands semantic meaning, not just exact word matches
            2. Handles synonyms and related concepts
            3. Can interpret complex, multi-faceted queries
            """)
            
            with st.expander("View Raw Search Context"):
                if search_mode == "RAG using Gemini":
                    st.code(result['context'], language='markdown')
            
            with st.expander("View Technical Metrics"):
                st.json({
                    'query': query,
                    'search_time_ms': f"{search_time * 1000:.1f}",
                    'num_results': len(results),
                    'avg_similarity': f"{sum(r['similarity_score'] for r in results) / len(results):.3f}" if results else "N/A",
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'vector_db': 'ChromaDB',
                    'llm_model': 'Gemini 1.5 Flash' if search_mode == "RAG using Gemini" else 'N/A'
                })

if __name__ == "__main__":
    main()