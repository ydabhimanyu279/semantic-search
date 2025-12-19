import pandas as pd
import json
import ast
from typing import List, Dict
import numpy as np

class DataProcessor:
    def __init__(self, movies_path='tmdb_5000_movies.csv', credits_path='tmdb_5000_credits.csv'):
        self.movies_path = movies_path
        self.credits_path = credits_path
    
    def safe_parse_json(self, json_str):
        if pd.isna(json_str) or json_str == '':
            return []
        try:
            # Try parsing as JSON first
            return json.loads(json_str)
        except:
            try:
                # Try literal_eval for Python-style strings
                return ast.literal_eval(json_str)
            except:
                return []
    
    def extract_names(self, json_list, key='name', limit=None):
        if not json_list:
            return []
        
        names = [item.get(key, '') for item in json_list if isinstance(item, dict)]
        
        if limit:
            names = names[:limit]
        
        return [n for n in names if n]
    
    def load_and_process(self):
        print("Loading movies dataset\n")
        movies_df = pd.read_csv(self.movies_path)
        
        print(f"Loaded {len(movies_df)} movies")
        print(f"Columns: {movies_df.columns.tolist()}")
        
        # Load credits if available
        try:
            credits_df = pd.read_csv(self.credits_path)
            print(f"Loaded credits for {len(credits_df)} movies")
            
            # Merge on movie_id/id
            if 'movie_id' in credits_df.columns and 'id' in movies_df.columns:
                movies_df = movies_df.merge(
                    credits_df[['movie_id', 'cast', 'crew']], 
                    left_on='id', 
                    right_on='movie_id', 
                    how='left'
                )
        except FileNotFoundError:
            print("Credits file not found, proceeding without cast/crew data")
            movies_df['cast'] = None
            movies_df['crew'] = None
        
        # Process the data
        print("\nProcessing movie data")
        processed_df = self.process_movies(movies_df)
        
        print(f"\nProcessed {len(processed_df)} movies")
        print(f"  Columns: {processed_df.columns.tolist()}")
        
        return processed_df
    
    def process_movies(self, df):
        # Keep only relevant columns
        columns_to_keep = [
            'id', 'title', 'overview', 'genres', 'keywords', 
            'release_date', 'vote_average', 'vote_count', 
            'popularity', 'runtime', 'original_language'
        ]
        
        # Add cast/crew if available
        if 'cast' in df.columns:
            columns_to_keep.extend(['cast', 'crew'])
        
        # Filter existing columns
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        df = df[columns_to_keep].copy()
        
        # Remove movies without overview
        df = df.dropna(subset=['overview'])
        df = df[df['overview'].str.strip() != '']
        
        # Parse JSON fields
        print("  Parsing genres...")
        df['genres'] = df['genres'].apply(self.safe_parse_json)
        df['genre_list'] = df['genres'].apply(lambda x: self.extract_names(x))
        df['genres_str'] = df['genre_list'].apply(lambda x: ', '.join(x))
        
        if 'keywords' in df.columns:
            print("  Parsing keywords...")
            df['keywords'] = df['keywords'].apply(self.safe_parse_json)
            df['keyword_list'] = df['keywords'].apply(lambda x: self.extract_names(x, limit=10))
            df['keywords_str'] = df['keyword_list'].apply(lambda x: ', '.join(x))
        else:
            df['keyword_list'] = [[] for _ in range(len(df))]
            df['keywords_str'] = ''
        
        if 'cast' in df.columns:
            print("  Parsing cast...")
            df['cast'] = df['cast'].apply(self.safe_parse_json)
            df['cast_list'] = df['cast'].apply(lambda x: self.extract_names(x, limit=5))
            df['cast_str'] = df['cast_list'].apply(lambda x: ', '.join(x))
            
            print("  Parsing crew (directors)...")
            df['crew'] = df['crew'].apply(self.safe_parse_json)
            df['director'] = df['crew'].apply(
                lambda x: ', '.join([
                    person['name'] for person in x 
                    if isinstance(person, dict) and person.get('job') == 'Director'
                ][:2])  # Top 2 directors
            )
        else:
            df['cast_list'] = [[] for _ in range(len(df))]
            df['cast_str'] = ''
            df['director'] = ''
        
        # Extract year from release_date
        if 'release_date' in df.columns:
            df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        else:
            df['year'] = None
        
        # Clean ratings
        if 'vote_average' in df.columns:
            df['rating'] = df['vote_average'].fillna(0)
        else:
            df['rating'] = 0
        
        # Filter out movies with very low vote counts (likely unreliable)
        if 'vote_count' in df.columns:
            df = df[df['vote_count'] >= 10]
        
        # Create rich search text (for embeddings)
        print("Creating search text")
        df['search_text'] = df.apply(self.create_search_text, axis=1)
        
        # Final cleanup
        df = df.dropna(subset=['title', 'overview'])
        df = df.reset_index(drop=True)
        
        return df
    
    def create_search_text(self, row):
        parts = []
        
        # Title (appears 2x for emphasis)
        parts.append(f"{row['title']} {row['title']}")
        
        # Overview (main content)
        parts.append(row['overview'])
        
        # Genres (important for categorization)
        if row['genres_str']:
            parts.append(f"Genres: {row['genres_str']}")
        
        # Keywords (semantic tags)
        if row.get('keywords_str'):
            parts.append(f"Keywords: {row['keywords_str']}")
        
        # Cast (people search for actor names)
        if row.get('cast_str'):
            parts.append(f"Starring: {row['cast_str']}")
        
        # Director (auteur searches)
        if row.get('director'):
            parts.append(f"Directed by: {row['director']}")
        
        # Year/Era context
        if pd.notna(row.get('year')):
            decade = (int(row['year']) // 10) * 10
            parts.append(f"{decade}s film")
        
        return ' '.join(parts)
    
    def save_processed_data(self, df, output_path='movies_processed.csv'):
        # Select final columns for output
        output_columns = [
            'id', 'title', 'overview', 'search_text',
            'genres_str', 'keywords_str', 'cast_str', 'director',
            'year', 'rating', 'vote_count', 'popularity', 'runtime'
        ]
        
        # Filter existing columns
        output_columns = [col for col in output_columns if col in df.columns]
        
        output_df = df[output_columns].copy()
        output_df.to_csv(output_path, index=False)
        
        print(f"\nSaved processed data to {output_path}")
        return output_df
    
    def get_statistics(self, df):
        print("\nDataset statistics")
        
        print(f"\nTotal movies: {len(df)}")
        
        if 'year' in df.columns:
            print(f"Year range: {df['year'].min():.0f} - {df['year'].max():.0f}")
        
        if 'genres_str' in df.columns:
            all_genres = []
            for genres in df['genre_list']:
                all_genres.extend(genres)
            from collections import Counter
            genre_counts = Counter(all_genres)
            print(f"\nTop 10 genres:")
            for genre, count in genre_counts.most_common(10):
                print(f"  {genre}: {count}")
        
        if 'rating' in df.columns:
            print(f"\nRating statistics:")
            print(f"  Mean: {df['rating'].mean():.2f}")
            print(f"  Median: {df['rating'].median():.2f}")
            print(f"  Min: {df['rating'].min():.2f}")
            print(f"  Max: {df['rating'].max():.2f}")
        
        if 'overview' in df.columns:
            avg_overview_length = df['overview'].str.len().mean()
            print(f"\nAverage overview length: {avg_overview_length:.0f} characters")

def main():
    processor = DataProcessor(
        movies_path='tmdb_5000_movies.csv',
        credits_path='tmdb_5000_credits.csv'
    )
    
    # Load and process
    df = processor.load_and_process()
    
    # Show statistics
    processor.get_statistics(df)
    
    # Save processed data
    output_df = processor.save_processed_data(df, 'movies_processed.csv')
    
    # Show sample
    print("\nSample processed movies:\n")
    for idx in range(min(3, len(output_df))):
        row = output_df.iloc[idx]
        print(f"\n{idx+1}. {row['title']} ({row.get('year', 'N/A')})")
        print(f"   Genres: {row.get('genres_str', 'N/A')}")
        print(f"   Rating: {row.get('rating', 0)}/10")
        if 'cast_str' in row and row['cast_str']:
            print(f"   Cast: {row['cast_str']}")
        print(f"   {row['overview'][:150]}")
    
    return output_df

if __name__ == "__main__":
    df = main()