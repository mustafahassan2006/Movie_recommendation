import os
import joblib
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import linear_kernel




# 1. Load local .env file (for local development)
load_dotenv()

# 2. Prefer Streamlit Cloud Secrets, fallback to local os.getenv
if "TMDB_API_KEY" in st.secrets:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
else:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

# 2. Page Configuration
st.set_page_config(page_title="Movie Recommender", layout="wide")

# 3. Load Models & Generate Cosine Matrix Dynamically on Launch
@st.cache_resource
def load_models():
    df = joblib.load('movie_df.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    tfidf_matrix = joblib.load('tfidf_matrix.pkl')
    
    # Calculate matrix in-memory on startup
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return df, tfidf, tfidf_matrix, cosine_sim

df, tfidf, tfidf_matrix, cosine_sim = load_models()
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# 4. Fetch Poster Helper Function
@st.cache_data
def fetch_poster_by_title(title):
    if not TMDB_API_KEY:
        return "https://placehold.co/150x225/333333/ffffff?text=No+API+Key"
        
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title
    }
    # User-Agent header prevents TMDb from blocking python-requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                first_match = data['results'][0]
                poster_path = first_match.get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            return f"https://placehold.co/150x225/333333/ffffff?text=HTTP+{response.status_code}"
    except Exception:
        pass
    
    return "https://placehold.co/150x225/333333/ffffff?text=No+Poster"

# 5. Search & Recommendation Core Logic
def search_movies(query, top_n=10):
    query_vec = tfidf.transform([query])
    sim_scores = linear_kernel(query_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1][:top_n]
    
    return df[['title', 'vote_average', 'genres_clean', 'overview']].iloc[top_indices]

def get_recommendations(title, cosine_sim=cosine_sim, top_n=10):
    if title not in indices:
        return search_movies(title, top_n=top_n)
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    return df[['title', 'vote_average', 'genres_clean', 'overview']].iloc[movie_indices]

# 6. Header Bar & Navigation
top_col1, top_col2 = st.columns([5, 1])

with top_col1:
    st.title("🎬 Movie Recommender System")

with top_col2:
    st.link_button("Home", "https://mustafahassan.site", use_container_width=True)

st.markdown("---")

# 7. Search Interface
user_input = st.text_input("🔍 Search by movie title or plot concept (e.g., 'Inception' or 'space travel and robots'):")

if st.button("Search", type="primary") and user_input.strip():
    query_str = user_input.strip()
    results = get_recommendations(query_str)
    
    if query_str in indices:
        st.markdown(f"### Top Recommendations for **{query_str}**:")
    else:
        st.markdown(f"### Top Results matching: *\"{query_str}\"*")
    
    # Display Movie Cards
    for _, row in results.iterrows():
        with st.container(border=True):
            col_img, col_info = st.columns([1, 4])
            
            with col_img:
                poster_url = fetch_poster_by_title(row['title'])
                st.image(poster_url, width=130)
            
            with col_info:
                st.subheader(row['title'])
                st.markdown(f"**Rating:** ⭐ `{row['vote_average']}/10` | **Genres:** `{row['genres_clean']}`")
                st.write(row['overview'])