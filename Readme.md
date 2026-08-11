# 🎬 Movie Recommender System

A content-based movie recommendation web app built with **Python**, **Streamlit**, and **Scikit-Learn**. The system offers a unified search interface that allows users to find similar movies either by title lookup or by entering natural language plot concepts and keywords.

---

## ✨ Features

* **Dual Search Modes in One Bar:**
  * **Title Recommendations:** Look up an exact movie title to get the top 10 most similar films based on genre, tagline, and overview features.
  * **Free-Text Concept Search:** Search by themes, plot tropes, or arbitrary keywords (e.g., *"space travel and artificial intelligence"*) using vectorized TF-IDF similarity.
* **Dynamic Poster Fetching:** Fetches official high-resolution posters on the fly using the **TMDb (The Movie Database) API**.
* **Memory & Storage Optimized:** Pre-calculates TF-IDF representations offline, while computing similarity matrices dynamically in-memory on startup to keep repo size minimal (<25 MB).

---

## 🛠️ Tech Stack

* **Frontend / Web UI:** [Streamlit](https://streamlit.io/)
* **Machine Learning & NLP:** [Scikit-Learn](https://scikit-learn.org/) (`TfidfVectorizer`, `linear_kernel`)
* **Data Processing:** [Pandas](https://pandas.pydata.org/), [Joblib](https://joblib.readthedocs.io/)
* **External API:** [TMDb API](https://www.themoviedb.org/documentation/api)

---

## 📁 Repository Structure

```text
├── app.py                  # Main Streamlit application
├── movie_df.pkl            # Cleaned movie metadata DataFrame
├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer model
├── tfidf_matrix.pkl        # TF-IDF feature matrix (10,000 movies)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # Project documentation
