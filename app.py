# ====================== Imports ======================
import streamlit as st
import arxiv
import requests
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie

# ====================== Page Config ======================
st.set_page_config(page_title="AI Research Paper Finder", layout="wide")

# ====================== Lottie Loader ======================
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_ai = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_tno6cg2w.json")
if lottie_ai:
    st_lottie(lottie_ai, height=200, key="header_lottie")
else:
    st.warning("⚠️ Lottie animation failed to load.")

# ====================== NLP Model ======================
@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

# ====================== Helper Functions ======================
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def fetch_arxiv(query, max_results=5):
    try:
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        papers = []
        for result in search.results():
            papers.append({
                "title": result.title or "",
                "authors": ", ".join([a.name for a in result.authors]) or "Unknown",
                "summary": result.summary or "",
                "url": result.entry_id,
                "source": "ArXiv"
            })
        return papers
    except Exception as e:
        st.error(f"ArXiv fetch error: {e}")
        return []

def fetch_semantic_scholar(query, limit=5):
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={limit}&fields=title,authors,url,abstract"
        response = requests.get(url, timeout=10)
        data = response.json()
        papers = []
        for paper in data.get("data", []):
            papers.append({
                "title": paper.get("title") or "No title",
                "authors": ", ".join([a.get("name", "") for a in paper.get("authors", [])]) or "Unknown",
                "summary": paper.get("abstract") or "No abstract",
                "url": paper.get("url") or "#",
                "source": "Semantic Scholar"
            })
        return papers
    except Exception as e:
        st.error(f"Semantic Scholar fetch error: {e}")
        return []

def fetch_researchgate(query):
    # Placeholder since ResearchGate doesn't provide a public API
    return [{
        "title": f"ResearchGate Placeholder for '{query}'",
        "authors": "N/A",
        "summary": "ResearchGate data not accessible via API.",
        "url": "https://www.researchgate.net/",
        "source": "ResearchGate"
    }]

def semantic_search(query, papers):
    # Handle NoneType values safely
    docs = [
        clean_text((p.get("title") or "") + " " + (p.get("summary") or ""))
        for p in papers
        if p.get("title") or p.get("summary")
    ]
    if not docs:
        return []
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs + [query])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    sorted_idx = cosine_sim.argsort()[::-1]
    return [papers[i] for i in sorted_idx]

# ====================== Sidebar Inputs ======================
st.sidebar.header("🔎 Search Papers")

query = st.sidebar.text_input("Enter a research topic", "large language models in healthcare", key="main_query")
sources = st.sidebar.multiselect(
    "Choose sources", ["arxiv", "semanticscholar", "researchgate"],
    default=["arxiv", "semanticscholar", "researchgate"], key="source_selection"
)
papers_per_source = st.sidebar.slider("Papers per source", 5, 25, 10, key="papers_slider")

# Extract keywords using spaCy
keywords = []
if query:
    doc = nlp(query)
    keywords = list({chunk.text.lower().strip() for chunk in doc.noun_chunks})[:5]

selected_keywords = st.sidebar.multiselect(
    "Refine by keywords", options=keywords, default=keywords[:2], key="keyword_selection"
)
year_filter = st.sidebar.text_input("Filter by Year (optional)", "", key="year_filter_input")
author_filter = st.sidebar.text_input("Filter by Author (optional)", "", key="author_filter_input")

# Build final search query
search_query = query
if selected_keywords:
    search_query += " " + " ".join(selected_keywords)
if author_filter:
    search_query += f" author:{author_filter}"
if year_filter:
    search_query += f" year:{year_filter}"

st.sidebar.markdown(f"**Final Search Query:** `{search_query}`")

# ====================== Fetch & Display Papers ======================
if st.button("Search") and query:
    with st.spinner("Fetching papers..."):
        all_papers = []
        if "arxiv" in sources:
            all_papers += fetch_arxiv(search_query, papers_per_source)
        if "semanticscholar" in sources:
            all_papers += fetch_semantic_scholar(search_query, papers_per_source)
        if "researchgate" in sources:
            all_papers += fetch_researchgate(search_query)

        if not all_papers:
            st.warning("No papers found.")
        else:
            ranked_papers = semantic_search(search_query, all_papers)
            if not ranked_papers:
                st.warning("No papers with titles or summaries found.")
            else:
                st.success(f"Found {len(ranked_papers)} papers!")

                for paper in ranked_papers:
                    st.markdown(f"### [{paper['title']}]({paper['url']})")
                    st.write(f"**Authors:** {paper['authors']}")
                    st.write(f"**Source:** {paper['source']}")
                    st.write(paper['summary'])
                    st.markdown("---")

# ====================== Saved Papers Section ======================
if "saved_papers" not in st.session_state:
    st.session_state.saved_papers = []

if st.session_state.saved_papers:
    st.subheader("📂 Saved Papers")
    df = pd.DataFrame(st.session_state.saved_papers)
    st.dataframe(df, use_container_width=True)
    with open("saved_papers.csv", "wb") as f:
        df.to_csv("saved_papers.csv", index=False)
    st.download_button("⬇️ Download Saved Papers", "saved_papers.csv", file_name="saved_papers.csv")

