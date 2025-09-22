import streamlit as st
import requests
from typing import List, Dict, Any

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Chatbot UI", page_icon="💬", layout="wide")

st.title("💬 Chatbot UI")
st.caption("Powered by FastAPI backend and sentence-transformers model")

@st.cache_data(show_spinner=False)
def load_sample_questions() -> List[str]:
    try:
        resp = requests.get(f"{API_BASE}/questions", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("questions", [])
        return []
    except Exception:
        return []

with st.sidebar:
    st.header("Sample Questions")
    samples = load_sample_questions()
    if not samples:
        st.info("Start the FastAPI server to see sample questions.")
    else:
        for q in samples:
            if st.button(q, use_container_width=True):
                st.session_state["current_question"] = q

if "current_question" not in st.session_state:
    st.session_state["current_question"] = ""

st.subheader("Ask a question")
question = st.text_input("Your question", value=st.session_state["current_question"], key="question_input")
col_ask, col_clear = st.columns([1,1])

placeholder = st.empty()
result_container = st.container()

with col_ask:
    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Getting answer..."):
                try:
                    resp = requests.post(f"{API_BASE}/ask", json={"question": question}, timeout=60)
                    if resp.status_code == 200:
                        data: Dict[str, Any] = resp.json()
                        # Handle answer structure (Series or string)
                        answer = data.get("answer")
                        if isinstance(answer, list):
                            answer = answer[0] if answer else "(No answer)"
                        result_container.markdown(f"### ✅ Answer\n{answer}")
                        result_container.markdown(f"**Most Similar Question:** {data.get('similar_question')}  ")
                        result_container.markdown(f"**Similarity Score:** {data.get('similarity_score'):.4f}")
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

with col_clear:
    if st.button("Clear"):
        st.session_state["current_question"] = ""
        st.experimental_rerun()

st.markdown("---")
st.caption("Ensure the FastAPI server is running: python api.py")
