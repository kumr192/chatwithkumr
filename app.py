import os
import re
from urllib.parse import urlparse, parse_qs
from typing import List, Tuple

import streamlit as st
from openai import OpenAI

# youtube_transcript_api (same as your code)
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
except ImportError:
    st.error("Please install youtube-transcript-api: pip install youtube-transcript-api")
    st.stop()

# -----------------------------
# PURE BLACK THEME + MATRIX TEXT
# -----------------------------
st.set_page_config(page_title="Chat with KumR", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
      /* Global black */
      html, body, .stApp { background: #000 !important; color: #fff !important; }

      /* Main container to remove panel tint and keep things flat black */
      .main .block-container { background: transparent !important; }

      /* Sidebar */
      section[data-testid="stSidebar"] {
        background: #000 !important;
        border-right: 1px solid #111 !important;
      }
      .stSidebar .stMarkdown, .stSidebar label, .stSidebar p, .stSidebar span { color: #fff !important; }

      /* Inputs */
      .stTextInput input, .stChatInput textarea, .stSelectbox div[data-baseweb="select"] div {
        background: #000 !important;
        color: #fff !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
      }

      /* Buttons */
      .stButton button {
        background: #0a0a0a !important;
        color: #fff !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
      }
      .stButton button:hover { border-color: #00ff41 !important; }

      /* Messages */
      [data-testid="stChatMessage"] { background: transparent !important; border: none !important; }
      /* optional subtle box around messages */
      .bubble { border: 1px solid #111; background:#000; border-radius: 12px; padding: 10px 12px; }

      /* Success/Info boxes (dark) */
      .stSuccess, .stInfo, .stError, .stWarning {
        background: #000 !important;
        border: 1px solid #111 !important;
        color: #fff !important;
      }

      /* Matrix-style streaming text for assistant */
      .matrix-text {
        color: #00ff41 !important;
        font-family: "Courier New", monospace;
        text-shadow: 0 0 6px #00ff41;
        animation: matrix-glow 2s ease-in-out infinite alternate;
      }
      @keyframes matrix-glow {
        from { text-shadow: 0 0 6px #00ff41, 0 0 10px #00ff41; }
        to   { text-shadow: 0 0 10px #00ff41, 0 0 16px #00ff41, 0 0 22px #00ff41; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Utilities (unchanged)
# -----------------------------
def video_id_from_url(url: str) -> str:
    u = urlparse(url.strip())
    if u.netloc in {"youtu.be"}:
        return u.path.lstrip("/")
    if "youtube" in u.netloc:
        qs = parse_qs(u.query).get("v", [""])[0]
        return qs
    if re.fullmatch(r"[A-Za-z0-9_-]{6,}", url.strip()):
        return url.strip()
    return ""

def fetch_transcript_text(youtube_url: str) -> Tuple[str | None, str | None]:
    vid = video_id_from_url(youtube_url)
    if not vid:
        return None, "Could not parse a video ID from the URL."

    try:
        # your instance-based approach kept
        ytt_api = YouTubeTranscriptApi()
        try:
            fetched_transcript = ytt_api.fetch(vid, languages=['en', 'en-US', 'en-GB'])
        except NoTranscriptFound:
            try:
                transcript_list = ytt_api.list(vid)
                if transcript_list:
                    transcript = list(transcript_list)[0]
                    fetched_transcript = transcript.fetch()
                else:
                    return None, "No transcripts available for this video."
            except Exception:
                return None, "No transcripts available for this video."

        text_parts = []
        for snippet in fetched_transcript:
            text_parts.append(snippet.text)
        text = " ".join(text_parts)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return None, "Transcript was empty."
        return text, None

    except TranscriptsDisabled:
        return None, "Captions are disabled for this video."
    except VideoUnavailable:
        return None, "Video is unavailable or private."
    except Exception as e:
        return None, f"Transcript fetch failed: {str(e)}"

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        period = text.rfind(". ", start, end)
        if period != -1 and period > start + 200:
            end = period + 1
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end)
    return [c for c in chunks if c]

_STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","is","it","that",
    "this","as","at","by","be","are","was","were","from","we","you","i"
}

def score_chunk(query: str, chunk: str) -> int:
    token = lambda s: {w for w in re.findall(r"[A-Za-z0-9]+", s.lower())
                       if w not in _STOPWORDS and len(w) > 2}
    q = token(query)
    c = token(chunk)
    return len(q & c)

def top_k_chunks(query: str, chunks: List[str], k: int = 4) -> List[str]:
    scored = sorted(chunks, key=lambda c: score_chunk(query, c), reverse=True)
    return scored[: max(1, k)]

# -----------------------------
# State (unchanged)
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "video_title" not in st.session_state:
    st.session_state.video_title = None

# -----------------------------
# Sidebar (logos/banner removed)
# -----------------------------
st.sidebar.header("Setup")
yt_url = st.sidebar.text_input("YouTube URL or Video ID", placeholder="https://www.youtube.com/watch?v=...")
api_key = st.sidebar.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")

if st.sidebar.button("Load Video", use_container_width=True):
    if not api_key:
        st.sidebar.error("Please provide your OpenAI API key.")
    elif not yt_url.strip():
        st.sidebar.error("Please paste a YouTube URL or ID.")
    else:
        with st.spinner("Loading video transcript..."):
            text, err = fetch_transcript_text(yt_url)
        if err:
            st.sidebar.error(err)
            st.session_state.transcript = None
            st.session_state.chunks = None
            st.session_state.video_title = None
        else:
            chunks = chunk_text(text, max_chars=1500, overlap=200)
            st.session_state.transcript = text
            st.session_state.chunks = chunks
            st.session_state.video_title = f"YouTube Video ({video_id_from_url(yt_url)})"
            st.session_state.messages = []
            st.sidebar.success("Video loaded successfully!")

# -----------------------------
# Main (logos removed, black UI)
# -----------------------------
st.title("Chat with KumR")

if st.session_state.transcript:
    st.success("Ready to chat about the video!")

    # history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(f'<div class="matrix-text bubble">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bubble">{message["content"]}</div>', unsafe_allow_html=True)

    # chat input
    if prompt := st.chat_input("Ask about the video..."):
        if not api_key:
            st.error("Please provide your OpenAI API key in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(f'<div class="bubble">{prompt}</div>', unsafe_allow_html=True)

            relevant = top_k_chunks(prompt, st.session_state.chunks, k=4)
            context = "\n\n".join(f"[Section {i+1}]\n{chunk}" for i, chunk in enumerate(relevant))

            # optional short conversation memory
            conversation_context = ""
            if len(st.session_state.messages) > 1:
                recent = st.session_state.messages[-6:]
                conversation_context = "\n\nPrevious conversation:\n"
                for msg in recent[:-1]:
                    conversation_context += f"{msg['role'].title()}: {msg['content']}\n"

            try:
                client = OpenAI(api_key=api_key)
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""

                    for response in client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0.3,
                        stream=True,
                        messages=[
                            {"role": "system", "content": f"""You are KumR, the presenter in this YouTube video. You answer questions about your video content clearly and helpfully for a smart 15-year-old.

YOUR VIDEO TRANSCRIPT:
{context}{conversation_context}
"""},
                            {"role": "user", "content": prompt}
                        ],
                    ):
                        if response.choices[0].delta.content is not None:
                            full_response += response.choices[0].delta.content
                            message_placeholder.markdown(
                                f'<div class="matrix-text bubble">{full_response}▊</div>',
                                unsafe_allow_html=True
                            )

                    message_placeholder.markdown(
                        f'<div class="matrix-text bubble">{full_response}</div>',
                        unsafe_allow_html=True
                    )

                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error: {str(e)}")

else:
    st.info("Paste a YouTube URL in the sidebar and click **Load Video** to start.")

# Clear chat
if st.session_state.messages:
    if st.sidebar.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
