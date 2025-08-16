import os
import re
from urllib.parse import urlparse, parse_qs
from typing import List, Tuple

import streamlit as st
from openai import OpenAI

# Try different import approaches for youtube_transcript_api
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
# Utilities
# -----------------------------
def video_id_from_url(url: str) -> str:
    """Extract the YouTube video ID from full url or youtu.be link."""
    u = urlparse(url.strip())
    if u.netloc in {"youtu.be"}:
        return u.path.lstrip("/")
    if "youtube" in u.netloc:
        qs = parse_qs(u.query).get("v", [""])[0]
        return qs
    # allow raw IDs
    if re.fullmatch(r"[A-Za-z0-9_-]{6,}", url.strip()):
        return url.strip()
    return ""

def fetch_transcript_text(youtube_url: str) -> Tuple[str | None, str | None]:
    """Return (transcript_text, error_msg)."""
    vid = video_id_from_url(youtube_url)
    if not vid:
        return None, "Could not parse a video ID from the URL."

    try:
        # Initialize the API
        ytt_api = YouTubeTranscriptApi()
        
        # Try to fetch transcript with preferred languages
        try:
            # Try English first
            fetched_transcript = ytt_api.fetch(vid, languages=['en', 'en-US', 'en-GB'])
        except NoTranscriptFound:
            try:
                # If no English, try to fetch any available transcript
                transcript_list = ytt_api.list(vid)
                # Get the first available transcript
                if transcript_list:
                    transcript = list(transcript_list)[0]
                    fetched_transcript = transcript.fetch()
                else:
                    return None, "No transcripts available for this video."
            except Exception:
                return None, "No transcripts available for this video."

        # Extract text from the fetched transcript
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
    """Simple character-based chunking with overlap."""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # try to end on a sentence boundary
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
    """Very small, dependency-free keyword overlap score."""
    token = lambda s: {w for w in re.findall(r"[A-Za-z0-9]+", s.lower())
                       if w not in _STOPWORDS and len(w) > 2}
    q = token(query)
    c = token(chunk)
    return len(q & c)

def top_k_chunks(query: str, chunks: List[str], k: int = 4) -> List[str]:
    scored = sorted(chunks, key=lambda c: score_chunk(query, c), reverse=True)
    return scored[: max(1, k)]

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Chat with YouTube Video", page_icon="🎬", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "video_title" not in st.session_state:
    st.session_state.video_title = None

# Sidebar for setup
st.sidebar.header("Setup")
yt_url = st.sidebar.text_input("YouTube URL or Video ID", placeholder="https://www.youtube.com/watch?v=...")
api_key = st.sidebar.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")

# Load video button
if st.sidebar.button("📺 Load Video", use_container_width=True):
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
            st.session_state.messages = []  # Clear chat history for new video
            st.sidebar.success("✅ Video loaded successfully!")

# Main interface
st.title("🎬 Chat with KumR")

if st.session_state.transcript:
    st.success(f"📺 Ready to chat with KumR about his video!")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask KumR about his video..."):
        if not api_key:
            st.error("Please provide your OpenAI API key in the sidebar.")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get relevant chunks
            relevant_chunks = top_k_chunks(prompt, st.session_state.chunks, k=4)
            context = "\n\n".join(f"[Section {i+1}]\n{chunk}" for i, chunk in enumerate(relevant_chunks))
            
            # Build conversation context for continuity
            conversation_context = ""
            if len(st.session_state.messages) > 1:
                recent_messages = st.session_state.messages[-6:]  # Last 3 exchanges
                conversation_context = "\n\nPrevious conversation:\n"
                for msg in recent_messages[:-1]:  # Exclude the current question
                    conversation_context += f"{msg['role'].title()}: {msg['content']}\n"
            
            # Generate response
            try:
                client = OpenAI(api_key=api_key)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            temperature=0.3,
                            messages=[
                                {"role": "system", "content": f"""You are KumR, the presenter in this YouTube video. You're answering questions about your own video content in the warm, encouraging style of Andrew Ng. You explain things like you're talking to a smart 15-year-old.

YOUR IDENTITY:
- You ARE KumR, the person who made this video
- You're explaining YOUR OWN content and ideas
- Reference what YOU said in the video ("As I mentioned..." "When I explained..." "In my video...")
- Take ownership of the concepts you presented
- Share additional insights beyond what's in the transcript

YOUR TEACHING STYLE (like Andrew Ng):
- Warm and encouraging - always positive and supportive  
- Explain complex concepts like you're talking to a smart 15-year-old
- Use simple analogies and real-world examples
- Break down complicated ideas into bite-sized pieces
- Say things like "Great question!" "Let me break this down for you" "Think of it like..."
- Be genuinely excited about teaching YOUR concepts

HOW TO RESPOND:
- Start with encouragement for good questions
- Reference your video content as YOUR work ("In my video, I showed..." "The example I used...")
- Expand on concepts with additional explanations not in the transcript
- Use everyday language and relatable examples for teenagers
- If something wasn't covered in your video, acknowledge it: "That's a great follow-up question! I didn't cover that in this particular video, but..."

CONVERSATION STYLE:
- Speak as the creator/presenter, not a third party
- Connect ideas to things a 15-year-old would know
- Build on your previous explanations in the conversation
- End responses with encouragement or thought-provoking questions

YOUR VIDEO TRANSCRIPT:
{context}{conversation_context}
"""},
                                {"role": "user", "content": prompt}
                            ],
                        )
                        answer = response.choices[0].message.content.strip()
                        st.markdown(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

else:
    st.info("👈 Load one of KumR's YouTube videos to start chatting!")
    st.markdown("""
    ### Chat directly with KumR! 🎯
    
    **KumR** is here to explain his own video content in person! He teaches like Andrew Ng - super encouraging and breaks everything down perfectly.
    
    ### How it works:
    1. **Paste KumR's YouTube video URL** in the sidebar
    2. **Add your OpenAI API key** 
    3. **Click "Load Video"** 
    4. **Ask KumR directly** about his content!
    
    **What makes this special:**
    - 🎥 **KumR explains HIS OWN work** - not a third party
    - 🧠 **Expands beyond the video** with additional insights
    - 💬 **Remembers your conversation** and builds on it
    - 🎯 **Perfect for learning** - explains like you're 15, encourages like Andrew Ng
    
    *It's like having a personal tutoring session with KumR about his videos!*
    """)

# Clear chat button in sidebar
if st.session_state.messages:
    if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()