import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi as yta
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from pytubefix import YouTube

hide_github_icon = """
#MainMenu {
  visibility: hidden;
}
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)

st.title("YouTube Video Chapter Generator")

model = ChatGroq(model="llama3-70b-8192")

radio = st.radio("Select model:", ["GPT-4o", "Llama3-70b"], index=1)

def get_video_id(url):
    video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return video_id.group(1) if video_id else None

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def process_chunk(title,chunk, chunk_index, total_chunks, chunk_start_time, chunk_end_time):
    chain = prompt | model
    response = chain.invoke({
        "title":title,
        "chunk": chunk,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "chunk_start_time": format_time(chunk_start_time),
        "chunk_end_time": format_time(chunk_end_time)
    })
    return response.content

st.text("Sample Video URL")
st.code("https://www.youtube.com/watch?v=vCPKIw43NFU", language="string")

url = st.text_input("Enter YouTube video URL (currently supports videos with transcript):")

if radio == "GPT-4o":
    openapi_key = st.text_input("For GPT-4o model, please enter your OpenAI API key:", type="password")
    model = ChatOpenAI(model="gpt-4o", api_key=openapi_key)


if st.button("Generate Chapters"):
    if radio == "GPT-4o" and not openapi_key:
        st.error("Please enter your OpenAI API key.")
        st.stop()
    video_id = get_video_id(url)
    if video_id:
        yt = YouTube(url)
        title = yt.title
        try:
            transcript = yta.get_transcript(video_id, languages=['en', 'hi'])
            
            data = [t['text'] for t in transcript]
            times = [t['start'] for t in transcript]
            
            video_duration = times[-1]
            
            text = " ".join(data)
            
            target_chunks = 15
            chunk_size = max(4000, len(text) // target_chunks)
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
            documents = splitter.split_text(text)
            
            prompt = PromptTemplate.from_template(
                "You are a seasoned expert in creating precise and engaging YouTube video chapters for a youtuber called Harkirat Singh, He is a software developer and sells courses on 100xdevs.com . "
                "Based on the provided transcript chunk, generate a concise and relevant chapter title. "
                "Return 'None' if there isn't a significant topic shift or noteworthy content in this segment. "
                "Only return the chapter title without quotation marks. No need for additional details. "
                "The given video is about {title}. "
                "The first chapter title should start from the beginning of the video. If the first chunk doesn't contain a significant topic shift, return one of the following Introduction or Context for the Video, or Upcoming in the video. "
                "Consider the following details: "
                "This is chunk {chunk_index} of {total_chunks}, covering the time from {chunk_start_time} to {chunk_end_time}. "
                "Transcript: {chunk}. "
                "The chapter title should be impactful, short(not more than 4-5 words), concise and reflect the essence of the content."
            )

            progress_bar = st.progress(0)
            status_text = st.empty()
            chapters = []
            
            with ThreadPoolExecutor() as executor:
                future_to_chunk = {executor.submit(
                    process_chunk, 
                    title,
                    chunk, 
                    i, 
                    len(documents),
                    times[min(i * len(times) // len(documents), len(times) - 1)],
                    times[min((i + 1) * len(times) // len(documents) - 1, len(times) - 1)]
                ): (i, chunk) for i, chunk in enumerate(documents)}
                
                for i, future in enumerate(as_completed(future_to_chunk)):
                    try:
                        chunk_index, _ = future_to_chunk[future]
                        response = future.result()
                        if response.lower() != 'none':
                            chunk_start_time = times[min(chunk_index * len(times) // len(documents), len(times) - 1)]
                            chunk_end_time = times[min((chunk_index + 1) * len(times) // len(documents) - 1, len(times) - 1)]
                            chapters.append((chunk_start_time, chunk_end_time, response))
                    except Exception as e:
                        st.warning(f"Error processing chunk: {str(e)}")
                    finally:
                        progress_bar.progress((i + 1) / len(documents))
                        status_text.text(f"Processed {i + 1}/{len(documents)} chunks...")

            chapters.sort(key=lambda x: x[0])

            st.subheader("Generated Chapters:")
            for start_time, end_time, title in chapters:
                st.text(format_time(start_time) + " " + title)
                
            
            st.write(f"Total chapters generated: {len(chapters)}")
            
            st.video(url)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Invalid YouTube URL. Please enter a valid URL.")
