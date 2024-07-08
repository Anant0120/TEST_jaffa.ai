import streamlit as st
import openai
import re
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import time
from pinecone import Pinecone, ServerlessSpec
from openai.error import ServiceUnavailableError

# Set your OpenAI API key
openai.api_key = ""

# Set Pinecone API key and create an instance of Pinecone client
pinecone_api_key = ""
pinecone_client = Pinecone(api_key=pinecone_api_key)
index_name = "transcript"
index = None

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    number_of_pages = len(reader.pages)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text, number_of_pages

# Function to extract metadata from text using OpenAI
def extract_metadata_from_text(text):
    prompt = f"Extract the following metadata from the given transcript:\n\n{text}\n\nMetadata:\nCompany name:\nQuarter:\nConference call date:\nManagement info:"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that extracts metadata from conference call transcripts."},
            {"role": "user", "content": prompt}
        ]
    )
    
    metadata_text = response.choices[0].message['content'].strip()
    return metadata_text

# Function to extract opening remarks from PDF text
def get_opening_remarks(text):
    match = re.search(r'(.*?)(?=\bfirst question\b)', text, re.DOTALL | re.IGNORECASE)
    if match:
        opening_remarks = match.group(0).strip()
    else:
        opening_remarks = "Could not find the opening remarks section."
    return opening_remarks

# Function to extract question and answer section from PDF text
def get_question_answer_section(text):
    match = re.search(r'\bfirst question\b(.*?)(?=closing remarks|\Z)', text, re.DOTALL | re.IGNORECASE)
    if match:
        question_answer_section = match.group(1).strip()
    else:
        question_answer_section = "Could not find the question and answer section."
    return question_answer_section

# Function to process PDF and store embeddings in Pinecone index
def process_pdf(uploaded_file):
    global index
    # Clear the existing index
    if not index:
        index = pinecone_client.Index(index_name)
    else:
        index.delete(delete_all=True)
    
    text, _ = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    for chunk in chunks:
        embedding = get_embedding(chunk)
        index.upsert([(str(hash(chunk)), embedding, {'text': chunk})])

# Function to chunk text
def chunk_text(text, chunk_size=2000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to get embedding using OpenAI
def get_embedding(text, retries=5):
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(
                input=[text],
                model="text-embedding-ada-002"  # Use an appropriate model for embeddings
            )
            return response['data'][0]['embedding']
        except ServiceUnavailableError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

# Function to extract topics and summaries from content using OpenAI
def extract_topics_and_summaries(content, retries=5):
    for i in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Extract the main topics from the following content and provide a summary for each topic:\n\n{content}"}
                ],
                max_tokens=1024,
                temperature=0.7,
            )
            return response.choices[0].message['content'].strip()
        except openai.error.APIConnectionError as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(2 ** i)
    raise Exception("Failed to connect to OpenAI API after several retries.")

# Function to summarize content using OpenAI
def summarize_content(content):
    summary = extract_topics_and_summaries(content)
    return summary

# Function to query Pinecone and retrieve relevant document chunks
def query_pinecone(query, top_k=5):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [res['metadata']['text'] for res in results['matches']]

# Function to generate response using GPT-3.5-turbo based on retrieved documents
def generate_response(query):
    retrieved_texts = query_pinecone(query)
    combined_text = "\n\n".join(retrieved_texts)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Answer the question based on the following documents:\n\n{combined_text}\n\nQuestion: {query}\nAnswer:"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message['content'].strip()

# Streamlit app layout
def main():
    st.title("Earning-Trancript-Summarizer")

    # Sidebar navigation with radio buttons
    page = st.sidebar.radio("Select an option", ["Welcome", "Opening Remarks Analyzer", "Q&A Section Analyzer", "PDF Question Answering"])

    # Handle file upload and global PDF storage
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Initialize global variables
    global text, opening_remarks, question_answer_section

    # Page 1: Metadata Extractor
    if page == "Welcome":
        if uploaded_file is not None:
            text, number_of_pages = extract_text_from_pdf(uploaded_file)
            st.write("Metadata from processed transcript is given below:")
            st.write(extract_metadata_from_text(text))
            st.write(f"Number of pages in transcript: {number_of_pages}")

    # Page 2: Opening Remarks Analyzer
    elif page == "Opening Remarks Analyzer":
        if uploaded_file is not None:
            text = extract_text_from_pdf(uploaded_file)[0]
            opening_remarks = get_opening_remarks(text)
            st.subheader("Opening Remarks")
            preview = opening_remarks.split('\n')[0] + "..." if len(opening_remarks) > 100 else opening_remarks
            if st.button("Show full opening remarks"):
                st.write(opening_remarks)
            else:
                st.write(preview)
            if st.button("Get Topics and Summaries"):
                topics_summaries = extract_topics_and_summaries(opening_remarks)
                st.subheader("Topics and Summaries")
                st.write(topics_summaries)
                st.download_button(
                    label="Download Summary",
                    data=topics_summaries,
                    file_name='opening_remarks_summary.txt',
                    mime='text/plain'
                )

    # Page 3: Q&A Section Analyzer
    elif page == "Q&A Section Analyzer":
        if uploaded_file is not None:
            text = extract_text_from_pdf(uploaded_file)[0]
            question_answer_section = get_question_answer_section(text)
            st.subheader("Question and Answer Section")
            preview = question_answer_section.split('\n')[0] + "..." if len(question_answer_section) > 100 else question_answer_section
            if st.button("Show full Q&A section"):
                st.write(question_answer_section)
            else:
                st.write(preview)
            if st.button("Get Topics and Summaries for Q&A Section"):
                qa_topics_summaries = summarize_content(question_answer_section)
                st.subheader("Topics and Summaries for Q&A Section")
                st.write(qa_topics_summaries)
                st.download_button(
                    label="Download Summary",
                    data=qa_topics_summaries,
                    file_name='qa_summary.txt',
                    mime='text/plain'
                )

    # Page 4: PDF Question Answering System
    elif page == "PDF Question Answering":
        if uploaded_file is not None:
            with st.spinner('Processing the PDF...'):
                process_pdf(uploaded_file)
                st.success("PDF processed and embeddings stored in Pinecone index.")

        query = st.text_input("Enter your query:")
        if query:
            with st.spinner('Generating response...'):
                response = generate_response(query)
                st.success("Response generated:")
                st.write(response)


# Run the app
if __name__ == "__main__":
    main()
