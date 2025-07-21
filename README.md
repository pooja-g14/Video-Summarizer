# Video-Summarizer

Upload a video file and generate a transcript and summary for it.

Tech Stack:
- LLMs for transcript generation and summary generation (Whisper for speech-to-text and LLaMA for text summarization)
- Fast API as the interface
- PostgreSQL as the backend for storing data
- HTML and CSS as the frontend for design

To run this application:

1. Create a virtual environment (venv)
2. Activate the virtual environment (venv)
3. Install the libraries in requirements.txt
4. Create the database in PostgreSQL and add the Database URL to the .env file
5. In terminal, run: python app.py
6. Go to http://localhost:8000 to check out the app
