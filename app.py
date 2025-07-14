from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
from main import process_video_for_transcript, generate_summary_from_transcript
from sqlalchemy.orm import Session
from database import get_db, Video
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Video Transcription and Summarisation API",
    description="API for transcribing video files using Whisper and Silero VAD and summarisation using Llama",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Create necessary directories
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/upload")
async def upload_file(video: UploadFile = File(...), db: Session = Depends(get_db)):
    if not video:
        raise HTTPException(status_code=400, detail="No video file provided")
    if not allowed_file(video.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    try:
        with open(video_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)

        # Save to database with all summary fields as None initially
        db_video = Video(
            filename=video.filename,
            transcript=None,
            summary_short=None,
            summary_detailed=None,
            summary_bullets=None
        )
        db.add(db_video)
        db.commit()
        db.refresh(db_video)

        return {"message": "Video uploaded successfully", "video_id": db_video.id}
    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/transcripts/{video_id}",
    summary="Get a transcript and summary by video ID",
    response_description="Returns the transcript and summary")
async def get_transcript(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return JSONResponse(content={
        'filename': video.filename,
        'transcript': video.transcript,
        'summary_short': video.summary_short,
        'summary_detailed': video.summary_detailed,
        'summary_bullets': video.summary_bullets
    })

@app.get("/videos",
    summary="List all videos",
    response_description="Returns list of all videos with their transcripts and summaries")
async def list_videos(db: Session = Depends(get_db)):
    videos = db.query(Video).all()
    return JSONResponse(content={
        'videos': [
            {
                'id': video.id,
                'filename': video.filename,
                'transcript': video.transcript,
                'summary_short': video.summary_short,
                'summary_detailed': video.summary_detailed,
                'summary_bullets': video.summary_bullets
            } for video in videos
        ]
    })

@app.post("/process_transcript/{video_id}")
async def process_transcript(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    transcript = process_video_for_transcript(video_path)
    if transcript is None:
        raise HTTPException(status_code=500, detail="Failed to generate transcript")
    video.transcript = transcript
    db.commit()
    return {"transcript": transcript}

@app.post("/process_summary/{video_id}")
async def process_summary(video_id: int, summary_type: str = "short", db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.transcript:
        raise HTTPException(status_code=400, detail="Transcript not available. Generate transcript first.")
    
    summary = generate_summary_from_transcript(video.transcript, summary_type)
    if summary is None:
        raise HTTPException(status_code=500, detail="Failed to generate summary")
    
    # Update the appropriate summary field based on type
    if summary_type == "short":
        video.summary_short = summary
    elif summary_type == "detailed":
        video.summary_detailed = summary
    elif summary_type == "bullets":
        video.summary_bullets = summary
    
    db.commit()
    return {"summary": summary}

@app.get("/", response_class=FileResponse)
async def read_root():
    return "index.html"

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
