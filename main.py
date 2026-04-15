from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from pathlib import Path
from ultralytics import YOLO
import tempfile
import os
import json

from omr_logic import grade_omr_image, draw_graded_overlay, generate_random_answer_key, load_answer_key

app = FastAPI(title="OMR Grading SaaS", version="1.0.0")

# Enable CORS for the Next.js/Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
model = None
answer_key = {}
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best.pt"
KEY_PATH = BASE_DIR / "answer_key.json"

@app.on_event("startup")
def startup_event():
    global model, answer_key
    print(f"Loading model from {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"WARNING: Model {MODEL_PATH} not found. Ensure training is complete.")
    else:
        model = YOLO(str(MODEL_PATH))
    
    if KEY_PATH.exists():
        answer_key = load_answer_key(KEY_PATH)
        print(f"Loaded answer key from {KEY_PATH}")
    else:
        answer_key = generate_random_answer_key(n_questions=100)
        print("Generated random 100-q answer key.")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "OMR Grading API is running."}

@app.post("/api/grade")
async def grade_endpoint(
    file: UploadFile = File(...),
    answer_key_file: UploadFile = File(None)
):
    """
    Takes an OMR sheet image, runs YOLO inference, calculates the score
    based on the answer key, and returns the result metadata + base64 annotated image.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="YOLO model not loaded.")
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Read the uploaded file into a temporary file since ultralytics accepts paths/numpy
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_cv2 is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        # Save to temp file to send to the grade_omr_image logic
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        
        cv2.imwrite(tmp_path, img_cv2)

        # Check if a custom answer key was uploaded
        current_answer_key = answer_key
        if answer_key_file is not None and answer_key_file.filename:
            try:
                contents_ak = await answer_key_file.read()
                raw_json = json.loads(contents_ak.decode('utf-8'))
                current_answer_key = {int(k): v.upper() for k, v in raw_json.items()}
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid answer key JSON: {str(e)}")

        # 1. Run the inference and logic
        omr_result = grade_omr_image(
            image_path=tmp_path,
            model=model,
            answer_key=current_answer_key,
            conf_thresh=0.25,
            device="cpu"   # ✅ ADD THIS
        )

        # 2. Draw overlay
        annotated_img = draw_graded_overlay(image_path=tmp_path, omr_result=omr_result, show_question_boxes=True)

        # Cleanup temp file
        os.remove(tmp_path)

        # 3. Base64 encode the annotated image
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        response_data = {
            "metadata": {
                "filename": file.filename,
                "total_questions": omr_result.total_questions,
                "score": omr_result.score,
                "percentage": omr_result.percentage,
                "answered": omr_result.answered,
                "blank": omr_result.blank,
                "multi_marked": omr_result.multi_marked,
                "correct": omr_result.correct,
                "wrong": omr_result.wrong,
            },
            "questions": [
                {
                    "q": r.question_number,
                    "detected": r.detected_answer,
                    "correct": r.correct_answer,
                    "is_correct": r.is_correct,
                    "status": r.status,
                    "confidence": round(r.confidence, 4)
                } for r in omr_result.question_results
            ],
            "annotated_image": f"data:image/jpeg;base64,{img_base64}"
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
