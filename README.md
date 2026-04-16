# OMR Grading SaaS Backend

This repository contains the backend service for the automated Optical Mark Recognition (OMR) Grading application. Built with **FastAPI** and **Ultralytics YOLOv8**, this API analyzes scanned OMR sheets, dynamically detects and grades answers against an answer key, and returns structured result data along with a base64-encoded annotated image.

## 🚀 Key Features

- **FastAPI Framework**: Provides a highly performant and auto-documented asynchronous REST API.
- **YOLOv8 Edge Detection**: Leverages an Ultralytics YOLOv8 model (`best.pt`) to precisely detect questions and user-marked responses.
- **Dynamic Grading Logic**: Calculates scores, evaluates multiple-choice questions, and handles edge cases like blank submissions or multi-marked bubbles.
- **Custom Answer Key Uploads**: Supports overriding the default answer key at runtime via an uploaded JSON file.
- **On-the-fly Image Processing**: Uses OpenCV headless to overlay bounding boxes, correctness indicators, and scores on the uploaded sheet, returning it directly as a base64 string without relying on permanent disk storage.

---

## 📂 Project Structure

```text
backend/
├── main.py              # Application entry point & FastAPI routes
├── omr_logic.py         # Core OMR image processing, OpenCV drawing, and YOLO inference
├── requirements.txt     # Python dependencies
├── answer_key.json      # Default ground-truth answer key mapping questions to options
├── model/               # Directory containing the trained YOLO model
│   └── best.pt          # The actual YOLOv8 weights (Must be present for inference)
```

---

## 🛠️ Installation & Setup

1. **Verify Python & Virtual Environment Setup**  
   It is recommended to run this inside a virtual environment (e.g., `venv` or `conda`). Ensure Python 3.9+ is installed.

2. **Install Dependencies**  
   Install the required libraries listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   > **Note:** The project uses `opencv-python-headless` and CPU-optimized PyTorch.

3. **Verify Model Weights**  
   Ensure that your trained YOLOv8 model file is placed in `model/best.pt` relative to the backend directory. The application will warn you on startup if the model is missing.

4. **Run the Server**  
   Start the FastAPI app via Uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **View API Logs / Documentation**  
   Once running, you can open `http://localhost:8000/docs` in your browser to access the interactive Swagger UI and test the `/api/grade` endpoint.

---

## 🔌 API Endpoints

### `GET /`
Health check endpoint.
**Response**: `{"status": "ok", "message": "OMR Grading API is running."}`

### `POST /api/grade`
The core inference endpoint. 

**Request (Form-Data)**
- `file`: The OMR sheet image (`.jpg`, `.jpeg`, `.png`).
- `answer_key_file` *(optional)*: A custom JSON file mapping question numbers to letters (e.g., `{"1": "A", "2": "C"}`).

**Response (JSON)**
- `metadata`: Contains overall scoring information including total questions, score, percentage, correct/wrong counts, and blank/multi-marked counts.
- `questions`: An array detailing the exact parsing details per question limit (question number, detected answer, correctness state, confidence score).
- `annotated_image`: A Data-URI (base64) containing the processed JPG with colored bounding overlays.

---

## 💡 Troubleshooting

- **503 YOLO model not loaded**: The `best.pt` file was missing during startup. Refer to the `model` folder structure above.
- **Dependency Issues**: Make sure that Torch version matches your compute platform (e.g. CUDA vs CPU). The `requirements.txt` is geared towards a CPU-based inference via the `download.pytorch.org/whl/cpu` index.
