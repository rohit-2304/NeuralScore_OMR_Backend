"""
omr_logic.py  —  OMR Detection → Answer Extraction → Grading
=============================================================

Pipeline
--------
1. Run YOLOv8 inference on the OMR sheet image.
2. Cluster `question` boxes into columns (by x-centre, gap > threshold).
3. Within each column, sort question rows top-to-bottom (by y-centre).
4. Assign global question numbers by reading columns left-to-right.
5. For each `marked` bubble, find which question row it sits inside
   (y-overlap + same column) and determine the option (A/B/C/D) from
   its relative x-position within the question bbox.
6. Handle edge cases: multiple marks per row → flag as "multi-marked";
   no mark → flag as "blank".
7. Compare against an answer key and compute the score.

Classes (YOLO internal, 0-indexed):
    0 → ROI       (the overall sheet bounding box)
    1 → marked    (individual filled bubble)
    2 → question  (one row of 4 option bubbles)

Coordinate convention used internally: pixel xyxy (x1, y1, x2, y2).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ── constants ────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["ROI", "marked", "question"]      # YOLO class index → name
CLS_ROI      = 0
CLS_MARKED   = 1
CLS_QUESTION = 2

OPTIONS      = ["A", "B", "C", "D"]

# How wide a gap (pixels) between question-column x-centres marks a new column
COLUMN_GAP_PX    = 50
# Fraction of question-box height that a marked bubble must overlap to be "in" that row
MIN_Y_OVERLAP_FRAC = 0.25
# How far (fraction of question-box width) from each option centre before we
# consider the bubble ambiguously placed
OPTION_TOL_FRAC  = 0.30


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class BubbleBox:
    """A detected bounding box from YOLO."""
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


@dataclass
class QuestionResult:
    question_number: int              # 1-indexed
    detected_answer: Optional[str]   # "A", "B", "C", "D", or None
    correct_answer: Optional[str]    # from answer key
    is_correct: Optional[bool]       # None if answer key not provided
    status: str                      # "answered" | "blank" | "multi-marked"
    confidence: float                # avg conf of the marked bubble(s)
    question_bbox: tuple             # (x1, y1, x2, y2)  px
    marked_bboxes: list              # list of (x1,y1,x2,y2) for filled bubbles


@dataclass
class OMRResult:
    image_path: str
    total_questions: int
    answer_key: dict                         # {q_num: option}
    question_results: list[QuestionResult]
    answered: int = field(init=False)
    blank: int = field(init=False)
    multi_marked: int = field(init=False)
    correct: int = field(init=False)
    wrong: int = field(init=False)
    score: int = field(init=False)
    percentage: float = field(init=False)

    def __post_init__(self):
        self.answered     = sum(1 for r in self.question_results if r.status == "answered")
        self.blank        = sum(1 for r in self.question_results if r.status == "blank")
        self.multi_marked = sum(1 for r in self.question_results if r.status == "multi-marked")
        self.correct      = sum(1 for r in self.question_results if r.is_correct is True)
        self.wrong        = sum(1 for r in self.question_results
                                if r.is_correct is False and r.status == "answered")
        self.score        = self.correct
        total_keyed       = sum(1 for r in self.question_results if r.correct_answer is not None)
        self.percentage   = round((self.correct / total_keyed * 100) if total_keyed else 0, 2)

    def to_dict(self) -> dict:
        return {
            "image_path":       self.image_path,
            "total_questions":  self.total_questions,
            "score":            self.score,
            "percentage":       self.percentage,
            "answered":         self.answered,
            "blank":            self.blank,
            "multi_marked":     self.multi_marked,
            "correct":          self.correct,
            "wrong":            self.wrong,
            "questions":        [
                {
                    "q":              r.question_number,
                    "detected":       r.detected_answer,
                    "correct":        r.correct_answer,
                    "is_correct":     r.is_correct,
                    "status":         r.status,
                    "confidence":     round(r.confidence, 4),
                }
                for r in self.question_results
            ],
        }


# ── answer key helpers ───────────────────────────────────────────────────────

def generate_random_answer_key(n_questions: int = 100, seed: int = 42) -> dict[int, str]:
    """
    Generate a reproducible random answer key.
    Returns {1: 'A', 2: 'C', ...}
    """
    rng = random.Random(seed)
    return {q: rng.choice(OPTIONS) for q in range(1, n_questions + 1)}


def load_answer_key(path: str | Path) -> dict[int, str]:
    """
    Load an answer key from a JSON file.
    Expected format: {"1": "A", "2": "B", ...}
    """
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v.upper() for k, v in raw.items()}


def save_answer_key(key: dict[int, str], path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in sorted(key.items())}, f, indent=2)


# ── core spatial logic ───────────────────────────────────────────────────────

def _parse_detections(result) -> tuple[list[BubbleBox], list[BubbleBox], list[BubbleBox]]:
    """Convert a YOLO result object into typed BubbleBox lists."""
    rois, marked_boxes, question_boxes = [], [], []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return rois, marked_boxes, question_boxes

    for box in boxes:
        cls  = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        b = BubbleBox(cls, conf, x1, y1, x2, y2)
        if   cls == CLS_ROI:      rois.append(b)
        elif cls == CLS_MARKED:   marked_boxes.append(b)
        elif cls == CLS_QUESTION: question_boxes.append(b)

    return rois, marked_boxes, question_boxes


def _cluster_into_columns(question_boxes: list[BubbleBox],
                           gap_px: float = COLUMN_GAP_PX) -> list[list[BubbleBox]]:
    """
    Group question rows into vertical columns by their x-centre.
    Returns a list of columns ordered left-to-right, each column sorted top-to-bottom.
    """
    if not question_boxes:
        return []

    sorted_by_x = sorted(question_boxes, key=lambda b: b.cx)
    columns: list[list[BubbleBox]] = [[sorted_by_x[0]]]

    for box in sorted_by_x[1:]:
        if box.cx - columns[-1][-1].cx > gap_px:
            columns.append([box])
        else:
            columns[-1].append(box)

    # Sort each column top-to-bottom
    for col in columns:
        col.sort(key=lambda b: b.cy)

    return columns


def _determine_option(marked_cx: float,
                      question_x1: float,
                      question_x2: float) -> Optional[str]:
    """
    Given the x-centre of a filled bubble and the x-extents of its question row,
    determine which option (A/B/C/D) was selected.

    The question row is assumed to contain 4 equally-spaced option slots.
    """
    q_w = question_x2 - question_x1
    slot_w = q_w / 4.0  # width of each option slot

    # Relative x position within the question row
    rel_x = marked_cx - question_x1
    # Clamp to valid range
    rel_x = max(0.0, min(q_w - 0.01, rel_x))

    slot_idx = int(rel_x / slot_w)          # 0, 1, 2, 3
    slot_idx = min(slot_idx, 3)             # guard against floating-point edge case

    return OPTIONS[slot_idx]


def _y_overlap_fraction(marked: BubbleBox, question: BubbleBox) -> float:
    """Fraction of the marked bubble's height that overlaps the question row's y-range."""
    ov_y1 = max(marked.y1, question.y1)
    ov_y2 = min(marked.y2, question.y2)
    overlap = max(0.0, ov_y2 - ov_y1)
    return overlap / marked.h if marked.h > 0 else 0.0


def _x_within_question(marked: BubbleBox, question: BubbleBox,
                        column_x1: float, column_x2: float) -> bool:
    """
    Check that the marked bubble's x-centre falls within the question row's x-range
    (with a small tolerance equal to half a bubble width on each side).
    """
    tol = marked.w * 0.5
    return (column_x1 - tol) <= marked.cx <= (column_x2 + tol)


def _assign_marked_to_questions(
        marked_boxes: list[BubbleBox],
        columns: list[list[BubbleBox]],
) -> dict[int, list[BubbleBox]]:
    """
    Map each marked bubble to its question number (1-indexed, read left-to-right,
    top-to-bottom across columns).

    Returns {question_number: [list of BubbleBox that were marked in that row]}.
    """
    # Build a flat list of (question_number, BubbleBox) for all detected question rows
    question_map: dict[int, BubbleBox] = {}  # q_num → question BubbleBox
    q_num = 1
    for col in columns:
        for q_box in col:
            question_map[q_num] = q_box
            q_num += 1

    # For each column, determine its x-range
    col_x_ranges: list[tuple[float, float]] = []
    for col in columns:
        x1s = [b.x1 for b in col]
        x2s = [b.x2 for b in col]
        col_x_ranges.append((min(x1s), max(x2s)))

    # Pre-compute which column each question belongs to
    q_to_col_range: dict[int, tuple[float, float]] = {}
    q_num = 1
    for col_idx, col in enumerate(columns):
        for _ in col:
            q_to_col_range[q_num] = col_x_ranges[col_idx]
            q_num += 1

    assignments: dict[int, list[BubbleBox]] = {qn: [] for qn in question_map}

    for m in marked_boxes:
        best_q: Optional[int] = None
        best_overlap: float   = 0.0

        for qn, q_box in question_map.items():
            col_x1, col_x2 = q_to_col_range[qn]

            # The marked bubble must be horizontally within this column
            if not _x_within_question(m, q_box, col_x1, col_x2):
                continue

            y_ov = _y_overlap_fraction(m, q_box)
            if y_ov >= MIN_Y_OVERLAP_FRAC and y_ov > best_overlap:
                best_overlap = y_ov
                best_q = qn

        if best_q is not None:
            assignments[best_q].append(m)

    return assignments


# ── main grading function ─────────────────────────────────────────────────────

def grade_omr_image(
    image_path: str | Path,
    model: YOLO,
    answer_key: dict[int, str],
    conf_thresh: float = 0.25,
    iou_thresh:  float = 0.45,
    img_size:    int   = 640,
    device: int | str  = 0,
    column_gap_px:     float = COLUMN_GAP_PX,
) -> OMRResult:
    """
    Grade a single OMR sheet image.

    Parameters
    ----------
    image_path  : path to the input image
    model       : loaded YOLO model
    answer_key  : {question_number: option}   e.g. {1: 'A', 2: 'C', ...}
    conf_thresh : minimum detection confidence
    iou_thresh  : NMS IoU threshold
    img_size    : inference image size
    device      : CUDA device index or 'cpu'
    column_gap_px : pixel gap threshold to separate question columns

    Returns
    -------
    OMRResult dataclass with full per-question breakdown and summary statistics.
    """
    image_path = str(image_path)

    # ── 1. inference ────────────────────────────────────────────────────────
    preds  = model.predict(
        source  = image_path,
        imgsz   = img_size,
        conf    = conf_thresh,
        iou     = iou_thresh,
        device  = device,
        verbose = False,
    )
    result = preds[0]

    _, marked_boxes, question_boxes = _parse_detections(result)

    if not question_boxes:
        # No questions detected at all — return empty result
        return OMRResult(
            image_path      = image_path,
            total_questions = len(answer_key),
            answer_key      = answer_key,
            question_results = [],
        )

    # ── 2. cluster questions into columns ────────────────────────────────────
    columns = _cluster_into_columns(question_boxes, gap_px=column_gap_px)

    # ── 3. assign marked bubbles to question rows ────────────────────────────
    assignments = _assign_marked_to_questions(marked_boxes, columns)

    # ── 4. build per-question results ────────────────────────────────────────
    question_results: list[QuestionResult] = []

    q_num = 1
    for col in columns:
        for q_box in col:
            marks       = assignments.get(q_num, [])
            correct_ans = answer_key.get(q_num)

            if len(marks) == 0:
                # Blank — no bubble filled
                qr = QuestionResult(
                    question_number = q_num,
                    detected_answer = None,
                    correct_answer  = correct_ans,
                    is_correct      = False if correct_ans else None,
                    status          = "blank",
                    confidence      = 0.0,
                    question_bbox   = (q_box.x1, q_box.y1, q_box.x2, q_box.y2),
                    marked_bboxes   = [],
                )

            elif len(marks) > 1:
                # Multi-marked — penalise as wrong
                options_marked = [
                    _determine_option(m.cx, q_box.x1, q_box.x2) for m in marks
                ]
                avg_conf = float(np.mean([m.conf for m in marks]))
                qr = QuestionResult(
                    question_number = q_num,
                    detected_answer = "+".join(sorted(set(options_marked))),
                    correct_answer  = correct_ans,
                    is_correct      = False,
                    status          = "multi-marked",
                    confidence      = avg_conf,
                    question_bbox   = (q_box.x1, q_box.y1, q_box.x2, q_box.y2),
                    marked_bboxes   = [(m.x1, m.y1, m.x2, m.y2) for m in marks],
                )

            else:
                # Single mark — determine option
                mark   = marks[0]
                option = _determine_option(mark.cx, q_box.x1, q_box.x2)
                is_correct = (option == correct_ans) if correct_ans else None
                qr = QuestionResult(
                    question_number = q_num,
                    detected_answer = option,
                    correct_answer  = correct_ans,
                    is_correct      = is_correct,
                    status          = "answered",
                    confidence      = mark.conf,
                    question_bbox   = (q_box.x1, q_box.y1, q_box.x2, q_box.y2),
                    marked_bboxes   = [(mark.x1, mark.y1, mark.x2, mark.y2)],
                )

            question_results.append(qr)
            q_num += 1

    return OMRResult(
        image_path       = image_path,
        total_questions  = len(question_results),
        answer_key       = answer_key,
        question_results = question_results,
    )


# ── visual overlay ────────────────────────────────────────────────────────────

def draw_graded_overlay(
    image_path: str | Path,
    omr_result: OMRResult,
    out_path: Optional[str | Path] = None,
    show_question_boxes: bool = True,
) -> np.ndarray:
    """
    Draw a colour-coded overlay on the OMR image:
        Green   = correct answer
        Red     = wrong answer
        Orange  = multi-marked
        Grey    = blank

    Returns the annotated image as a BGR numpy array.
    Optionally saves it to out_path.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    STATUS_COLORS = {
        "correct":      (50,  200,  50),    # green
        "wrong":        (50,   50, 220),    # red
        "multi-marked": (50,  165, 255),    # orange
        "blank":        (160, 160, 160),    # grey
    }

    def resolve_color(r: QuestionResult) -> tuple:
        if r.status == "blank":        return STATUS_COLORS["blank"]
        if r.status == "multi-marked": return STATUS_COLORS["multi-marked"]
        return STATUS_COLORS["correct"] if r.is_correct else STATUS_COLORS["wrong"]

    for r in omr_result.question_results:
        color = resolve_color(r)

        # Draw question-row bounding box
        if show_question_boxes:
            qx1, qy1, qx2, qy2 = [int(v) for v in r.question_bbox]
            cv2.rectangle(img, (qx1, qy1), (qx2, qy2), color, 1)
            cv2.putText(img, f"Q{r.question_number}", (qx1, qy1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

        # Draw marked bubble(s)
        for (mx1, my1, mx2, my2) in r.marked_bboxes:
            mx1, my1, mx2, my2 = int(mx1), int(my1), int(mx2), int(my2)
            cv2.rectangle(img, (mx1, my1), (mx2, my2), color, 2)
            # Label the detected option
            label = r.detected_answer or "?"
            cv2.putText(img, label, (mx1, my1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # ── score banner at top ─────────────────────────────────────────────────
    h, w = img.shape[:2]
    banner_h = 40
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)
    txt = (f"Score: {omr_result.score}/{omr_result.total_questions}  "
           f"({omr_result.percentage}%)  "
           f"Correct: {omr_result.correct}  Wrong: {omr_result.wrong}  "
           f"Blank: {omr_result.blank}  Multi: {omr_result.multi_marked}")
    cv2.putText(banner, txt, (8, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)
    img = np.vstack([banner, img])

    if out_path is not None:
        cv2.imwrite(str(out_path), img)

    return img


# ── CLI / quick test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ROOT      = Path(__file__).parent
    MODEL_PT  = ROOT / "runs" / "omr_finetune" / "run1" / "weights" / "best.pt"
    RESULTS   = ROOT / "results"
    KEY_PATH  = RESULTS / "answer_key.json"
    OUT_DIR   = RESULTS / "graded"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Grade OMR sheets")
    parser.add_argument("images", nargs="*",
                        help="Image file(s) to grade. Defaults to test set.")
    parser.add_argument("--key",   default=str(KEY_PATH),
                        help="Path to answer key JSON (generated if missing).")
    parser.add_argument("--model", default=str(MODEL_PT),
                        help="Path to YOLO weights.")
    parser.add_argument("--conf",  type=float, default=0.25)
    parser.add_argument("--questions", type=int, default=100,
                        help="Total number of questions on the sheet.")
    args = parser.parse_args()

    # ── load / generate answer key ──────────────────────────────────────────
    key_path = Path(args.key)
    if key_path.exists():
        answer_key = load_answer_key(key_path)
        print(f"Loaded answer key from {key_path}")
    else:
        answer_key = generate_random_answer_key(n_questions=args.questions)
        save_answer_key(answer_key, key_path)
        print(f"Generated random answer key → {key_path}")

    # ── load model ──────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # ── resolve images ──────────────────────────────────────────────────────
    if args.images:
        images = [Path(p) for p in args.images]
    else:
        images = sorted((ROOT / "data" / "test" / "images").glob("*.jpg"))
    
    print(f"Grading {len(images)} image(s) …\n")

    all_results = []
    for img_path in images:
        print(f"  → {img_path.name}")
        omr_result = grade_omr_image(
            image_path  = img_path,
            model       = model,
            answer_key  = answer_key,
            conf_thresh = args.conf,
        )

        # Print per-question table (first 20 only for brevity)
        print(f"     Questions detected : {omr_result.total_questions}")
        print(f"     Score              : {omr_result.score}/{omr_result.total_questions}"
              f"  ({omr_result.percentage}%)")
        print(f"     Correct/Wrong/Blank/Multi: "
              f"{omr_result.correct} / {omr_result.wrong} / "
              f"{omr_result.blank} / {omr_result.multi_marked}")

        # First 20 questions detailed
        print("     Q#   Detected  Expected  Status")
        print("     " + "-" * 40)
        for r in omr_result.question_results[:20]:
            mark = "✓" if r.is_correct else ("✗" if r.is_correct is False else "?")
            print(f"     Q{r.question_number:<3}  {str(r.detected_answer):<8}  "
                  f"{str(r.correct_answer):<8}  {r.status}  {mark}")
        if len(omr_result.question_results) > 20:
            print(f"     ... ({len(omr_result.question_results) - 20} more questions)")

        # Save annotated image
        out_img_path = OUT_DIR / f"graded_{img_path.stem}.jpg"
        draw_graded_overlay(img_path, omr_result, out_path=out_img_path)
        print(f"     Overlay saved → {out_img_path}\n")

        all_results.append(omr_result.to_dict())

    # ── save combined JSON ──────────────────────────────────────────────────
    out_json = RESULTS / "grading_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined grading JSON → {out_json}")
