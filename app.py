import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import time
import pytesseract
import re

# Load YOLOv8 model
model = YOLO(r"D:\final_st\best.pt")  # your yolov8 model

# Load OCR
#reader = easyocr.Reader(['en'])



st.set_page_config(page_title=" Number Plate Detection", layout="wide")


st.markdown("""
<style>
div[data-baseweb="tab-list"] {
    justify-content: center !important;
}
</style>
""", unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(["🏠 Home", "📷 Upload Image", "🎥 Webcam"])

with tab1:
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #0D47A1, #42A5F5);
        padding: 5px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 5px;">
        <h1 style="margin-bottom: 10px;">🚘 Number Plate Detection & OCR</h1>
        
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2,3])

    with col1:
        st.image(r"D:\final_st\env\whatisanpr_1.png", use_container_width=True)

    with col2:
        st.markdown("""
        ### ✨ What This App Can Do
        <div style="font-size:17px; line-height:1.6;">
            • 🔍 Detect vehicle number plates using <b>YOLOv8</b>  
            • 🔤 Extract clean text using <b>EasyOCR</b>  
            • 📷 Upload image and get results instantly  
            • 🎥 Real-time webcam detection  
        </div>
        <br>

        ### 🚀 How to Use
        <div style="font-size:17px; line-height:1.6;">
            1️⃣ Go to <b>"📷 Upload Image"</b> to upload a picture  
            <br>
            2️⃣ Go to <b>"🎥 Webcam"</b> to scan plates live  
            <br>
            3️⃣ View cropped plates + extracted text  
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <br><hr>
    <div style="text-align:center; color:gray; font-size:15px;">
        Developed using <b>YOLOv8</b>, <b>EasyOCR</b>, and <b>Streamlit</b>.
    </div>
    <br>
    """, unsafe_allow_html=True)

# OCR function
# If installed in default Windows path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


reader = easyocr.Reader(['en'], gpu=False)

def ocr_on_plate_crop(crop):

    # preprocessing (BEST for number plates) ---
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # VERY MILD cleaning — keeps real characters intact
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Resize 2× for Ocr clarity
    gray_big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # EasyOCR
    easy_texts = reader.readtext(gray_big, detail=0)
    easy_text = "".join(easy_texts).replace(" ", "") if easy_texts else ""

    # Tesseract
    tess_text = pytesseract.image_to_string(
        gray_big,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ).strip().replace(" ", "")

    # Pick longer result (more characters = more correct)
    final = easy_text if len(easy_text) > len(tess_text) else tess_text

    # Clean output
    final = re.sub(r"[^A-Z0-9]", "", final)
    
#plate formatting: KL21T8413 → KL 21 T 8413
    pattern = r"^([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{3,4})$"
    match = re.match(pattern, final)
    if match:
        final = f"{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}"

    return {
        "easyocr_raw": easy_texts,
        "tesseract_raw": tess_text,
        "plate_text": final
    }

with tab2:
    st.title("📸 Upload Image - Number Plate Detection")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = model(img_bgr)

        start_time = time.time()
        results = model(img_bgr)
        end_time = time.time()

        detected_plates = []
        r = results[0]

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            crop = img_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            ocr_result = ocr_on_plate_crop(crop)

            detected_plates.append({
                "crop": crop_rgb,
                "confidence": conf,
                "easyocr_raw": ocr_result["easyocr_raw"],
                "tesseract_raw": ocr_result["tesseract_raw"],
                "plate_text": ocr_result["plate_text"],
            })

        st.success(f"Detected {len(detected_plates)} plate(s) in {end_time - start_time:.2f} seconds")

        for idx, plate in enumerate(detected_plates):
            st.markdown(f"### Plate {idx+1}: **{plate['plate_text'] or 'No OCR'}**")
            st.image(plate["crop"], width=300)

            with st.expander(f"OCR Details for Plate {idx+1}"):
                st.write("EasyOCR:", plate["easyocr_raw"])
                st.write("Tesseract:", plate["tesseract_raw"])


# LIVE WEBCAM PAGE


with tab3:
    st.title("🎥 Live Webcam - Real-time Number Plate Detection")

    run_webcam = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.empty()

    if run_webcam:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Webcam not found!")
        else:
            st.info("Webcam running...")

        while run_webcam:

            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from webcam")
                break

            # --- YOLO detection ---
            results = model.predict(frame)

            annotated = frame.copy()
            detections = results[0].boxes

            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # --- Crop plate ---
                crop = frame[y1:y2, x1:x2]

                # --- OCR ---
                ocr_out = ocr_on_plate_crop(crop)
                plate_text = ocr_out["plate_text"]

                # Overlay OCR text
                cv2.putText(annotated, plate_text, (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Show annotated webcam feed
            FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        cap.release()


















