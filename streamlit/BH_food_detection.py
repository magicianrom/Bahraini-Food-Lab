
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Bahrain Cuisine App",
    page_icon="🍗",
    layout="wide"
)

st.title("🇧🇭 Bahrain Cuisine Detector")


# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
    
except Exception as e:
    st.error(f"Could not load YOLO model: {e}")
    st.stop()

# -----------------------------
# UPLOAD IMAGE
# -----------------------------
st.subheader("📸 Upload Your Food Image")

file = st.file_uploader(
    "Upload the picture here",
    type=["jpg", "jpeg", "png", "WEBP"]
)

if file is not None:

    image = Image.open(file).convert("RGB")

    st.subheader("Uploaded Image 🖼️")

    st.image(
        image,
        caption="Your uploaded food image",
        width=300
    )

    # -----------------------------
    # DETECT FOOD BUTTON
    # -----------------------------
    if st.button("🔍 Detect Food", type="primary"):

        with st.spinner("AI is detecting your food..."):

            # Convert image to NumPy array
            image_array = np.array(image)

            # Run YOLO prediction
            results = model.predict(
                source=image_array,
                conf=0.25,
                verbose=False
            )

            result = results[0]

            # Keep original image colors
            annotated_image = np.array(image).copy()

            # -----------------------------
            # DRAW BLACK BOXES
            # -----------------------------
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:

                for box in boxes:

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    food_name = model.names[class_id]

                    # Draw black bounding box
                    cv2.rectangle(
                        annotated_image,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 0),
                        2
                    )

                    # Food name and confidence
                    label = f"{food_name} {confidence:.2f}"

                    cv2.putText(
                        annotated_image,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2
                    )

                # -----------------------------
                # DISPLAY RESULTS
                # -----------------------------
                st.success("Food detection completed!")

                st.subheader("🍽️ Detection Results")

                st.image(
                    annotated_image,
                    caption="Detected Food",
                    width=600
                )

                # -----------------------------
                # DISPLAY FOOD DETAILS
                # -----------------------------
                st.subheader("📋 Detected Food Details")

                for box in boxes:

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    food_name = model.names[class_id]

                    st.write(f"🍗 **Food:** {food_name}")

                    st.write(
                        f"🎯 **Confidence:** {confidence * 100:.2f}%"
                    )

                    st.divider()

            else:
                st.warning(
                    "No food detected. Please try another image."
                )


