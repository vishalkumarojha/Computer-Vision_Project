import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageFilter

st.set_page_config(page_title="Edge Detection Playground", layout="centered")

st.title("🧱 Edge Detection Playground")
st.write("Upload an image and try different edge detection methods.")

# --- Helper functions ---------------------------------------------------------
def apply_laplacian_pil(pil_img: Image.Image) -> Image.Image:
    """Laplacian edge detection using custom kernel (PIL)."""
    gray = pil_img.convert("L")
    kernel = (-1, -1, -1,
              -1,  8, -1,
              -1, -1, -1)
    final = gray.filter(ImageFilter.Kernel((3, 3), kernel, 1, 0))
    return final

def apply_pil_find_edges(pil_img: Image.Image) -> Image.Image:
    """PIL built-in FIND_EDGES filter."""
    gray = pil_img.convert("L")
    edged = gray.filter(ImageFilter.FIND_EDGES)
    return edged

def pil_to_cv_gray(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV grayscale uint8 array."""
    gray = pil_img.convert("L")
    arr = np.array(gray, dtype=np.uint8)
    return arr

def apply_sobel(cv_gray: np.ndarray):
    """Sobel X, Y, XY on grayscale OpenCV image."""
    img_blur = cv2.GaussianBlur(cv_gray, (3, 3), 0, 0)

    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.convertScaleAbs(sobelxy)

    return sobelx, sobely, sobelxy

def apply_canny(cv_gray: np.ndarray, t1: int = 100, t2: int = 200) -> np.ndarray:
    """Canny edge detection on grayscale OpenCV image."""
    img_blur = cv2.GaussianBlur(cv_gray, (3, 3), 0, 0)
    edges = cv2.Canny(image=img_blur, threshold1=t1, threshold2=t2)
    return edges

# --- Sidebar controls ---------------------------------------------------------
st.sidebar.header("⚙️ Options")
method = st.sidebar.radio(
    "Select edge detection method:",
    [
        "Laplacian (PIL Kernel)",
        "PIL FIND_EDGES",
        "Sobel (X / Y / XY)",
        "Canny"
    ]
)

if method == "Canny":
    st.sidebar.subheader("Canny Parameters")
    t1 = st.sidebar.slider("Threshold 1", 0, 255, 100, 1)
    t2 = st.sidebar.slider("Threshold 2", 0, 255, 200, 1)
else:
    t1, t2 = 100, 200  # default, not used for others

# --- File uploader ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is None:
    st.info("👆 Upload an image to get started.")
    st.stop()

# Load image as PIL (for display and PIL methods)
pil_img = Image.open(uploaded_file).convert("RGB")

st.subheader("Original Image")
st.image(pil_img, use_container_width=True)

# --- Apply selected method ----------------------------------------------------
st.subheader("Edge Detection Result")

if method == "Laplacian (PIL Kernel)":
    result = apply_laplacian_pil(pil_img)
    st.image(result, caption="Laplacian Edge Detection (PIL)", use_container_width=True)

elif method == "PIL FIND_EDGES":
    result = apply_pil_find_edges(pil_img)
    st.image(result, caption="PIL FIND_EDGES", use_container_width=True)

elif method == "Sobel (X / Y / XY)":
    cv_gray = pil_to_cv_gray(pil_img)
    sobelx, sobely, sobelxy = apply_sobel(cv_gray)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(sobelx, clamp=True, caption="Sobel X", use_container_width=True)
    with col2:
        st.image(sobely, clamp=True, caption="Sobel Y", use_container_width=True)
    with col3:
        st.image(sobelxy, clamp=True, caption="Sobel XY", use_container_width=True)

elif method == "Canny":
    cv_gray = pil_to_cv_gray(pil_img)
    edges = apply_canny(cv_gray, t1=t1, t2=t2)
    st.image(edges, clamp=True, caption=f"Canny Edges (t1={t1}, t2={t2})", use_container_width=True)

st.caption("Built with Streamlit + OpenCV + PIL.")
