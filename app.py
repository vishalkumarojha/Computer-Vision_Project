import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageFilter

st.set_page_config(page_title="Edge Detection – Comparison", layout="wide")

st.title("🧱 Edge Detection – Comparison Dashboard")
st.write("Upload an image, tweak preprocessing, and compare multiple edge detection methods side by side.")

# ----------------- Helper Functions ----------------- #

def load_image(file):
    """Load uploaded image as RGB PIL Image."""
    img = Image.open(file).convert("RGB")
    return img

def pil_to_cv_gray(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV grayscale."""
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray

def apply_preprocessing(gray_img: np.ndarray, blur_ksize: int, sharpen: bool,
                        use_threshold: bool, thresh_value: int) -> np.ndarray:
    """
    Apply Gaussian blur, optional sharpening, and optional binary threshold
    to a grayscale image.
    """
    processed = gray_img.copy()

    # Gaussian blur
    if blur_ksize > 1:
        processed = cv2.GaussianBlur(processed, (blur_ksize, blur_ksize), 0)

    # Sharpen (simple kernel)
    if sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        processed = cv2.filter2D(processed, ddepth=-1, kernel=kernel)

    # Binary threshold
    if use_threshold:
        _, processed = cv2.threshold(processed, thresh_value, 255, cv2.THRESH_BINARY)

    return processed

def laplacian_edges(gray_img: np.ndarray) -> np.ndarray:
    """Laplacian edge detection using 3x3 kernel."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=np.float32)
    edges = cv2.filter2D(gray_img, ddepth=-1, kernel=kernel)
    edges = cv2.convertScaleAbs(edges)
    return edges

def pil_find_edges(gray_img: np.ndarray) -> np.ndarray:
    """PIL FIND_EDGES applied on a grayscale image (converted from numpy)."""
    pil_gray = Image.fromarray(gray_img)
    edges = pil_gray.filter(ImageFilter.FIND_EDGES)
    return np.array(edges)

def sobel_xy(gray_img: np.ndarray) -> np.ndarray:
    """Sobel combined (XY) edge detection."""
    sobelx = cv2.Sobel(src=gray_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobely = cv2.Sobel(src=gray_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobelxy = cv2.magnitude(sobelx, sobely)
    sobelxy = cv2.convertScaleAbs(sobelxy)
    return sobelxy

def canny_edges(gray_img: np.ndarray, t1: int, t2: int) -> np.ndarray:
    """Canny edge detection."""
    edges = cv2.Canny(image=gray_img, threshold1=t1, threshold2=t2)
    return edges

# ----------------- Sidebar Controls ----------------- #

st.sidebar.header("⚙️ Preprocessing Controls")

# Blur kernel size (odd only)
blur_slider = st.sidebar.slider("Gaussian Blur Kernel Size", 1, 15, 3, 2)
if blur_slider % 2 == 0:
    blur_slider += 1  # ensure odd

sharpen_flag = st.sidebar.checkbox("Apply Sharpening", value=False)

use_threshold = st.sidebar.checkbox("Apply Binary Threshold", value=False)
thresh_value = st.sidebar.slider("Threshold Value", 0, 255, 128, 1, disabled=not use_threshold)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Canny Parameters")
canny_t1 = st.sidebar.slider("Canny Threshold 1", 0, 255, 50, 1)
canny_t2 = st.sidebar.slider("Canny Threshold 2", 0, 255, 150, 1)

st.sidebar.markdown("---")
display_mode = st.sidebar.radio("Original Image Display", ["Color", "Grayscale"])

# ----------------- File Uploader ----------------- #

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is None:
    st.info("👆 Upload an image to start comparing edge detection methods.")
    st.stop()

# Load image
pil_img = load_image(uploaded_file)

# Original display (color or grayscale)
if display_mode == "Color":
    st.subheader("Original Image (Color)")
    st.image(pil_img, use_container_width=True)
else:
    st.subheader("Original Image (Grayscale)")
    st.image(pil_img.convert("L"), use_container_width=True)

# ----------------- Preprocessing ----------------- #

cv_gray = pil_to_cv_gray(pil_img)
preprocessed = apply_preprocessing(
    cv_gray,
    blur_ksize=blur_slider,
    sharpen=sharpen_flag,
    use_threshold=use_threshold,
    thresh_value=thresh_value,
)

st.subheader("Preprocessed Image (Grayscale)")
st.caption("This is the image after blur/sharpen/threshold. All edge methods use this as input.")
st.image(preprocessed, clamp=True, use_container_width=True)

# ----------------- Edge Detection Comparison ----------------- #

st.subheader("🔍 Edge Detection Comparison")

lap_img = laplacian_edges(preprocessed)
sobel_img = sobel_xy(preprocessed)
canny_img = canny_edges(preprocessed, canny_t1, canny_t2)
pil_edges_img = pil_find_edges(preprocessed)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Laplacian (Kernel)**")
    st.image(lap_img, clamp=True, use_container_width=True)

with col2:
    st.markdown("**Sobel (XY)**")
    st.image(sobel_img, clamp=True, use_container_width=True)

with col3:
    st.markdown("**Canny**")
    st.image(canny_img, clamp=True, use_container_width=True)

with col4:
    st.markdown("**PIL FIND_EDGES**")
    st.image(pil_edges_img, clamp=True, use_container_width=True)

st.caption("Adjust preprocessing and Canny parameters from the sidebar to see how each method responds.")
