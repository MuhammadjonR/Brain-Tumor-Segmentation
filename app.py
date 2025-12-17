import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from skimage import filters, morphology, measure, feature, segmentation
from skimage.morphology import disk, remove_small_objects
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
import nibabel as nib
import pydicom
from collections import deque
import segmentation_models_pytorch as smp
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Brain MRI Segmentation",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'tumor_stats' not in st.session_state:
    st.session_state.tumor_stats = None
if 'previous_method' not in st.session_state:
    st.session_state.previous_method = None
if 'previous_category' not in st.session_state:
    st.session_state.previous_category = None
if 'previous_params' not in st.session_state:
    st.session_state.previous_params = {}
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'preprocessed_image' not in st.session_state:
    st.session_state.preprocessed_image = None
if 'mask' not in st.session_state:
    st.session_state.mask = None
if 'overlay' not in st.session_state:
    st.session_state.overlay = None
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None

# ==================== IMAGE LOADING FUNCTIONS ====================

def load_image(uploaded_file):
    """Load image from various formats"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension in ['jpg', 'jpeg', 'png']:
            img = Image.open(uploaded_file).convert('L')
            return np.array(img)
        
        elif file_extension in ['nii', 'gz']:
            nii_img = nib.load(uploaded_file)
            img_data = nii_img.get_fdata()
            if len(img_data.shape) == 3:
                img_data = img_data[:, :, img_data.shape[2]//2]
            return img_data.astype(np.float32)
        
        elif file_extension == 'dcm':
            dcm = pydicom.dcmread(uploaded_file)
            return dcm.pixel_array.astype(np.float32)
        
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
            
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def preprocess_image(img):
    """Apply preprocessing pipeline"""
    # Convert to float
    img = img.astype(np.float32)
    
    # Resize to 256x256 using Lanczos interpolation
    img_pil = Image.fromarray(img)
    img_pil = img_pil.resize((256, 256), Image.LANCZOS)
    img = np.array(img_pil)
    
    # Normalize to [0, 1]
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    
    # Apply Gaussian smoothing
    img = filters.gaussian(img, sigma=0.5)
    
    return img

# ==================== TRADITIONAL SEGMENTATION METHODS ====================

def otsu_thresholding(img):
    """Otsu's method for binary segmentation"""
    threshold = filters.threshold_otsu(img)
    mask = img > threshold
    return mask.astype(np.uint8)

def adaptive_thresholding(img, block_size=35, c_constant=2):
    """Adaptive thresholding with local statistics"""
    img_uint8 = (img * 255).astype(np.uint8)
    mask = cv2.adaptiveThreshold(
        img_uint8, 
        1, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_constant
    )
    return mask

def kmeans_clustering(img, n_clusters=3):
    """K-means clustering returning brightest cluster"""
    h, w = img.shape
    img_flat = img.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(img_flat)
    
    # Find brightest cluster
    cluster_means = []
    for i in range(n_clusters):
        cluster_pixels = img_flat[labels == i]
        cluster_means.append(cluster_pixels.mean())
    
    brightest_cluster = np.argmax(cluster_means)
    mask = (labels == brightest_cluster).reshape(h, w)
    
    return mask.astype(np.uint8)

def region_growing(img, intensity_threshold=0.05):
    """Seed-based region growing with BFS"""
    h, w = img.shape
    
    # Find seed (brightest pixel)
    seed_y, seed_x = np.unravel_index(np.argmax(img), img.shape)
    seed_value = img[seed_y, seed_x]
    
    # Initialize
    mask = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)
    
    # BFS with 8-connected neighborhood
    queue = deque([(seed_y, seed_x)])
    visited[seed_y, seed_x] = True
    mask[seed_y, seed_x] = 1
    
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    while queue:
        y, x = queue.popleft()
        
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                
                if abs(img[ny, nx] - seed_value) < intensity_threshold:
                    mask[ny, nx] = 1
                    queue.append((ny, nx))
    
    return mask

def watershed_segmentation(img):
    """Watershed segmentation with distance transform"""
    # Binary threshold
    threshold = filters.threshold_otsu(img)
    binary = img > threshold
    
    # Morphological operations
    binary = morphology.binary_closing(binary, disk(3))
    binary = remove_small_objects(binary, min_size=50)
    
    # Distance transform
    distance = distance_transform_edt(binary)
    
    # Find markers
    coords = peak_local_max(distance, min_distance=20, labels=binary)
    mask_markers = np.zeros(distance.shape, dtype=bool)
    mask_markers[tuple(coords.T)] = True
    markers = measure.label(mask_markers)
    
    # Watershed
    labels = segmentation.watershed(-distance, markers, mask=binary)
    mask = labels > 0
    
    return mask.astype(np.uint8)

def morphological_skull_stripping(img):
    """Complete skull stripping pipeline"""
    # Otsu threshold
    threshold = filters.threshold_otsu(img)
    mask = img > threshold
    
    # Binary closing
    mask = morphology.binary_closing(mask, disk(5))
    
    # Fill holes
    mask = ndi.binary_fill_holes(mask)
    
    # Remove small objects
    mask = remove_small_objects(mask, min_size=1000)
    
    # Erosion and dilation
    mask = morphology.erosion(mask, disk(3))
    mask = morphology.dilation(mask, disk(3))
    
    # Select largest component
    labeled = measure.label(mask)
    if labeled.max() > 0:
        largest = np.argmax(np.bincount(labeled.flat)[1:]) + 1
        mask = labeled == largest
    
    return mask.astype(np.uint8)

def canny_edge_detection(img, sigma=1.0, low_threshold=0.05, high_threshold=0.15):
    """Canny edge detection with morphological post-processing"""
    edges = feature.canny(img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    
    # Post-processing
    edges = morphology.closing(edges, disk(2))
    edges = ndi.binary_fill_holes(edges)
    
    return edges.astype(np.uint8)

def sobel_edge_detection(img):
    """Sobel edge detection with gradient magnitude"""
    sobel_h = filters.sobel_h(img)
    sobel_v = filters.sobel_v(img)
    
    # Gradient magnitude
    magnitude = np.hypot(sobel_h, sobel_v)
    
    # Threshold with Otsu
    threshold = filters.threshold_otsu(magnitude)
    edges = magnitude > threshold
    
    # Post-processing
    edges = morphology.closing(edges, disk(2))
    edges = ndi.binary_fill_holes(edges)
    
    return edges.astype(np.uint8)

def laplacian_of_gaussian(img, sigma=2.0):
    """LoG edge detection with zero crossings"""
    # Apply Gaussian then Laplacian
    gaussian_img = filters.gaussian(img, sigma=sigma)
    laplacian = filters.laplace(gaussian_img)
    
    # Find zero crossings
    zero_crossings = np.zeros_like(laplacian, dtype=bool)
    
    # Check for sign changes in all directions
    for i in range(1, laplacian.shape[0]-1):
        for j in range(1, laplacian.shape[1]-1):
            patch = laplacian[i-1:i+2, j-1:j+2]
            if patch.min() < 0 and patch.max() > 0:
                zero_crossings[i, j] = True
    
    # Morphological processing
    edges = morphology.closing(zero_crossings, disk(1))
    edges = ndi.binary_fill_holes(edges)
    
    return edges.astype(np.uint8)

def active_contour_segmentation(img, alpha=0.015, beta=10):
    """Active contour (snakes) segmentation"""
    h, w = img.shape
    
    # Initialize circular contour
    center_y, center_x = h // 2, w // 2
    radius = int(0.6 * min(h, w) / 2)
    
    theta = np.linspace(0, 2*np.pi, 400)
    init_x = center_x + radius * np.cos(theta)
    init_y = center_y + radius * np.sin(theta)
    init = np.array([init_x, init_y]).T
    
    # Apply active contour
    try:
        snake = segmentation.active_contour(
            filters.gaussian(img, 3),
            init,
            alpha=alpha,
            beta=beta,
            gamma=0.001,
            max_iterations=2500
        )
        
        # Fill contour
        from skimage.draw import polygon
        rr, cc = polygon(snake[:, 1], snake[:, 0], img.shape)
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask[rr, cc] = 1
        
        return mask
    except:
        return np.zeros(img.shape, dtype=np.uint8)

def chan_vese_segmentation(img, mu=0.25, iterations=100):
    """Chan-Vese active contour segmentation"""
    # Checkerboard initialization
    init_level_set = segmentation.checkerboard_level_set(img.shape, 6)
    
    # Apply Chan-Vese
    cv = segmentation.chan_vese(
        img,
        mu=mu,
        lambda1=1,
        lambda2=1,
        tol=1e-3,
        max_num_iter=iterations,
        dt=0.5,
        init_level_set=init_level_set,
        extended_output=False
    )
    
    return cv.astype(np.uint8)

def threshold_connected_components(img):
    """Threshold with largest connected component selection"""
    threshold = filters.threshold_otsu(img)
    binary = img > threshold
    
    # Label connected components
    labeled = measure.label(binary)
    
    # Select largest component (ignore background)
    if labeled.max() > 0:
        largest = np.argmax(np.bincount(labeled.flat)[1:]) + 1
        mask = labeled == largest
    else:
        mask = binary
    
    return mask.astype(np.uint8)

# ==================== DEEP LEARNING FUNCTIONS ====================

# Global variables for model
DL_MODEL = None
DL_DEVICE = None
DL_TRANSFORM = None
CLASSIFIER_MODEL = None
CLASSIFIER_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_TRANSFORM = None


def load_classifier_model(
    model_path="models/tumor_classifier2.pth"
):
    global CLASSIFIER_MODEL, CLASSIFIER_TRANSFORM

    if CLASSIFIER_MODEL is not None:
        return CLASSIFIER_MODEL

    # 🔹 Rebuild architecture (MUST MATCH TRAINING)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    state = torch.load(model_path, map_location=CLASSIFIER_DEVICE)
    model.load_state_dict(state, strict=True)

    model.to(CLASSIFIER_DEVICE)
    model.eval()

    CLASSIFIER_MODEL = model

    # 🔹 SAME normalization as training
    CLASSIFIER_TRANSFORM = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    print("✅ Classifier loaded")
    return CLASSIFIER_MODEL
def predict_tumor_class(img_np, threshold=0.5):
    """
    Returns:
        is_tumor (bool)
        probability (float)
    """
    model = load_classifier_model()

    if not isinstance(img_np, np.ndarray):
        raise TypeError("img_np must be np.ndarray")

    # grayscale → RGB
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)

    # uint8 safety
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)

    transformed = CLASSIFIER_TRANSFORM(image=img_np)
    x = transformed["image"].unsqueeze(0).to(CLASSIFIER_DEVICE)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    return prob > threshold, prob

def load_dl_model(model_path=r"models/brain_mri_unet_state.pth"):
    global DL_MODEL, DL_DEVICE, DL_TRANSFORM

    try:
        DL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 🔹 REBUILD ARCHITECTURE (MUST MATCH TRAINING)
        DL_MODEL = smp.Unet(
            encoder_name="resnet34",     # change ONLY if trained differently
            encoder_weights=None,
            in_channels=3,
            classes=1
        )

        state = torch.load(model_path, map_location=DL_DEVICE)
        DL_MODEL.load_state_dict(state)

        DL_MODEL.to(DL_DEVICE)
        DL_MODEL.eval()

        DL_TRANSFORM = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        return True

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return False

def deep_learning_segmentation(img):
    """Two-stage safe segmentation: segment only if tumor detected"""
    global DL_MODEL, DL_DEVICE, DL_TRANSFORM
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img)}")
    
    try:
        # Initialize transform if not loaded
        if DL_TRANSFORM is None:
            DL_TRANSFORM = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        # Load model if needed
        if DL_MODEL is None:
            model_loaded = load_dl_model()
            if not model_loaded:
                st.error("❌ Could not load model. Using fallback segmentation.")
                threshold = filters.threshold_otsu(img)
                return (img > threshold).astype(np.uint8), None
        
        # Convert grayscale to RGB (model expects RGB)
        img_rgb = np.stack([img, img, img], axis=-1)
        img_rgb_uint8 = (img_rgb * 255).astype(np.uint8)
        
        # Apply transform
        transformed = DL_TRANSFORM(image=img_rgb_uint8)
        input_tensor = transformed["image"].unsqueeze(0).to(DL_DEVICE)
        
        # Get prediction probabilities
        with torch.no_grad():
            output = DL_MODEL(input_tensor)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Stage 1: check tumor presence
        tumor_detected = prob.max() > 0.05  # tweak threshold if needed
        
        if not tumor_detected:
            # No tumor → empty mask
            mask = np.zeros_like(prob, dtype=np.uint8)
            tumor_stats = {
    'detected': True,
    'tumor_area_percent': (mask.sum()/mask.size)*100,
    'tumor_percentage': (mask.sum()/mask.size)*100,  # add this for backward compatibility
    'max_probability': float(prob.max()),
    'mean_probability': float(prob[mask>0].mean()),
    'tumor_pixels': int(mask.sum()),
    'probability_map': prob
}

        else:
            # Stage 2: create mask
            mask = (prob > 0.5).astype(np.uint8)
            tumor_stats = {
                'detected': True,
                'tumor_area_percent': (mask.sum()/mask.size)*100,
                'max_probability': float(prob.max()),
                'mean_probability': float(prob[mask>0].mean()),
                'tumor_pixels': int(mask.sum()),
                'probability_map': prob
            }
        
        return mask, tumor_stats

    except ImportError as e:
        st.warning(f"⚠️ Missing dependencies: {str(e)}")
        st.info("💡 Install required packages: `pip install torch albumentations`")
        threshold = filters.threshold_otsu(img)
        return (img > threshold).astype(np.uint8), None
    
    except Exception as e:
        st.error(f"❌ Error during deep learning segmentation: {str(e)}")
        threshold = filters.threshold_otsu(img)
        return (img > threshold).astype(np.uint8), None
def deep_learning_segmentation_with_classification(img):
    """
    Stage 1: classification
    Stage 2: segmentation ONLY if tumor
    """

    is_tumor, cls_prob = predict_tumor_class(img, threshold=0.6)

    if not is_tumor:
        mask = np.zeros_like(img, dtype=np.uint8)
        tumor_stats = {
            "detected": False,
            "classifier_probability": cls_prob,
            "tumor_percentage": 0.0,
            "tumor_pixels": 0,
            "probability_map": None
        }
        return mask, tumor_stats

    # Tumor detected → segmentation
    mask, seg_stats = deep_learning_segmentation(img)

    seg_stats["classifier_probability"] = cls_prob
    return mask, seg_stats


# ==================== VISUALIZATION FUNCTIONS ====================

def create_overlay(img, mask):
    img_rgb = np.stack([img, img, img], axis=-1)

    mask_bin = (mask > 0).astype(bool)

    overlay = img_rgb.copy()
    overlay[mask_bin] = (
        overlay[mask_bin] * 0.6 + np.array([1.0, 0.0, 0.0]) * 0.4
    )

    return np.clip(overlay, 0, 1)


def display_results(img, mask, overlay, filename, tumor_stats=None):
    """Display results in three-column layout"""
    st.markdown("---")
    st.markdown("### 📊 Segmentation Results")
    
    # Three-column layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.imshow(img, cmap='gray')
        ax1.set_title("Original MRI", fontsize=14, fontweight='bold')
        ax1.axis('off')
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.imshow(mask, cmap='gray')
        ax2.set_title("Segmentation Mask", fontsize=14, fontweight='bold')
        ax2.axis('off')
        st.pyplot(fig2)
        plt.close()
    
    with col3:
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.imshow(overlay)
        ax3.set_title("Mask Overlay", fontsize=14, fontweight='bold')
        ax3.axis('off')
        st.pyplot(fig3)
        plt.close()
    
    # Show probability heatmap if available (Deep Learning)
    if (
    isinstance(tumor_stats, dict)
    and tumor_stats.get("probability_map") is not None
):

        st.markdown("---")
        st.markdown("### 🔥 Prediction Probability Heatmap")
        
        col1, col2, col3 = st.columns(3)
        
        with col2:
            fig_heat, ax_heat = plt.subplots(figsize=(6, 6))
            im = ax_heat.imshow(tumor_stats['probability_map'], cmap='hot')
            ax_heat.set_title("Tumor Probability (Deep Learning)", fontsize=14, fontweight='bold')
            ax_heat.axis('off')
            plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
            st.pyplot(fig_heat)
            plt.close()
    
    # Statistics
    st.markdown("---")
    st.markdown("### 📈 Segmentation Statistics")
    
    total_pixels = mask.size
    segmented_pixels = np.sum(mask > 0)
    segmentation_pct = (segmented_pixels / total_pixels) * 100
    
    # Standard statistics
    if not tumor_stats or not isinstance(tumor_stats, dict):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Pixels", f"{total_pixels:,}")
        with col2:
            st.metric("Segmented Pixels", f"{segmented_pixels:,}")
        with col3:
            st.metric("Segmentation %", f"{segmentation_pct:.2f}%")
        with col4:
            st.metric("Image Size", "256×256")

    else:
        st.markdown("#### 🧠 Tumor Detection Analysis")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Tumor Detected", '1' if tumor_stats["detected"] else "❌ No")

        with col2:
            st.metric("Tumor Area", f"{tumor_stats['tumor_area_percent']:.2f}%")



        with col3:
            st.metric("Max Probability", f"{tumor_stats['max_probability']:.3f}")

        with col4:
            mean_prob = (
                f"{tumor_stats['mean_probability']:.3f}"
                if tumor_stats["detected"] else "N/A"
            )
            st.metric("Mean Prob (Tumor)", mean_prob)

        with col5:
            st.metric("Tumor Pixels", f"{tumor_stats['tumor_pixels']:,}")

        if tumor_stats["detected"]:
            st.info(
            f"""
🎯 **Detection Summary**
- Tumor area: **{tumor_stats.get('tumor_percentage', tumor_stats.get('tumor_area_percent', 0)):.2f}%**
- Max confidence: **{tumor_stats['max_probability']:.1%}**
- Mean tumor confidence: **{tumor_stats['mean_probability']:.1%}**
- Tumor pixels: **{tumor_stats['tumor_pixels']:,} / {total_pixels:,}**
"""
        )
        else:
            st.success("✅ No significant tumor regions detected")

    
    # General image info
    st.markdown("---")
    st.markdown("### 🖼️ Image Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Image Size", "256×256")
    with col2:
        st.metric("Total Pixels", f"{total_pixels:,}")
    with col3:
        st.metric("Segmented Pixels", f"{segmented_pixels:,}")
    
    # Download buttons
    st.markdown("---")
    st.markdown("### 💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        buf = BytesIO()
        mask_img.save(buf, format='PNG')
        st.download_button(
            label="⬇️ Download Mask",
            data=buf.getvalue(),
            file_name=f"mask_{filename}",
            mime="image/png"
        )
    
    with col2:
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        buf = BytesIO()
        overlay_img.save(buf, format='PNG')
        st.download_button(
            label="⬇️ Download Overlay",
            data=buf.getvalue(),
            file_name=f"overlay_{filename}",
            mime="image/png"
        )
    
    with col3:
        original_img = Image.fromarray((img * 255).astype(np.uint8))
        buf = BytesIO()
        original_img.save(buf, format='PNG')
        st.download_button(
            label="⬇️ Download Original",
            data=buf.getvalue(),
            file_name=f"original_{filename}",
            mime="image/png"
        )

# ==================== MAIN APPLICATION ====================

def main():
    # Title
    st.title("🧠 Brain MRI Segmentation")
    st.markdown("Advanced medical image segmentation using traditional methods and deep learning")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload & Configure")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an MRI image",
            type=['jpg', 'jpeg', 'png', 'nii', 'dcm'],
            help="Upload brain MRI in JPG, PNG, NIfTI, or DICOM format"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Loaded: {uploaded_file.name}")
            st.session_state.uploaded_filename = uploaded_file.name
        
        st.markdown("---")
        st.header("🔬 Segmentation Method")
        
        # Method category
        method_category = st.radio(
            "Select Category",
            ["Traditional Methods", "Deep Learning"],
            help="Choose between classical image processing or deep learning"
        )
        
        # Method selection
        current_method = None
        current_params = {}
        
        if method_category == "Traditional Methods":
            st.session_state.tumor_stats = None,
            methods = [
                "Otsu Thresholding",
                "Adaptive Thresholding",
                "K-means (k=2)",
                "K-means (k=3)",
                "K-means (k=4)",
                "Region Growing",
                "Watershed Segmentation",
                "Morphological Skull Stripping",
                "Canny Edge Detection",
                "Sobel Edge Detection",
                "Laplacian of Gaussian (LoG)",
                "Active Contour (Snakes)",
                "Chan-Vese Segmentation",
                "Threshold + Connected Components"
            ]
            
            current_method = st.selectbox(
                "Choose Method",
                methods,
                help="Select a traditional segmentation algorithm"
            )
            
            # Method-specific parameters
            st.markdown("---")
            st.header("⚙️ Method Parameters")
            
            if current_method == "Adaptive Thresholding":
                current_params['block_size'] = st.slider(
                    "Block Size",
                    min_value=3,
                    max_value=99,
                    value=35,
                    step=2,
                    key="adaptive_block_size",
                    help="Size of pixel neighborhood for threshold calculation (must be odd)"
                )
                current_params['c_constant'] = st.slider(
                    "C Constant",
                    min_value=-10,
                    max_value=10,
                    value=2,
                    step=1,
                    key="adaptive_c_constant",
                    help="Constant subtracted from weighted mean"
                )
            
            elif current_method == "Region Growing":
                current_params['intensity_threshold'] = st.slider(
                    "Intensity Threshold",
                    min_value=0.01,
                    max_value=0.20,
                    value=0.05,
                    step=0.01,
                    key="region_threshold",
                    help="Maximum intensity difference from seed pixel"
                )
            
            elif current_method == "Canny Edge Detection":
                current_params['sigma'] = st.slider(
                    "Gaussian Sigma",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    key="canny_sigma",
                    help="Standard deviation of Gaussian filter"
                )
                current_params['low_threshold'] = st.slider(
                    "Low Threshold",
                    min_value=0.01,
                    max_value=0.20,
                    value=0.05,
                    step=0.01,
                    key="canny_low",
                    help="Lower bound for hysteresis thresholding"
                )
                current_params['high_threshold'] = st.slider(
                    "High Threshold",
                    min_value=0.10,
                    max_value=0.50,
                    value=0.15,
                    step=0.01,
                    key="canny_high",
                    help="Upper bound for hysteresis thresholding"
                )
            
            elif current_method == "Laplacian of Gaussian (LoG)":
                current_params['sigma'] = st.slider(
                    "Gaussian Sigma",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.1,
                    key="log_sigma",
                    help="Standard deviation for Gaussian smoothing"
                )
            
            elif current_method == "Active Contour (Snakes)":
                current_params['alpha'] = st.slider(
                    "Alpha (Elasticity)",
                    min_value=0.001,
                    max_value=0.050,
                    value=0.015,
                    step=0.001,
                    key="snake_alpha",
                    help="Snake length shape parameter (elasticity)"
                )
                current_params['beta'] = st.slider(
                    "Beta (Rigidity)",
                    min_value=1,
                    max_value=20,
                    value=10,
                    step=1,
                    key="snake_beta",
                    help="Snake smoothness shape parameter (rigidity)"
                )
            
            elif current_method == "Chan-Vese Segmentation":
                current_params['mu'] = st.slider(
                    "Mu (Smoothing)",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.25,
                    step=0.01,
                    key="chanvese_mu",
                    help="Smoothing parameter (length penalty)"
                )
                current_params['iterations'] = st.slider(
                    "Iterations",
                    min_value=50,
                    max_value=300,
                    value=100,
                    step=10,
                    key="chanvese_iter",
                    help="Number of iterations"
                )
        
        else:  # Deep Learning
            current_method = "Deep Learning Model"
            st.info("📦 **Model Configuration**")
            st.markdown("- **Input:** 256×256 grayscale")
            st.markdown("- **Output:** 256×256 binary mask")
            st.markdown("- **Architecture:** U-Net")
            st.markdown("- **Model Path:** `./models/unet_brain.pth`")
    
    # Main content area
    if uploaded_file is None:
        st.info("👈 Please upload a brain MRI image to begin")
        st.markdown("### Supported Formats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- **Standard Images:** JPG, PNG")
            st.markdown("- **Medical Formats:** NIfTI (.nii)")
        with col2:
            st.markdown("- **DICOM:** .dcm files")
            st.markdown("- **Size:** Auto-resized to 256×256")
        return
    
    # Check if reprocessing is needed
    needs_processing = False
    
    if (st.session_state.previous_method != current_method or
        st.session_state.previous_category != method_category or
        st.session_state.previous_params != current_params or
        st.session_state.preprocessed_image is None):
        needs_processing = True
    
    # Process image
    if needs_processing:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load image
        status_text.text("📥 Loading image...")
        progress_bar.progress(10)
        
        raw_img = load_image(uploaded_file)
        if raw_img is None:
            return
        
        # Preprocess
        status_text.text("🔧 Preprocessing image...")
        progress_bar.progress(30)
        
        preprocessed_img = preprocess_image(raw_img)
        st.session_state.preprocessed_image = preprocessed_img
        
        # Apply segmentation
        status_text.text(f"🎯 Applying {current_method}...")
        progress_bar.progress(50)
        
        try:
            if method_category == "Traditional Methods":
                if current_method == "Otsu Thresholding":
                    mask = otsu_thresholding(preprocessed_img)
                elif current_method == "Adaptive Thresholding":
                    mask = adaptive_thresholding(
                        preprocessed_img,
                        current_params['block_size'],
                        current_params['c_constant']
                    )
                elif current_method == "K-means (k=2)":
                    mask = kmeans_clustering(preprocessed_img, n_clusters=2)
                elif current_method == "K-means (k=3)":
                    mask = kmeans_clustering(preprocessed_img, n_clusters=3)
                elif current_method == "K-means (k=4)":
                    mask = kmeans_clustering(preprocessed_img, n_clusters=4)
                elif current_method == "Region Growing":
                    mask = region_growing(preprocessed_img, current_params['intensity_threshold'])
                elif current_method == "Watershed Segmentation":
                    mask = watershed_segmentation(preprocessed_img)
                elif current_method == "Morphological Skull Stripping":
                    mask = morphological_skull_stripping(preprocessed_img)
                elif current_method == "Canny Edge Detection":
                    mask = canny_edge_detection(
                        preprocessed_img,
                        current_params['sigma'],
                        current_params['low_threshold'],
                        current_params['high_threshold']
                    )
                elif current_method == "Sobel Edge Detection":
                    mask = sobel_edge_detection(preprocessed_img)
                elif current_method == "Laplacian of Gaussian (LoG)":
                    mask = laplacian_of_gaussian(preprocessed_img, current_params['sigma'])
                elif current_method == "Active Contour (Snakes)":
                    mask = active_contour_segmentation(
                        preprocessed_img,
                        current_params['alpha'],
                        current_params['beta']
                    )
                elif current_method == "Chan-Vese Segmentation":
                    mask = chan_vese_segmentation(
                        preprocessed_img,
                        current_params['mu'],
                        current_params['iterations']
                    )
                elif current_method == "Threshold + Connected Components":
                    mask = threshold_connected_components(preprocessed_img)
            else:
                mask, tumor_stats = deep_learning_segmentation(preprocessed_img)
                st.session_state.tumor_stats = tumor_stats



            
            progress_bar.progress(80)
            
            # Create overlay
            overlay = create_overlay(preprocessed_img, mask)
            
            # Store results
            st.session_state.mask = mask
            st.session_state.overlay = overlay
            st.session_state.show_results = True
            
            # Update session state
            st.session_state.previous_method = current_method
            st.session_state.previous_category = method_category
            st.session_state.previous_params = current_params.copy()
            
            status_text.text("✅ Processing complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            import time
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
        except Exception as e:
            st.error(f"❌ Error during segmentation: {str(e)}")
            return
    
    # Display results
    if st.session_state.show_results:
        display_results(
        st.session_state.preprocessed_image,
        st.session_state.mask,
        st.session_state.overlay,
        st.session_state.uploaded_filename,
        st.session_state.tumor_stats
)


if __name__ == "__main__":
    main()