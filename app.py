import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import create_kew_cnn
from src.data_loader import preprocess_image
from src.utils import load_config

# Page config
st.set_page_config(
    page_title="🌺 Kew-MNIST Botanical Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config
config = load_config('config.yaml')

# Class names
CLASS_NAMES = ['flower', 'fruit', 'leaf', 'plant_tag', 'stem', 'whole_plant']

@st.cache_resource
def load_models():
    """Load both trained models"""
    try:
        original_model = tf.keras.models.load_model('models/kew_mnist_original_model.h5')
        synthetic_model = tf.keras.models.load_model('models/kew_mnist_synthetic_model.h5')
        return original_model, synthetic_model
    except:
        # If saved models don't exist, create dummy models for demo
        st.warning("Pre-trained models not found. Creating demo models...")
        model = create_kew_cnn(config['model'])
        return model, model

def predict_image(image, model):
    """Make prediction on single image"""
    processed = preprocess_image(image)
    prediction = model.predict(np.expand_dims(processed, axis=0))
    return prediction[0]

def plot_predictions(predictions, model_name):
    """Plot prediction probabilities"""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(CLASS_NAMES)))
    bars = ax.bar(CLASS_NAMES, predictions, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title(f'{model_name} Predictions')
    
    # Add value labels on bars
    for bar, pred in zip(bars, predictions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pred:.2%}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Main app
st.title("🌺 Kew-MNIST Botanical Classifier")
st.markdown("""
### Compare Original vs Synthetic-Enhanced Models
This demo showcases the performance difference between CNN models trained on:
- **Original Model**: Trained only on original Kew-MNIST data
- **Synthetic Model**: Enhanced with AI-generated botanical images
""")

# Sidebar
with st.sidebar:
    st.header("📊 Model Performance")
    
    # Performance metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Original': [91.97, 91.83, 91.97, 91.87],
        'Synthetic': [93.42, 93.38, 93.42, 93.37]
    })
    
    st.dataframe(metrics_df.set_index('Metric'))
    
    st.markdown("---")
    st.header("📈 Improvement")
    st.metric("Accuracy Gain", "+1.45%", "+1.45%")
    st.metric("F1-Score Gain", "+1.50%", "+1.50%")
    
    st.markdown("---")
    st.header("🔍 About")
    st.markdown("""
    The synthetic-enhanced model shows:
    - Better performance on underrepresented classes
    - More robust feature learning
    - Improved generalization
    """)

# Load models
original_model, synthetic_model = load_models()

# File uploader
uploaded_file = st.file_uploader(
    "Upload a botanical image (flower, fruit, leaf, plant tag, stem, or whole plant)",
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    
    # Create three columns
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        st.subheader("📷 Input Image")
        st.image(image, use_column_width=True)
    
    # Make predictions
    original_pred = predict_image(image, original_model)
    synthetic_pred = predict_image(image, synthetic_model)
    
    with col2:
        st.subheader("🔵 Original Model")
        predicted_class_orig = CLASS_NAMES[np.argmax(original_pred)]
        confidence_orig = np.max(original_pred)
        
        st.metric("Prediction", predicted_class_orig)
        st.metric("Confidence", f"{confidence_orig:.2%}")
        
        fig_orig = plot_predictions(original_pred, "Original Model")
        st.pyplot(fig_orig)
    
    with col3:
        st.subheader("🟢 Synthetic-Enhanced Model")
        predicted_class_syn = CLASS_NAMES[np.argmax(synthetic_pred)]
        confidence_syn = np.max(synthetic_pred)
        
        st.metric("Prediction", predicted_class_syn)
        st.metric("Confidence", f"{confidence_syn:.2%}")
        
        fig_syn = plot_predictions(synthetic_pred, "Synthetic Model")
        st.pyplot(fig_syn)
    
    # Comparison
    st.markdown("---")
    st.subheader("📊 Prediction Comparison")
    
    comparison_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Original Model': original_pred,
        'Synthetic Model': synthetic_pred,
        'Difference': synthetic_pred - original_pred
    })
    
    # Highlight differences
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: lightgreen' if v else '' for v in is_max]
    
    styled_df = comparison_df.style.apply(highlight_max, subset=['Original Model', 'Synthetic Model'])
    st.dataframe(styled_df)

else:
    # Show example results when no image is uploaded
    st.info("👆 Upload an image to see model predictions")
    
    st.markdown("---")
    st.subheader("📸 Example Results")
    
    # Create example visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("docs/images/sample_images_original.jpeg", 
                caption="Sample Original Images", use_column_width=True)
    
    with col2:
        st.image("docs/images/synthetic_sample_images.png",
                caption="Sample Synthetic Images", use_column_width=True)
    
    st.markdown("---")
    st.subheader("📈 Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("docs/images/comparison_class_accuracy.png",
                caption="Per-Class Accuracy Comparison", use_column_width=True)
    
    with col2:
        st.image("docs/images/training_comparison.png",
                caption="Training History Comparison", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | 
    <a href='https://github.com/yusufmo1/kew-mnist-synthetic'>GitHub</a> | 
    <a href='notebooks/'>View Notebooks</a></p>
</div>
""", unsafe_allow_html=True)