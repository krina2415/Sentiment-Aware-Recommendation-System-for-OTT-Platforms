import streamlit as st
import pandas as pd
import re
from textblob import TextBlob
import random
import urllib.parse
import os
import subprocess
import sys
import json
import shutil
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
from torch.utils.data import Dataset, DataLoader

# --- CHECK AND INSTALL REQUIRED PACKAGES ---
def install_required_packages():
    """Install required packages if missing"""
    required_packages = {
        'scikit-learn': 'sklearn',
        'plotly': 'plotly',
        'torch': 'torch',
        'transformers': 'transformers',
        'vaderSentiment': 'vaderSentiment',
        'seaborn': 'seaborn',
        'matplotlib': 'matplotlib'
    }
    
    for pip_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            st.warning(f"Installing {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

# Run installation check
install_required_packages()

# --- UPDATED IMPORTS FOR TRANSFORMERS ---
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    pipeline
)
# --------------------------------------------------

# Page Config
st.set_page_config(
    page_title="Sentiment-Aware OTT Recommender",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tag {
        display: inline-block;
        background-color: #e6f7ff;
        color: #0066cc;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .time-big {
        font-size: 3rem;
        font-weight: bold;
        color: #333;
        margin: 0;
        padding: 0;
    }
    
    /* Rank badges */
    .rank-badge {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 18px;
        margin: auto;
        color: white;
    }
    
    /* Footer Styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #666;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        border-top: 1px solid #eee;
        display: flex;
        justify-content: space-around;
        z-index: 100;
    }
    
    .movie-info-card {
        background-color: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .similarity-badge {
        background-color: #e7f5ff;
        color: #0066cc;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Training Progress */
    .training-progress {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        height: 6px;
        border-radius: 3px;
        margin: 10px 0;
    }
    
    /* Evaluation Metrics */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    
    /* Model Score Indicators */
    .model-score {
        background-color: #f0f8ff;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 5px 0;
        border-left: 3px solid #4169E1;
    }
    
    /* Training Graph Container */
    .graph-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Graph Image Styling */
    .graph-image {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Helpers & Maps
# --------------------------------------------------
MOOD_GENRE_MAP = {
    "Happy": ["Comedy", "Animation", "Musical", "Adventure", "Fantasy"],
    "Sad": ["Drama", "Romance", "Biography", "History", "Family"],
    "Angry": ["Action", "Thriller", "Horror", "Crime"],
    "Romantic": ["Romance", "Drama", "Comedy", "Musical"],
    "Fear": ["Comedy", "Animation", "Family", "Musical"],
    "Surprised": ["Sci-Fi", "Mystery", "Documentary", "Fantasy", "Thriller"]
}

# Emotion labels
EMOTION_LABELS = ["Sad", "Happy", "Romantic", "Angry", "Fear", "Surprised"]
EMOTION_MAPPING = {
    0: "Sad",
    1: "Happy", 
    2: "Romantic",
    3: "Angry",
    4: "Fear",
    5: "Surprised"
}

# Sentiment labels with ONLY Positive and Negative (2 classes)
SENTIMENT_LABELS = ["Negative", "Positive"]
SENTIMENT_MAPPING = {
    0: "Negative",
    1: "Positive"
}

# Samples per class for training
SAMPLES_PER_CLASS = 3000

# --------------------------------------------------
# Custom Dataset Class for Hugging Face
# --------------------------------------------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --------------------------------------------------
# NEW FUNCTION: Sample 3000 per class randomly
# --------------------------------------------------
def sample_3000_per_class(df, text_col, label_col, random_state=42):
    """
    Randomly sample exactly 3000 samples from each class
    """
    st.write("### 📊 Sampling 3000 samples per class...")
    
    sampled_dfs = []
    class_details = []
    
    # Get unique classes
    unique_classes = df[label_col].unique()
    
    for class_val in unique_classes:
        class_df = df[df[label_col] == class_val]
        class_size = len(class_df)
        
        if class_size >= SAMPLES_PER_CLASS:
            # Randomly sample 3000 from this class without replacement
            sampled_class = class_df.sample(n=SAMPLES_PER_CLASS, random_state=random_state)
            status = "✅"
            method = "random sampling"
        else:
            # If less than 3000, use all available and sample with replacement to reach 3000
            sampled_class = class_df.sample(n=SAMPLES_PER_CLASS, 
                                           replace=True, 
                                           random_state=random_state)
            status = "⚠️"
            method = "sampling WITH replacement"
        
        sampled_dfs.append(sampled_class)
        
        # Store details for display
        if label_col == 'sentiment':
            class_name = "Positive" if class_val == 1 else "Negative" if class_val == 0 else str(class_val)
        else:
            # For emotion labels
            if isinstance(class_val, (int, np.integer)):
                class_name = EMOTION_LABELS[class_val] if class_val < len(EMOTION_LABELS) else str(class_val)
            else:
                class_name = str(class_val)
        
        class_details.append({
            'class': class_name,
            'original': class_size,
            'sampled': SAMPLES_PER_CLASS,
            'status': status,
            'method': method
        })
    
    # Combine all sampled classes
    balanced_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle the final dataset
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Display sampling results
    for detail in class_details:
        if detail['status'] == '✅':
            st.success(f"{detail['status']} {detail['class']}: Sampled {detail['sampled']} from {detail['original']} available ({detail['method']})")
        else:
            st.warning(f"{detail['status']} {detail['class']}: Only {detail['original']} available, used {detail['method']} to reach {detail['sampled']}")
    
    st.info(f"📊 Total balanced dataset size: {len(balanced_df)} samples")
    
    return balanced_df

# --------------------------------------------------
# Load Data & Initialize Models
# --------------------------------------------------
@st.cache_resource
def load_models():
    """Load pre-trained DistilBERT models for sentiment and emotion"""
    try:
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Sentiment model (2 classes: negative, positive)
        sentiment_model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        
        # Emotion model (6 classes)
        emotion_model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=6,
            ignore_mismatched_sizes=True
        )
        
        return tokenizer, sentiment_model, emotion_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

@st.cache_data
def load_data():
    # --- PATH CONFIGURATION ---
    base_path = r"C:\Users\patel\OneDrive\Desktop\sentiment"
    
    path_imdb = f"{base_path}\\imdb_reviews.csv"
    path_movies = f"{base_path}\\movies.csv"
    path_ratings = f"{base_path}\\ratings.csv"
    path_tags = f"{base_path}\\tags.csv"
    path_links = f"{base_path}\\links.csv"
    path_emotion = f"{base_path}\\Emotion.csv"
    
    try:
        # 1. Load IMDB Reviews for Sentiment
        sentiment_data = None
        if os.path.exists(path_imdb):
            imdb = pd.read_csv(path_imdb)
            # Check if it has sentiment labels
            if 'sentiment' in imdb.columns:
                sentiment_data = imdb[['review', 'sentiment']].copy()
                # Map positive=1, negative=0
                sentiment_data['sentiment'] = sentiment_data['sentiment'].map({'positive': 1, 'negative': 0})
                st.sidebar.success(f"✅ Loaded sentiment dataset: {len(sentiment_data)} samples")
            else:
                st.warning("IMDB dataset doesn't have 'sentiment' column")
                sentiment_data = pd.DataFrame(columns=['review', 'sentiment'])
        else:
            st.error(f"File not found: {path_imdb}")
            sentiment_data = pd.DataFrame(columns=['review', 'sentiment'])
        
        # 2. Load Emotion Dataset
        emotion_data = None
        if os.path.exists(path_emotion):
            emotion_df = pd.read_csv(path_emotion)
            # Check required columns
            if 'text' in emotion_df.columns and 'label' in emotion_df.columns:
                emotion_data = emotion_df[['text', 'label']].copy()
                emotion_data = emotion_data.rename(columns={'text': 'review'})
                st.sidebar.success(f"✅ Loaded emotion dataset: {len(emotion_data)} samples")
                
                # Convert string labels to numeric if needed
                if emotion_data['label'].dtype == 'object':
                    label_mapping = {
                        'sad': 0, 'happy': 1, 'romantic': 2, 
                        'angry': 3, 'fear': 4, 'surprised': 5
                    }
                    emotion_data['label'] = emotion_data['label'].astype(str).str.lower().map(label_mapping)
                    emotion_data = emotion_data.dropna()
                    emotion_data['label'] = emotion_data['label'].astype(int)
            else:
                st.warning("Emotion dataset doesn't have required columns 'text' and 'label'")
                emotion_data = pd.DataFrame(columns=['review', 'label'])
        else:
            st.warning(f"Emotion dataset not found at: {path_emotion}")
            emotion_data = pd.DataFrame(columns=['review', 'label'])
        
        # 3. Load Movies, Ratings
        movies = pd.read_csv(path_movies)
        ratings = pd.read_csv(path_ratings)
        
        # Calculate average ratings
        avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
        movies = movies.merge(avg_ratings, on="movieId", how="left")
        movies["rating"] = movies["rating"].fillna(0)
        
        # 4. Load Links
        if os.path.exists(path_links):
            links = pd.read_csv(path_links)
            movies = movies.merge(links, on="movieId", how="left")
            movies["imdbId"] = movies["imdbId"].fillna(0).astype(int)
        else:
            movies["imdbId"] = 0
        
        # 5. Handle Tags
        if os.path.exists(path_tags):
            tags = pd.read_csv(path_tags)
            tags['tag'] = tags['tag'].astype(str)
            movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: '|'.join(x)).reset_index()
            movies = movies.merge(movie_tags, on='movieId', how='left')
            movies["tag"] = movies["tag"].fillna("No tags")
        else:
            movies["tag"] = "No tags"
        
        # Extract Year and Simulate Runtime
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
        movies['year'] = pd.to_numeric(movies['year'], errors='coerce').fillna(2000).astype(int)
        
        np.random.seed(42) 
        movies['runtime'] = np.random.randint(70, 180, size=len(movies))
        movies.loc[movies['title'].str.contains('Time Traveler'), 'runtime'] = 84

        return sentiment_data, movies, emotion_data

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# --------------------------------------------------
# Helper function to save training plots permanently
# --------------------------------------------------
def save_training_plots(model_type, metrics, epoch_details=None):
    """Save all training plots permanently to disk"""
    plot_dir = "./training_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    saved_files = {}
    
    # 1. Loss Curves
    if 'loss_history' in metrics and 'val_loss_history' in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(metrics['loss_history']) + 1)
        ax.plot(epochs, metrics['loss_history'], 'b-', label='Training Loss', marker='o', linewidth=2, markersize=8)
        ax.plot(epochs, metrics['val_loss_history'], 'r-', label='Validation Loss', marker='s', linewidth=2, markersize=8)
        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.set_title(f'{model_type.capitalize()} Model: Training & Validation Loss', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        
        loss_path = f"{plot_dir}/{model_type}_loss_curves.png"
        fig.savefig(loss_path, dpi=150, bbox_inches='tight')
        saved_files['loss_curves'] = loss_path
        plt.close(fig)
    
    # 2. Accuracy Curves
    if 'val_accuracy_history' in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(metrics['val_accuracy_history']) + 1)
        ax.plot(epochs, metrics['val_accuracy_history'], 'g-', marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_title(f'{model_type.capitalize()} Model: Validation Accuracy', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        
        # Add value labels
        for i, (x, y) in enumerate(zip(epochs, metrics['val_accuracy_history'])):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
        
        acc_path = f"{plot_dir}/{model_type}_accuracy_curve.png"
        fig.savefig(acc_path, dpi=150, bbox_inches='tight')
        saved_files['accuracy_curve'] = acc_path
        plt.close(fig)
    
    # 3. Confusion Matrix
    if 'confusion_matrix' in metrics and metrics['confusion_matrix'] is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = metrics['confusion_matrix']
        
        if model_type == "sentiment":
            labels = SENTIMENT_LABELS
            cmap = 'Blues'
        else:
            labels = EMOTION_LABELS
            cmap = 'YlOrRd'
        
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                    xticklabels=labels, 
                    yticklabels=labels,
                    ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('Actual', fontsize=14)
        ax.set_title(f'{model_type.capitalize()} Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        cm_path = f"{plot_dir}/{model_type}_confusion_matrix.png"
        fig.savefig(cm_path, dpi=150, bbox_inches='tight')
        saved_files['confusion_matrix'] = cm_path
        plt.close(fig)
    
    # 4. Per-class Performance (for emotion model)
    if model_type == "emotion" and 'classification_report' in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        report = metrics['classification_report']
        classes = list(report.keys())[:-3]  # Exclude accuracy, macro avg, weighted avg
        
        precision = [report[cls]['precision'] * 100 for cls in classes]
        recall = [report[cls]['recall'] * 100 for cls in classes]
        f1 = [report[cls]['f1-score'] * 100 for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        ax.bar(x, recall, width, label='Recall', color='#2ecc71')
        ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
        
        ax.set_xlabel('Emotion Classes', fontsize=14)
        ax.set_ylabel('Score (%)', fontsize=14)
        ax.set_title('Emotion Model: Per-class Performance', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#f8f9fa')
        
        # Add value labels
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            ax.text(i - width, p + 1, f'{p:.1f}', ha='center', va='bottom', fontsize=9)
            ax.text(i, r + 1, f'{r:.1f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width, f + 1, f'{f:.1f}', ha='center', va='bottom', fontsize=9)
        
        per_class_path = f"{plot_dir}/{model_type}_per_class_performance.png"
        fig.savefig(per_class_path, dpi=150, bbox_inches='tight')
        saved_files['per_class_performance'] = per_class_path
        plt.close(fig)
    
    return saved_files

# --------------------------------------------------
# Function to load trained models from disk
# --------------------------------------------------
def load_trained_models_from_disk():
    """Load trained models from disk if available"""
    try:
        models_dir = "./trained_models"
        plots_dir = "./training_plots"
        
        # Check if directory exists
        if not os.path.exists(models_dir):
            st.sidebar.info("No trained models found on disk. Train models first.")
            return False
        
        sentiment_model_path = f"{models_dir}/sentiment_model"
        emotion_model_path = f"{models_dir}/emotion_model"
        
        models_loaded = False
        
        # Load sentiment model
        if os.path.exists(sentiment_model_path) and os.path.exists(f"{sentiment_model_path}/config.json"):
            try:
                # Load models with proper device handling
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Load model and tokenizer
                st.session_state.trained_sentiment_model = DistilBertForSequenceClassification.from_pretrained(
                    sentiment_model_path,
                    local_files_only=True
                )
                st.session_state.sentiment_tokenizer = DistilBertTokenizer.from_pretrained(
                    sentiment_model_path,
                    local_files_only=True
                )
                
                # Move model to device and set to eval mode
                st.session_state.trained_sentiment_model.to(device)
                st.session_state.trained_sentiment_model.eval()
                
                st.session_state.sentiment_model_trained = True
                st.sidebar.success("✅ Loaded trained sentiment model from disk")
                models_loaded = True
                
                # Load metrics if available
                metrics_path = f"{sentiment_model_path}/metrics.json"
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        st.session_state.sentiment_metrics = json.load(f)
                
                # Load plot paths
                if os.path.exists(plots_dir):
                    sentiment_plots = [f for f in os.listdir(plots_dir) if f.startswith('sentiment_')]
                    if sentiment_plots:
                        st.session_state.sentiment_plot_paths = {
                            plot.split('.')[0]: f"{plots_dir}/{plot}"
                            for plot in sentiment_plots
                        }
                
            except Exception as e:
                st.sidebar.warning(f"Could not load sentiment model: {e}")
                st.session_state.trained_sentiment_model = None
                st.session_state.sentiment_tokenizer = None
                st.session_state.sentiment_model_trained = False
        else:
            st.sidebar.info("Sentiment model not found on disk. Please train it first.")
        
        # Load emotion model
        if os.path.exists(emotion_model_path) and os.path.exists(f"{emotion_model_path}/config.json"):
            try:
                # Load models with proper device handling
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Load model and tokenizer
                st.session_state.trained_emotion_model = DistilBertForSequenceClassification.from_pretrained(
                    emotion_model_path,
                    local_files_only=True
                )
                st.session_state.emotion_tokenizer = DistilBertTokenizer.from_pretrained(
                    emotion_model_path,
                    local_files_only=True
                )
                
                # Move model to device and set to eval mode
                st.session_state.trained_emotion_model.to(device)
                st.session_state.trained_emotion_model.eval()
                
                st.session_state.emotion_model_trained = True
                st.sidebar.success("✅ Loaded trained emotion model from disk")
                models_loaded = True
                
                # Load metrics if available
                metrics_path = f"{emotion_model_path}/metrics.json"
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        st.session_state.emotion_metrics = json.load(f)
                
                # Load plot paths
                if os.path.exists(plots_dir):
                    emotion_plots = [f for f in os.listdir(plots_dir) if f.startswith('emotion_')]
                    if emotion_plots:
                        st.session_state.emotion_plot_paths = {
                            plot.split('.')[0]: f"{plots_dir}/{plot}"
                            for plot in emotion_plots
                        }
                
            except Exception as e:
                st.sidebar.warning(f"Could not load emotion model: {e}")
                st.session_state.trained_emotion_model = None
                st.session_state.emotion_tokenizer = None
                st.session_state.emotion_model_trained = False
        else:
            st.sidebar.info("Emotion model not found on disk. Please train it first.")
        
        return models_loaded
            
    except Exception as e:
        st.sidebar.warning(f"Could not load models from disk: {e}")
        return False

# --------------------------------------------------
# SIMPLE TRAINING FUNCTION with 3000 samples per class
# --------------------------------------------------
def simple_train_model(model, tokenizer, train_texts, train_labels, val_texts, val_labels, model_type="sentiment"):
    """Simple training function with 3000 samples per class"""
    
    # Show class distribution
    st.write("### 📊 Training Data Distribution:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for class_val, count in zip(unique, counts):
        if model_type == "sentiment":
            class_name = SENTIMENT_LABELS[class_val]
        else:
            class_name = EMOTION_LABELS[class_val]
        st.write(f"- {class_name}: {count} samples")
    
    st.info(f"🚀 Training {model_type} model with {len(train_texts)} samples...")
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Training parameters
    num_epochs = 2
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_history = []
    val_loss_history = []
    val_accuracy_history = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress
            current_progress = (epoch * len(train_loader) + batch_idx + 1) / (num_epochs * len(train_loader))
            progress_bar.progress(current_progress)
            accuracy = correct / total * 100 if total > 0 else 0
            status_text.text(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {accuracy:.1f}%")
        
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        loss_history.append(avg_loss)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_accuracy = val_correct / val_total * 100 if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_loss_history.append(avg_val_loss)
        val_accuracy_history.append(val_accuracy)
        
        st.info(f"📊 Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.1f}%")
    
    progress_bar.empty()
    status_text.empty()
    
    # Calculate final metrics
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    if len(all_preds) > 0 and len(all_labels) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        # Get classification report
        if model_type == "sentiment":
            report = classification_report(all_labels, all_preds, target_names=SENTIMENT_LABELS, output_dict=True, zero_division=0)
        else:
            report = classification_report(all_labels, all_preds, target_names=EMOTION_LABELS, output_dict=True, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'loss_history': loss_history,
            'val_loss_history': val_loss_history,
            'val_accuracy_history': val_accuracy_history,
            'predictions': all_preds,
            'true_labels': all_labels
        }
    else:
        if model_type == "sentiment":
            cm_shape = (2, 2)
        else:
            cm_shape = (6, 6)
            
        metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'confusion_matrix': np.zeros(cm_shape),
            'classification_report': {},
            'loss_history': loss_history,
            'val_loss_history': val_loss_history,
            'val_accuracy_history': val_accuracy_history,
            'predictions': [],
            'true_labels': []
        }
    
    # Save model and plots permanently
    try:
        models_dir = "./trained_models"
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = f"{models_dir}/{model_type}_model"
        
        # Remove old folder if exists
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        os.makedirs(model_path, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(model_path, safe_serialization=False)
        tokenizer.save_pretrained(model_path)
        
        # Save metrics (excluding numpy arrays that can't be JSON serialized)
        metrics_to_save = {}
        for k, v in metrics.items():
            if k not in ['predictions', 'true_labels', 'confusion_matrix']:
                if isinstance(v, dict):
                    # Handle nested dict (classification_report)
                    metrics_to_save[k] = v
                elif isinstance(v, (np.ndarray, list)):
                    if isinstance(v, np.ndarray):
                        metrics_to_save[k] = v.tolist()
                    else:
                        metrics_to_save[k] = v
                else:
                    metrics_to_save[k] = v
        
        # Save confusion matrix separately as list
        if 'confusion_matrix' in metrics and metrics['confusion_matrix'] is not None:
            metrics_to_save['confusion_matrix'] = metrics['confusion_matrix'].tolist()
        
        with open(f"{model_path}/metrics.json", "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        
        # Save plots permanently
        plot_paths = save_training_plots(model_type, metrics)
        
        # Update session state
        model.to('cpu')
        model.eval()
        
        if model_type == "sentiment":
            st.session_state.trained_sentiment_model = model
            st.session_state.sentiment_tokenizer = tokenizer
            st.session_state.sentiment_model_trained = True
            st.session_state.sentiment_metrics = metrics_to_save
            st.session_state.sentiment_plot_paths = plot_paths
        else:
            st.session_state.trained_emotion_model = model
            st.session_state.emotion_tokenizer = tokenizer
            st.session_state.emotion_model_trained = True
            st.session_state.emotion_metrics = metrics_to_save
            st.session_state.emotion_plot_paths = plot_paths
        
        st.success(f"✅ {model_type.capitalize()} model, metrics, and plots saved permanently!")
        
        # Show saved plot locations
        st.info("**📁 Saved plots:**")
        for plot_name, plot_path in plot_paths.items():
            st.write(f"- {plot_name}: {plot_path}")
        
    except Exception as e:
        st.error(f"❌ Error saving model: {str(e)}")
        return None, None
    
    return model, metrics

# Initialize models and data
tokenizer, sentiment_model, emotion_model = load_models()
sentiment_df, movies_df, emotion_df = load_data()

# Initialize all Session States
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42

if 'similar_movies_results' not in st.session_state:
    st.session_state.similar_movies_results = None

if 'trigger_similar_search' not in st.session_state:
    st.session_state.trigger_similar_search = None

if 'sentiment_model_trained' not in st.session_state:
    st.session_state.sentiment_model_trained = False

if 'emotion_model_trained' not in st.session_state:
    st.session_state.emotion_model_trained = False

if 'sentiment_metrics' not in st.session_state:
    st.session_state.sentiment_metrics = None

if 'emotion_metrics' not in st.session_state:
    st.session_state.emotion_metrics = None

if 'sentiment_plot_paths' not in st.session_state:
    st.session_state.sentiment_plot_paths = {}

if 'emotion_plot_paths' not in st.session_state:
    st.session_state.emotion_plot_paths = {}

# Initialize trained models and tokenizers in session state
if 'trained_sentiment_model' not in st.session_state:
    st.session_state.trained_sentiment_model = None
    
if 'trained_emotion_model' not in st.session_state:
    st.session_state.trained_emotion_model = None
    
if 'sentiment_tokenizer' not in st.session_state:
    st.session_state.sentiment_tokenizer = None
    
if 'emotion_tokenizer' not in st.session_state:
    st.session_state.emotion_tokenizer = None

# Load trained models from disk at startup
load_trained_models_from_disk()

# Helper function for IMDB link
def get_imdb_link(imdb_id, title):
    if imdb_id and imdb_id != 0:
        id_str = str(int(imdb_id)).zfill(7)
        return f"https://www.imdb.com/title/tt{id_str}/"
    else:
        encoded = urllib.parse.quote(title)
        return f"https://www.imdb.com/find?q={encoded}"

# --- Similarity Function ---
def find_similar_movies(movie_title, movies_df, top_n=5):
    if movie_title not in movies_df['title'].values:
        return pd.DataFrame()
    
    target_movie = movies_df[movies_df['title'] == movie_title].iloc[0]
    target_genres = set(str(target_movie['genres']).split('|'))
    target_rating = target_movie['rating']
    
    similarity_scores = []
    for idx, movie in movies_df.iterrows():
        if movie['title'] == movie_title:
            continue
        
        movie_genres = set(str(movie['genres']).split('|'))
        
        if target_genres and movie_genres:
            genre_similarity = len(target_genres.intersection(movie_genres)) / len(target_genres.union(movie_genres))
        else:
            genre_similarity = 0
        
        rating_similarity = 1 - abs(target_rating - movie['rating']) / 5.0
        total_score = 0.7 * genre_similarity + 0.3 * rating_similarity
        
        similarity_scores.append({
            'title': movie['title'],
            'genres': movie['genres'],
            'rating': movie['rating'],
            'total_similarity': total_score,
            'year': movie.get('year', 2000),
            'runtime': movie.get('runtime', 120),
            'imdbId': movie.get('imdbId', 0)
        })
    
    similar_df = pd.DataFrame(similarity_scores)
    if not similar_df.empty:
        similar_df = similar_df.sort_values('total_similarity', ascending=False).head(top_n)
    
    return similar_df

# --- HELPER FUNCTIONS TO CHECK MODEL AVAILABILITY ---
def is_sentiment_model_ready():
    """Check if sentiment model is properly loaded and ready"""
    return (st.session_state.get('sentiment_model_trained', False) and 
            st.session_state.get('trained_sentiment_model') is not None and
            st.session_state.get('sentiment_tokenizer') is not None)

def is_emotion_model_ready():
    """Check if emotion model is properly loaded and ready"""
    return (st.session_state.get('emotion_model_trained', False) and 
            st.session_state.get('trained_emotion_model') is not None and
            st.session_state.get('emotion_tokenizer') is not None)

# --- SENTIMENT AND EMOTION PREDICTION FUNCTIONS ---
def predict_sentiment_distilbert(text):
    """Predict sentiment using trained DistilBERT model - ALWAYS use DistilBERT if available"""
    # Force check for trained model
    if is_sentiment_model_ready():
        try:
            model = st.session_state.trained_sentiment_model
            tokenizer = st.session_state.sentiment_tokenizer
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            
            # Tokenize input
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=128,
                padding='max_length'
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][predicted_class].item()
            
            # Map to sentiment
            if predicted_class in SENTIMENT_MAPPING:
                sentiment = SENTIMENT_MAPPING[predicted_class]
            else:
                sentiment = SENTIMENT_LABELS[min(predicted_class, len(SENTIMENT_LABELS)-1)]
            
            # Get probabilities
            probs = predictions[0].cpu().numpy()
            prob_dict = {SENTIMENT_LABELS[i]: float(probs[i]) for i in range(len(SENTIMENT_LABELS))}
            
            # Debug info
            st.sidebar.success(f"✅ Using DistilBERT for sentiment: {sentiment} ({confidence:.2f})")
            
            return sentiment, confidence, prob_dict
            
        except Exception as e:
            st.sidebar.error(f"DistilBERT sentiment error: {e}")
            # Fallback to TextBlob only if DistilBERT fails
            return fallback_sentiment_textblob(text)
    else:
        # Model not ready, use fallback
        st.sidebar.warning("⚠️ Sentiment model not trained. Using TextBlob fallback.")
        return fallback_sentiment_textblob(text)

def fallback_sentiment_textblob(text):
    """Fallback sentiment analysis using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = "Positive"
        confidence = polarity
    else:
        sentiment = "Negative"
        confidence = abs(polarity)
    
    return sentiment, confidence, None

def predict_emotion_distilbert(text):
    """Predict emotion using trained DistilBERT model - ALWAYS use DistilBERT if available"""
    # Force check for trained model
    if is_emotion_model_ready():
        try:
            model = st.session_state.trained_emotion_model
            tokenizer = st.session_state.emotion_tokenizer
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            
            # Tokenize input
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=128,
                padding='max_length'
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][predicted_class].item()
            
            # Map to emotion
            if predicted_class in EMOTION_MAPPING:
                emotion = EMOTION_MAPPING[predicted_class]
            else:
                emotion = EMOTION_LABELS[min(predicted_class, len(EMOTION_LABELS)-1)]
            
            # Debug info
            st.sidebar.success(f"✅ Using DistilBERT for emotion: {emotion} ({confidence:.2f})")
            
            return emotion, confidence
            
        except Exception as e:
            st.sidebar.error(f"DistilBERT emotion error: {e}")
            # Fallback to TextBlob only if DistilBERT fails
            return fallback_emotion_textblob(text)
    else:
        # Model not ready, use fallback
        st.sidebar.warning("⚠️ Emotion model not trained. Using TextBlob fallback.")
        return fallback_emotion_textblob(text)

def fallback_emotion_textblob(text):
    """Fallback emotion detection using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Map TextBlob polarity to emotions
    if polarity > 0.3:
        emotion = "Happy"
        confidence = polarity
    elif polarity > 0.1:
        emotion = "Romantic"
        confidence = polarity
    elif polarity < -0.3:
        emotion = "Sad"
        confidence = abs(polarity)
    elif polarity < -0.1:
        emotion = "Angry"
        confidence = abs(polarity)
    else:
        # Neutral sentiment - could be Surprised or Fear
        if subjectivity > 0.5:
            emotion = "Surprised"
            confidence = subjectivity
        else:
            emotion = "Fear"
            confidence = 0.5
    
    return emotion, confidence

# --- CALLBACK FUNCTIONS ---
def run_analysis_callback():
    user_text = st.session_state.get("user_text", "")
    
    if not user_text.strip():
        st.warning("Please enter some text!")
        return

    st.session_state.random_seed = np.random.randint(1, 10000)

    # Text correction (optional)
    blob = TextBlob(user_text)
    corrected_text = str(blob.correct())
    final_input = corrected_text if user_text.lower() != corrected_text.lower() else user_text
    
    # Show model status in sidebar
    with st.sidebar:
        st.divider()
        st.subheader("🔍 Model Status")
        if is_sentiment_model_ready():
            st.success("✅ Sentiment Model: DistilBERT")
        else:
            st.warning("⚠️ Sentiment Model: TextBlob (fallback)")
        
        if is_emotion_model_ready():
            st.success("✅ Emotion Model: DistilBERT")
        else:
            st.warning("⚠️ Emotion Model: TextBlob (fallback)")
    
    # ALWAYS try DistilBERT first if available
    if is_sentiment_model_ready():
        sentiment, sentiment_conf, sentiment_probs = predict_sentiment_distilbert(final_input)
        sentiment_model_used = "DistilBERT"
    else:
        # Fallback to TextBlob
        sentiment, sentiment_conf, sentiment_probs = fallback_sentiment_textblob(final_input)
        sentiment_model_used = "TextBlob (fallback)"
    
    # ALWAYS try DistilBERT for emotion first if available
    if is_emotion_model_ready():
        emotion, emotion_conf = predict_emotion_distilbert(final_input)
        emotion_model_used = "DistilBERT"
    else:
        # Fallback to TextBlob
        emotion, emotion_conf = fallback_emotion_textblob(final_input)
        emotion_model_used = "TextBlob (fallback)"
    
    detected_mood = emotion
    
    st.session_state.mood_input = detected_mood
    st.session_state.selected_genres = MOOD_GENRE_MAP.get(detected_mood, ["Comedy", "Drama"])
    
    st.session_state.analysis_results = {
        "original_text": user_text,
        "corrected_text": corrected_text,
        "emotion": emotion,
        "sentiment": sentiment,
        "sentiment_confidence": sentiment_conf,
        "sentiment_probabilities": sentiment_probs,
        "emotion_confidence": emotion_conf,
        "model_used": {
            "sentiment": sentiment_model_used,
            "emotion": emotion_model_used
        },
        "has_run": True
    }

def search_similar_movies():
    search_query = st.session_state.get("movie_search_input", "").strip()
    if not search_query:
        st.warning("Please enter a movie title!")
        return
    
    matching_movies = movies_df[movies_df['title'].str.contains(search_query, case=False, na=False)]
    
    if matching_movies.empty:
        st.warning(f"No movie found with title containing '{search_query}'")
        st.session_state.similar_movies_results = None
        return
    
    selected_movie = matching_movies.iloc[0]
    similar_movies = find_similar_movies(selected_movie['title'], movies_df, top_n=10)
    
    st.session_state.similar_movies_results = {
        'search_query': search_query,
        'selected_movie': selected_movie.to_dict(),
        'similar_movies': similar_movies.to_dict('records') if not similar_movies.empty else [],
        'has_run': True
    }

# --------------------------------------------------
# Sidebar Settings
# --------------------------------------------------
with st.sidebar:
    st.header(" Settings")
    
    # Theme Selector
    st.markdown(" **Theme**")
    theme_option = st.selectbox(
        "Select Theme",
        ["Light", "Dark"],
        label_visibility="collapsed"
    )

    if st.button("Clear All Filters", type="secondary", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    st.divider()

    # Theme CSS
    if theme_option == "Dark":
        st.markdown("""
            <style>
                :root { color-scheme: dark; }
                .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
                    background-color: #000000 !important;
                    color: #ffffff !important;
                }
                section[data-testid="stSidebar"] {
                    background-color: #000000 !important;
                    border-right: 1px solid #333;
                }
                h1, h2, h3 { color: #E50914 !important; }
                h4, h5, h6, p, label, span, div, li { color: #ffffff !important; }
                .stTabs [data-baseweb="tab-list"] button {
                    font-size: 18px !important;
                    font-weight: bold !important;
                    color: white !important;
                }
                .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
                    color: #E50914 !important; 
                    border-bottom-color: #E50914 !important;
                }
                .stTextInput > div > div, .stSelectbox > div > div, .stMultiSelect > div > div, .stTextArea > div > div {
                    background-color: #141414 !important;
                    color: white !important;
                    border: 1px solid #333 !important;
                }
                .stTextInput input, .stTextArea textarea {
                    color: #ffffff !important;
                    -webkit-text-fill-color: #ffffff !important;
                    background-color: #141414 !important;
                    caret-color: #E50914 !important;
                }
                .stTextInput input::placeholder, .stTextArea textarea::placeholder {
                    color: #bbbbbb !important;
                    -webkit-text-fill-color: #bbbbbb !important;
                }
                div[data-baseweb="popover"], div[data-baseweb="popover"] * {
                    background-color: #141414 !important;
                    color: #ffffff !important;
                    border-color: #333 !important;
                }
                li[data-baseweb="option"]:hover, li[data-baseweb="option"]:hover *,
                li[aria-selected="true"], li[aria-selected="true"] * {
                    background-color: #E50914 !important;
                    color: #ffffff !important;
                }
                .stButton > button, .stDownloadButton > button, button[kind="secondary"] {
                    background-color: #E50914 !important;
                    color: white !important;
                    border: none !important;
                    font-weight: bold !important;
                }
                .stButton > button:hover, button[kind="secondary"]:hover { 
                    background-color: #b20710 !important; 
                }
                .tag { background-color: #E50914 !important; color: white !important; border: 1px solid white !important; }
                .yellow-circle {
                    background-color: #FFD700 !important;
                    border-radius: 50%;
                    width: 35px; height: 35px;
                    display: flex; align-items: center; justify-content: center;
                    font-weight: bold;
                }
                .yellow-circle * {
                    color: #000000 !important;
                    -webkit-text-fill-color: #000000 !important;
                    font-family: sans-serif !important;
                    margin: 0 !important;
                    padding: 0 !important;
                }
                svg { fill: white !important; stroke: white !important; }
                .graph-image { border: 2px solid #333; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
                    background-color: #ffffff !important;
                    color: #000000 !important;
                }
                section[data-testid="stSidebar"] {
                    background-color: #f0f2f6 !important;
                    color: #000000 !important;
                }
                h1, h2, h3, h4, h5, h6, p, span, div, label, li {
                    color: #000000 !important;
                }
                .tag {
                    background-color: #ffe6e6 !important;
                    color: #FF4B4B !important;
                    border: 1px solid #FF4B4B !important;
                    font-weight: bold !important;
                }
                .stTextInput > div > div, .stSelectbox > div > div, .stMultiSelect > div > div, .stTextArea > div > div {
                    background-color: #ffffff !important;
                    color: #000000 !important;
                    border: 1px solid #cccccc !important;
                }
                ul[data-baseweb="menu"], div[data-baseweb="popover"] {
                    background-color: #ffffff !important;
                    border: 1px solid #cccccc !important;
                }
                li[data-baseweb="option"] {
                    background-color: #ffffff !important;
                    color: #000000 !important;
                }
                li[data-baseweb="option"]:hover, li[aria-selected="true"] {
                    background-color: #ffcccc !important;
                    color: #000000 !important;
                }
                .stButton > button, .stDownloadButton > button { 
                    background-color: #FF4B4B !important; 
                    color: white !important; 
                    border: none !important;
                }
                .stButton > button:hover {
                    background-color: #b20710 !important;
                }
                .stMultiSelect span[data-baseweb="tag"] {
                    background-color: #FF4B4B !important;
                    color: white !important;
                }
                svg { fill: #000000 !important; stroke: #000000 !important; }
                .graph-image { border: 2px solid #ddd; }
            </style>
        """, unsafe_allow_html=True)

    # Recommendation & Rating Sliders
    num_recs = st.slider("Number of recommendations", 1, 10, 5)
    
    rating_range = st.slider(
        "Select Rating Range", 
        0.0, 5.0, (1.5, 4.5), 0.5
    )
    
    st.divider()
    
    # Advanced Filters
    with st.expander("Advanced Filters", expanded=True):
        st.markdown("**Decade**")
        decade_options = ["1980s", "1990s", "2000s", "2010s", "2020s"]
        selected_decades = st.multiselect(
            "Select Decades",
            decade_options,
            default=["2000s", "2010s"],
            label_visibility="collapsed"
        )
        
        st.markdown("**Duration**")
        duration_options = ["Any", "Short (<90 min)", "Medium (90-120 min)", "Long (>120 min)"]
        selected_duration = st.selectbox(
            "Select Duration",
            duration_options,
            label_visibility="collapsed"
        )
    
    st.divider()

    def on_mood_change():
        new_mood = st.session_state.mood_input
        st.session_state.selected_genres = MOOD_GENRE_MAP.get(new_mood, [])

    st.subheader("Current Mood")
    mood_options = list(MOOD_GENRE_MAP.keys())
    mood = st.radio(
        "How are you feeling?",
        mood_options,
        index=0,
        key="mood_input",
        on_change=on_mood_change,
        label_visibility="collapsed"
    )
    
    if 'selected_genres' not in st.session_state:
        st.session_state.selected_genres = MOOD_GENRE_MAP["Happy"]
    
    st.divider()
    
    st.subheader("Genre Preferences")
    
    all_genres = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", 
                  "Documentary", "Drama", "Fantasy", "Horror", "Musical", 
                  "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
                  "Family", "History", "Music", "Sport", "Film-Noir"]
    
    current_selected = st.session_state.get("selected_genres", ["Comedy", "Drama"])
    valid_selected = [g for g in current_selected if g in all_genres]
    if not valid_selected: valid_selected = ["Comedy", "Drama"]
    
    selected_genres = st.multiselect(
        "Select preferred genres:",
        all_genres,
        default=valid_selected,
        key="selected_genres"
    )
    
    st.divider()
    # Display Options (always True for simplicity)
    show_score = True
    show_tags = True
    show_imdb = True
    show_model_info = True
    show_eval_metrics = True
    show_prob_details = True
    
    st.divider()

# --------------------------------------------------
# Main Content Area (TABS SYSTEM)
# --------------------------------------------------
st.markdown('<h1 class="main-header"> Sentiment-Aware Recommendation System </h1>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analyze & Recommend", "Find Similar Movies", "Favorites", "Top Movies", "Train & Evaluate"])

# --- TAB 1: Main Analysis ---
with tab1:
    
    user_input = st.text_area(
        "How are you feeling or write a review:",
        placeholder=" ",
        height=135,
        key="user_text"
    )

    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    with analyze_col2:
        st.button("Analyze & Recommend", type="primary", use_container_width=True, on_click=run_analysis_callback)

    # Display Results
    if st.session_state.get("analysis_results") and st.session_state.analysis_results["has_run"]:
        
        res = st.session_state.analysis_results
        if isinstance(res, tuple):
            res = res[0]
        # Preview
        with st.expander("Analysis Preview", expanded=True):
            preview_text = res["original_text"]
            word_count = len(preview_text.split()) if preview_text else 0
            char_count = len(preview_text)
            
            # Quick sentiment analysis
            if is_sentiment_model_ready():
                quick_sent, quick_conf, _ = predict_sentiment_distilbert(preview_text)
            else:
                blob = TextBlob(preview_text)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    quick_sent = "Positive"
                else:
                    quick_sent = "Negative"
            
            color = "green" if quick_sent == "Positive" else "red" if quick_sent == "Negative" else "gray"
            
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                st.markdown(f"Quick Analysis: <span style='color:{color}; font-weight:bold'>{quick_sent}</span>", unsafe_allow_html=True)
            with p_col2:
                st.metric("Words", word_count)
            with p_col3:
                st.metric("Characters", char_count)
        
        if res["original_text"].lower() != res["corrected_text"].lower():
            with st.container(border=True):
                 st.info(f"Corrected: {res['corrected_text']}")

        st.divider()
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown("### Sentiment Analysis")
            sentiment = res["sentiment"]
            if sentiment == "Positive":
                st.success(f"**{sentiment}**")
            else:
                st.error(f"**{sentiment}**")
                
            if show_score:
                confidence = res['sentiment_confidence']
                st.metric("Confidence", f"{confidence*100:.2f}%")
                
                # Show model info
                model_type = res["model_used"]["sentiment"]
                if "DistilBERT" in model_type:
                    color = "#28a745"
                else:
                    color = "#6c757d"
                
                st.markdown(f'<div style="background-color: {color}20; padding: 10px; border-radius: 8px; text-align: center; margin-top: 10px;">'
                            f'<div style="font-size: 12px; color: #666;">Model: {model_type}</div>'
                            f'</div>', unsafe_allow_html=True)
        
        with result_col2:
            st.markdown("### Emotion Detection")
            emotion_text = res["emotion"]
            emotion_colors = {
                "Happy": "#FFD700",
                "Sad": "#4169E1",
                "Angry": "#FF4500",
                "Fear": "#8B0000",
                "Romantic": "#FF69B4",
                "Surprised": "#FFA500"
            }
            color = emotion_colors.get(emotion_text, "#4CAF50")
            st.markdown(f"<span style='color:{color}; font-weight:bold; font-size:20px;'>{emotion_text}</span>", unsafe_allow_html=True)
            if show_score:
                st.metric("Confidence", f"{res['emotion_confidence']*100:.2f}%")
        
        # Model Information
        if show_model_info:
            with st.expander("Model Information", expanded=False):
                st.markdown("### Model Details")
                if st.session_state.get("analysis_results"):
                    analysis_res = st.session_state.analysis_results
                    model_info = analysis_res.get("model_used", {})
                st.write(f"**Sentiment Model:** {model_info.get('sentiment', 'Default')}")
                st.write(f"**Emotion Model:** {model_info.get('emotion', 'Default')}")
                
                if is_sentiment_model_ready():
                    st.success("Custom sentiment model is active and loaded (trained with 3000 samples per class)")
                else:
                    st.warning("Using TextBlob fallback for sentiment analysis")
                
                if is_emotion_model_ready():
                    st.success("Custom emotion model is active and loaded (trained with 3000 samples per class)")
                else:
                    st.warning("Using keyword-based emotion detection")
                
                if sentiment_df is not None:
                    st.write(f"**Sentiment Training Data:** {len(sentiment_df)} samples")
                if emotion_df is not None:
                    st.write(f"**Emotion Training Data:** {len(emotion_df)} samples")
        
        current_genre_filter = st.session_state.selected_genres
        
        def genre_match(genres):
            if not current_genre_filter: return True
            movie_genres = str(genres).lower()
            return any(g.lower() in movie_genres for g in current_genre_filter)
        
        filtered = movies_df[movies_df["genres"].apply(genre_match)]
        min_bound, max_bound = rating_range
        filtered = filtered[(filtered["rating"] >= min_bound) & (filtered["rating"] <= max_bound)]

        if selected_decades:
            decade_map = {
                "1980s": (1980, 1989), "1990s": (1990, 1999),
                "2000s": (2000, 2009), "2010s": (2010, 2019), "2020s": (2020, 2029)
            }
            mask = pd.Series([False] * len(filtered), index=filtered.index)
            for dec in selected_decades:
                start, end = decade_map[dec]
                mask |= (filtered["year"] >= start) & (filtered["year"] <= end)
            filtered = filtered[mask]

        if selected_duration != "Any":
            if "Short" in selected_duration:
                filtered = filtered[filtered["runtime"] < 90]
            elif "Medium" in selected_duration:
                filtered = filtered[(filtered["runtime"] >= 90) & (filtered["runtime"] <= 120)]
            elif "Long" in selected_duration:
                filtered = filtered[filtered["runtime"] > 120]
        
        if not filtered.empty:
            if len(filtered) > num_recs:
                 final_recs = filtered.sample(n=num_recs, random_state=st.session_state.random_seed)
            else:
                 final_recs = filtered
        else:
            final_recs = filtered
        
        st.subheader("Recommended Movies")
        
        if not final_recs.empty:
            cols_per_row = 2
            rows = [st.columns(cols_per_row) for _ in range((len(final_recs) + cols_per_row - 1) // cols_per_row)]
            
            for idx, (_, row) in enumerate(final_recs.iterrows()):
                row_idx = idx // cols_per_row
                col_idx = idx % cols_per_row
                with rows[row_idx][col_idx]:
                    with st.container(border=True):
                        t_col, f_col = st.columns([5, 1])
                        with t_col:
                            st.markdown(f"### {row['title'][:40]}")
                        with f_col:
                            is_favorite = any(m['title'] == row['title'] for m in st.session_state.favorites)
                            if is_favorite:
                                if st.button("❤️", key=f"fav_{idx}", help="Remove from Favorites"):
                                    st.session_state.favorites = [m for m in st.session_state.favorites if m['title'] != row['title']]
                                    st.rerun()
                            else:
                                if st.button("🤍", key=f"fav_{idx}", help="Add to Favorites"):
                                    st.session_state.favorites.append({
                                        'title': row['title'],
                                        'rating': row['rating'],
                                        'genres': row['genres'],
                                        'year': row['year'],
                                        'runtime': row['runtime'],
                                        'imdbId': row.get('imdbId', 0)
                                    })
                                    st.rerun()
                        
                        rating_value = row['rating']
                        stars = "⭐" * min(5, int(rating_value))
                        st.markdown(f"{stars} **{rating_value:.1f}/5.0**")
                        genres_list = row['genres'].split("|")
                        genres_html = " ".join([f'<span class="tag">{genre}</span>' for genre in genres_list[:4]])
                        st.markdown(f"**Genres:** {genres_html}", unsafe_allow_html=True)
                        
                        if show_tags and pd.notna(row['tag']) and row['tag'] != "No tags":
                            st.caption(f"**Tags:** {row['tag'][:50]}...")
                        
                        if show_imdb:
                            direct_link = get_imdb_link(row.get('imdbId'), row['title'])
                            st.markdown(f"[![IMDB](https://img.icons8.com/color/20/000000/imdb.png) View on IMDB]({direct_link})")

            # Export button
            st.divider()
            csv_data = final_recs[['title', 'genres', 'rating', 'tag', 'year', 'runtime']].to_csv(index=False).encode('utf-8')
            
            exp_col1, exp_col2, exp_col3 = st.columns([1, 2, 1])
            with exp_col2:
                st.download_button(
                    label="Export Recommendations", 
                    data=csv_data,
                    file_name='my_movie_recommendations.csv',
                    mime='text/csv',
                    use_container_width=True 
                )
        else:
            st.warning(f"No movies found with selected filters.")

# --- TAB 2: Find Similar Movies ---
with tab2:
    if hasattr(st.session_state, 'trigger_similar_search') and st.session_state.trigger_similar_search:
        search_query = st.session_state.trigger_similar_search
        
        matching_movies = movies_df[movies_df['title'].str.contains(search_query, case=False, na=False)]
        
        if not matching_movies.empty:
            selected_movie = matching_movies.iloc[0]
            similar_movies = find_similar_movies(selected_movie['title'], movies_df, top_n=10)
            
            st.session_state.similar_movies_results = {
                'search_query': search_query,
                'selected_movie': selected_movie.to_dict(),
                'similar_movies': similar_movies.to_dict('records') if not similar_movies.empty else [],
                'has_run': True
            }
        
        del st.session_state.trigger_similar_search
    
    st.markdown('<p style="font-size: 16px; font-weight: 600; margin-bottom: 5px;">Search for a Movie</p>', unsafe_allow_html=True)
    
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_input = st.text_input(
            "Enter movie title:",
            placeholder="Search movie (e.g., Time Traveler)",
            key="movie_search_input",
            label_visibility="collapsed"
        )
    
    with search_col2:
        search_clicked = st.button("Search", type="primary", use_container_width=True, on_click=search_similar_movies)
    
    if st.session_state.get("similar_movies_results") and st.session_state.similar_movies_results["has_run"]:
        res = st.session_state.similar_movies_results
        movie = res['selected_movie']
        
        st.markdown('<div class="movie-info-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {movie['title']}")
            rating_value = movie['rating']
            full_stars = int(rating_value)
            half_star = 1 if rating_value - full_stars >= 0.5 else 0
            empty_stars = 5 - full_stars - half_star
            
            stars_html = "★" * full_stars + "½" * half_star + "☆" * empty_stars
            st.markdown(f"**{stars_html}** {rating_value:.1f}/5.0")
            
            year_val = movie.get('year', 1980)
            runtime_val = movie.get('runtime', 84)
            st.markdown(f"**Year:** {year_val} • **Runtime:** {runtime_val} min")
            
            genres_list = str(movie['genres']).split('|')
            genres_html = " ".join([f'<span class="tag">{genre}</span>' for genre in genres_list[:4]])
            st.markdown(f"**Genres:** {genres_html}", unsafe_allow_html=True)
            
        with col2:
            is_favorite = any(m['title'] == movie['title'] for m in st.session_state.favorites)
            if is_favorite:
                if st.button("❤️ Remove from Favorites", key="remove_fav_search", use_container_width=True):
                    st.session_state.favorites = [m for m in st.session_state.favorites if m['title'] != movie['title']]
                    st.rerun()
            else:
                if st.button("🤍 Add to Favorites", key="add_fav_search", type="secondary", use_container_width=True):
                    st.session_state.favorites.append({
                        'title': movie['title'],
                        'rating': rating_value,
                        'genres': movie['genres'],
                        'year': year_val,
                        'runtime': runtime_val,
                        'imdbId': movie.get('imdbId', 0)
                    })
                    st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
            direct_link = get_imdb_link(movie.get('imdbId'), movie['title'])
            st.markdown(f"<div style='text-align: center;'><a href='{direct_link}' target='_blank' style='text-decoration: none; color: #E50914; font-weight: bold;'>View on IMDB <img src='https://img.icons8.com/color/20/000000/imdb.png'/></a></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        similar_count = len(res['similar_movies'])
        st.divider()
        
        if similar_count > 0:
            st.subheader("Similar Movies Found:")
            num_similar_to_show = st.slider("Show top similar movies:", 1, min(10, similar_count), min(5, similar_count))
            
            for i, sim_movie in enumerate(res['similar_movies'][:num_similar_to_show]):
                with st.container(border=True):
                    sim_col1, sim_col2, sim_col3 = st.columns([3, 1, 1])
                    
                    with sim_col1:
                        st.markdown(f"**{i+1}. {sim_movie['title']}**")
    
                        sim_genres = str(sim_movie['genres']).split('|')
                        common_genres = set(genres_list).intersection(set(sim_genres))
    
                        if common_genres:
                            common_genres_html = " ".join([f'<span class="tag" style="background-color:#d4edda;">{genre}</span>' 
                                     for genre in list(common_genres)[:3]])
                            st.markdown(f"**Common genres:** {common_genres_html}", unsafe_allow_html=True)
    
                        sim_rating = sim_movie['rating']
                        sim_stars = "★" * min(5, int(sim_rating))
    
                        st.markdown(f"{sim_stars} **{sim_rating:.1f}/5.0**")
    
                        st.caption(f"Year: {sim_movie.get('year', 2000)} • Runtime: {sim_movie.get('runtime', 120)} min")
                    
                    with sim_col2:
                        similarity_percent = sim_movie['total_similarity'] * 100
                        similarity_color = "#28a745" if similarity_percent > 70 else "#ffc107" if similarity_percent > 50 else "#dc3545"
                        
                        st.markdown(
                            f'<div style="text-align: center; padding: 10px; border-radius: 8px; background-color: {similarity_color}20;">'
                            f'<div style="font-size: 12px; color: #666;">Similarity</div>'
                            f'<div style="color: {similarity_color}; font-size: 24px; font-weight: bold;">{similarity_percent:.0f}%</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    with sim_col3:
                        is_favorite_sim = any(m['title'] == sim_movie['title'] for m in st.session_state.favorites)
                        fav_text = "❤️ Added" if is_favorite_sim else "🤍 Add"
                        fav_key = f"sim_fav_{i}"
                        
                        if st.button(fav_text, key=fav_key, use_container_width=True):
                            if not is_favorite_sim:
                                st.session_state.favorites.append({
                                    'title': sim_movie['title'],
                                    'rating': sim_rating,
                                    'genres': sim_movie['genres'],
                                    'year': sim_movie.get('year', 2000),
                                    'runtime': sim_movie.get('runtime', 120),
                                    'imdbId': sim_movie.get('imdbId', 0)
                                })
                                st.rerun()
                        
                        if show_imdb:
                            direct_link = get_imdb_link(sim_movie.get('imdbId'), sim_movie['title'])
                            st.markdown(f"[View on IMDB]({direct_link})")
            
            st.divider()
            if st.button("Export Similar Movies List", use_container_width=True):
                similar_df = pd.DataFrame(res['similar_movies'])[['title', 'genres', 'rating', 'year', 'runtime', 'total_similarity']]
                similar_df['similarity_percent'] = (similar_df['total_similarity'] * 100).round(1)
                csv_data = similar_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"similar_to_{movie['title'].replace(' ', '_')}.csv",
                    mime='text/csv'
                )
        else:
            st.info("No similar movies found for this title.")

# --- TAB 3: Favorites ---
with tab3:
    st.markdown('<p style="font-size: 16px; font-weight: 600; margin-bottom: 10px;">⭐ Your Favorites</p>', unsafe_allow_html=True)
    
    if st.session_state.favorites:
        clear_col1, clear_col2, clear_col3 = st.columns([1, 2, 1])
        with clear_col2:
            if st.button("Clear All Favorites", type="secondary", use_container_width=True):
                st.session_state.favorites = []
                st.rerun()
        
        st.divider()
        
        for i, movie in enumerate(st.session_state.favorites):
            with st.container(border=True):
                col_left, col_right = st.columns([3, 1])
                
                with col_left:
                    title = movie.get('title', 'Unknown Movie')
                    genres = movie.get('genres', 'Unknown Genres')
                    rating = movie.get('rating', 0)
                    year_val = movie.get('year', 2000)
                    runtime_val = movie.get('runtime', 120)
                    
                    st.markdown(f"### {title}")
                    
                    if genres and genres != "Unknown Genres":
                        if '|' in genres:
                            genres_display = genres.replace('|', ', ')
                        else:
                            genres_display = genres
                        st.markdown(f"**Genres:** {genres_display}")
                    
                    st.markdown(f"⭐ **{rating:.1f}/5.0**")
                    
                    decade_str = f"{str(year_val)[:3]}0s"
                    dur_str = "Short (<90m)" if runtime_val < 90 else "Medium (90-120m)" if runtime_val <= 120 else "Long (>120m)"
                    
                    st.caption(f"**Decade:** {decade_str}")
                    st.caption(f"**Duration:** {dur_str}")
                
                with col_right:
                    remove_key = f"remove_fav_{i}"
                    if st.button("Remove", key=remove_key, use_container_width=True):
                        if i < len(st.session_state.favorites):
                            st.session_state.favorites.pop(i)
                            st.rerun()                  
                    direct_link = get_imdb_link(movie.get('imdbId'), title)
                    st.markdown(f"[View on IMDB]({direct_link})")

    else:
        st.info("You haven't added any favorites yet.")

# --- TAB 4: Top Movies by Genre ---
with tab4:
    st.subheader("🎬 Top 5 Movies by Genre")
    
    # Get all unique genres
    all_genres_list = []
    for genres in movies_df['genres'].dropna():
        if isinstance(genres, str):
            all_genres_list.extend(genres.split('|'))
    
    unique_genres = sorted(set(all_genres_list))
    
    # Genre selection with unique key (selectbox supports key)
    selected_genre = st.selectbox(
        "Select a genre to see top 5 movies:",
        unique_genres,
        key="genre_top5_selector_tab4"
    )
    
    if selected_genre:
        # Filter movies by selected genre
        genre_movies = movies_df[movies_df['genres'].str.contains(selected_genre, na=False)].copy()
        
        # Get top 5 by rating
        top_5 = genre_movies.nlargest(5, 'rating')[['title', 'rating', 'genres', 'year', 'runtime', 'imdbId']]
        
        st.markdown(f"### Top 5 {selected_genre} Movies")
        
        if not top_5.empty:
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                # Create a unique container key for each movie (container supports key)
                movie_container = st.container(border=True, key=f"genre_container_{selected_genre}_{i}_{hash(row['title'])}")
                
                with movie_container:
                    col1, col2, col3, col4 = st.columns([0.5, 3, 2, 1])
                    
                    with col1:
                        # Rank badge
                        rank_colors = {
                            1: "#FFD700",  # Gold
                            2: "#C0C0C0",  # Silver
                            3: "#CD7F32",  # Bronze
                            4: "#4A90E2",  # Blue
                            5: "#50C878"    # Green
                        }
                        color = rank_colors.get(i, "#4A90E2")
                        st.markdown(
                            f'<div class="rank-badge" style="background-color: {color};">{i}</div>',
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(f"**{row['title']}**")
                        if pd.notna(row.get('year')):
                            st.caption(f"Year: {row['year']}")
                    
                    with col3:
                        # Rating with stars
                        rating = row['rating']
                        stars = "⭐" * min(5, int(rating))
                        st.markdown(f"{stars} **{rating:.1f}/5.0**")
                        
                        if pd.notna(row.get('runtime')):
                            st.caption(f"Runtime: {row['runtime']} min")
                    
                    with col4:
                        # Action buttons with completely unique keys (buttons support key)
                        is_favorite = any(m['title'] == row['title'] for m in st.session_state.favorites)
                        
                        # Create a truly unique key using multiple identifiers
                        unique_id = f"{selected_genre}_{i}_{row['title'][:10]}_{hash(row['title'])}".replace(" ", "_")
                        
                        if is_favorite:
                            if st.button("❤️", key=f"remove_fav_tab4_{unique_id}", help="Remove from Favorites"):
                                st.session_state.favorites = [m for m in st.session_state.favorites if m['title'] != row['title']]
                                st.rerun()
                        else:
                            if st.button("🤍", key=f"add_fav_tab4_{unique_id}", help="Add to Favorites"):
                                st.session_state.favorites.append({
                                    'title': row['title'],
                                    'rating': rating,
                                    'genres': row['genres'],
                                    'year': row.get('year', 2000),
                                    'runtime': row.get('runtime', 120),
                                    'imdbId': row.get('imdbId', 0)
                                })
                                st.rerun()
                        
                        # IMDB link - no key parameter
                        if show_imdb:
                            direct_link = get_imdb_link(row.get('imdbId'), row['title'])
                            st.markdown(f"[![IMDB](https://img.icons8.com/color/20/000000/imdb.png) View on IMDB]({direct_link})")
            
            # Export option with unique key (download_button supports key)
            st.divider()
            csv_data = top_5.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download Top 5 {selected_genre} Movies",
                data=csv_data,
                file_name=f"top5_{selected_genre.lower()}_movies.csv",
                mime='text/csv',
                use_container_width=True,
                key=f"download_tab4_{selected_genre}_{hash(selected_genre)}"
            )
        else:
            st.info(f"No movies found for genre: {selected_genre}")
    
    st.divider()
    
    # Quick stats by genre
    st.subheader("📊 Genre Statistics")
    
    # Create genre statistics
    genre_stats = []
    for idx, genre in enumerate(unique_genres[:10]):  # Show top 10 genres
        genre_movies = movies_df[movies_df['genres'].str.contains(genre, na=False)]
        if not genre_movies.empty:
            genre_stats.append({
                'Genre': genre,
                'Total Movies': len(genre_movies),
                'Average Rating': genre_movies['rating'].mean(),
                'Top Rated': genre_movies.nlargest(1, 'rating')['title'].iloc[0] if not genre_movies.empty else 'N/A',
                'Top Rating': genre_movies['rating'].max() if not genre_movies.empty else 0
            })
    
    if genre_stats:
        stats_df = pd.DataFrame(genre_stats)
        stats_df['Average Rating'] = stats_df['Average Rating'].round(2)
        stats_df['Top Rating'] = stats_df['Top Rating'].round(2)
        
        # Display as interactive dataframe with unique key (dataframe supports key)
        st.dataframe(
            stats_df.style.format({
                'Average Rating': '{:.2f}',
                'Top Rating': '{:.2f}'
            }).background_gradient(cmap='YlOrRd', subset=['Average Rating', 'Top Rating']),
            use_container_width=True,
            hide_index=True,
            key=f"genre_stats_df_tab4_{hash(str(genre_stats))}"
        )
    
    st.divider()
    
    # Alternative: Show top movies across all genres
    with st.expander("🏆 Top 10 Movies Across All Genres"):
        top_all = movies_df.nlargest(10, 'rating')[['title', 'genres', 'rating', 'year']]
        
        for j, (_, row) in enumerate(top_all.iterrows(), 1):
            st.markdown(f"{j}. **{row['title']}** - ⭐ {row['rating']:.1f}")
            st.caption(f"   Genres: {row['genres']} | Year: {row['year']}")

# --- TAB 5: Train & Evaluate - COMPLETE EVALUATION ---
with tab5:
    st.header("Train & Evaluate Models")
    
    st.markdown(f"""
    This tab allows you to train custom sentiment and emotion models using your data.
    - **Sentiment Model:** Binary classification (Positive/Negative)
    - **Emotion Model:** 6-class classification
    - **Samples per class:** {SAMPLES_PER_CLASS} random samples from each class
    
    All models and graphs are saved permanently in the `trained_models` and `training_plots` folders.
    """)
    
    tab5_col1, tab5_col2 = st.columns(2)
    
    with tab5_col1:
        st.subheader("Sentiment Model")
        
        st.info("Binary sentiment analysis (2 classes: Positive, Negative)")
        
        if sentiment_df is not None and not sentiment_df.empty:
            st.write(f"**Available samples:** {len(sentiment_df)}")
            if 'sentiment' in sentiment_df.columns:
                unique_sentiments = sentiment_df['sentiment'].unique()
                st.write(f"**Sentiment classes:** {len(unique_sentiments)}")
                
                # Show class distribution
                class_counts = sentiment_df['sentiment'].value_counts()
                for class_val, count in class_counts.items():
                    class_name = "Positive" if class_val == 1 else "Negative"
                    status = "✅" if count >= SAMPLES_PER_CLASS else "⚠️"
                    st.write(f"{status} {class_name}: {count} samples")
        
        train_sentiment = st.button("🚀 Train Sentiment Model (3000 per class)", use_container_width=True, key="train_sent_tab5")
        
        if train_sentiment:
            with st.spinner("Training sentiment model..."):
                try:
                    if sentiment_df is not None and not sentiment_df.empty:
                        sentiment_data = sentiment_df.dropna()
                        
                        # Sample 3000 per class
                        balanced_data = sample_3000_per_class(
                            sentiment_data, 
                            'review', 
                            'sentiment'
                        )
                        
                        # Split data
                        train_texts, val_texts, train_labels, val_labels = train_test_split(
                            balanced_data['review'].tolist(),
                            balanced_data['sentiment'].tolist(),
                            test_size=0.2,
                            random_state=42,
                            stratify=balanced_data['sentiment']
                        )
                        
                        st.write(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
                        
                        # Create a fresh model for training
                        from transformers import DistilBertForSequenceClassification
                        fresh_model = DistilBertForSequenceClassification.from_pretrained(
                            'distilbert-base-uncased',
                            num_labels=2,
                            ignore_mismatched_sizes=True
                        )
                        
                        # Train model
                        trained_model, metrics = simple_train_model(
                            fresh_model, 
                            tokenizer, 
                            train_texts, 
                            train_labels, 
                            val_texts, 
                            val_labels,
                            model_type="sentiment"
                        )
                        
                        if trained_model is not None:
                            st.success("✅ Sentiment model trained successfully with 3000 samples per class!")
                            # Force reload models from disk
                            st.session_state.sentiment_model_trained = True
                            st.rerun()
                    else:
                        st.error("No sentiment data available.")
                    
                except Exception as e:
                    st.error(f"Error training sentiment model: {str(e)}")
    
    with tab5_col2:
        st.subheader("🎭 Emotion Model")
        
        st.info("Emotion detection model (6 emotions)")
        
        if emotion_df is not None and not emotion_df.empty:
            st.write(f"**Available samples:** {len(emotion_df)}")
            if 'label' in emotion_df.columns:
                # Show class distribution
                class_counts = emotion_df['label'].value_counts()
                for class_val, count in class_counts.items():
                    if isinstance(class_val, (int, np.integer)):
                        class_name = EMOTION_LABELS[class_val] if class_val < len(EMOTION_LABELS) else str(class_val)
                    else:
                        class_name = str(class_val)
                    status = "✅" if count >= SAMPLES_PER_CLASS else "⚠️"
                    st.write(f"{status} {class_name}: {count} samples")
        
        train_emotion = st.button("Train Emotion Model (3000 per class)", use_container_width=True, key="train_emo_tab5")
        
        if train_emotion:
            with st.spinner("Training emotion model..."):
                try:
                    if emotion_df is not None and not emotion_df.empty:
                        emotion_samples = emotion_df.dropna().copy()
                        
                        # Convert string labels to numeric if needed
                        if emotion_samples['label'].dtype == 'object':
                            label_mapping = {
                                'sad': 0, 'happy': 1, 'romantic': 2, 
                                'angry': 3, 'fear': 4, 'surprised': 5
                            }
                            emotion_samples['label'] = emotion_samples['label'].astype(str).str.lower().map(label_mapping)
                            emotion_samples = emotion_samples.dropna()
                        
                        emotion_samples['label'] = emotion_samples['label'].astype(int)
                        
                        # Sample 3000 per class
                        balanced_data = sample_3000_per_class(
                            emotion_samples, 
                            'review', 
                            'label'
                        )
                        
                        # Split data
                        train_texts, val_texts, train_labels, val_labels = train_test_split(
                            balanced_data['review'].tolist(),
                            balanced_data['label'].tolist(),
                            test_size=0.2,
                            random_state=42,
                            stratify=balanced_data['label']
                        )
                        
                        st.write(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
                        
                        # Create a fresh model for training
                        from transformers import DistilBertForSequenceClassification
                        fresh_model = DistilBertForSequenceClassification.from_pretrained(
                            'distilbert-base-uncased',
                            num_labels=6,
                            ignore_mismatched_sizes=True
                        )
                        
                        # Train model
                        trained_model, metrics = simple_train_model(
                            fresh_model,
                            tokenizer,
                            train_texts,
                            train_labels,
                            val_texts,
                            val_labels,
                            model_type="emotion"
                        )
                        
                        if trained_model is not None:
                            st.success("✅ Emotion model trained successfully with 3000 samples per class!")
                            # Force reload models from disk
                            st.session_state.emotion_model_trained = True
                            st.rerun()
                    else:
                        st.error("No emotion data available.")
                    
                except Exception as e:
                    st.error(f"Error training emotion model: {str(e)}")
    
    st.divider()
    
    st.subheader("Model Evaluation Metrics")
    
    # Create main tabs for Sentiment and Emotion models
    model_main_tabs = st.tabs(["Sentiment Model", "Emotion Model"])
    
    # ============================================
    # SENTIMENT MODEL TAB
    # ============================================
    with model_main_tabs[0]:
        if st.session_state.sentiment_metrics or st.session_state.sentiment_plot_paths:
            
            # Create sub-tabs for different graphs
            sent_tabs = st.tabs([
                "Accuracy Metrics", 
                "Loss Curves", 
                "Accuracy Curve", 
                "Evaluation Performance", 
                "Confusion Matrix"
            ])
            
            # Tab 1: Accuracy Metrics
            with sent_tabs[0]:
                st.markdown("### Accuracy Metrics")
                
                if st.session_state.sentiment_metrics:
                    metrics = st.session_state.sentiment_metrics
                    
                    # Overall metrics in cards
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
                    with col2:
                        st.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
                    with col3:
                        st.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
                    with col4:
                        st.metric("F1-Score", f"{metrics.get('f1', 0)*100:.2f}%")

            # Tab 2: Loss Curves
            with sent_tabs[1]:
                st.markdown("### Loss Function History")
                
                # Check for saved plot first
                if 'loss_curves' in st.session_state.sentiment_plot_paths:
                    st.image(st.session_state.sentiment_plot_paths['loss_curves'], 
                            use_column_width=True, 
                            caption="Loss Curves")
                elif 'loss_history' in st.session_state.sentiment_metrics and 'val_loss_history' in st.session_state.sentiment_metrics:
                    loss_history = st.session_state.sentiment_metrics['loss_history']
                    val_loss_history = st.session_state.sentiment_metrics['val_loss_history']
                    epochs = range(1, len(loss_history) + 1)
                    
                    fig, ax = plt.subplots(figsize=(7, 3))
                    ax.plot(epochs, loss_history, 'b-', marker='o', linewidth=2, markersize=8, label='Training Loss')
                    ax.plot(epochs, val_loss_history, 'r-', marker='s', linewidth=2, markersize=8, label='Validation Loss')
                    ax.set_xlabel('Epochs', fontsize=12)
                    ax.set_ylabel('Loss', fontsize=12)
                    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=12)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show loss values in table
                    loss_df = pd.DataFrame({
                        'Epoch': list(epochs),
                        'Training Loss': [f'{x:.4f}' for x in loss_history],
                        'Validation Loss': [f'{x:.4f}' for x in val_loss_history]
                    })
                    st.dataframe(loss_df, use_container_width=True)
            
            # Tab 3: Accuracy Curve
            with sent_tabs[2]:
                st.markdown("### Accuracy Curve")
                
                # Check for saved plot first
                if 'accuracy_curve' in st.session_state.sentiment_plot_paths:
                    st.image(st.session_state.sentiment_plot_paths['accuracy_curve'], 
                            use_column_width=True, 
                            caption="Accuracy Curve")
                elif 'val_accuracy_history' in st.session_state.sentiment_metrics:
                    acc_history = st.session_state.sentiment_metrics['val_accuracy_history']
                    epochs = range(1, len(acc_history) + 1)
                    
                    fig, ax = plt.subplots(figsize=(7,3))
                    ax.plot(epochs, acc_history, 'g-', marker='o', linewidth=2, markersize=8, label='Validation Accuracy')
                    ax.set_xlabel('Epochs', fontsize=12)
                    ax.set_ylabel('Accuracy (%)', fontsize=12)
                    ax.set_title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=12)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show accuracy values in table
                    acc_df = pd.DataFrame({
                        'Epoch': list(epochs),
                        'Validation Accuracy (%)': [f'{x:.2f}%' for x in acc_history]
                    })
                    st.dataframe(acc_df, use_container_width=True)
            
            # Tab 4: Per-Class Performance
            with sent_tabs[3]:
                st.markdown("### Evaluation Performance")
                
                # Check for saved plot first
                if 'per_class_performance' in st.session_state.sentiment_plot_paths:
                    st.image(st.session_state.sentiment_plot_paths['per_class_performance'], 
                            use_column_width=True, 
                            caption="Per-Class Performance")
                elif st.session_state.sentiment_metrics and 'classification_report' in st.session_state.sentiment_metrics:
                    report = st.session_state.sentiment_metrics['classification_report']
                    
                    # Prepare data for plotting
                    classes = list(report.keys())[:2]  # Negative, Positive
                    
                    # Create DataFrame
                    perf_data = []
                    for cls in classes:
                        if cls in report and isinstance(report[cls], dict):
                            perf_data.append({
                                'Class': cls,
                                'Precision': report[cls]['precision'] * 100,
                                'Recall': report[cls]['recall'] * 100,
                                'F1-Score': report[cls]['f1-score'] * 100
                            })
                    
                    if perf_data:
                        df_perf = pd.DataFrame(perf_data)
                        
                        # Create bar graph
                        fig, ax = plt.subplots(figsize=(7,4))
                        
                        x = np.arange(len(df_perf['Class']))
                        width = 0.25
                        
                        # Colors: Blue for Precision, Green for Recall, Red for F1
                        bars1 = ax.bar(x - width, df_perf['Precision'], width, label='Precision', color='#4169E1')
                        bars2 = ax.bar(x, df_perf['Recall'], width, label='Recall', color='#2ecc71')
                        bars3 = ax.bar(x + width, df_perf['F1-Score'], width, label='F1-Score', color='#e74c3c')
                        
                        ax.set_xlabel('Sentiment Classes', fontsize=12)
                        ax.set_ylabel('Score (%)', fontsize=12)
                        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
                        ax.set_xticks(x)
                        ax.set_xticklabels(df_perf['Class'])
                        ax.legend(fontsize=10)
                        ax.set_ylim([0, 100])
                        
                        # Add value labels
                        for bars in [bars1, bars2, bars3]:
                            for bar in bars:
                                height = bar.get_height()
                                if height > 0:
                                    ax.annotate(f'{height:.1f}',
                                               xy=(bar.get_x() + bar.get_width() / 2, height),
                                               xytext=(0, 3),
                                               textcoords="offset points",
                                               ha='center', va='bottom', fontsize=9)
                        
                        ax.grid(True, alpha=0.3, axis='y')
                        ax.set_facecolor('#f8f9fa')
                        ax.set_ylim([0, 105])
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Show performance table
                        st.markdown("#### Performance Table")
                        st.dataframe(
                            df_perf.style.format({
                                'Precision': '{:.1f}%',
                                'Recall': '{:.1f}%',
                                'F1-Score': '{:.1f}%'
                            }).background_gradient(cmap='Blues', subset=['Precision'])
                             .background_gradient(cmap='Greens', subset=['Recall'])
                             .background_gradient(cmap='Reds', subset=['F1-Score']),
                            use_container_width=True
                        )
            
            # Tab 5: Confusion Matrix
            with sent_tabs[4]:
                st.markdown("### Confusion Matrix")
                
                # Check for saved plot first
                if 'confusion_matrix' in st.session_state.sentiment_plot_paths:
                    st.image(st.session_state.sentiment_plot_paths['confusion_matrix'], 
                            use_column_width=True, 
                            caption="Confusion Matrix")
                elif st.session_state.sentiment_metrics and 'confusion_matrix' in st.session_state.sentiment_metrics:
                    cm = np.array(st.session_state.sentiment_metrics['confusion_matrix'])
                    
                    fig, ax = plt.subplots(figsize=(7, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=SENTIMENT_LABELS, 
                               yticklabels=SENTIMENT_LABELS, ax=ax)
                    ax.set_xlabel('Predicted', fontsize=12)
                    ax.set_ylabel('Actual', fontsize=12)
                    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                    plt.close(fig)
        
        else:
            st.info("No sentiment model metrics found. Train the sentiment model first.")
    
    # ============================================
    # EMOTION MODEL TAB
    # ============================================
    with model_main_tabs[1]:
        if st.session_state.emotion_metrics or st.session_state.emotion_plot_paths:
            
            # Create sub-tabs for different graphs
            emotion_tabs = st.tabs([
                "Accuracy Metrics", 
                "Loss Curves", 
                "Accuracy Curve", 
                "Evaluation Performance", 
                "Confusion Matrix"
            ])
            
            # Tab 1: Accuracy Metrics
            with emotion_tabs[0]:
                st.markdown("### Accuracy Metrics")
                
                if st.session_state.emotion_metrics:
                    metrics = st.session_state.emotion_metrics
                    
                    # Overall metrics in cards
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
                    with col2:
                        st.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
                    with col3:
                        st.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
                    with col4:
                        st.metric("F1-Score", f"{metrics.get('f1', 0)*100:.2f}%")
                    
            # Tab 2: Loss Curves
            with emotion_tabs[1]:
                st.markdown("### Loss Function History")
                
                # Check for saved plot first
                if 'loss_curves' in st.session_state.emotion_plot_paths:
                    st.image(st.session_state.emotion_plot_paths['loss_curves'], 
                            use_column_width=True, 
                            caption="Loss Curves")
                elif 'loss_history' in st.session_state.emotion_metrics and 'val_loss_history' in st.session_state.emotion_metrics:
                    loss_history = st.session_state.emotion_metrics['loss_history']
                    val_loss_history = st.session_state.emotion_metrics['val_loss_history']
                    epochs = range(1, len(loss_history) + 1)
                    
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(epochs, loss_history, 'b-', marker='o', linewidth=2, markersize=8, label='Training Loss')
                    ax.plot(epochs, val_loss_history, 'r-', marker='s', linewidth=2, markersize=8, label='Validation Loss')
                    ax.set_xlabel('Epochs', fontsize=12)
                    ax.set_ylabel('Loss', fontsize=12)
                    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=12)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show loss values in table
                    loss_df = pd.DataFrame({
                        'Epoch': list(epochs),
                        'Training Loss': [f'{x:.4f}' for x in loss_history],
                        'Validation Loss': [f'{x:.4f}' for x in val_loss_history]
                    })
                    st.dataframe(loss_df, use_container_width=True)
            
            # Tab 3: Accuracy Curve
            with emotion_tabs[2]:
                st.markdown("### Accuracy Curve")
                
                # Check for saved plot first
                if 'accuracy_curve' in st.session_state.emotion_plot_paths:
                    st.image(st.session_state.emotion_plot_paths['accuracy_curve'], 
                            use_column_width=True, 
                            caption="Accuracy Curve")
                elif 'val_accuracy_history' in st.session_state.emotion_metrics:
                    acc_history = st.session_state.emotion_metrics['val_accuracy_history']
                    epochs = range(1, len(acc_history) + 1)
                    
                    fig, ax = plt.subplots(figsize=(7, 3))
                    ax.plot(epochs, acc_history, 'g-', marker='o', linewidth=2, markersize=8, label='Validation Accuracy')
                    ax.set_xlabel('Epochs', fontsize=12)
                    ax.set_ylabel('Accuracy (%)', fontsize=12)
                    ax.set_title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=12)  
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show accuracy values in table
                    acc_df = pd.DataFrame({
                        'Epoch': list(epochs),
                        'Validation Accuracy (%)': [f'{x:.2f}%' for x in acc_history]
                    })
                    st.dataframe(acc_df, use_container_width=True)
            
            # Tab 4: Per-Class Performance
            with emotion_tabs[3]:
                st.markdown("### Evaluation Performance")
                
                # Check for saved plot first
                if 'per_class_performance' in st.session_state.emotion_plot_paths:
                    st.image(st.session_state.emotion_plot_paths['per_class_performance'], 
                            use_column_width=True, 
                            caption="Per-Class Performance")
                elif st.session_state.emotion_metrics and 'classification_report' in st.session_state.emotion_metrics:
                    report = st.session_state.emotion_metrics['classification_report']
                    
                    # Prepare data for plotting
                    classes = list(report.keys())[:6]  # All 6 emotions
                    
                    # Create DataFrame
                    perf_data = []
                    for cls in classes:
                        if cls in report and isinstance(report[cls], dict):
                            # Format class name for display
                            display_name = cls.capitalize()
                            perf_data.append({
                                'Emotion': display_name,
                                'Precision': report[cls]['precision'] * 100,
                                'Recall': report[cls]['recall'] * 100,
                                'F1-Score': report[cls]['f1-score'] * 100
                            })
                    
                    if perf_data:
                        df_perf = pd.DataFrame(perf_data)
                        
                        # Create bar graph
                        fig, ax = plt.subplots(figsize=(14, 7))
                        
                        x = np.arange(len(df_perf['Emotion']))
                        width = 0.25
                        
                        # Colors
                        bars1 = ax.bar(x - width, df_perf['Precision'], width, label='Precision', color='#4169E1')
                        bars2 = ax.bar(x, df_perf['Recall'], width, label='Recall', color='#2ecc71')
                        bars3 = ax.bar(x + width, df_perf['F1-Score'], width, label='F1-Score', color='#e74c3c')
                        
                        ax.set_xlabel('Emotion Classes', fontsize=14, fontweight='bold')
                        ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
                        ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
                        ax.set_xticks(x)
                        ax.set_xticklabels(df_perf['Emotion'], fontsize=12)
                        ax.legend(fontsize=12, loc='upper right')
                        ax.set_ylim([0, 100])
                        
                        # Add value labels on bars
                        for bars in [bars1, bars2, bars3]:
                            for bar in bars:
                                height = bar.get_height()
                                if height > 0:
                                    ax.annotate(f'{height:.1f}',
                                               xy=(bar.get_x() + bar.get_width() / 2, height),
                                               xytext=(0, 5),
                                               textcoords="offset points",
                                               ha='center', va='bottom', fontsize=10, fontweight='bold')
                        
                        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                        ax.set_facecolor('#f8f9fa')
                        ax.set_ylim([0, 105])
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Show performance table
                        st.markdown("#### Performance Table")
                        
                        # Create styled table
                        styled_df = df_perf.style.format({
                            'Precision': '{:.2f}',
                            'Recall': '{:.2f}',
                            'F1-Score': '{:.2f}'
                        }).background_gradient(cmap='Blues', subset=['Precision'])\
                          .background_gradient(cmap='Greens', subset=['Recall'])\
                          .background_gradient(cmap='Reds', subset=['F1-Score'])
                        
                        st.dataframe(styled_df, use_container_width=True)
            
            # Tab 5: Confusion Matrix
            with emotion_tabs[4]:
                st.markdown("### Confusion Matrix")
                
                # Check for saved plot first
                if 'confusion_matrix' in st.session_state.emotion_plot_paths:
                    st.image(st.session_state.emotion_plot_paths['confusion_matrix'], 
                            use_column_width=True, 
                            caption="Confusion Matrix")
                elif st.session_state.emotion_metrics and 'confusion_matrix' in st.session_state.emotion_metrics:
                    cm = np.array(st.session_state.emotion_metrics['confusion_matrix'])
                    
                    fig, ax = plt.subplots(figsize=(7, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                               xticklabels=EMOTION_LABELS, 
                               yticklabels=EMOTION_LABELS, ax=ax)
                    ax.set_xlabel('Predicted', fontsize=12)
                    ax.set_ylabel('Actual', fontsize=12)
                    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)    
                    st.pyplot(fig)
                    plt.close(fig)
        
        else:
            st.info("No emotion model metrics found. Train the emotion model first.")
    
    st.divider()
    
    st.subheader("Test Your Models")
    
    test_text = st.text_area("Enter text to test models:", "I was completely surprised by the plot twists in this movie!")
    
    test_col1, test_col2 = st.columns(2)
    
    with test_col1:
        if st.button("Test Sentiment Model", use_container_width=True, key="test_sentiment_tab5"):
            if test_text.strip():
                if is_sentiment_model_ready():
                    try:
                        sentiment, sentiment_conf, sentiment_probs = predict_sentiment_distilbert(test_text)
                        if sentiment == "Positive":
                            st.success(f"**Sentiment:** {sentiment} (Confidence: {sentiment_conf*100:.1f}%)")
                        else:
                            st.error(f"**Sentiment:** {sentiment} (Confidence: {sentiment_conf*100:.1f}%)")
                        
                        # Show probabilities
                        if sentiment_probs and show_prob_details:
                            st.markdown("**Sentiment Probabilities:**")
                            for sent_label, prob in sentiment_probs.items():
                                st.write(f"- {sent_label}: {prob*100:.1f}%")
                    except Exception as e:
                        st.warning(f"Error testing sentiment model: {e}")
                else:
                    st.warning("Sentiment model not trained. Using fallback.")
                    blob = TextBlob(test_text)
                    polarity = blob.sentiment.polarity
                    if polarity > 0.1:
                        sentiment = "Positive"
                    else:
                        sentiment = "Negative"
                    st.write(f"TextBlob Sentiment: **{sentiment}**")
    
    with test_col2:
        if st.button("Test Emotion Model", use_container_width=True, key="test_emotion_tab5"):
            if test_text.strip():
                if is_emotion_model_ready():
                    try:
                        emotion, emotion_conf = predict_emotion_distilbert(test_text)
                        emotion_colors = {
                            "Happy": "#FFD700",
                            "Sad": "#4169E1",
                            "Angry": "#FF4500",
                            "Fear": "#8B0000",
                            "Romantic": "#FF69B4",
                            "Surprised": "#FFA500"
                        }
                        color = emotion_colors.get(emotion, "#4CAF50")
                        
                        # Display emotion with color
                        st.markdown(f"**Emotion:** <span style='color:{color}; font-weight:bold; font-size:20px;'>{emotion}</span>", unsafe_allow_html=True)
                        st.metric("Confidence", f"{emotion_conf*100:.1f}%")
                        
                    except Exception as e:
                        st.warning(f"Error testing emotion model: {e}")
                else:
                    st.warning("Emotion model not trained. Using fallback.")
                    
                    # Use TextBlob for fallback
                    blob = TextBlob(test_text)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    
                    # Map TextBlob polarity to emotions
                    if polarity > 0.3:
                        emotion = "Happy"
                        emotion_conf = polarity
                    elif polarity > 0.1:
                        emotion = "Romantic"
                        emotion_conf = polarity
                    elif polarity < -0.3:
                        emotion = "Sad"
                        emotion_conf = abs(polarity)
                    elif polarity < -0.1:
                        emotion = "Angry"
                        emotion_conf = abs(polarity)
                    else:
                        # Neutral sentiment
                        if subjectivity > 0.5:
                            emotion = "Surprised"
                            emotion_conf = subjectivity
                        else:
                            emotion = "Fear"
                            emotion_conf = 0.5
                    
                    # Display fallback result
                    st.write(f"TextBlob Emotion: **{emotion}**")
            else:
                st.warning("Please enter some text to test.")