"""
Advanced Stacking Ensemble for Sentiment Analysis - GPU Optimized
This implementation leverages GPU acceleration where possible to improve performance.
"""

import numpy as np
import pandas as pd
import time
import cudf
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm  # Ensure tqdm is imported

# Try importing GPU libraries, but don't crash if they're not available
GPU_AVAILABLE = False
try:
    import torch
    TORCH_GPU_AVAILABLE = torch.cuda.is_available()
    
    if TORCH_GPU_AVAILABLE:
        try:
            import cupy as cp
            # Import cudf here
            import cudf
            from cuml import LogisticRegression as cuLogisticRegression
            from cuml.svm import LinearSVC as cuLinearSVC
            from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
            RAPIDS_AVAILABLE = True
        except ImportError:
            RAPIDS_AVAILABLE = False
            
        try:
            import xgboost as xgb
            XGBOOST_GPU_AVAILABLE = True
        except ImportError:
            XGBOOST_GPU_AVAILABLE = False
            
        # Only consider GPU available if at least one of the libraries is available
        GPU_AVAILABLE = RAPIDS_AVAILABLE or XGBOOST_GPU_AVAILABLE
    else:
        RAPIDS_AVAILABLE = False
        XGBOOST_GPU_AVAILABLE = False
except ImportError:
    TORCH_GPU_AVAILABLE = False
    RAPIDS_AVAILABLE = False
    XGBOOST_GPU_AVAILABLE = False

# Try to import XGBoost for CPU if not already imported
if not XGBOOST_GPU_AVAILABLE:
    try:
        import xgboost as xgb
        XGBOOST_CPU_AVAILABLE = True
    except ImportError:
        XGBOOST_CPU_AVAILABLE = False
else:
    XGBOOST_CPU_AVAILABLE = XGBOOST_GPU_AVAILABLE


# Check if GPU is available
def check_gpu():
    """Check if GPU is available and print device information."""
    print("\n----- GPU Availability Check -----")
    
    # Check PyTorch GPU
    print(f"PyTorch: CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch: CUDA device count: {torch.cuda.device_count()}")
        print(f"PyTorch: CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Check CuPy/RAPIDS
    try:
        print(f"CuPy version: {cp.__version__}")
        print(f"CuPy CUDA device: {cp.cuda.runtime.getDeviceCount()} device(s) available")
    except:
        print("CuPy not properly installed or no CUDA device available")
    
    # Check XGBoost
    try:
        print(f"XGBoost version: {xgb.__version__}")
        print(f"XGBoost has GPU support: {xgb.config.build_info()['USE_CUDA']}")
    except:
        print("Could not verify XGBoost GPU support")
    
    print("---------------------------------------\n")

# 1. Feature Engineering: Extract additional features with GPU acceleration if available

def extract_additional_features(X_data, processed_texts):
    """Extract text-based features to complement TF-IDF vectors with GPU acceleration when possible"""
    start_time = time.time()
    print("Extracting additional features...")
    
    # Initialize flag for GPU usage
    use_gpu = False
    
    # Only try GPU if CUDA is available
    if torch.cuda.is_available():
        try:
            # Explicitly import cudf here to ensure it's available
            import cudf
            use_gpu = True
            print("Using GPU acceleration with cuDF")
            
            # Convert processed_texts to a list if it's not already
            if not isinstance(processed_texts, list):
                processed_texts = processed_texts.tolist()
                
            # Text length
            text_lengths = [len(text) if isinstance(text, str) else 0 for text in processed_texts]
            
            # Word count
            word_counts = [len(text.split()) if isinstance(text, str) else 0 for text in processed_texts]
            
            # Average word length & Negation count
            avg_word_lengths = []
            negation_counts = []
            
            for text in processed_texts:
                if isinstance(text, str):
                    words = text.split()
                    if words:
                        avg_word_lengths.append(sum(len(word) for word in words) / len(words))
                        negation_counts.append(sum(1 for word in words if '_NEG' in word))
                    else:
                        avg_word_lengths.append(0)
                        negation_counts.append(0)
                else:
                    avg_word_lengths.append(0)
                    negation_counts.append(0)
            
            # Create DataFrame with GPU acceleration
            feature_df = cudf.DataFrame({
                'text_length': text_lengths,
                'word_count': word_counts,
                'avg_word_length': avg_word_lengths,
                'negation_count': negation_counts
            })
            
            # Scale features
            scaler = MaxAbsScaler()
            # Convert to CPU for scaling, then back to GPU
            cpu_features = feature_df.to_pandas()
            scaled_features = scaler.fit_transform(cpu_features)
            
            # Convert back to sparse matrix
            sparse_features = csr_matrix(scaled_features)
            
        except (ImportError, ModuleNotFoundError, NameError, AttributeError) as e:
            print(f"GPU libraries not available or error occurred, falling back to CPU processing: {e}")
            use_gpu = False
    else:
        print("CUDA not available, using CPU processing")
    
    # Fall back to CPU if GPU processing fails or isn't available
    if not use_gpu:
        # Initialize empty arrays for new features
        text_lengths = []
        word_counts = []
        avg_word_length = []
        negation_counts = []

        # Process each text entry
        for text in processed_texts:
            if isinstance(text, str):
                # Text length
                text_lengths.append(len(text))

                # Word count
                words = text.split()
                word_counts.append(len(words))

                # Average word length
                if words:
                    avg_word_length.append(sum(len(word) for word in words) / len(words))
                else:
                    avg_word_length.append(0)

                # Negation count
                negation_counts.append(sum(1 for word in words if '_NEG' in word))
            else:
                # Handle non-string entries
                text_lengths.append(0)
                word_counts.append(0)
                avg_word_length.append(0)
                negation_counts.append(0)

        # Create a DataFrame of engineered features
        feature_df = pd.DataFrame({
            'text_length': text_lengths,
            'word_count': word_counts,
            'avg_word_length': avg_word_length,
            'negation_count': negation_counts
        })

        # Scale numeric features and convert to sparse for compatibility with TF-IDF
        scaler = MaxAbsScaler()
        scaled_features = scaler.fit_transform(feature_df)
        sparse_features = csr_matrix(scaled_features)

    # Combine with TF-IDF features
    combined_features = hstack([X_data, sparse_features])
    
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
    return combined_features

# 2. Create a stacking ensemble with meta-learning and GPU acceleration

def create_stacking_ensemble(X_train, y_train, X_val, y_val, feature_names):
    """
    Creates a stacking ensemble with multiple base models and a meta-learner,
    using GPU acceleration where possible
    """
    start_time = time.time()
    use_gpu = torch.cuda.is_available()
    
    # Base models with optimized hyperparameters
    base_models = {
        'naive_bayes': MultinomialNB(alpha=0.1),
        'logistic_regression': LogisticRegression(C=2.0, solver='saga', penalty='l1', max_iter=5000, class_weight='balanced', n_jobs=-1, random_state=42),
        'linear_svc': LogisticRegression(C=1.0, solver='liblinear', penalty='l2', max_iter=10000, class_weight='balanced', n_jobs=-1, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', bootstrap=True, class_weight='balanced', n_jobs=-1, random_state=42),
        'xgboost': xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=2, gamma=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1, reg_alpha=0.01, reg_lambda=1, use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)
    }

    # Train base models and generate meta-features
    meta_features_train = np.zeros((X_train.shape[0], len(base_models)))
    meta_features_val = np.zeros((X_val.shape[0], len(base_models)))

    print("\nTraining base models for stacking...")

    # Use tqdm for progress tracking
    for i, (name, model) in enumerate(tqdm(base_models.items(), desc="Training Base Models", unit="model")):
        model_start_time = time.time()
        print(f"\nTraining {name}...")
        
        # Training logic remains the same
        model.fit(X_train, y_train)
        meta_features_train[:, i] = model.predict_proba(X_train)[:, 1]
        meta_features_val[:, i] = model.predict_proba(X_val)[:, 1]
        
        elapsed_time = time.time() - model_start_time
        print(f"  {name} trained in {elapsed_time:.2f} seconds")

    print(f"Ensemble training completed in {time.time() - start_time:.2f} seconds")
    return base_models, meta_learner, val_accuracy

# 3. Final prediction function for test data

def predict_with_stacking_ensemble(base_models, meta_learner, X_test):
    """Generate predictions using the stacking ensemble with GPU acceleration if available"""
    start_time = time.time()
    use_gpu = torch.cuda.is_available()
    
    # Generate meta-features for test data
    meta_features_test = np.zeros((X_test.shape[0], len(base_models)))

    print("\nGenerating predictions for base models...")
    for i, (name, model) in enumerate(tqdm(base_models.items(), desc="Predicting with Base Models", unit="model")):
        model_start_time = time.time()
        
        # Prediction logic remains the same
        if hasattr(model, 'predict_proba'):
            meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]
        else:
            meta_features_test[:, i] = model.decision_function(X_test)
        
        elapsed_time = time.time() - model_start_time
        print(f"  Predictions for {name} completed in {elapsed_time:.2f} seconds")

    print("\nMaking final predictions with meta-learner...")
    test_predictions = meta_learner.predict(meta_features_test)
    print(f"Prediction completed in {time.time() - start_time:.2f} seconds")
    return test_predictions

# 4. Main execution function

def improved_sentiment_analysis(X_train, y_train, X_test, y_test, X_val, y_val, processed_texts_train, processed_texts_test, processed_texts_val, feature_names):
    """
    Main function to execute the improved sentiment analysis pipeline with GPU acceleration where possible
    """
    overall_start_time = time.time()
    
    # Limit datasets to 100k samples each
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_val = X_val[:10000]
    y_val = y_val[:10000]
    X_test = X_test[:10000]
    y_test = y_test[:10000]
    processed_texts_train = processed_texts_train[:10000]
    processed_texts_val = processed_texts_val[:10000]
    processed_texts_test = processed_texts_test[:10000]

    y_train = y_train - 1
    y_val = y_val - 1
    y_test = y_test - 1
    
    # Check for GPU availability
    check_gpu()
    
    print("Enhancing features with text metadata...")
    X_train_enhanced = extract_additional_features(X_train, processed_texts_train)
    X_val_enhanced = extract_additional_features(X_val, processed_texts_val)
    X_test_enhanced = extract_additional_features(X_test, processed_texts_test)

    print(f"Original feature shape: {X_train.shape}")
    print(f"Enhanced feature shape: {X_train_enhanced.shape}")

    print("Unique labels in y_train:", np.unique(y_train))
    print("Unique labels in y_val:", np.unique(y_val))
    print("Unique labels in y_test:", np.unique(y_test))

    # Create and train the stacking ensemble
    base_models, meta_learner, val_accuracy = create_stacking_ensemble(
        X_train_enhanced, y_train, X_val_enhanced, y_val, feature_names
    )

    # Make predictions on the test set
    test_predictions = predict_with_stacking_ensemble(
        base_models, meta_learner, X_test_enhanced
    )

    # Ensure y_test is a numpy array
    if hasattr(y_test, 'values'):
        y_test_np = y_test.values
    else:
        y_test_np = y_test
        
    # Evaluate performance on test set
    test_accuracy = accuracy_score(y_test_np, test_predictions)

    print("\n----- Final Test Results -----")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_test_np, test_predictions))
    print(f"\nTotal execution time: {time.time() - overall_start_time:.2f} seconds")

    return test_accuracy, base_models, meta_learner

# Installation helper function
def install_required_packages():
    """
    Installs the required packages for GPU acceleration.
    This should be run in a code cell before the main script.
    """
    import sys
    import subprocess
    
    # Check if running in Colab
    is_colab = 'google.colab' in sys.modules
    
    if is_colab:
        print("Installing required packages for GPU acceleration in Google Colab...")
        
        # Install packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "xgboost", "cupy-cuda11x", "cudf-cu11", "cuml-cu11", "torch"])
        
        print("\nInstallation complete. Please restart the runtime to ensure all packages are properly loaded.")
        print("After restarting, run your script to utilize GPU acceleration.")
    else:
        print("This function is designed for Google Colab. Please install packages manually in your environment.")