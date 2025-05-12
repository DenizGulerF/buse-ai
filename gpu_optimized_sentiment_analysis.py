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
    base_models = {}
    
    # Check if we can use GPU for model training
    if use_gpu:
        try:
            print("\nTraining models with GPU acceleration...")

            # Add GPU-accelerated models
            base_models['naive_bayes'] = MultinomialNB(alpha=0.1)  # No GPU version available

            # GPU-accelerated LogisticRegression
            from cuml import LogisticRegression  # Corrected import
            base_models['logistic_regression'] = LogisticRegression(
                C=2.0,
                penalty='l1',
                solver='saga',
                max_iter=5000
            )

            # GPU-accelerated LinearSVC
            base_models['linear_svc'] = cuLinearSVC(
                C=1.0,
                loss='squared_hinge',
                penalty='l2',
                max_iter=10000
            )
            
            # GPU-accelerated RandomForest
            base_models['random_forest'] = cuRandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                n_streams=4  # Parallel streams for GPU
            )
            
            # XGBoost with GPU acceleration
            base_models['xgboost'] = xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=2,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                reg_alpha=0.01,
                reg_lambda=1,
                use_label_encoder=False,
                tree_method='gpu_hist',  # Use GPU acceleration
                predictor='gpu_predictor',  # Use GPU for prediction
                gpu_id=0
            )
            
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Error initializing GPU models: {e}")
            print("Falling back to CPU models")
            use_gpu = False
    
    # Fall back to CPU models if GPU initialization failed
    if not use_gpu:
        print("\nTraining models on CPU...")
        base_models = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'logistic_regression': LogisticRegression(
                C=2.0,
                solver='saga',
                penalty='l1',
                max_iter=5000,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ),
            'linear_svc': LogisticRegression(  # Replace LinearSVC with LogisticRegression
                C=1.0,
                solver='liblinear',
                penalty='l2',
                max_iter=10000,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=2,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                reg_alpha=0.01,
                reg_lambda=1,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1,
                random_state=42
            )
        }

    # Train base models and generate meta-features
    meta_features_train = np.zeros((X_train.shape[0], len(base_models)))
    meta_features_val = np.zeros((X_val.shape[0], len(base_models)))

    print("\nTraining base models for stacking...")

    # Convert y_train and y_val to numpy arrays if they're not already
    if hasattr(y_train, 'values'):
        y_train_np = y_train.values
    else:
        y_train_np = y_train
        
    if hasattr(y_val, 'values'):
        y_val_np = y_val.values
    else:
        y_val_np = y_val

    for i, (name, model) in enumerate(base_models.items()):
        model_start_time = time.time()
        print(f"Training {name}...")

        # For XGBoost and cuML models, handle the sparse matrix differently
        if name == 'xgboost':
            # Convert to dense matrix for XGBoost
            if hasattr(X_train, 'toarray'):
                x_train_dense = X_train.toarray()
                x_val_dense = X_val.toarray()
                
                # If using GPU and more than 1GB of data, use DMatrix for better performance
                if use_gpu and (x_train_dense.nbytes > 1e9):
                    dtrain = xgb.DMatrix(x_train_dense, y_train_np)
                    dval = xgb.DMatrix(x_val_dense, y_val_np)
                    
                    # Use native API instead of sklearn wrapper for better GPU performance
                    params = {
                        'max_depth': 6,
                        'learning_rate': 0.05,
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'tree_method': 'gpu_hist',
                        'predictor': 'gpu_predictor',
                        'gpu_id': 0
                    }
                    
                    # Train model
                    bst = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=300
                    )
                    
                    # Generate predictions
                    meta_features_train[:, i] = bst.predict(dtrain)
                    meta_features_val[:, i] = bst.predict(dval)
                else:
                    # Use sklearn wrapper
                    model.fit(x_train_dense, y_train_np)
                    meta_features_train[:, i] = model.predict_proba(x_train_dense)[:, 1]
                    meta_features_val[:, i] = model.predict_proba(x_val_dense)[:, 1]
            else:
                model.fit(X_train, y_train_np)
                meta_features_train[:, i] = model.predict_proba(X_train)[:, 1]
                meta_features_val[:, i] = model.predict_proba(X_val)[:, 1]
        
        # For RAPIDS cuML models
        elif use_gpu and name in ['logistic_regression', 'linear_svc', 'random_forest']:
            try:
                # Convert to dense for cuML
                if hasattr(X_train, 'toarray'):
                    X_train_np = X_train.toarray()
                    X_val_np = X_val.toarray()
                else:
                    X_train_np = X_train
                    X_val_np = X_val
                
                # Convert to CuPy arrays for GPU processing
                X_train_gpu = cp.array(X_train_np)
                y_train_gpu = cp.array(y_train_np)
                X_val_gpu = cp.array(X_val_np)
                
                # Fit model on GPU
                model.fit(X_train_gpu, y_train_gpu)
                
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    proba_train = model.predict_proba(X_train_gpu)
                    proba_val = model.predict_proba(X_val_gpu)
                    
                    # Extract probability for positive class
                    if proba_train.shape[1] > 1:  # If binary classification
                        meta_features_train[:, i] = cp.asnumpy(proba_train[:, 1])
                        meta_features_val[:, i] = cp.asnumpy(proba_val[:, 1])
                    else:  # If only one probability returned
                        meta_features_train[:, i] = cp.asnumpy(proba_train)
                        meta_features_val[:, i] = cp.asnumpy(proba_val)
                else:
                    # For models without predict_proba
                    meta_features_train[:, i] = cp.asnumpy(model.decision_function(X_train_gpu))
                    meta_features_val[:, i] = cp.asnumpy(model.decision_function(X_val_gpu))
                    
            except Exception as e:
                print(f"Error using GPU for {name}: {e}")
                print("Falling back to CPU implementation")
                
                # Fall back to CPU implementation
                if name == 'logistic_regression':
                    model = LogisticRegression(C=2.0, solver='saga', penalty='l1', 
                                              max_iter=5000, class_weight='balanced', n_jobs=-1)
                elif name == 'linear_svc':
                    model = LogisticRegression(C=1.0, solver='liblinear', penalty='l2', 
                                              max_iter=10000, class_weight='balanced', n_jobs=-1)
                elif name == 'random_forest':
                    model = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5,
                                                 min_samples_leaf=2, max_features='sqrt', n_jobs=-1)
                
                # Handle sparse matrices
                model.fit(X_train, y_train_np)
                
                if hasattr(model, 'predict_proba'):
                    meta_features_train[:, i] = model.predict_proba(X_train)[:, 1]
                    meta_features_val[:, i] = model.predict_proba(X_val)[:, 1]
                else:
                    meta_features_train[:, i] = model.decision_function(X_train)
                    meta_features_val[:, i] = model.decision_function(X_val)
        
        else:
            # Standard sklearn models
            if hasattr(model, 'predict_proba'):
                # Use cross_val_predict for more robust meta-features
                meta_features_train[:, i] = cross_val_predict(
                    model, X_train, y_train_np, cv=5, method='predict_proba'
                )[:, 1]

                model.fit(X_train, y_train_np)
                meta_features_val[:, i] = model.predict_proba(X_val)[:, 1]
            else:
                # For models without predict_proba
                model.fit(X_train, y_train_np)
                meta_features_train[:, i] = cross_val_predict(
                    model, X_train, y_train_np, cv=5, method='decision_function'
                )
                meta_features_val[:, i] = model.decision_function(X_val)
        
        print(f"  {name} trained in {time.time() - model_start_time:.2f} seconds")

    # Meta learner: Try GPU-accelerated version first, fall back to CPU if needed
    if use_gpu:
        try:
            # Convert meta-features to GPU
            meta_features_train_gpu = cp.array(meta_features_train)
            meta_features_val_gpu = cp.array(meta_features_val)
            y_train_gpu = cp.array(y_train_np)
            
            # Create and train GPU meta-learner
            meta_learner = cuLogisticRegression(
                C=3.0,
                penalty='l2',
                max_iter=2000
            )
            meta_learner.fit(meta_features_train_gpu, y_train_gpu)
            
            # Get validation predictions
            val_predictions_gpu = meta_learner.predict(meta_features_val_gpu)
            val_predictions = cp.asnumpy(val_predictions_gpu)
            
        except Exception as e:
            print(f"Error using GPU for meta-learner: {e}")
            print("Falling back to CPU for meta-learner")
            use_gpu = False
            
    # Fall back to CPU meta-learner
    if not use_gpu:
        meta_learner = LogisticRegression(
            C=3.0,
            solver='liblinear',
            max_iter=2000,
            class_weight='balanced',
            random_state=42
        )
        meta_learner.fit(meta_features_train, y_train_np)
        val_predictions = meta_learner.predict(meta_features_val)

    # Evaluate meta-learner on validation set
    val_accuracy = accuracy_score(y_val_np, val_predictions)

    print(f"\nStacking ensemble validation accuracy: {val_accuracy:.4f}")
    print("\nClassification Report (Validation):")
    print(classification_report(y_val_np, val_predictions))
    print(f"Ensemble training completed in {time.time() - start_time:.2f} seconds")

    return base_models, meta_learner, val_accuracy

# 3. Final prediction function for test data

def predict_with_stacking_ensemble(base_models, meta_learner, X_test):
    """Generate predictions using the stacking ensemble with GPU acceleration if available"""
    start_time = time.time()
    use_gpu = torch.cuda.is_available()
    
    # Generate meta-features for test data
    meta_features_test = np.zeros((X_test.shape[0], len(base_models)))

    for i, (name, model) in enumerate(base_models.items()):
        # For XGBoost, handle sparse matrix differently
        if name == 'xgboost':
            if hasattr(X_test, 'toarray'):
                if use_gpu and hasattr(model, 'get_booster'):  # XGBoost sklearn API with GPU
                    # Use DMatrix for faster prediction
                    dtest = xgb.DMatrix(X_test.toarray())
                    meta_features_test[:, i] = model.get_booster().predict(dtest)
                else:  # Standard prediction
                    meta_features_test[:, i] = model.predict_proba(X_test.toarray())[:, 1]
            else:
                meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]
        
        # For RAPIDS cuML models
        elif use_gpu and name in ['logistic_regression', 'linear_svc', 'random_forest']:
            try:
                # Convert to dense for cuML
                if hasattr(X_test, 'toarray'):
                    X_test_np = X_test.toarray()
                else:
                    X_test_np = X_test
                
                # Convert to CuPy array for GPU processing
                X_test_gpu = cp.array(X_test_np)
                
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    proba_test = model.predict_proba(X_test_gpu)
                    
                    # Extract probability for positive class
                    if proba_test.shape[1] > 1:  # If binary classification
                        meta_features_test[:, i] = cp.asnumpy(proba_test[:, 1])
                    else:  # If only one probability returned
                        meta_features_test[:, i] = cp.asnumpy(proba_test)
                else:
                    # For models without predict_proba
                    meta_features_test[:, i] = cp.asnumpy(model.decision_function(X_test_gpu))
                    
            except Exception as e:
                print(f"Error using GPU for prediction with {name}: {e}")
                # Fall back to CPU prediction
                if hasattr(X_test, 'toarray'):
                    meta_features_test[:, i] = model.predict_proba(X_test.toarray())[:, 1]
                else:
                    meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]
        
        else:  # Standard sklearn models
            if hasattr(model, 'predict_proba'):
                meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]
            else:
                meta_features_test[:, i] = model.decision_function(X_test)

    # Make final predictions with GPU acceleration if possible
    if use_gpu and not isinstance(meta_learner, LogisticRegression):  # If meta_learner is cuML
        try:
            # Convert to CuPy array
            meta_features_test_gpu = cp.array(meta_features_test)
            
            # Predict with GPU
            test_predictions_gpu = meta_learner.predict(meta_features_test_gpu)
            test_predictions = cp.asnumpy(test_predictions_gpu)
        except Exception as e:
            print(f"Error using GPU for meta-learner prediction: {e}")
            # Fall back to CPU
            test_predictions = meta_learner.predict(meta_features_test)
    else:
        # Standard prediction
        test_predictions = meta_learner.predict(meta_features_test)
    
    print(f"Prediction completed in {time.time() - start_time:.2f} seconds")
    return test_predictions

# 4. Main execution function

def improved_sentiment_analysis(X_train, y_train, X_test, y_test, X_val, y_val, processed_texts_train, processed_texts_test, processed_texts_val, feature_names):
    """
    Main function to execute the improved sentiment analysis pipeline with GPU acceleration where possible
    """
    overall_start_time = time.time()
    
    # Check for GPU availability
    check_gpu()
    
    print("Enhancing features with text metadata...")
    X_train_enhanced = extract_additional_features(X_train, processed_texts_train)
    X_val_enhanced = extract_additional_features(X_val, processed_texts_val)
    X_test_enhanced = extract_additional_features(X_test, processed_texts_test)

    print(f"Original feature shape: {X_train.shape}")
    print(f"Enhanced feature shape: {X_train_enhanced.shape}")

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
