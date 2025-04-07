import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load

# -------------------------------
# 🔢 IMAGE CLASSIFIER (Zernike)
# -------------------------------

# Add pyfeats path for local import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyfeats import zernikes_moments  # Ensure pyfeats is installed or cloned locally

DATASET_PATH = "datasets/shapes/"
radius = 64  # Zernike Moments parameter

def extract_features(dataset_path):
    features, labels = [], []
    
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue

        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image = cv2.resize(image, (128, 128))
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            feats, _ = zernikes_moments(binary, radius=radius)
            features.append(feats)
            labels.append(label)
    
    return np.array(features), np.array(labels)

# Load and prepare data
X, y = extract_features(DATASET_PATH)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.33, random_state=42, stratify=y_encoded)

# Train image-based SVM classifier
clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, C=10, gamma='scale'))
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n🧠 Image-Based Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and encoder
dump(clf, "zernike_shape_svm_model.joblib")
dump(label_encoder, "label_encoder.joblib")

# Visualizations
def visualize_predictions(X_test, y_test, model, encoder):
    probas = model.predict_proba(X_test)
    for i, prob in enumerate(probas):
        true_label = encoder.inverse_transform([y_test[i]])[0]
        pred_label = encoder.inverse_transform([np.argmax(prob)])[0]
        confidence = np.max(prob) * 100
        print(f"Image {i+1}: True = {true_label}, Predicted = {pred_label} ({confidence:.2f}% confidence)")

def save_confidence_plot(X_test, model, output_path="confidence_plot.png"):
    probas = model.predict_proba(X_test)
    confidences = np.max(probas, axis=1)
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(confidences)), confidences * 100, marker='o', linestyle='--')
    plt.title('Prediction Confidence (%) per Test Sample')
    plt.xlabel('Sample Index')
    plt.ylabel('Confidence')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_classwise_confidence_histogram(X_test, model, encoder, output_path="confidence_histogram.png"):
    probas = model.predict_proba(X_test)
    confidences = np.max(probas, axis=1)
    predicted_labels = model.predict(X_test)
    decoded_labels = encoder.inverse_transform(predicted_labels)

    plt.figure(figsize=(8, 5))
    for label in np.unique(decoded_labels):
        label_conf = [conf for conf, l in zip(confidences, decoded_labels) if l == label]
        plt.hist(np.array(label_conf)*100, bins=10, alpha=0.6, label=label)

    plt.title("Class-wise Prediction Confidence Histogram")
    plt.xlabel("Confidence (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Run image-based visualizations
visualize_predictions(X_test, y_test, clf, label_encoder)
save_confidence_plot(X_test, clf)
save_classwise_confidence_histogram(X_test, clf, label_encoder)

# -------------------------------------
# ✨ NLP ENHANCEMENT: TEXT CLASSIFIER
# -------------------------------------

print("\n🧾 Training NLP Classifier from Text Descriptions...\n")

# Example descriptions and labels
descriptions = [
    "A round shape with no edges or corners",  # circle
    "A shape that is perfectly round",         # circle
    "Three straight sides and three angles",   # triangle
    "A geometric figure with three corners",   # triangle
    "Four equal sides and right angles",       # square
    "A shape with four sides of the same length" # square
]
text_labels = [
    "circle", "circle",
    "triangle", "triangle",
    "square", "square"
]

# Train TF-IDF + LinearSVC pipeline
text_clf = make_pipeline(TfidfVectorizer(), LinearSVC())
text_clf.fit(descriptions, text_labels)

# Evaluate NLP classifier
text_preds = text_clf.predict(descriptions)
print("📄 Text Description Classification Report:\n")
print(classification_report(text_labels, text_preds))

# Save NLP classifier
dump(text_clf, "text_to_shape_classifier.joblib")
