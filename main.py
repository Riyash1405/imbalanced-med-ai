from src.train_tabular import train_tabular_models
from src.evaluate import evaluate_all_models
from src.multimodal_model import train_multimodal_model

if __name__ == "__main__":
    print("\n=== TRAINING TABULAR MODELS (Heart + Parkinson's) ===")
    train_tabular_models()

    print("\n=== EVALUATION ===")
    evaluate_all_models()

    print("\n=== TRAINING MULTIMODAL CNN + TRANSFORMER ===")
    train_multimodal_model()

    print("\nAll tasks completed successfully.")
