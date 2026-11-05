import glob
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

class ClassifierTrainer:
    def __init__(
            dataset_path: str = "dataset/*.csv",
            model_output_path: str = "sign_model.pkl"
        ):
        X = []
        y = []

        # Load all csv files in the dataset folder
        for file_path in glob.glob(dataset_path):
            sign_name = os.path.basename(file_path).replace(".csv", "")

            with open(file_path, "r") as f:
                for line in f:
                    values = [float(val) for val in line.strip().split(",")]
                    X.append(values)
                    y.append(sign_name)

        X = np.array(X)
        y = np.array(y)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train an SVM Classifier
        clf = SVC(kernel="rbf", C=10, probability=True)
        clf.fit(X_train, y_train)

        # Evaluate accuracy
        y_pred = clf.predict(X_test)
        print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

        # Save the trained model
        joblib.dump(clf, model_output_path)
        print(f"Model saved as '{model_output_path}'")

def main():
    ClassifierTrainer()

if __name__ == "__main__":
    main()