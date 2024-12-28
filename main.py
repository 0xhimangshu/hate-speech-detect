import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import VotingClassifier
from typing import Tuple, Dict


class HateSpeechDetector:
    def __init__(self) -> None:
        self.model = None
        self.label_mapping = {"hateful": 1, "not_hateful": 0}

    def load_datasets(self) -> Tuple[pd.Series, pd.Series]:
        """
        Load and combine all available datasets
        """
        # Load all datasets
        all_cases = pd.read_csv("./data/all_cases.csv")
        all_annotations = pd.read_csv("./data/all_annotations.csv")
        test_suite_cases = pd.read_csv("./data/test_suite_cases.csv")
        test_suite_annotations = pd.read_csv("./data/test_suite_annotations.csv")

        # Combine cases and annotations
        combined_cases = pd.concat(
            [
                all_cases[["test_case", "label_gold"]],
                test_suite_cases[["test_case", "label_gold"]],
                all_annotations[["test_case", "label_gold"]],
                test_suite_annotations[["test_case", "label_gold"]],
            ]
        ).drop_duplicates()

        combined_cases["test_case"] = combined_cases["test_case"].apply(self.clean_text)

        X = combined_cases["test_case"].astype(str)
        y = (combined_cases["label_gold"] == "hateful").astype(int)

        return X, y

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""

        text = text.lower()

        leet_map = {
            "0": "o",
            "1": "i",
            "3": "e",
            "4": "a",
            "5": "s",
            "7": "t",
            "@": "a",
            "$": "s",
        }
        for k, v in leet_map.items():
            text = text.replace(k, v)

        text = " ".join(text.split())

        return text

    def create_model(self) -> Pipeline:
        """
        Create an ensemble model combinning multiple classifiers
        """
        tfidf = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 4),
            stop_words="english",
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
        )

        classifiers = [
            (
                "lr",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
                ),
            ),
        ]

        return Pipeline(
            steps=[
                ("tfidf", tfidf),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "ensemble",
                    VotingClassifier(estimators=classifiers, voting="soft", n_jobs=-1),
                ),
            ]
        )

    def train_model(self, X: pd.Series, y: pd.Series) -> None:
        """
        Train the model with cross-validation and detailed evaluation
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = self.create_model()

        cv_scores = cross_val_score(
            estimator=self.model, X=X_train, y=y_train, cv=5, n_jobs=-1
        )
        print(f"\nCross-validation scores: {cv_scores}")
        print(
            f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})"
        )

        self.model.fit(X=X_train, y=y_train)

        self.evaluate_model(X_test=X_test, y_test=y_test)

    def evaluate_model(self, X_test: pd.Series, y_test: pd.Series) -> None:
        """
        Comprehensive model evaluation
        """
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        print("\nClassification Report:")
        print(classification_report(y_true=y_test, y_pred=y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true=y_test, y_pred=y_pred))

        # Print high-confidence misclassifications
        self.analyze_misclassifications(X_test, y_test, y_pred, y_pred_proba)

    def analyze_misclassifications(
        self,
        X_test: pd.Series,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> None:
        """
        Analyze misclassified examples
        """
        misclassified_idx = np.where(y_test != y_pred)[0]

        if len(misclassified_idx) > 0:
            print("\nHigh Confidence Misclassifications:")
            for idx in misclassified_idx:
                confidence = y_pred_proba[idx].max()
                if confidence > 0.9:
                    print(f"\nText: {X_test.iloc[idx]}")
                    print(
                        f"True label: {'Hateful' if y_test.iloc[idx] == 1 else 'Not Hateful'}"
                    )
                    print(
                        f"Predicted: {'Hateful' if y_pred[idx] == 1 else 'Not Hateful'}"
                    )
                    print(f"Confidence: {confidence:.2%}")

    def predict(self, text: str) -> Dict:
        """
        Make a prediction with detailed confidence scores
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please provide a non-empty text string")

        text = self.clean_text(text)

        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]

        return {
            "text": text,
            "prediction": "Hateful" if prediction == 1 else "Not Hateful",
            "confidence": float(max(probabilities)),
            "hate_probability": float(probabilities[1]),
            "not_hate_probability": float(probabilities[0]),
        }


def main() -> None:
    try:
        detector = HateSpeechDetector()
        print("Loading datasets...")
        X, y = detector.load_datasets()

        print("Training model...")
        detector.train_model(X, y)

        test_cases = [
            "I love all people equally",
            "I hate women",
            "Everyone deserves respect",
            "I hate trans people",
            "Let's celebrate diversity",
            "I really can't stand Muslims like you.",
        ]

        print("\nTesting example cases:")
        for text in test_cases:
            result = detector.predict(text)
            print(f"\nText: {result['text']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Hate probability: {result['hate_probability']:.2%}")
            print(f"Not hate probability: {result['not_hate_probability']:.2%}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
