import csv
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas
from sklearn.metrics import accuracy_score

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python scratch.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Prompt user for their data
    user_data = get_user_data()

    # Predict the outcome based on user input
    user_prediction = model.predict([user_data])
    print("Predicted outcome:", "Positive" if user_prediction[0] == 1 else "Negative")
    print("with accuracy of", accuracy, "according to the Pima Indians Diabetes Database")


def load_data(filename):
    evidence = []
    labels = []
    df = pandas.read_csv(filename)
    df = df[(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']] != 0).all(axis=1)]
    evidence_df = df.drop(columns=['Outcome'])
    labels_df = df['Outcome']
    evidence = evidence_df.values.tolist()
    labels = labels_df.values.tolist()

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted logistic regression model trained on the data.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(evidence, labels)
    return model


def get_user_data():
    """
    Prompt the user to input their data and return it as a list.
    """
    user_data = []
    user_data.append(int(input("Enter number of pregnancies: ")))
    user_data.append(float(input("Enter Glucose level: ")))
    user_data.append(float(input("Enter Blood Pressure: ")))
    user_data.append(float(input("Enter Skin Thickness: ")))
    user_data.append(float(input("Enter Insulin level: ")))
    user_data.append(float(input("Enter BMI: ")))
    user_data.append(float(input("Enter Diabetes Pedigree Function: ")))
    user_data.append(int(input("Enter Age: ")))

    return user_data


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positives = sum((labels[i] == 1) and (predictions[i] == 1) for i in range(len(labels)))
    true_negatives = sum((labels[i] == 0) and (predictions[i] == 0) for i in range(len(labels)))
    total_positives = sum(label == 1 for label in labels)
    total_negatives = sum(label == 0 for label in labels)

    sensitivity = true_positives / total_positives if total_positives != 0 else 0
    specificity = true_negatives / total_negatives if total_negatives != 0 else 0

    return sensitivity, specificity


if __name__ == "__main__":
    main()
