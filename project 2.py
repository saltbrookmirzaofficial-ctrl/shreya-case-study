import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==========================================
# 1. Generate Training Dataset
# ==========================================

np.random.seed(42)

students = 500

# Random student data
study_hours = np.random.uniform(1, 12, students)
attendance = np.random.uniform(40, 100, students)
assignment = np.random.uniform(30, 100, students)
previous_exam = np.random.uniform(30, 100, students)

# Combine features
X = np.column_stack((
    study_hours,
    attendance,
    assignment,
    previous_exam
))

# ==========================================
# 2. Create Target Output
# ==========================================

# Calculate performance score
performance_score = (
    study_hours * 5 +
    attendance * 0.25 +
    assignment * 0.25 +
    previous_exam * 0.25
)

# Pass / Fail
y = (performance_score >= 55).astype(int)

# ==========================================
# 3. Normalize Data
# ==========================================

X[:, 0] = X[:, 0] / 12``
X[:, 1] = X[:, 1] / 100
X[:, 2] = X[:, 2] / 100
X[:, 3] = X[:, 3] / 100

# ==========================================
# 4. Create Neural Network
# ==========================================

model = keras.Sequential([

    layers.Input(shape=(4,)),

    layers.Dense(32, activation="relu"),

    layers.Dense(16, activation="relu"),

    layers.Dense(8, activation="relu"),

    layers.Dense(1, activation="sigmoid")
])

# ==========================================
# 5. Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 6. Train Model
# ==========================================

print("\nTraining Neural Network...")

model.fit(
    X,
    y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print("Training Completed!\n")

# ==========================================
# 7. Dynamic Prediction Function
# ==========================================

def predict_student():

    print("\n==============================")
    print(" STUDENT PERFORMANCE PREDICTOR")
    print("==============================")

    name = input("Student Name: ")

    study = float(input("Study Hours per Day: "))
    attend = float(input("Attendance (%): "))
    assignment_score = float(input("Assignment Score (%): "))
    previous_score = float(input("Previous Exam Score (%): "))

    # Normalize
    student_data = np.array([[
        study / 12,
        attend / 100,
        assignment_score / 100,
        previous_score / 100
    ]])

    # Prediction
    prediction = model.predict(
        student_data,
        verbose=0
    )[0][0]

    probability = prediction * 100

    print("\n------------------------------")
    print("Student:", name)

    print(
        "Pass Probability:",
        round(probability, 2),
        "%"
    )

    if probability >= 50:

        print("Result: PASS")

        if probability >= 80:
            print("Performance: Excellent")

        elif probability >= 65:
            print("Performance: Good")

        else:
            print("Performance: Average")

    else:

        print("Result: FAIL")
        print("Performance: Needs Improvement")

    print("------------------------------")


# ==========================================
# 8. Dynamic Menu
# ==========================================

while True:

    print("\n========== MENU ==========")

    print("1. Predict Student Performance")
    print("2. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":

        predict_student()

    elif choice == "2":

        print("\nProgram Ended.")
        break

    else:

        print("Invalid choice!")