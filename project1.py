import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------------
# 1. Training Data
# -------------------------------

# [Study Hours, Attendance %]
X = np.array([
    [1, 40],
    [2, 45],
    [2, 50],
    [3, 55],
    [3, 60],
    [4, 65],
    [5, 70],
    [5, 75],
    [6, 80],
    [7, 85],
    [8, 90],
    [9, 95]
], dtype=float)

# 0 = Fail, 1 = Pass
y = np.array([
    0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1
])




# -------------------------------
# 2. Normalize Data
# -------------------------------

X[:, 0] = X[:, 0] / 10
X[:, 1] = X[:, 1] / 100

# -------------------------------
# 3. Create Neural Network
# -------------------------------

model = keras.Sequential([
    layers.Dense(8, activation="relu", input_shape=(2,)),
    layers.Dense(4, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# -------------------------------
# 4. Compile Model
# -------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# 5. Train Model
# -------------------------------

print("Training Neural Network...")

model.fit(
    X,
    y,
    epochs=200,
    verbose=0
)

print("Training Completed!\n")

# -------------------------------
# 6. Dynamic Prediction
# -------------------------------

while True:

    print("-----------------------------")
    print(" Student Performance AI")
    print("-----------------------------")

    study_hours = input(
        "Enter study hours per day (or 'q' to quit): "
    )

    if study_hours.lower() == "q":
        print("Program stopped.")
        break

    attendance = input(
        "Enter attendance percentage: "
    )

    try:
        study_hours = float(study_hours)
        attendance = float(attendance)

        # Validate input
        if study_hours < 0 or study_hours > 24:
            print("Study hours must be between 0 and 24.")
            continue

        if attendance < 0 or attendance > 100:
            print("Attendance must be between 0 and 100.")
            continue

        # Normalize user input
        new_student = np.array([
            [study_hours / 10, attendance / 100]
        ])

        # Neural Network Prediction
        prediction = model.predict(
            new_student,
            verbose=0
        )

        probability = prediction[0][0]

        # Result
        print("\nAI Result")
        print("-----------------------------")
        print(
            f"Pass Probability: {probability * 100:.2f}%"
        )

        if probability >= 0.5:
            print("Prediction: PASS ✅")
        else:
            print("Prediction: FAIL ❌")

    except ValueError:
        print("Please enter valid numbers.")
