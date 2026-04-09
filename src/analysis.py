import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Load Dataset
# -----------------------------
# df = pd.read_csv('../data/students.csv')
base_dir = os.path.dirname(os.path.abspath(__file__))

# Correct path to CSV
data_path = os.path.join(base_dir, '..', 'data', 'students.csv')

print(" Loading file from:", data_path)

df = pd.read_csv(data_path)

print("🔹 First 5 rows:")
print(df.head())

print("\n🔹 Dataset Info:")
print(df.info())

print("\n🔹 Statistical Summary:")
print(df.describe())




# -----------------------------
# 2. Data Cleaning
# -----------------------------
print("\n🔹 Checking Missing Values:")
print(df.isnull().sum())

df = df.dropna()

# -----------------------------
# 3. Basic Statistics
# -----------------------------
math_avg = df['math score'].mean()
reading_avg = df['reading score'].mean()
writing_avg = df['writing score'].mean()

print("\n🔹 Average Scores:")
print("Math:", math_avg)
print("Reading:", reading_avg)
print("Writing:", writing_avg)

# -----------------------------
# 4. Visualization
# -----------------------------
# plt.figure()
# plt.bar(['Math', 'Reading', 'Writing'],
#         [math_avg, reading_avg, writing_avg])

# plt.title("Average Student Scores")
# plt.xlabel("Subjects")
# plt.ylabel("Average Score")

# # Save image
# plt.savefig('../images/output.png')
# plt.show()
import os
import matplotlib.pyplot as plt

# Get current file directory (analysis.py location)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Create images folder path
image_path = os.path.join(base_dir, '..', 'images')
os.makedirs(image_path, exist_ok=True)

# Full file path
file_path = os.path.join(image_path, 'output.png')

# Plot
plt.figure()
plt.bar(['Math', 'Reading', 'Writing'],
        [math_avg, reading_avg, writing_avg])

plt.title("Average Student Scores")

# Save
plt.savefig(file_path)

print(" Image saved at:", file_path)

plt.show()

# -----------------------------
# 5. Extra Visualization
# -----------------------------
plt.figure()
df['math score'].hist()
plt.title("Math Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# 6. Filter Example
# -----------------------------
top_students = df[df['math score'] > 80]

print("\n🔹 Students with Math Score > 80:")
print(top_students.head())

# -----------------------------
# 7. Machine Learning
# -----------------------------
X = df[['reading score']]
y = df['writing score']

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[70]])

print("\n🔹 ML Prediction:")
print("Predicted Writing Score for Reading Score 70:", prediction[0])