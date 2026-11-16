
import numpy as np
import sys

# ==============================
# LAB SHEET 1 - PYTHON
# Array Creation & Basics
# ==============================

# Q1. Multiples of 7 between 7 and 100
arr = np.arange(7, 101, 7)
print("Q1:", arr)
print("Length:", len(arr))


# Q2. 4x4 array filled with 9
arr = np.full((4, 4), 9)
print("\nQ2:\n", arr)


# Q3. Convert integer array into float
arr = np.arange(1, 11)
float_arr = arr.astype(float)
print("\nQ3:", float_arr)
print("Data type:", float_arr.dtype)


# Q4. Compare memory size of list and NumPy array
lst = [10, 20, 30, 40, 50]
arr = np.array(lst)
print("\nQ4: List memory:", sys.getsizeof(lst))
print("NumPy array memory:", sys.getsizeof(arr))


# ==============================
# Indexing & Slicing
# ==============================

# Q5. Extract four corner elements
arr = np.arange(1, 37).reshape(6, 6)
corners = arr[[0, 0, -1, -1], [0, -1, 0, -1]]
print("\nQ5:\n", arr)
print("Corners:", corners)


# Q6. Every third element starting from index 1
arr = np.arange(1, 13)
print("\nQ6:", arr[1::3])


# Q7. Reverse rows of 5x5 array
arr = np.arange(1, 26).reshape(5, 5)
print("\nQ7:\n", arr[::-1])


# Q8. Replace elements divisible by 3 with -1
arr = np.arange(1, 11)
arr[arr % 3 == 0] = -1
print("\nQ8:", arr)


# ==============================
# Array Operations
# ==============================

# Q9. Normalize array between 0 and 1
arr = np.array([10, 20, 30, 40, 50])
normalized = (arr - arr.min()) / (arr.max() - arr.min())
print("\nQ9:", normalized)


# Q10. Transpose of a matrix
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
transposed = arr.T
print("\nQ10 Transpose:\n", transposed)
print("Original restored:", np.array_equal(transposed.T, arr))


# Q11. Maximum value and its index
arr = np.random.randint(1, 100, 10)
index = np.argmax(arr)
print("\nQ11:", arr)
print("Maximum:", arr[index])
print("Index:", index)


# Q12. Element-wise remainder
a = np.array([10, 20, 30, 40])
b = np.array([3, 6, 7, 9])
print("\nQ12:", np.remainder(a, b))


# ==============================
# Concatenation & Reshaping
# ==============================

# Q13. Stack two 2x3 arrays
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])
result = np.stack((a, b))
print("\nQ13:\n", result)
print("Shape:", result.shape)


# Q14. Reshape and concatenate three arrays
a = np.array([[1], [2]])
b = np.array([[3], [4], [5]])
c = np.array([[6]])
result = np.concatenate((a.reshape(-1),
                         b.reshape(-1),
                         c.reshape(-1)))
print("\nQ14:", result)


# Q15. Reshape 1D array into 4x4 and transpose
arr = np.arange(1, 17)
matrix = arr.reshape(4, 4)
column_major = np.transpose(matrix)
print("\nQ15 Matrix:\n", matrix)
print("Column-major equivalent:\n", column_major)


# ==============================
# Splitting
# ==============================

# Q16. Split 3D array into two equal parts
arr = np.arange(32).reshape(2, 4, 4)
parts = np.split(arr, 2, axis=0)
print("\nQ16 Part 1:\n", parts[0])
print("Part 2:\n", parts[1])


# Q17. Split 17 elements into 4 parts
arr = np.arange(1, 18)
parts = np.array_split(arr, 4)
print("\nQ17:")
for part in parts:
    print(part, "Size:", len(part))


# ==============================
# Statistics
# ==============================

# Q18. Remove outliers beyond 2 standard deviations
arr = np.array([10, 12, 11, 13, 12, 14, 15, 100])
mean = np.mean(arr)
std = np.std(arr)
filtered = arr[np.abs(arr - mean) <= 2 * std]
print("\nQ18 Original:", arr)
print("Without outliers:", filtered)


# Q19. Student and subject averages
marks = np.array([
    [80, 75, 90, 85],
    [70, 65, 80, 75],
    [90, 85, 95, 90],
    [60, 70, 65, 75],
    [85, 80, 90, 95]
])
print("\nQ19 Student averages:", np.mean(marks, axis=1))
print("Subject averages:", np.mean(marks, axis=0))


# Q20. Correlation coefficient
a = np.array([10, 20, 30, 40, 50])
b = np.array([15, 25, 35, 45, 55])
correlation = np.corrcoef(a, b)
print("\nQ20:\n", correlation)


# ==============================
# File I/O
# ==============================

# Q21. Save and load CSV file
arr = np.random.randint(1, 100, (5, 5))
np.savetxt("data.csv", arr, delimiter=",", fmt="%d")
loaded = np.loadtxt("data.csv", delimiter=",", dtype=int)
print("\nQ21 Original:\n", arr)
print("Loaded:\n", loaded)
print("Arrays match:", np.array_equal(arr, loaded))


# Q22. Save three arrays in NPZ file
a = np.array([1, 2, 3])
b = np.array([[4, 5], [6, 7]])
c = np.array([8, 9, 10, 11])

np.savez("arrays.npz", array1=a, array2=b, array3=c)

data = np.load("arrays.npz")
print("\nQ22 Stored array names:", data.files)
data.close()


# ==============================
# NumPy Random Module
# ==============================

# Q23. Random 3x3 matrix and sort each row
arr = np.random.randint(10, 51, (3, 3))
print("\nQ23 Original:\n", arr)
print("Sorted rows:\n", np.sort(arr, axis=1))


# Q24. Biased coin simulation (70% heads)
heads = np.random.binomial(1, 0.7, 200)
count = np.sum(heads)
print("\nQ24 Heads:", count)
print("Observed proportion:", count / 200)


# Q25. Normal distribution and count outside [-4, 4]
arr = np.random.normal(0, 2, 2000)
count = np.sum((arr < -4) | (arr > 4))
print("\nQ25 Values outside [-4, 4]:", count)


# Q26. Poisson distribution for 60 minutes
visits = np.random.poisson(5, 60)
minute = np.argmax(visits) + 1
print("\nQ26 Website visits:", visits)
print("Minute with highest visits:", minute)
print("Highest visits:", visits[minute - 1])


# Q27. Multinomial distribution for 200 respondents
products = np.random.multinomial(200, [0.5, 0.3, 0.2])
print("\nQ27 Product 1:", products[0])
print("Product 2:", products[1])
print("Product 3:", products[2])


# Q28. Random permutation of 52 cards
deck = np.arange(1, 53)
shuffled = np.random.permutation(deck)
hand = shuffled[:5]
print("\nQ28 Shuffled deck:", shuffled)
print("Five cards hand:", hand)
