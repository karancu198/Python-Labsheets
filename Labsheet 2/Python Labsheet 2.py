
import pandas as pd

# Creating DataFrame
data = {
    "RollNo": [101, 102, 103, 104, 105, 106, 107, 108],
    "Name": ["Amit", "Neha", "Rahul", "Priya",
             "Rohan", "Kavya", "Arjun", "Simran"],
    "Department": ["BCA", "BCA", "MCA", "BCA",
                   "MCA", "MCA", "BCA", "MCA"],
    "Age": [20, 21, 22, 20, 23, 22, 21, 24],
    "Marks": [85, 92, 67, 78, 88, 95, 55, 73]
}

df = pd.DataFrame(data)

# LEVEL 1 - BASIC

# Q1. Display complete DataFrame
print("\nQ1:\n", df)

# Q2. First 3 records
print("\nQ2:\n", df.head(3))

# Q3. Last 2 records
print("\nQ3:\n", df.tail(2))

# Q4. Number of rows and columns
print("\nQ4:", df.shape)

# Q5. Display column names
print("\nQ5:", df.columns)

# Q6. Data types
print("\nQ6:\n", df.dtypes)

# Q7. Display Name column
print("\nQ7:\n", df["Name"])

# Q8. Display Name and Marks
print("\nQ8:\n", df[["Name", "Marks"]])


# LEVEL 2 - SELECTION AND FILTERING

# Q9. Marks greater than 80
print("\nQ9:\n", df[df["Marks"] > 80])

# Q10. Marks less than 70
print("\nQ10:\n", df[df["Marks"] < 70])

# Q11. Marks between 70 and 90
print("\nQ11:\n", df[df["Marks"].between(70, 90)])

# Q12. BCA students
print("\nQ12:\n", df[df["Department"] == "BCA"])

# Q13. MCA students with marks above 80
print("\nQ13:\n", df[(df["Department"] == "MCA") &
                    (df["Marks"] > 80)])

# Q14. Age greater than 21
print("\nQ14:\n", df[df["Age"] > 21])


# LEVEL 3 - DATA ANALYSIS

# Q15. Average marks
print("\nQ15:", df["Marks"].mean())

# Q16. Maximum marks
print("\nQ16:", df["Marks"].max())

# Q17. Minimum marks
print("\nQ17:", df["Marks"].min())

# Q18. Median marks
print("\nQ18:", df["Marks"].median())

# Q19. Students scoring above 75
print("\nQ19:", (df["Marks"] > 75).sum())

# Q20. Average marks of BCA students
print("\nQ20:", df[df["Department"] == "BCA"]["Marks"].mean())

# Q21. Average marks of MCA students
print("\nQ21:", df[df["Department"] == "MCA"]["Marks"].mean())

# Q22. Highest-scoring student
print("\nQ22:\n", df.loc[df["Marks"].idxmax()])


# LEVEL 4 - DATA MANIPULATION

# Q23. Add Bonus column
df["Bonus"] = 5
print("\nQ23:\n", df)

# Q24. Create FinalMarks column
df["FinalMarks"] = df["Marks"] + df["Bonus"]
print("\nQ24:\n", df)

# Q25. Create Result column
df["Result"] = df["Marks"].apply(
    lambda x: "Pass" if x >= 40 else "Fail"
)
print("\nQ25:\n", df)

# Q26. Create Grade column
def grade(marks):
    if marks >= 90:
        return "A+"
    elif marks >= 80:
        return "A"
    elif marks >= 70:
        return "B"
    elif marks >= 60:
        return "C"
    else:
        return "D"

df["Grade"] = df["Marks"].apply(grade)
print("\nQ26:\n", df)


# LEVEL 5 - RANKING

# Q27. Sort marks highest to lowest
print("\nQ27:\n", df.sort_values("Marks", ascending=False))

# Q28. Top 3 students
print("\nQ28:\n", df.nlargest(3, "Marks"))

# Q29. Add Rank column
df["Rank"] = df["Marks"].rank(
    ascending=False, method="min"
).astype(int)
print("\nQ29:\n", df)


# LEVEL 6 - GROUP ANALYSIS

# Q30. Number of students in each department
print("\nQ30:\n", df.groupby("Department").size())

# Q31. Average marks of each department
print("\nQ31:\n", df.groupby("Department")["Marks"].mean())

# Q32. Maximum marks in each department
print("\nQ32:\n", df.groupby("Department")["Marks"].max())

# Q33. Minimum marks in each department
print("\nQ33:\n", df.groupby("Department")["Marks"].min())


# CHALLENGE QUESTION - MERIT LIST

merit_list = df.sort_values(
    "Marks", ascending=False
)[["Rank", "RollNo", "Name", "Department",
   "Marks", "Grade", "Result"]]

print("\nCHALLENGE - FINAL MERIT LIST:\n", merit_list)

# Export merit list to Excel
merit_list.to_excel("student_merit_list.xlsx", index=False)

print("\nMerit list exported successfully!")
  
