import tkinter as tk
from tkinter import messagebox
from sklearn.linear_model import LinearRegression

def recommend_spending():
    try:
        salary = float(salary_entry.get())

        # Predict ideal spending targets
        predicted_targets = model.predict([[salary]])
        
        recommendations = {
            'Rent': f"Recommended: ${predicted_targets[0][0]:.2f}",
            'Leisure': f"Recommended: ${predicted_targets[0][1]:.2f}",
            'Saving': f"Recommended: ${predicted_targets[0][2]:.2f}"
        }
        messagebox.showinfo("Recommendations", "\n".join([f"{category}: {recommendation}" for category, recommendation in recommendations.items()]))
    
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid numeric salary.")

root = tk.Tk()
root.title("Spending Recommendation")

root.configure(bg="#f0f0f0")

input_frame = tk.Frame(root, padx=20, pady=20, bg="#f0f0f0")
input_frame.pack()

# Create labels and entry fields for input
label_font = ("Arial", 12, "bold")
entry_font = ("Arial", 12)

salary_label = tk.Label(input_frame, text="Salary:", font=label_font, bg="#f0f0f0")
salary_label.grid(row=0, column=0, sticky="w")
salary_entry = tk.Entry(input_frame, font=entry_font)
salary_entry.grid(row=0, column=1)

# model training
# Simulated data for demonstration
# Salary data
X = [[30000], [40000], [50000], [60000], [70000]]
# Ideal spending targets (rent, leisure, saving)
y = [[9000, 6000, 9000], [12000, 8000, 12000], [15000, 10000, 15000], [18000, 12000, 18000], [21000, 14000, 21000]]

model = LinearRegression()
model.fit(X, y)

recommend_button = tk.Button(root, text="Get Recommendations", command=recommend_spending, font=label_font, bg="#4CAF50", fg="white")
recommend_button.pack(pady=10)

root.mainloop()
