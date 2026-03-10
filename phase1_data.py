import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_student_data(num_students=300):
    """Generates synthetic student performance data based on the project parameters."""
    np.random.seed(42)
    
    # 1. Attendance (typically right-skewed, most students attend >60%)
    # Using normal distribution that we'll clip
    attendance = np.random.normal(loc=78, scale=12, size=num_students)
    attendance = np.clip(attendance, 30, 100)
    
    # 2. Assignment Scores (highly correlated with attendance)
    # Base score + attendance factor + noise
    assignment_base = np.random.normal(loc=40, scale=10, size=num_students)
    assignment = 0.4 * attendance + assignment_base
    assignment = np.clip(assignment, 20, 100)
    
    # 3. CAT Scores (correlated with both)
    cat1_base = np.random.normal(loc=35, scale=12, size=num_students)
    cat_1_score = 0.3 * attendance + 0.2 * assignment + cat1_base
    cat_1_score = np.clip(cat_1_score, 10, 100)
    
    cat2_base = np.random.normal(loc=35, scale=12, size=num_students)
    cat_2_score = 0.3 * attendance + 0.2 * assignment + cat2_base
    cat_2_score = np.clip(cat_2_score, 10, 100)
    
    # 4. Final Exam Score
    final_base = np.random.normal(loc=20, scale=8, size=num_students)
    final_score = 0.2 * cat_1_score + 0.2 * cat_2_score + 0.3 * assignment + 0.2 * attendance + final_base
    final_score = np.clip(final_score, 0, 100)
    
    # Departments
    departments = ['Computing and Informatics', 'Business', 'Engineering', 'Applied Sciences']
    selected_departments = np.random.choice(departments, size=num_students, p=[0.4, 0.3, 0.2, 0.1])
    
    # Create DataFrame
    df = pd.DataFrame({
        'StudentID': [f'KNP/STUD/{i:04d}' for i in range(1, num_students + 1)],
        'Department': selected_departments,
        'Attendance_Pct': attendance.round(1),
        'Assignment_Score': assignment.round(1),
        'CAT_1_Score': cat_1_score.round(1),
        'CAT_2_Score': cat_2_score.round(1),
        'Final_Exam_Score': final_score.round(1)
    })
    
    # Define Pass/Fail (Pass = Final Score >= 50)
    df['Status'] = np.where(df['Final_Exam_Score'] >= 50, 'Pass', 'Fail')
    
    return df

def perform_eda(df, output_dir='eda_outputs'):
    """Performs Exploratory Data Analysis and saves plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Basic Stats
    print("\n--- Descriptive Statistics ---")
    print(df.describe().round(2))
    print("\n--- Class Balance ---")
    print(df['Status'].value_counts(normalize=True).round(3) * 100)
    
    # 2. Correlation Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[['Attendance_Pct', 'Assignment_Score', 'CAT_1_Score', 'CAT_2_Score', 'Final_Exam_Score']].corr(), 
                annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Correlation between Academic Features')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png')
    plt.close()
    
    print("\nEDA completed. Plots saved to 'eda_outputs' directory.")

if __name__ == "__main__":
    # In the future, this is where you swap for real data
    # Example: df = pd.read_csv('actual_student_data.csv')
    
    print("Generating simulated student data...")
    df = generate_student_data(300)
    
    # Save the simulated data so we have it for the next phases
    df.to_csv('simulated_student_data.csv', index=False)
    print("Saved 'simulated_student_data.csv'")
    
    perform_eda(df)
