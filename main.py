import pandas as pd
import time
import ollama


def categorize_description(description):
    """
    Use LLama3 via ollama to predict the category based on the description.
    If the model is uncertain, it will return 'Uncertain', prompting the user for input.
    """
    prompt = f"""
    
    Classify the following transaction description into an appropriate category. Only response with the category name.
    
    Transaction description: "{description}"
    
    Possible categories: Food, Bills & Utilities, Groceries, Shopping, Others
    
    If unsure, respond with only 'Uncertain'.
    
    Category:
    """
    
    response = ollama.chat(model="llama3", 
                           messages=[
                               {"role": "system", 
                                "content": "You are a financial categorization assistant."},
                                {"role": "user", 
                                 "content": prompt}])
    
    category = response["message"]["content"].strip()
    return category if category not in ["Uncertain", ""] else None

def user_review(description):
    """Prompt the user for a category when the model is uncertain."""
    print(f"Description: {description}")
    user_input = ""
    while not user_input:
        user_input = input("Enter correct category: ").strip()
    return user_input

def process_csv(file_path, output_path):
    """Read a CSV file, add a category column, prompt user only for uncertain cases, and save the updated file."""
    df = pd.read_csv(file_path)
    
    if "description" not in df.columns:
        raise ValueError("The CSV file must contain a 'description' column.")
    
    df["category"] = df["description"].apply(lambda desc: categorize_description(desc) or user_review(desc))
    df.to_csv(output_path, index=False)
    print(f"Processed file saved to {output_path}")

if __name__ == "__main__":
    input_file = "./processed data/data.csv"  # Path to the original CSV file
    output_file = "./output/output_data.csv"  # Path to save the updated CSV file
    start_time = time.time()
    process_csv(input_file, output_file)
    end_time = time.time()
    print(f'Time: {end_time - start_time} s')