import pandas as pd
import ollama  # Replace this with your model library if needed
import os
import json
import sqlparse
import time
from torch.amp import autocast

# Generate synthetic data using a prompt template
def generate_synthetic_data(batch_size):
    prompt = """
        Generate a JSON array containing 20 examples of natural language phrases paired with simple SQL SELECT queries. 

        Constraints:
        1. Queries must not include subqueries or JOIN operations.
        2. Use only basic SQL components: SELECT, WHERE, GROUP BY, HAVING, ORDER BY, ASC, DESC, LIMIT, OFFSET, LIKE, BETWEEN, IN, IS NULL, IS NOT NULL.
        3. Ensure queries retrieve realistic and varied data selections based on real-world scenarios and are unique
        4. Do not attempt to include all possible components in every query. Focus on simplicity and clarity.

        Format:
        Each object in the JSON array should have:
        - "phrase": a natural language description of the data request.
        - "sql": the corresponding SQL SELECT query.

        Example:
        [
            {"phrase": "Show all customer names and email addresses", "sql": "SELECT name, email FROM customers;"},
            {"phrase": "List all product names with their prices", "sql": "SELECT product_name, price FROM products;"},
            {"phrase": "Display order dates for all completed orders", "sql": "SELECT order_date FROM orders WHERE status = 'completed';"},
            {"phrase": "Get employee names who work in the Sales department", "sql": "SELECT name FROM employees WHERE department = 'Sales';"},
            {"phrase": "Retrieve the names of products with prices over $50", "sql": "SELECT product_name FROM products WHERE price > 50;"}
        ]

        Output:
        Generate only the JSON array as a response, strictly adhering to the format and rules outlined above. Do not generate help text. Only the JSON

    """
    try:
        with autocast('cuda'):
            response = ollama.generate(model='llama3:8b', prompt=prompt)
        # Parse the response as JSON
        return json.loads(response["response"])
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}. Skipping this batch.")
        return []  # Return an empty list to skip invalid responses
    except Exception as e:
        print(f"Unexpected error during generation: {e}")
        return []


# Validate SQL to ensure it is a valid SELECT statement
def validate_sql(sql_query):
    try:
        parsed = sqlparse.parse(sql_query)
        if parsed:
            statement = parsed[0]
            if statement.get_type() == "SELECT":
                return True
        return False
    except Exception:
        print(f"Invalid SQL")
        return False

# Prepare data from generated output with validation
def prepare_data(data_batch):
    processed_data = []
    for item in data_batch:
        # Validate SQL before adding it
        if validate_sql(item['sql']):
            processed_data.append([item['phrase'], item['sql']])
    return processed_data

# Append data to CSV
def append_to_csv(data_batch, filename="synthetic_data.csv"):
    # Convert data to DataFrame
    data = pd.DataFrame(data_batch, columns=["Phrase", "SQL"])
    # Write to CSV, appending if file exists
    data.to_csv(
        filename,
        mode='a',  # Append mode
        header=not os.path.exists(filename),  # Write header only if file does not exist
        index=False  # No index
    )

# Main loop to generate and save synthetic data
def main(num_samples, batch_size=20):
    for batch_num in range(num_samples // batch_size):
        try:
            start_time = time.time()  # Start timer
            # Generate batch of phrases and SQL queries
            generated_data = generate_synthetic_data(batch_size)

            # Process and append data to CSV
            if generated_data:
                processed_data = prepare_data(generated_data)
                if processed_data:  # Only append if there are valid entries
                    append_to_csv(processed_data)

            # Log the elapsed time for each batch
            elapsed_time = time.time() - start_time
            print(f"Batch {batch_num + 1} processed. Elapsed time: {elapsed_time:.2f} seconds")
        except Exception as e:
            print(f"Error in batch {batch_num + 1}: {e}. Skipping this batch.")


print("Starting")
# Run the main function
num_samples = 200000 # Total number of samples to generate
main(num_samples)
           