import os
import pandas as pd

def extractLabels(sql_text):
    """Extract SQL labels relevant to basic SELECT statements and aggregations."""
    labels = [
        'SELECT', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY',
        'ASC', 'DESC', 'LIMIT', 'OFFSET', 
        'LIKE', 'BETWEEN', 'IN', 
        'IS NULL', 'IS NOT NULL'
    ]
    
    sql_upper = sql_text.upper()
    
    syntax = []
    for label in labels:
        if label in sql_upper:
            syntax.append(label)
    
    return syntax

all_labels = [
    'SELECT', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY',
    'ASC', 'DESC', 'LIMIT', 'OFFSET', 
    'LIKE', 'BETWEEN', 'IN', 
    'IS NULL', 'IS NOT NULL'
]
final_df = pd.DataFrame(columns=['Phrase', 'SQL'] + all_labels)
folder_path = "../data"

# Initialize an empty DataFrame
final_df = pd.DataFrame(columns=['Phrase', 'SQL'])
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        # Read the dataset
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # Check if necessary columns exist
        if 'Phrase' in df.columns and 'SQL' in df.columns:
            for _, row in df.iterrows():
                phrase = row['Phrase']
                sql_query = row['SQL']
                labels = extractLabels(sql_query)
                
                # Create a dictionary for this row
                row_data = {label: 1 if label in labels else 0 for label in all_labels}
                row_data['Phrase'] = phrase
                row_data['SQL'] = sql_query
                
                # Append the row to the final DataFrame
                final_df = pd.concat(
                    [final_df, pd.DataFrame([row_data])],
                    ignore_index=True
                )
final_output_path = "../dataset.csv"
final_df.to_csv(final_output_path, index=False)

print(f"Final dataset saved to: {final_output_path}")