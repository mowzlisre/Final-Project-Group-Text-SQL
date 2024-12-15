import re
import nltk
from nltk.tokenize import word_tokenize
from difflib import get_close_matches
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
# Function to parse schema from text file
def parse_schema_from_text(file_path):
    schema = {}
    with open(file_path, 'r') as file:
        schema_text = file.read()
    
    # Regex to match table definitions
    tables = re.findall(r"CREATE TABLE (\w+) \((.*?)\);", schema_text, re.DOTALL)
    for table_name, table_definition in tables:
        # Extract column names from the table definition
        columns = re.findall(r"(\w+)\s+\w+", table_definition)
        schema[table_name] = columns
    return schema

# Function to replace synonyms in the prompt with schema terms
stemmer = PorterStemmer()

def replace_tokens_with_schema(prompt, schema_synonyms):
    # Tokenize the prompt
    tokens = word_tokenize(prompt.lower())
    
    # Create a reverse mapping from synonyms to schema terms, considering stems
    reverse_mapping = {}
    for key, synonyms in schema_synonyms.items():
        for synonym in synonyms:
            reverse_mapping[stemmer.stem(synonym)] = key  # Use stems for better matching
    
    # Replace tokens with schema terms
    replaced_tokens = [
        reverse_mapping.get(stemmer.stem(token), token)  # Replace if stemmed token exists in synonyms
        for token in tokens
    ]
    
    return ' '.join(replaced_tokens)

# Function for fuzzy matching of tokens with schema terms
def fuzzy_replace_tokens(prompt, schema):
    tokens = word_tokenize(prompt.lower())
    schema_terms = [term for terms in schema.values() for term in terms]
    
    replaced_tokens = []
    for token in tokens:
        matches = get_close_matches(token, schema_terms, n=1, cutoff=0.8)
        replaced_tokens.append(matches[0] if matches else token)
    
    return ' '.join(replaced_tokens)

# Validation function to check if the prompt is relevant to the schema
def validate_prompt_against_schema(prompt, schema):
    tokens = word_tokenize(prompt.lower())
    for table, columns in schema.items():
        if table in tokens or any(col in tokens for col in columns):
            return True
    return False

# Function to validate and normalize the prompt
def validate_and_replace_prompt(prompt, schema, schema_synonyms):
    # Replace synonyms with schema terms
    normalized_prompt = replace_tokens_with_schema(prompt, schema_synonyms)
    
    # Validate against schema
    if validate_prompt_against_schema(normalized_prompt, schema):
        return normalized_prompt
    return None


# Function to generate synonyms using WordNet
def generate_synonyms(term):
    synonyms = set()
    for synset in wordnet.synsets(term):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace("_", " "))  # Add lemma names as synonyms
    return list(synonyms)

# Function to create schema synonyms dynamically
def create_schema_synonyms(schema):
    schema_synonyms = {}
    
    # Iterate through schema terms to generate synonyms
    for table, columns in schema.items():
        for column in columns:
            # Generate synonyms for each column name
            synonyms = generate_synonyms(column)
            schema_synonyms[column] = synonyms or []  # Default to an empty list if no synonyms found
    
    return schema_synonyms

# Full pipeline for replacing, validating, and preparing the prompt
def process_prompt(prompt):
    schema = parse_schema_from_text('./schema.txt')
    
    # Create dynamic schema synonyms
    schema_synonyms = create_schema_synonyms(schema)
    
    # Normalize and validate the prompt
    normalized_prompt = validate_and_replace_prompt(prompt, schema, schema_synonyms)
    if normalized_prompt:
        return normalized_prompt
    else:
        print("Prompt does not match schema.")
        return "Nothing"
