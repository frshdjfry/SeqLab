# import os
#
#
# def extract_kern_sequences(kern_spine):
#     # Initialize an empty list to hold the sequences
#     sequences = []
#
#     # Temporary list to hold the current sequence of **kern values
#     current_sequence = []
#
#     # Iterate through each **kern value in the spine
#     for value in kern_spine:
#         # Check if the line is empty or consists only of whitespace characters
#         if value.strip() == '':
#             continue  # Skip empty lines
#
#         # Check if the line starts with '*>', marking the beginning of a new sequence
#         if value.startswith('*>'):
#             # If the current sequence is not empty, it means we've reached the end of a sequence
#             if current_sequence:
#                 sequences.append(current_sequence)  # Add the completed sequence to the list
#                 current_sequence = []  # Reset the current sequence for the next one
#             continue  # Move on to the next line
#
#         # If the line is not a sequence marker or empty, add it to the current sequence
#         if (not value.startswith('*') and not value.startswith('=') and not value.startswith('.')
#                 and not value.startswith('1r')):
#             clean_value = value.replace('.', '').replace(';', '').replace(' ', '_').upper()
#             current_sequence.append(clean_value)
#
#     # After processing all lines, check if there's a sequence that hasn't been added yet
#     if current_sequence:
#         sequences.append(current_sequence)  # Add the last sequence to the list
#
#     return sequences
#
#
# def extract_kern_spine(dataset):
#     # print(dataset)
#     # Split the dataset into lines
#     lines = dataset.split("\n")
#
#     # Initialize an empty list to hold the **kern values
#     kern_values = []
#
#     # Locate the index of the **kern column in the first (or header) line
#     # Assuming the first line contains the column identifiers
#     spines_index = 0
#     for i, v in enumerate(lines):
#         if '**kern' in v:
#             spines_index = i
#     column_identifiers = lines[spines_index].split("\t")  # Adjust split character if necessary (e.g., space or tab)
#     kern_index = column_identifiers.index("**kern")
#
#     # Iterate over each line in the dataset
#     for line in lines:
#         # Split the line into columns (adjust split character if necessary)
#         columns = line.split("\t")
#
#         # Ensure the line has enough columns and extract the **kern value
#         if len(columns) > kern_index:
#             kern_values.append(columns[kern_index])
#
#     # Return the list of **kern values
#     return kern_values
#
#
# def extract_kern_spine_from_files(folder_path):
#     # Ensure the `extract_kern_spine` function is defined here or imported if it's defined elsewhere
#
#     # Initialize a dictionary to hold the results: file names and their extracted **kern spines
#     kern_spines = {}
#
#     # Iterate through each file in the specified folder
#     for filename in os.listdir(folder_path):
#         # Check if the file is a .krn file
#         if filename.endswith('.krn'):
#             # Construct the full path to the file
#             file_path = os.path.join(folder_path, filename)
#
#             # Open and read the content of the .krn file
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 content = file.read()
#             # Extract the **kern spine from the content
#             kern_spine = extract_kern_spine(content)
#
#             # Store the result associated with the filename
#             kern_spines[filename] = kern_spine
#
#     # Return the dictionary containing all the results
#     return kern_spines
#
# def process_and_write_kern_sequences(folder_path, output_file_path):
#     results = extract_kern_spine_from_files(folder_path)
#
#     with open(output_file_path, 'w', encoding='utf-8') as output_file:
#         for file, spine in results.items():
#             sequences = extract_kern_sequences(spine)
#
#             for i, seq in enumerate(sequences, start=1):
#                 # Convert the sequence list to a string representation
#                 seq_str = ' '.join(seq)
#                 # Write the file name, sequence number, and sequence to the output file
#                 output_file.write(f"{seq_str}\n")
#
#
# # Set the folder path containing your .krn files
# folder_path = 'cocopops/krn'
#
# # Set the path for the output file where the sequences will be written
# output_file_path = 'cocopops.txt'
#
# # Call the function to process the .krn files and write the sequences to the output file
# process_and_write_kern_sequences(folder_path, output_file_path)


#
# import os
# from collections import defaultdict
# import pandas as pd
# # Define the path to the directory containing the CSV files
# directory_path = 'cocopops/original'
#
# # Initialize an empty list to store DataFrames data and columns
# dataframes_data = []
# columns_set = set()
#
# # Function to rename duplicate columns
# def rename_duplicates(columns):
#     counts = defaultdict(int)
#     for i, col in enumerate(columns):
#         counts[col] += 1
#         if counts[col] > 1:
#             columns[i] = f"{col}_{counts[col] - 1}"
#     return columns
#
# # Read each file
# for filename in os.listdir(directory_path):
#     # if filename.endswith('.csv'):
#     file_path = os.path.join(directory_path, filename)
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#         data = []
#         for i, line in enumerate(lines):
#             # Skip comment lines
#             if line.startswith('!'):
#                 continue
#
#             # Split line by tabs, strip to remove newline characters
#             row = line.strip().split('\t')
#
#             if i == 0:  # Header row
#                 # Rename duplicate columns in header
#                 row = rename_duplicates(row)
#                 columns_set.update(row)
#             data.append(row)
#
#         dataframes_data.append(data)
# # Combine all data into a final structure and create a DataFrame
# final_data = []
# for data in dataframes_data:
#     for row in data:
#         final_data.append(row)
#
# # Ensure all columns are present in each row
# final_columns = list(columns_set)
# final_rows = []
# for row in final_data:
#     row_dict = {col: '' for col in final_columns}
#     for i, value in enumerate(row):
#         if i < len(final_columns):
#             row_dict[final_columns[i]] = value
#     final_rows.append(row_dict)
#
# # Convert final structure to DataFrame
# final_df = pd.DataFrame(final_rows, columns=final_columns)
#
# # Display the combined DataFrame (optional)
# # print(final_df)
#
# def meets_conditions(value):
#     if not value:
#         return False
#     if not isinstance(value, str):
#         return False
#     # Always include lines starting with '*>'
#     if value.startswith('*>'):
#         return True
#     # Apply existing conditions for other lines
#     return not (value.startswith('*') or value.startswith('=') or value.startswith('.') or value.startswith('1r'))
#
# # Filter the DataFrame based on conditions in the '**harte' column
# filtered_df = final_df[final_df['**harte'].apply(meets_conditions)]
#
# # Display the filtered DataFrame (optional)
# print(filtered_df)
#
#
# # To save the final DataFrame to a new CSV file
# filtered_df.to_csv('cocopops.csv', index=False)


import pandas as pd
import glob


def process_header(header_line):
    """Ensure unique column names by appending an index to duplicates."""
    header = header_line.strip().split('\t')
    unique_header = []
    col_counts = {}
    for col in header:
        new_col = col
        while new_col in unique_header:
            col_counts[col] = col_counts.get(col, 1) + 1
            new_col = f"{col}_{col_counts[col]}"
        unique_header.append(new_col)
    return unique_header


def read_custom_csv(file_path):
    """Read and process a custom CSV file, returning a DataFrame."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header = process_header(lines[0])
    data = [line.strip().split('\t') for line in lines[1:] if line.strip() and len(line.strip().split('\t')) == len(header)]

    # Creating DataFrame with refined column names and filtered rows
    return pd.DataFrame(data, columns=header)


# Globbing to find CSV files
file_paths = glob.glob('./cocopops/original/*')  # Adjust path as needed
dataframes = [read_custom_csv(file_path) for file_path in file_paths]

# Merging DataFrames with alignment on identical columns
final_df = pd.concat(dataframes, ignore_index=True)

# Final DataFrame is now ready
print(final_df.info())
print(final_df)



def meets_conditions(value):
    if not value:
        return False
    if not isinstance(value, str):
        return False
    # Always include lines starting with '*>'
    if value.startswith('*>'):
        return True
    # Apply existing conditions for other lines
    return not (value.startswith('*') or value.startswith('=') or value.startswith('.') or value.startswith('1r'))

# Filter the DataFrame based on conditions in the '**harte' column
filtered_df = final_df[final_df['**harte'].apply(meets_conditions)]

filtered_df = filtered_df[filtered_df['**kern'].apply(meets_conditions)]
# Display the filtered DataFrame (optional)
print(filtered_df)


# To save the final DataFrame to a new CSV file
filtered_df.to_csv('cocopops.csv', index=False)
