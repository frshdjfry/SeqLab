import os


def extract_kern_sequences(kern_spine):
    # Initialize an empty list to hold the sequences
    sequences = []

    # Temporary list to hold the current sequence of **kern values
    current_sequence = []

    # Iterate through each **kern value in the spine
    for value in kern_spine:
        # Check if the line is empty or consists only of whitespace characters
        if value.strip() == '':
            continue  # Skip empty lines

        # Check if the line starts with '*>', marking the beginning of a new sequence
        if value.startswith('*>'):
            # If the current sequence is not empty, it means we've reached the end of a sequence
            if current_sequence:
                sequences.append(current_sequence)  # Add the completed sequence to the list
                current_sequence = []  # Reset the current sequence for the next one
            continue  # Move on to the next line

        # If the line is not a sequence marker or empty, add it to the current sequence
        if (not value.startswith('*') and not value.startswith('=') and not value.startswith('.')
                and not value.startswith('1r')):
            clean_value = value.replace('.', '').replace(';', '').replace(' ', '_').upper()
            current_sequence.append(clean_value)

    # After processing all lines, check if there's a sequence that hasn't been added yet
    if current_sequence:
        sequences.append(current_sequence)  # Add the last sequence to the list

    return sequences


def extract_kern_spine(dataset):
    # print(dataset)
    # Split the dataset into lines
    lines = dataset.split("\n")

    # Initialize an empty list to hold the **kern values
    kern_values = []

    # Locate the index of the **kern column in the first (or header) line
    # Assuming the first line contains the column identifiers
    spines_index = 0
    for i, v in enumerate(lines):
        if '**kern' in v:
            spines_index = i
    column_identifiers = lines[spines_index].split("\t")  # Adjust split character if necessary (e.g., space or tab)
    kern_index = column_identifiers.index("**kern")

    # Iterate over each line in the dataset
    for line in lines:
        # Split the line into columns (adjust split character if necessary)
        columns = line.split("\t")

        # Ensure the line has enough columns and extract the **kern value
        if len(columns) > kern_index:
            kern_values.append(columns[kern_index])

    # Return the list of **kern values
    return kern_values


def extract_kern_spine_from_files(folder_path):
    # Ensure the `extract_kern_spine` function is defined here or imported if it's defined elsewhere

    # Initialize a dictionary to hold the results: file names and their extracted **kern spines
    kern_spines = {}

    # Iterate through each file in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .krn file
        if filename.endswith('.krn'):
            # Construct the full path to the file
            file_path = os.path.join(folder_path, filename)

            # Open and read the content of the .krn file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            # Extract the **kern spine from the content
            kern_spine = extract_kern_spine(content)

            # Store the result associated with the filename
            kern_spines[filename] = kern_spine

    # Return the dictionary containing all the results
    return kern_spines

# Replace 'path/to/krn/files' with the actual folder path containing your .krn files
# folder = 'cocopops/krn'
# results = extract_kern_spine_from_files(folder)
#
# # Output the results
# for file, spine in results.items():
#     print(f"File: {file}, **kern spine: {spine}")
#     # Call the function with the example **kern spine data
#     sequences = extract_kern_sequences(spine)
#
#     # Output the extracted sequences
#     for i, seq in enumerate(sequences, start=1):
#         print(f"{file} Sequence {i}: {seq}")
def process_and_write_kern_sequences(folder_path, output_file_path):
    results = extract_kern_spine_from_files(folder_path)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for file, spine in results.items():
            sequences = extract_kern_sequences(spine)

            for i, seq in enumerate(sequences, start=1):
                # Convert the sequence list to a string representation
                seq_str = ' '.join(seq)
                # Write the file name, sequence number, and sequence to the output file
                output_file.write(f"{seq_str}\n")


# Set the folder path containing your .krn files
folder_path = 'cocopops/krn'

# Set the path for the output file where the sequences will be written
output_file_path = 'cocopops.txt'

# Call the function to process the .krn files and write the sequences to the output file
process_and_write_kern_sequences(folder_path, output_file_path)
