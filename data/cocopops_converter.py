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
