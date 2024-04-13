import pandas as pd
import glob
import re

def get_duration_note(input_string):
    input_string = input_string.split()[0].strip(']').strip('[').strip('.').strip(';')
    match = re.match(r'^(\d+)', input_string)
    if match:
        number_part = int(match.group(1))
        text_part = input_string[len(match.group(1)):]
    else:
        number_part = 1
        text_part = input_string
    return number_part, text_part

def process_header(header_line):
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
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header_index = 0
    for i, line in enumerate(lines):
        if line.startswith('!'):
            continue
        else:
            header_index = i
            break
    header = process_header(lines[header_index])
    data = [line.strip().split('\t') for line in lines[header_index+1:] if line.strip() and len(line.strip().split('\t')) == len(header)]
    df = pd.DataFrame(data, columns=header)

    # Splitting **kern into **duration and **kern

    if '**kern' in df.columns:
        df[['**duration', '**kern']] = df['**kern'].apply(lambda x: pd.Series(get_duration_note(x)))
    return df

file_paths = glob.glob('./cocopops/original/*')
dataframes = [read_custom_csv(file_path) for file_path in file_paths]

file_paths = glob.glob('./cocopops/rollingstones/*')
dataframes_2 = [read_custom_csv(file_path) for file_path in file_paths]
dataframes.extend(dataframes_2)

final_df = pd.concat(dataframes, ignore_index=True)

def meets_conditions(value):
    if not value:
        return False
    if not isinstance(value, str):
        return False
    if value.startswith('*>'):
        return True
    return not (value.startswith('*') or value.startswith('=') or value.startswith('.') or value.startswith('1r') or value.startswith('!'))

filtered_df = final_df[final_df['**harte'].apply(meets_conditions)]
filtered_df = filtered_df[filtered_df['**kern'].apply(meets_conditions)]

# Display the filtered DataFrame (optional)
print(filtered_df.info())
print(filtered_df)

# To save the final DataFrame to a new CSV file
filtered_df.to_csv('cocopops_all.csv', index=False)
