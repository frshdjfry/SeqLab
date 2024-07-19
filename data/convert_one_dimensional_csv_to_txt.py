from many_to_many_data_preprocessing import read_csv, extract_sequences

if __name__ == '__main__':
    df = read_csv('rolling_stones.csv')
    sequences = extract_sequences(df,['**harte'], '**harte')
    lines = sequences['**harte']
    with open('rolling_stones.txt', 'w') as f:
        for line in lines:
            f.write(' '.join(line) + '\n')
        f.close()