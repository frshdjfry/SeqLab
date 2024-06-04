import re

chord_to_semitone = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4, 'F': 5,
    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11
}

chord_to_semitone_sharps = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7,
    'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

chord_to_semitone_flats = {
    'C': 0, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'Gb': 6, 'G': 7,
    'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11
}

semitone_to_chord_sharps = {v: k for k, v in chord_to_semitone_sharps.items()}
semitone_to_chord_flats = {v: k for k, v in chord_to_semitone_flats.items()}


def extract_root(chord):
    match = re.match(r'^[A-G][b#]?', chord)
    if match:
        return match.group(0)
    return None


def transpose_chord(chord, steps, use_flats=False):
    root = extract_root(chord)
    if root:
        new_root_semitone = (chord_to_semitone[root] + steps) % 12
        if use_flats:
            new_root = semitone_to_chord_flats[new_root_semitone]
        else:
            new_root = semitone_to_chord_sharps[new_root_semitone]
        return chord.replace(root, new_root, 1)
    return chord


def transpose_progression(progression, steps, use_flats=False):
    transposed_chords = [transpose_chord(chord, steps, use_flats) for chord in progression]
    return transposed_chords


def get_chord_seq_in_different_keys(sequence):
    augmented_sequences = []
    for steps_to_transpose in range(12):
        augmented_sequences.append(transpose_progression(sequence, steps_to_transpose))
        augmented_sequences.append(transpose_progression(sequence, steps_to_transpose, use_flats=True))
    return augmented_sequences
