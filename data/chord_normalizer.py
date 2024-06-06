import re


def simplify_chord_extensions(chord):
    # Define regex patterns for basic chord types and alterations
    chord_patterns = [
        r'(maj|min|dim|aug|sus|maj|min|7|9|11|13|b5|#5|b9|#9|b13|#11)[^/]*'  # Capture basic chord types and ignore alterations/extensions after them
    ]

    # Search for the first matching pattern and simplify the chord
    for pattern in chord_patterns:
        match = re.search(pattern, chord)
        if match:
            # Return the chord up to the end of the matched pattern
            return chord[:match.end()]
    return chord  # Return the original chord if no pattern matches


def normalize_inversions(chord):
    # Remove inversion and complex alterations by splitting at '/'
    return chord.split('/')[0]


def remove_last_char_if_number(input_string):
    if 'min' in input_string or 'maj' in input_string:
        while input_string and input_string[-1].isdigit():
            input_string = input_string[:-1]
    return input_string


def normalize_chord(chord):
    chord = simplify_chord_extensions(chord)
    chord = normalize_inversions(chord)
    chord = remove_last_char_if_number(chord)
    return chord


def normalize_chord_sequence(sequence):
    normalized_chord_sequence = []
    for chord in sequence:
        normalized_chord_sequence.append(normalize_chord(chord))
    return normalized_chord_sequence
