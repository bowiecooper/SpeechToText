def create_char_mappings():
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
    char2idx = {char: idx+1 for idx, char in enumerate(chars)}
    char2idx[''] = 0  # blank token for CTC
    idx2char = {v: k for k, v in char2idx.items()}
    return char2idx, idx2char