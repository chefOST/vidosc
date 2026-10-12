class Vocab:
    def __init__(self, tokens):
        self._itos = list(tokens)
        self._stoi = {t: i for i, t in enumerate(self._itos)}

    def get_itos(self):
        return list(self._itos)

    def get_stoi(self):
        return dict(self._stoi)

    def __len__(self):
        return len(self._itos)

    def __getitem__(self, token):
        return self._stoi[token]


def vocab(ordered_dict, min_freq=1, specials=None, special_first=True):
    # Same semantics as torchtext 0.13: keep dict insertion order (no sorting).
    specials = specials or []
    tokens = [t for t, f in ordered_dict.items() if f >= min_freq and t not in specials]
    tokens = (specials + tokens) if special_first else (tokens + specials)
    return Vocab(tokens)
