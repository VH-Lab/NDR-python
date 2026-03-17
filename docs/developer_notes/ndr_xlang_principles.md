# NDR Cross-Language (MATLAB/Python) Principles

- **Status:** Active
- **Scope:** Universal (Applies to all NDR implementations)

## 1. Indexing & Counting (The Semantic Parity Rule)
- Python uses 0-indexing internally.
- User-facing concepts (Epochs, Channels) use 1-based numbering in both languages.
- Python code accepts `channel_number=1` from user, maps to `data[0]` internally.

## 2. Data Containers
- Prefer NumPy over lists for numerical data.
- MATLAB `double` array -> `numpy.ndarray` in Python.

## 3. Multiple Returns
- Python returns multiple values as a tuple in MATLAB signature order.

## 4. Booleans
- MATLAB `1`/`0` (logical) -> Python `True`/`False`.

## 5. Strings
- MATLAB `char` and `string` -> Python `str`.
- MATLAB cell array of strings -> Python `list[str]`.

## 6. Error Philosophy
- No silent failures. If MATLAB errors, Python raises an exception.
