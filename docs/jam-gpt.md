
# Jam-gpt

documentation for using jam-gpt library

# 1 collecting data

```python
from jam_gpt import data

text_data=data.get("path-to-textfile")
```
just to get data from a text data file and return as one large single string for furthere pre processing 

# 2 Tokenization

```python
from jam_gpt import Tokenizer

tok = Tokenizer()
tok.get_encoding(text_data)

enc = tok.encode("test sample $^&~~data")
# [81, 66, 80, 81, 1, 80, 62, 74, 77, 73, 66, 1, 4, 60, 6, 90, 90, 65, 62, 81, 62]
dec = tok.decode(enc)
# test sample $^&~~data
```
A tokenizer is a tool that breaks down text into smaller units called tokens These tokens can then be processed by an LLM. The tokens can be words, characters, subwords, or other segments of text, depending on the type of LLM and the desired granularity of the text representation.
  

