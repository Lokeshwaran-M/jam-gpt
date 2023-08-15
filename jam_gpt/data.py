

def get(path: str) -> str:
    """
    just to get data from a text data file and
    return as one large single string for furthere pre processing 
    """
    with open(path, "r", encoding="utf-8") as f:
        text_data = f.read()
    # print("len of text data :" , len(text_data))
    # print(text_data[:10000])
    return text_data
