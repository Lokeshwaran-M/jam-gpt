class Data:
    def __init__(self) -> None:
        pass

    @classmethod
    def get(cls, path: str) -> str:
        """
        text data file -> string data
        
        """
        with open(path, "r", encoding="utf-8") as f:
            text_data = f.read()
        return text_data

    @classmethod
    def set(cls, path: str, data: str) -> None:
        """
        string data -> text data file
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
            print("writen data : ",len(data))



       




# def test(path):
#     Data.set(path,"ghsdubvujsjbnjsnjbnsjnbljnlbnls")
#     d= Data.get(path)
#     print(d)
#     Data.set_vocab(path,d)
#     l= Data.get_vocab(path)
#     print(l)
# test("data.txt")
   




