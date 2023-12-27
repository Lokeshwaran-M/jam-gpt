```py
eos_token = tok.encode(""" [eos]""")

eos = tok.encode(" [eos]")
print(tok.decode(model.generate(pmt,max_new_tokens=3000,eos_token=eos)))
```

