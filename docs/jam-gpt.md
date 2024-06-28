# Jam-gpt

documentation for using jam-gpt library

## 1 Setup :

### Installization

#### insatll from pip relese

```bash
pip insatll jam-gpt
```

#### install it in your local site-packages directory

```bash
pip install git+https://github.com/Lokeshwaran-M/jam-gpt.git
```

### Modified installization

To add your modification and install in your local site-packages directory

```bash
# clone in your project directory
git clone https://github.com/Lokeshwaran-M/jam-gpt.git
cd jam-gpt

# run pip to move inside your local site-packages directory
pip install .

#  to do modification editable mode
pip install -e .
```

## 2 Collecting Data :

```python
from jam_gpt import Data

# data collection
data=Data.get("path-to-textfile")
```

just to get data from a text data file and return as one large single string for furthere pre processing

## 3 Tokenization :

```python
from jam_gpt import Tokenizer

tok = Tokenizer()

model_name = "md-test"
# tokanization
tok.set_encoding(model_name, data)
tok.get_encoding(model_name)

vocab_size = tok.n_vocab

enc = tok.encode("test sample $^&~~data")
dec = tok.decode(enc)

# out :
# [81, 66, 80, 81, 1, 80, 62, 74, 77, 73, 66, 1, 4, 60, 6, 90, 90, 65, 62, 81, 62]
# test sample $^&~~data

#to store a pretrained model vocab to finetuned model or other model 
tok.store_vocab("source_md-name","md-name")
```

```python
import tiktoken

# tokanization using tiktoken
tok = tiktoken.get_encoding("gpt2")

vocab_size = 50257

enc = tok.encode("test sample $^&~~data")
dec = tok.decode(enc)

# out :
# [9288, 6291, 720, 61, 5, 4907, 7890]
# test sample $^&~~data
```

A tokenizer is a tool that breaks down text into smaller units called tokens These tokens can then be processed by an LLM. The tokens can be words, characters, subwords, or other segments of text, depending on the type of LLM and the desired granularity of the text representation.

## 4 configuration :

```python
from jam_gpt import config

# customizing parameter settings before initializing model 

args = config.pass_args()
config.vocab_size = 50257

print(args)
print(config.pass_args())

# out :
# [0, 16, 32, 5000, 100, 0.001, 'cuda', 200, 64, 4, 4, 0.0]
# [50257, 16, 32, 5000, 100, 0.001, 'cuda', 200, 64, 4, 4, 0.0]

# To store the customized config setting into model/config.json
config.store(model_name,args)
# To retrive the config settings from model/config.json
config.retrive(model_name)
```
The config custamization need to be done before initializing the model 

## 5 Language Model ( LM , Model ) :

### Initilizing model

```python
from jam_gpt import  lm
from jam_gpt import Model

# model instantiation
model = Model()

# setting model architecture

# GPT Language Model
model.set_model(lm.GPTLM())
```

### Traning

```python
# prepare data for training ( train , test )
model.set_data(Data.train_test_split(enc_data))

# traning
model.optimize()
model.train()
```

### Saving model

```python
# default bin
model.save_model(model_name)
# can edit model_format
# model_format  = pt or pkl
model.save_model(model_name,model_format)
```

### load model

```python
# retrive model parameter settings
config.retrive(md_name)

# model instantiation
model = Model()

model.load_model(model_name)
```

### Generate data using Model

```python
pmt = tok.encode("user prompt")
eos = tok.encode(" [eos] ")
res = tok.decode(model.generate(pmt,3000,eos))

print(res)
```

## 6 Model Fine Tuning :

```python

                Coming Soon ........

```
