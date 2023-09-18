
# Jam-gpt

documentation for using jam-gpt library

## 1 Setup :

### Installization 

install it in your local site-packages directory

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

# customizing parameter settings

args = config.pass_args()
config.vocab_size = 96

print(args)
print(config.pass_args())

# out :
# [0, 16, 32, 5000, 100, 0.001, 'cuda', 200, 64, 4, 4, 0.0]
# [96, 16, 32, 5000, 100, 0.001, 'cuda', 200, 64, 4, 4, 0.0]

```
## 5 Language Model ( LM , Model )  :

### Initilizing model

```python
from jam_gpt import  lm
from jam_gpt import Model

# model genration
test_model = Model()

# if needed Bigram Language Model
test_model.set_model(lm.BigramLM())
# elif needed GPT Language Model
test_model.set_model(lm.GPTLM())
```
### Traning 
```python
# prepare data for training ( train , test )
test_model.set_data(Data.train_test_split(enc_data))

# traning
test_model.optimize()
test_model.train()
```
### Saving model 
```python
# default bin
test_model.save_model(model_name)
# can edit model_format 
# model_format  = pt or pkl
test_model.save_model(model_name,model_format )
```
### load model 
```python
test_model.load_model(model_name)
```
### Generate data using Model
```python
pmt = tok.encode("user prompt")
print(tok.decode(test_model.generate(pmt)))
```
## 6 Model Fine Tuning :

```python

                Coming Soon ........

```