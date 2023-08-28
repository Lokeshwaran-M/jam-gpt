# Jam-gpt

An Experimental Reimplementation of [large language model (LLM)](https://en.wikipedia.org/wiki/Large_language_model#:~:text=References-,Large%20language%20model,-19%20languages) for research and development of it architectures and design process and create a simple and fast framework for training and fine-tuning medium-sized [Generative Pretrained Transformers (GPT)](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer#:~:text=Generative%20pre%2Dtrained,20%20languages) based on the paper ["Attention Is All You Need"](./docs/1706.03762.pdf)



<a  href="https://github.com/Lokeshwaran-M/Jam-AI.git">
<p align="center">
<img src="https://user-images.githubusercontent.com/80915494/263127835-0509942a-0528-4471-96fa-8eda3d4f159c.jpeg" width="50%" height="50%" >
<!-- <p align="center"><a  href="https://github.com/Lokeshwaran-M/Jam-AI.git">Jam-AI</a></p> -->

</p>
</a>

Jam-gpt is also utilized in building Core Knowledge Model of [Jam-AI](https://github.com/Lokeshwaran-M/Jam-AI.git) an open source AGI system for Linux machines

## Installation :

use pip and install it in your local site-packages directory

```bash
pip install git+https://github.com/Lokeshwaran-M/jam-gpt.git
```

## Usage :

Refere [test.py](test.py) or [test-bgLM.ipynb](test-bgLM.ipynb) for understanding the library

```python

from jam_gpt import Data, Tokenizer, Config, LM, Model

tok = Tokenizer()


model_name = "test-m"
path = "data.txt"

# data collection
data = Data.get(path)

# tokanization
tok.set_encoding(model_name, data)
tok.get_encoding(model_name)

# setting parameters
args = Config.pass_args()
args[0] = tok.n_vocab
print(tok.n_vocab)

# model genration
test_model = Model()
lm = LM()
test_model.set_parameters(args)
lm.set_parameters(args)

# if needed GPT Language Model
test_model.set_model(GPTLanguageModel())

# prepare data for training ( train , test )
test_model.set_data(Data.train_test_split(enc_data))

# traning
test_model.optimize()
test_model.train()

#Saving model 
# default bin
test_model.save_model(model_name)

# load model 
test_model.load_model(model_name)

# Generate data using Model

pmt = tok.encode("user prompt")
print(tok.decode(test_model.generate(pmt)))

```

## DOCS :

[Jam-gpt docs](./docs/jam-gpt.md) will give you the useage and explanation of the jam-gpt library

1 [ setup](./docs/jam-gpt.md#1-setup)  
2 [ Collecting data](./docs/jam-gpt.md#2-collecting-data)  
3 [ Tokenization](./docs/jam-gpt.md#3-tokenization)  
4 [ configuration](./docs/jam-gpt.md#4-configuration)  
5 [ Language Model ( LM , Model )](./docs/jam-gpt.md#5-language-model--lm--model)  
6 [ Model Fine Tuning](./docs/jam-gpt.md#6-model-fine-tuning)

## Contribution :
for contribution guidelines and terms and condition to contribute refere [jam-contribution](https://github.com/Lokeshwaran-M/jam-contribution.git) by rasing the PR you are accepting the terms and condition

Any form of contribution is accepted here 

    Submitting :    
        Issues  
        pull requests   
        feature requests    
        bug reports  
        documentation   



## credits :

kudos to [Andrej karpathy](https://github.com/karpathy) for his lectures on deep learning and [Open AI](https://github.com/openai) for GPT-2
