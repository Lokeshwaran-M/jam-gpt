# Jam-gpt

An Experimental Reimplementation and reinnovation of [large language model (LLM)](https://en.wikipedia.org/wiki/Large_language_model#:~:text=References-,Large%20language%20model,-19%20languages) for research and development of it architectures, design process, training, and fine-tuning the [Generative Pretrained Transformers (GPT)](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer#:~:text=Generative%20pre%2Dtrained,20%20languages) model based on the paper ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf)

<a  href="https://github.com/jam-agi">
<div align="center" style="box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);">
<img src="https://user-images.githubusercontent.com/80915494/265230650-6b3100d4-305e-414a-b7b5-cb2399fc197a.png" width="50%" height="50%" >
</div>
</a>

Jam-gpt is also utilized in building Core Knowledge Model of **[Jam-AGI](https://github.com/jam-agi)** an open source AGI system for Linux machines

## Installation :

### unstable release
> jam-gpt==0.0.3 may not have fine tuning as its still under development and may contain bug please report issues if any 

```bash
pip install jam-gpt
```

### latest version
> github pull will be clean if encountered with bugs please report issues
```bash
pip install git+https://github.com/Lokeshwaran-M/jam-gpt.git
```

## Usage :

Refere [Docs](./docs/jam-gpt.md) or [test-bgLM.ipynb](test-bgLM.ipynb) and [test-gptLM.ipynb](test-gptLM.ipynb) for code examples

```python

from jam_gpt.tokenizer import Tokenizer
from jam_gpt import config
from jam_gpt import lm
from jam_gpt.model import Model

md_name = "md-name"

tok = Tokenizer()
tok.get_encoding(md_name)

# model initilization
model = Model()

# load pretrined model
model.load_model(md_name)

# Generate data using Model
pmt = tok.encode("user prompt")
res = tok.decode(model.generate(pmt))
print(res)

```

## Docs :

[Jam-gpt docs](./docs/jam-gpt.md) will give you the complete useage and explanation of the jam-gpt library

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

