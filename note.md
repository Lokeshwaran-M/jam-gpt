

## to publish next release in pip jam-gpt-0.0.4

### run :

```
python setup.py sdist 
```
>include requirements.txt inside the ziped dist and also in SOURCES.txt

### then run

```
twine upload dist/jam-gpt-0.0.4.tar.gz
```