

## to publish

### run :

```
python setup.py sdist 

```
include requirements.txt inside the ziped dist and also in SOURCES.txt

then run

```
twine upload dist/jam-gpt-0.0.4.tar.gz
```