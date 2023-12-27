

## to publish next release in pip jam-gpt-0.0.5

### run :

```
python setup.py sdist 
```
>include requirements.txt inside the ziped 
./dist/jam-gpt-0.0.5.tar.gz/jam-gpt-0.0.5
./dist/jam-gpt-0.0.5.tar.gz/jam-gpt-0.0.5/jam_gpt.egg-info/SOURCES.txt

### then run

```
twine upload dist/jam-gpt-0.0.5.tar.gz
```