# BlackFoxPython documentation

Black Fox python client package documentation source. It can be built by using Sphinx document generator, which must be installed in the current Python virtualenv using [pip](https://pip.pypa.io/en/stable/quickstart/) or [pipenv](https://docs.pipenv.org/en/latest/).

```console
 $ pip install -U sphinx
```

Since both win and linux makefiles are available, docs can be build with
```console
 $ make html
```
which will build the documentation in a local _build folder and automatically copy html to docs where it is published.