# Tutorial
## Python
### Virtual environments (virtualenvs)
keep project dependencies separated, and help to avoid version conflicts between packages and different versions of the Python runtime.
Before creating & activating a virtualenv `python` and `pip` map to the system version of the Python interpreter (e.g. Python 2.7)
```shell
$ which python
/usr/local/bin/python
```
To create a fresh virtualenv using another version of Python (e.g. Python 3)
```shell
$ python3 -m venv ./venv
```
A virtualenv is just a Python environment in a folder
```shell
$ ls ./venv
bin      include    lib      pyvenv.cfg
```
Activating a virtualenv configures the current shell session to use the python (and pip) commands from the virtualenv folder instead of the global environment
```shell
$ source ./venv/bin/activate
```
Note activating a virtualenv modifies shell prompt with a little note showing the name of the virtualenv folder
```shell
(venv) $ echo "wee!"
```
With an active virtualenv, the `python` command maps to the interpreter binary *inside the active virtualenv*
```shell
(venv) $ which python
/Users/dan/my-project/venv/bin/python3
```
Installing new libraries and frameworks with `pip` now installs them *into the virtualenv sandbox*, leaving global environment (and any other virtualenvs) completely unmodified
```shell
(venv) $ pip install requests
```
To get back to the global Python environment, run the following command
```shell
(venv) $ deactivate
```
