# qgan_subspace

This code has been working on based on the repository "qWGAN" of yiminghwang. 

## License

Distributed under the MIT License. See LICENSE for more information.

## Usage
This section covers how to set up a local development environment for qgan_subspace and run the tests.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ayaka-usui/qgan_subspace.git
   cd qgan_subspace
   ```

2. **Sync dependencies**:
   - We maintain a list with all the needed dependencies in `requirements.txt`.
   - To create a local environment using `venv`, and install the necessary dependencies, run:
    ```
    python3 -m venv .venv
    source .venv/bin/activate 
    pip install -r requirements.txt   
    ```
    *(Exact command can vary depending on your shell and OS.)*

Now you can run all necessary commands (run, tests, etc.) within this environment.

3. **Running**:
    - The file to execute is `main.py`, and the only file you need to edit, for changing experiments is `config.py`.
    - For an execution, after the `config.py` has been set, run:
    ```
    .venv/bin/python src/main.py
    ```

4. **Testing**:
    If you need, to test anything (development), there are 2 ways:
    - Set the flag `testing=True` in `config.py`, and add any case you want to run in the end of the file, to the `test_configurations` dictionary.
    - There will also be unit test for parts of the code, to execute them, run:
    ```
    PYTHONPATH=src .venv/bin/pytest
    ```