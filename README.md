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

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate 
    pip install -r requirements.txt   
    ```

    *(Exact command can vary depending on your shell and OS.)*

    Now you can run all necessary commands (run, tests, etc.) within this environment.

3. **Running**:

    - The file to execute is `main.py`, and the only file you need to edit, for changing experiments is `config.py`.
    - For an execution, after the `config.py` has been set, run:

    ```bash
    .venv/bin/python src/main.py
    ```

4. **Testing**:

    If you need to test anything (for development), we use pytest:
    - There are unit test for diverse parts of the code,
    - And also several short integration tests, whose configuration you can set at the end of `config.py` in the `test_configurations` dictionary. These integration test will output their logs, figures, etc at the `generated_data/TESTING` directory (some graphs overwrite each other).

    To execute them, run:

    ```bash
    PYTHONPATH=src .venv/bin/pytest
    ```

    Also if you want VSCode to detect the imports from the testing module, add:

    ```json
    "python.analysis.extraPaths": [
        "./src"
    ],
    ```

    to your `Preferences: Open User Settings (JSON)`.
