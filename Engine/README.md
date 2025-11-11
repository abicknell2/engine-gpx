# Copernicus-Engine

Python 3 engine for the Copernicus modeling environment

## Ideas

[ ] SysML representation of the model

## Flows

### Creating and solving a model

- JSON is uploaded to one shot
- one-shot creates an interactive session
  - one-shot creates all of the elements in the interactive session
  - one-shot solves the interactive session and returns data to APIs

#### InteractiveModel

- interactive creates all of the objects from the front end with the serialization representation
- these interactive objects are then interpreted and translated into GPx objects

#### Traditional tasks

- parameters → GPx variables
- `copernicus.InteractiveModel` → `gpxModel`

## Questions

- Where are the variables tracked?

## Installation Instructions

### Initial Setup – Python 3.13

#### 1. Install Python 3.13

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-venv python3.13-dev
```

#### 2. Create a virtual environment

```bash
python3.13 -m venv path/to/venv
```

#### 3. Activate the virtual environment

```bash
source path/to/venv/bin/activate
```

#### 4. Install project dependencies

The project uses modular requirements files:

- `prod-requirements.in` – Core packages required for running the engine
- `dev-requirements.in` – Tooling and development dependencies
- `additional-requirements.txt` – Private or vendored packages (e.g. `ad`)

To install all dependencies:

```bash
pip install -r prod-requirements.txt
pip install -r dev-requirements.txt
pip install -r additional-requirements.txt
```

> When compiling from `.in` files using `pip-compile`, ensure all packages are pinned and include hashes if `--require-hashes` is used.

##### Manually adding a hash for setuptools

If a hash for `setuptools` is not available in the compiled requirements file, it can be generated using the following commands:

```bash
pip download setuptools==80.8.0
pip hash setuptools-80.8.0-py3-none-any.whl
```

This generates an output such as:

```text
--hash=sha256:<HASH_VALUE>
```

Add the pinned package with hash to the `.in` file:

```text
setuptools==80.8.0 \
    --hash=sha256:<HASH_VALUE>
```

Then regenerate the `.txt` file:

```bash
pip-compile --generate-hashes prod-requirements.in
```

## Running Tests

### Running individually

```bash
python -m pytest ./test/test.py
```

### Create test files from front end `/solve`

- Set the `SAVE_MODEL_DATA_FOR_TESTS` environment variable.
- Change the API call path to `http://localhost:5000/solve?save_model_data_for_tests=true`.
- Submit a model for solving.
- Input and output JSON data will be saved to `test/data/json_data`.

## Code Formatting, Linting, Type Checking and Processing LLX Models and Results

### Setup

#### Installing NPM (Dev Only)

```bash
sudo apt install npm
```

#### Installing `node_modules` (from root of Engine repository)

```bash
npm i
```

### Running the Tasks

Automation tasks are defined in `pavement.py` and executed using Paver:

```bash
paver <task-name> [options]
```

#### Formatting & Import Order

| Task                  | Purpose                                                             |
| --------------------- | ------------------------------------------------------------------- |
| `isort`               | Sort and group imports                                              |
| `add_trailing_commas` | Add trailing commas to all Python files (uses _add-trailing-comma_) |
| `yapf`                | Format code with `yapf` using `.style.yapf`                         |
| `prettier`            | Format `.json` and `.md` files using Prettier                       |

Example:

```bash
paver isort
paver yapf
```

### Static Analysis / Linting

| Task                     | Description                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `pylint` / `pylint_save` | Run **Pylint** and compare or refresh `pylint_results.txt`                          |
| `flake8` / `flake8_save` | Run **Flake8** and compare or refresh `flake8_results.txt`                          |
| `mypy` / `mypy_save`     | Run **mypy** and compare or refresh `mypy_results.txt`                              |
| `bandit` / `bandit_save` | Run **Bandit** for security checks and optionally save baseline                     |
| `process_llx`            | Convert `.llx` files to JSON and split `results`                                    |
| `lint`                   | Full stack: pylint → isort → prettier → flake8 → mypy → yapf → bandit → process_llx |
| `lint_save`              | Same as `lint`, but refreshes all result baselines                                  |

#### Test Coverage

| Task              | Description                               |
| ----------------- | ----------------------------------------- |
| `coverage_clean`  | Erase previous coverage data              |
| `coverage`        | Run `pytest` with branch coverage         |
| `coverage_report` | Output a text summary                     |
| `coverage_html`   | Generate HTML coverage in `htmlcov/`      |
| `coverage_all`    | Run full sequence: clean → test → reports |

Example:

```bash
paver coverage_all
xdg-open htmlcov/index.html
```

#### Test Model Regeneration

- The `test_regen_models.py` script rebuilds all model outputs in a specified directory.
- You can switch the target directory inside the script (e.g., `models`, `rate_ramps`, etc.).
- Newly generated JSON files are written to `test/data/json_data`.
- After generation, copy the JSON files into the corresponding test data folder (`models`, `rate_ramps`, etc.).
- Launch this script manually, via a `.vscode` launch configuration, or through the VS Code Testing sidebar.
- Using the Testing sidebar lets you regenerate a single model’s output instead of the entire folder.

## Deployment via Paver

### Packaging – GPX Library

| Task        | Description                                                                                                             |
| ----------- | ----------------------------------------------------------------------------------------------------------------------- |
| `build_ad`  | builds `ad-<ver>-py3-none-any.whl` → places it in `<engine-root>/ad-<ver>-py3-none-any.whl`                             |
| `build_gpx` | Interactive selection of GPX source folder → builds `gpx-<ver>.tar.gz` → places it in `<engine-root>/gpx-zipped.tar.gz` |

### Docker & Deployment

| Task                  | Description                                                                                                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `build_docker_image`  | Vendors private `ad` package into `vendor/`, generates a temporary `requirements.build.txt` with the vendored wheel, then builds the image `test-solve-engine:<branch>-<hash>` |
| `push_image_to_azure` | Tags and pushes the image to an Azure Container Registry; supports `--registry` and `--login` options                                                                          |
| `deploy`              | Runs the full deployment pipeline: `build_ad` → `build_gpx` → `build_docker_image` → `push_image_to_azure`                                                                     |
| `az_build_image`      | Builds the Docker image **directly inside Azure ACR** using `az acr build`; skips the local build → push step.                                                                 |
| `az_deploy`           | Alternative full pipeline: `build_ad` → `build_gpx` → `az_build_image` → `deploy_container_revision` (no local image build or manual push required).                           |

### Registry Short-cuts

These keys are supported by the `REGISTRIES` mapping in deployment tooling:

| Key      | Registry URL        |
| -------- | ------------------- |
| `lltest` | `lltest.azurecr.io` |
| `stage`  | `stage.azurecr.io`  |
| `prod`   | `prod.azurecr.io`   |

Examples:

```bash
# Interactive registry selection
paver push_image_to_azure

# Specify registry and perform ACR login
paver push_image_to_azure -r lltest --login
```

### Prerequisite – Azure CLI in WSL

Azure CLI is required to log in to ACR. Install with:

```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

Authenticate and login:

```bash
az login --use-device-code
az acr login --name <registryName>
```

## Quick Recipes

| Goal                         | Command                    |
| ---------------------------- | -------------------------- |
| Format & lint everything     | `paver lint`               |
| Update baseline lint results | `paver lint_save`          |
| Build GPX package            | `paver build_gpx`          |
| Build local Docker image     | `paver build_docker_image` |
| Build and push to ACR        | `paver deploy`             |
| Build on ACR                 | `paver az_deploy`          |
