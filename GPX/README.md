# GPX

The GPX module contains all the proprietary machinery we don't want to expose to the modeling engine.

## Ideas

- All production system mathematics
- Proprietary process models
- Cost-down curves
- Uncertainty and Risk characterization

## Packaging to compressed folder for use in the docker build

`tar -czvf gpx-zipped.tar.gz .`

## Code Formatting, Linting, Type Checking and Processing LLX Models and Results

### Installation

#### Installing NPM (Dev Only)

```bash
  sudo apt install npm
```

#### Installing node_packages - from root of Engine repo (Dev Only)

```bash
  npm i
```

### Running the Tasks

#### Run isort

```bash
  paver isort
```

#### Run yapf

```bash
  paver yapf
```

#### Run prettier

```bash
  paver prettier
```

#### Run flake8

```bash
  paver flake8
```

#### Run mypy

```bash
  paver mypy
```

#### Run all linting tasks

```bash
  paver lint
```

#### Run flake8 when issues fixed

```bash
  paver flake8_save
```

#### Run mypy when issues fixed

```bash
  paver mypy_save
```

#### Run all linting when issues fixed

```bash
  paver lint_save
```
