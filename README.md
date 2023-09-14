# MIDI Diffusion Model

Diffusion model for generating MIDI sequences.

### Model Training

```sh
python train.py --config-name config-default
```

```mermaid
flowchart TD
    A[MIDI Sequence] --> B(frozen conditioning model)
    A --> C(velocity)
    C --> D(velocity diffusion)
    B --> E(encoded)
    E --> F(noise prediction)
    D --> F 
```

### Sample Generation

```mermaid
flowchart TD
    A[MIDI Sequence] --> B(quantized piece)
    B --> E(encoded)
    C(conditional_embedding) --> F
    E --> F[generated velocity]
```

### Code Style

This repository uses pre-commit hooks with forced python formatting ([black](https://github.com/psf/black),
[flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/)):

```sh
pip install pre-commit
pre-commit install
```

Whenever you execute `git commit` the files altered / added within the commit will be checked and corrected.
`black` and `isort` can modify files locally - if that happens you have to `git add` them again.
You might also be prompted to introduce some fixes manually.

To run the hooks against all files without running `git commit`:

```sh
pre-commit run --all-files
```
