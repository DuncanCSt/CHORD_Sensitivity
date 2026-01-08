# CHORD Sensitivity

A Python package to compute sensitivity of CHORD to Radio Recombination Lines.

## Installation

### From PyPI (once published)

```bash
pip install chord-sensitivity
```

### From source

```bash
git clone https://github.com/DuncanCSt/CHORD_Sensitivity.git
cd CHORD_Sensitivity
pip install -e .
```

### For development

```bash
git clone https://github.com/DuncanCSt/CHORD_Sensitivity.git
cd CHORD_Sensitivity
pip install -e ".[dev]"
```

## Usage

```python
import chord_sensitivity

# Your code here
print(chord_sensitivity.__version__)
```

## Development

### Building the package

```bash
python -m build
```

### Running tests

```bash
pytest
```

### Code formatting

```bash
black chord_sensitivity/
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
