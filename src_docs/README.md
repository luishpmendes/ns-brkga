# NS-BRKGA Documentation Build

This directory contains the documentation source for **NS-BRKGA**, built using
the **Doxygen → Doxyrest → Sphinx** pipeline.

---

## Prerequisites

### Python packages (Sphinx)

```bash
pip install -r requirements.txt
```

### External tools

| Tool | Purpose | Install |
|------|---------|---------|
| `doxygen` | Extract C++ API docs / XML | `sudo apt install doxygen` |
| `doxyrest` | Convert Doxygen XML → RST | See [Doxyrest releases](https://github.com/vovkos/doxyrest/releases) |

### Environment variable

After installing Doxyrest, export its installation root:

```bash
export DOXYREST_DIR=/path/to/doxyrest   # e.g. /usr/local or ~/doxyrest-2.1.3
```

Typical Doxyrest layout:
```
$DOXYREST_DIR/
  bin/doxyrest
  share/doxyrest/frame/cfamily/
  share/doxyrest/frame/common/
  share/doxyrest/sphinx/
```

---

## Building

```bash
cd src_docs
export DOXYREST_DIR=/path/to/doxyrest
make clean && make all
```

The published output will be written to `../docs/` (the `docs/` folder at the
repository root used by GitHub Pages).

### Individual steps

```bash
make doxygen    # Run Doxygen → build/xml/
make doxyrest   # Run Doxyrest → build/rst/
make sphinx     # Run Sphinx  → build/html_sphinx/
make publish    # Copy to ../docs/ and add .nojekyll
```

### Clean

```bash
make clean      # Removes build/
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `NameError: name 'Path' is not defined` | Upgrade to the fixed `conf.py` (already done) |
| `DOXYREST_DIR is not set` | `export DOXYREST_DIR=...` before running `make` |
| `doxyrest: command not found` | Install Doxyrest and ensure `$DOXYREST_DIR/bin` is in `PATH` |
| `sphinx-build: command not found` | `pip install sphinx sphinx-rtd-theme` |
| `ModuleNotFoundError: No module named 'doxyrest'` | DOXYREST_DIR doesn't point to the correct location |
