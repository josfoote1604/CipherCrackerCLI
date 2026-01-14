# CipherCracker CLI

WORK IN PROGRESS: Last updated 01/13/2026
CipherCracker CLI is a set of classical cipher tools and cryptanalysis helpers. It allows you to analyze ciphertext, decrypt using known keys, and crack unknown ciphers using various techniques like frequency analysis and simulated annealing.

## Overview

The tool provides a command-line interface to work with common classical ciphers, including:
- **Monoalphabetic Ciphers**: Caesar, Affine, Atbash, and general Substitution.
- **Polyalphabetic Ciphers**: Vigenère and Periodic Substitution.

## Requirements

- Python >= 3.10
- Dependencies: `typer`

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CipherCrackerCLI
   ```

2. **Install the package**:
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install .
   ```

## Usage

After installation, you can use the `ciphercracker` command.

### General Commands

- **List available plugins**:
  ```bash
  ciphercracker plugins
  ```

- **Analyze text (IoC, frequency analysis)**:
  ```bash
  ciphercracker analyze "YOUR_CIPHERTEXT_HERE"
  ```
  Use `--iocmax N` to perform an Index of Coincidence scan up to key length N.

- **Decrypt with a known key**:
  ```bash
  ciphercracker decrypt -c caesar -k 3 "L FDPH, L VDZ, L FRQTX HUHG"
  ```

- **Crack an unknown cipher**:
  ```bash
  ciphercracker crack "YOUR_CIPHERTEXT_HERE"
  ```
  Optional flags:
  - `-t, --top N`: Show top N candidates (default: 5).
  - `-c, --cipher NAME`: Limit to specific cipher plugin(s).
  - `--iocmax N`: Show IoC scan before cracking.

## Scripts

The project defines the following entry points:
- `ciphercracker`: Main CLI application.

## Project Structure

```text
src/ciphercracker/
├── classical/          # Cipher implementations (mono/polyalphabetic)
├── core/               # Scoring, registry, and analysis logic
├── data/               # Language models (e.g., english_quadgrams.txt)
└── cli.py              # CLI entry point and commands
```

## Environment Variables

- TODO: Document any environment variables used by the system (none currently identified).

## Tests

- TODO: Automated tests are not yet fully implemented/configured. 
- You can run any manual test scripts if present in the `tests` directory (none currently identified in the root).

## License

- TODO: Specify license (e.g., MIT, Apache-2.0).
