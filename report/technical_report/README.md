# Report source

[Read the 12-page PDF](technical_report.pdf). The report covers the equation and manufactured source, finite elements, BE/Radau time stepping, GMRES and preconditioning, five result figures, reproduction and limitations.

| Path | Contents |
|---|---|
| `main.tex` | Editable text, mathematics and references |
| `technical_report.pdf` | Compiled report |
| `evidence/data.tex` | Generated tables and numerical values |
| `evidence/claims.json` | Source selectors and hashes for the reported data |
| `evidence/primary_sources.json` | Literature/documentation context |
| `evidence/report-build.json` | Build command, tool version and source/PDF hashes |
| `images/` | Exact copies of the five source figures |
| `.build/`, `.qa/` | Ignored local compiler and page-rendering files |

From the repository root:

```sh
python3 scripts/check_saved_results.py
python3 scripts/prepare_report.py --check
python3 scripts/build_report.py
```

The build requires `pdflatex` and `pdfinfo`; it uses saved measurements, with no numerical installation required. To regenerate table macros and figure copies, run `prepare_report.py` without `--check`. The original full-installation audit remains available through `scripts/check_report_evidence.py`.

The report revision changes explanations and presentation, not the numerical implementations, inputs or measurements. Figure regeneration is documented in [the reproduction guide](../../docs/reproducibility.md). Exact PDF bytes can vary with the TeX installation.
