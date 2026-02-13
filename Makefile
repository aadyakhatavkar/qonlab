.PHONY: help pdf tables figures quick run append-figures clean

help:
	@echo "Available commands:"
	@echo "  make pdf           - Build tables PDF (requires tables data)"
	@echo "  make figures       - Generate figures only (stores in outputs/figures/)"
	@echo "  make append-figures - Append figures to latest tables PDF"
	@echo "  make quick         - Run quick simulation (30 runs, stores results)"
	@echo "  make run           - Run full simulation (300 runs, stores results)"
	@echo "  make clean         - Clean output directories"

pdf:
	pixi run python scripts/build_pdfs.py --tables

figures:
	pixi run python scripts/generate_plots.py --all

append-figures:
	pixi run python scripts/build_pdfs.py --combined

quick:
	pixi run python main.py --quick

run:
	pixi run python main.py

clean:
	rm -rf outputs/figures/*.png
	rm -rf outputs/*.csv
	rm -rf outputs/pdf/*.pdf
	rm -rf outputs/logs/*
