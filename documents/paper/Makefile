# Makefile for LaTeX paper compilation
# Research Module: Forecasting Under Structural Breaks

MAIN = main
TEX = pdflatex
BIB = bibtex

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)
	@echo "PDF compiled successfully: $(MAIN).pdf"

# Full compilation with bibliography
full: $(MAIN).tex
	$(TEX) $(MAIN)
	$(BIB) $(MAIN) || true
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)
	@echo "Full PDF compiled with bibliography: $(MAIN).pdf"

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null || echo "Open $(MAIN).pdf manually"

# Watch for changes (requires latexmk)
watch:
	latexmk -pvc -pdf $(MAIN).tex
