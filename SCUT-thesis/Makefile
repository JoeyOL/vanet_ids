src = main

LATEXMK ?= $(shell command -v latexmk 2>/dev/null)
XELATEX ?= $(shell command -v xelatex 2>/dev/null)
PYGMENTIZE ?= $(shell command -v pygmentize 2>/dev/null)
PDF_VIEWER ?= $(shell command -v open 2>/dev/null || command -v xdg-open 2>/dev/null)

# Fallback paths for environments where PATH is not initialized.
ifeq ($(strip $(LATEXMK)),)
LATEXMK := /Users/bytedance/texlive/2026/bin/universal-darwin/latexmk
endif
ifeq ($(strip $(XELATEX)),)
XELATEX := /Users/bytedance/texlive/2026/bin/universal-darwin/xelatex
endif
ifeq ($(strip $(PYGMENTIZE)),)
PYGMENTIZE := /Users/bytedance/Library/Python/3.14/bin/pygmentize
endif

ifneq ($(wildcard $(dir $(XELATEX))xelatex),)
export PATH := $(dir $(XELATEX)):$(PATH)
endif
ifneq ($(wildcard $(dir $(PYGMENTIZE))pygmentize),)
export PATH := $(dir $(PYGMENTIZE)):$(PATH)
endif

pdf: clean main.tex main.bib
	@if [ ! -x "$(LATEXMK)" ]; then \
		echo "latexmk not found. Please install TeX Live or export LATEXMK=/path/to/latexmk"; \
		exit 1; \
	fi
	@if [ ! -x "$(XELATEX)" ]; then \
		echo "xelatex not found. Please install TeX Live or export XELATEX=/path/to/xelatex"; \
		exit 1; \
	fi
	$(LATEXMK) $(src) -shell-escape -xelatex

show:
	@if [ -z "$(PDF_VIEWER)" ]; then \
		echo "No PDF viewer found (open/xdg-open)."; \
		exit 1; \
	fi
	"$(PDF_VIEWER)" $(src).pdf

clean:
	-@rm -rf \
		*~ \
		*.aux \
		*.bbl \
		*.blg \
		*.dvi \
		*.ent \
		*.loa \
		*.lof \
		*.lot \
		*.toc \
		*.log \
		*.out \
		.fuse* \
		*.run.xml \
		*.bcf \
		*.xdv \
		*.synctex.gz \
		_minted-*
	-@find . -name "*.aux" -exec rm {} \;

update:
#自动下载最新cls并备份旧版替换
	-@mv ./scutthesis.cls ./scutthesis.cls.bak
	-@wget -O ./scutthesis.cls https://raw.githubusercontent.com/ShevonKuan/SCUT-thesis/main/scutthesis.cls
	-@echo "Update scutthesis.cls successfully!"
# install-fonts:
# 	cd fonts && chmod +x init_fonts.sh && ./init_fonts.sh
