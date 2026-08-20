
# Lecture pages that are authored in Markdown and shipped with a runnable twin.
#
# The Markdown file is the single source of truth: it is what you edit, what git
# diffs, and what the site publishes. The notebook is generated from it by
# jupytext, executed once on a real GPU, and committed with its outputs so the
# Colab badge works for readers who have no GPU of their own.
#
# The notebook is NOT published as a site page. mkdocs-jupyter renders notebook
# markdown cells without mkdocs' extension pipeline, so admonitions, relative
# image paths and internal .md links all break there. mkdocs.yml excludes
# lectures/*.ipynb from the build for that reason.
#
# Only ```python fences become code cells. Shell snippets that are meant to be
# read rather than run must use ```shell or ```console -- jupytext turns ```bash
# and ```sh into code cells, which would then execute as Python.

NB_LECTURES := docs/lectures/Lecture7

PY        := .venv/bin/python
JUPYTEXT  := .venv/bin/jupytext
NOTEBOOKS := $(addsuffix .ipynb,$(NB_LECTURES))

.PHONY: notebooks notebooks-exec serve build clean-notebooks

## Regenerate notebooks from their Markdown sources, without executing them.
notebooks: $(NOTEBOOKS)

# --update refreshes the notebook from the Markdown while KEEPING the outputs of
# code cells that did not change. Without it, regenerating after a prose edit
# would silently discard the measurements `notebooks-exec` just produced.
%.ipynb: %.md
	@if [ -f $@ ]; then $(JUPYTEXT) --to ipynb --update $< -o $@; \
	 else $(JUPYTEXT) --to ipynb $< -o $@; fi

## Regenerate and execute. Run this on a machine with a GPU: the outputs are
## the measurements the lecture quotes, so they must come from real hardware.
## On a laptop, run it on mains power -- these cells drive sustained peak load.
notebooks-exec: notebooks
	$(PY) -m nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1200 $(NOTEBOOKS)

serve: notebooks
	$(PY) -m mkdocs serve

build: notebooks
	$(PY) -m mkdocs build

clean-notebooks:
	rm -f $(NOTEBOOKS)
