.PHONY: run clean

run:
	@echo "Running experiments..."
	@for f in experiments/*.py; do \
		echo "→ $$f"; \
		python "$$f"; \
	done

clean:
	rm -rf data/processed/*
