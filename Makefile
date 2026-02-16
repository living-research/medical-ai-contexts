.PHONY: run clean

run:
	@echo "Running experiments..."
	@for f in experiments/*.py; do \
		[ -f "$$f" ] || continue; \
		echo "→ $$f"; \
		python "$$f"; \
	done

clean:
	rm -rf data/processed/*
