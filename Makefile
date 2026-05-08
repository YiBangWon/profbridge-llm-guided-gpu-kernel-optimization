.PHONY: smoke

smoke:
	python -m compileall profbridge scripts
	python scripts/smoke_test.py
