.PHONY: test test-integration

test:
	python -m pytest -m "not integration"

test-integration:
	DETLLM_RUN_INTEGRATION=1 python -m pytest -m integration
