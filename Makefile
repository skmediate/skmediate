.PHONY: flake lint

flake:
	flake8
	black --check .
	pydocstyle

lint: flake

test:
    # Unit testing using pytest
	pytest --pyargs skmediate --cov-report term-missing --cov-config .coveragerc --cov=skmediate

devtest:
    # Unit testing with the -x option, aborts testing after first failure
    # Useful for development when tests are long
	pytest -x --pyargs skmediate --cov-report term-missing --cov-config .coveragerc --cov=skmediate