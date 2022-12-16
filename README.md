# Lightning LLMs

This project contains all utility functions to train large language models on [lightning cloud](https://lightning.ai) using [lightning apps](https://github.com/Lightning-AI/lightning). Parts of this repository may later be upstreamed to the [lightning main repository](https://github.com/Lightning-AI/lightning).

[![CI testing](https://github.com/Lightning-AI/lightning-LLMs/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-LLMs/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-LLMs/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-LLMs/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-LLMs/badge/?version=latest)](https://lightning-LLMs.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-LLMs/main.svg?badge_token=mqheL1-cTn-280Vx4cJUdg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-LLMs/main?badge_token=mqheL1-cTn-280Vx4cJUdg)



## Tests / Docs notes

- We are using [Napoleon style,](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) and we shall use static types...
- It is nice to se [doctest](https://docs.python.org/3/library/doctest.html) as they are also generated as examples in documentation
- For wider and edge cases testing use [pytest parametrization](https://docs.pytest.org/en/stable/parametrize.html) :\]
