# Lightning LLMs

This is starter project template which shall simplify initial steps for each new PL project...

[![CI testing](https://github.com/Lightning-AI/lightning-LLMs/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-LLMs/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-LLMs/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-LLMs/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-LLMs/badge/?version=latest)](https://lightning-LLMs.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-LLMs/main.svg?badge_token=mqheL1-cTn-280Vx4cJUdg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-LLMs/main?badge_token=mqheL1-cTn-280Vx4cJUdg)

\* the Read-The-Docs is failing as this one leads to the public domain which requires the repo to be public too

## To be Done aka cross-check

You still need to enable some external integrations such as:

- [x] rename `pl_<sandbox>` to anu other name, simple find-replace shall work well
- [x] update path used in the badges to the repository
- [ ] lock the main breach in GH setting - no direct push without PR
- [x] set `gh-pages` as website and _docs_ as source folder in GH setting
- [ ] init Read-The-Docs (add this new project)
- [ ] add credentials for releasing package to PyPI
- [ ] specify license in `LICENSE` file and package init

## Tests / Docs notes

- We are using [Napoleon style,](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) and we shall use static types...
- It is nice to se [doctest](https://docs.python.org/3/library/doctest.html) as they are also generated as examples in documentation
- For wider and edge cases testing use [pytest parametrization](https://docs.pytest.org/en/stable/parametrize.html) :\]
