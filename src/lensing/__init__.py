"""Tier-0 strong-lens simulator for the substructure stress test (project 05).

Built directly on `lenstronomy` rather than `paltas` (which does not build on
Python 3.14 as of 2026-09 — its packaging pins a setuptools/pkg_resources
combination that assumes `pkgutil.ImpImporter`, removed in Python 3.12).
`lenstronomy` is the engine paltas itself wraps, so nothing physical is lost;
every choice below is explicit and cited against `../fulltext_findings.md`.
"""
