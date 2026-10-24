"""The tests of Leaseholder's engine variants for vLLM 0.32 (deploy/phastform/variants).

The package holds no runtime code: each variant builds its scheduler module from the engine's own vLLM at engine
start. The engine plugins it held, LeasePolicy and LeaseConnector, were built for the lease design and removed
after v0.3 (ADR-0006).
"""
