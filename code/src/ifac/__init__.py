"""Vendored copy of the IFAC uncertainty-aware clustered-identification pipeline.

Copied verbatim (client, clustering, rls, run, server, utils) from
paper_UncertaintyAwareClustering/src2 (last upstream change 52df1bb6, 2026-03-31) so that
paper_FBO_LQI is self-contained. The only edit: server.py imports `.client` instead of
`src2.client`. Do not modify these files; project code should import via `src.ifac_bridge`.
"""
