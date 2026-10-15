"""The canonical architecture: semantic identity, certified archive, generic parameter binding, action records.

Every component is behind an explicit switch in `canon.config.IntegrationConfig`; with every switch OFF the loop
`canon.search.beam_search` is behaviour-identical to `rewrite_search_latent.beam_search`.
"""
