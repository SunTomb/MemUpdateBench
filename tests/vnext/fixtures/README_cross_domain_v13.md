# Cross-domain v13 recorded fixture

`cross_domain_v13/` is a byte-preserving copy of the collected historical v13 Qdrant canary. Its result artifact index is `2b2b9dbfd7703f0a91cc66a9267344837f4a892bfac89e5e2db25a00b836a60a`, and its collection-audit hash is `5188cbbb7fbb3aacbf3a25994a75273060c9a2ab2857c180f3eacf7be3326eca`.

The fixture supports deterministic tests of hashes, bindings, normalized state/retrieval records and package safety. Tests do not start a provider or model. The data comprises public-source-derived scalar observations and hash-only runtime/HTTP provenance; it does not include raw request/response bodies, credentials, prompts, source captures or model weights. Credential and private-path checks were run before inclusion.

Do not regenerate or edit fixture members to make a test pass. Any future fixture must use a separate directory and its own explicit provenance. The two task release roots in `data/vnext/family_h_cross_domain_*/v1` likewise retain their original bytes.

Raw-source capture and human-review integration tests remain separate: when their historical `external/` and `results/` inputs are absent, they explicitly skip instead of downloading replacements or pretending to have run.
