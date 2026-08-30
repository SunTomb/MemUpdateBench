# Main-track v1 human audit

This packet prepares the selected synthetic main-track candidate for human review.

Review policy: post-core-data-audit-v1
Initial review status: NOT_STARTED

For every row, inspect the visible events, event roles and metadata, declared actions,
target objects, query selector, gold evidence, and version history. Record exactly one
value in `audit_decision`: `pass`, `needs_revision`, or `block`. Use `issue_category`
when a decision is not `pass`, add the reviewer identity and a concise review note,
and set `resolved_status` to `unresolved` or `resolved` as appropriate. Leave the
five audit fields empty until a human reviewer makes a decision.

Do not edit candidate task content in this packet. Do not infer model behavior, runtime
behavior, or benchmark approval from these records. This packet contains synthetic,
redistributable task data only and contains no model outputs. Completion of this packet
is not human-audit approval; a separate authenticated review gate is required.
