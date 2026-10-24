## Change

Describe the problem, resulting behavior and affected users. Include a concrete
before/after example when useful, and identify any compatibility or migration
change.

## Validation

List the checks actually run, their outcomes and any remaining coverage gap.
For a `fix:` PR, name the regression test; if unavailable, explain why and give
the alternative evidence. Follow [MAINTAINING.md](../MAINTAINING.md) for the
current CI and test procedures.

For scientific calculations, distinguish formula-based correctness checks from
returned-data compatibility. An aggregate with an unknown calculation procedure
is not a numerical oracle. For affected HFSS/EPR or batch paths (Tier A), record
assigned private-golden checks as `COMPLETED`, `NOT RUN` or `NOT APPLICABLE`, and
state the affected claim when data or access is unavailable. Keep private data,
values, names and paths out of public text.

## Delivery

Link the actual required CI results and identify any remaining delivery action.
Report Human acceptance only when explicitly given; passing CI does not imply
acceptance or native verification.
