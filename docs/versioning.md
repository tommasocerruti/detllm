# Versioning policy

detLLM follows semantic versioning:

- `0.x` may include breaking changes with notice.
- `1.x` and above follow SemVer rules.

## Artifact schema stability

Schema versioning is stable and forward compatible:

- Within a schema major version: only add fields, never remove or rename.
- Readers must ignore unknown fields.
- Breaking schema changes require a major schema version bump.
