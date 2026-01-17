# Versioning policy

detLLM follows semantic versioning with pre-1.0 rules:

- `0.x` may include breaking changes without notice.
- Patch releases fix bugs and documentation.
- Minor releases add features with best-effort backward compatibility.

Schema changes:
- Within a schema major version: only add fields, never remove or rename.
- Readers must ignore unknown fields.
