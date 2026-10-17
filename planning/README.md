# KernelWiki planning

Keep design notes, proposals, research plans, and decision records here. Link
plans to their Project Control task IDs when tasks are created.

Project Control workspace and repository ID: `kernelwiki`.

Todo Orchestrator is the transactional task authority, accessed through Project
Control (`next_task`, `inspect_task`, `coordinate_task`, and `finish_task`).
`todos.md` and `todo-status.md` are generated views; update task state through
the supported workflow rather than editing those files.

The project identity and portable snapshot live in `.todo-orchestrator/`.
The live SQLite authority lives under the Git common directory at
`todo-orchestrator/<project_uuid>/state.sqlite3`.

No implementation backlog has been seeded. Add tasks when project goals and
scope are defined.
