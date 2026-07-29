## Plan: Repository polish and best-practice cleanup

TL;DR: Improve the repository by sharpening onboarding, removing/generated artifact clutter, adding quality tooling and CI, and documenting build/test/data clearly.

**Steps**
1. Documentation and onboarding
   - Create or update `README.md` with a concise project summary, supported platforms, dependencies, quick-start build/test commands, and a short tree of key source/test directories.
   - Add a `LICENSE` file and reference it in `README.md`.
   - Add `CONTRIBUTING.md` with contribution guidance, testing expectations, and branch workflow.
   - Add `data/README.md` or `docs/data.md` describing dataset provenance, `git lfs` setup, and any non-tracked data generation steps.
   - Consolidate existing research notes into a `docs/` directory or keep `markdown/` but document the purpose.

2. Repository hygiene and structure
   - Remove or stop tracking generated artifacts and intermediate build outputs such as `bin/`, `obj/`, `libcore.a`, `libcore_baseline.a`, `tree.out`, and any transient `.d` or `.o` files.
   - Fix the duplicate/typo file `.gitinore` and ensure `.gitignore` covers build artifacts and editor files.
   - Keep data files that are intentionally tracked, but document which are source data versus derived outputs.
   - Optionally move large notebook outputs or derived CSVs out of the root if they are not critical to version control.

3. Build/test tooling
   - Add a dedicated `tests` or `check` target that builds and runs the test suite.
   - Add a `make help` target or top-level usage section in `makefile` to expose main targets.
   - Add GitHub Actions CI workflow to build `all` and run tests on Linux; optionally add matrix coverage for compiler flags or build types.
   - Add tooling configuration files such as `.clang-format` and optionally `.clang-tidy` to document code style.

4. Codebase clarity and maintenance
   - Document key directories: `include/`, `src/`, `tests/`, `experiments/`, `benchmarks/`, `data/`, `results/`.
   - Add a short `README` or `README.md` in subdirectories like `tests/` if the purpose is not obvious.
   - Consider a `CMakeLists.txt` migration later to improve portability, but keep the current Make-based build as the first priority.

**Relevant files**
- `README.md` — update project documentation and quick start.
- `makefile` and `make/*.mk` — add help/test targets and ensure build outputs are isolated.
- `.gitignore` / `.gitinore` — clean up ignore rules.
- `.gitattributes` / `.gitmodules` — verify LFS settings and submodule handling.
- `tests/` — ensure test automation and describe how to run tests.
- `data/` — document datasets and data management.

**Verification**
1. Confirm `make -j` completes successfully with the updated Makefile and documentation.
2. Confirm `make tests` or `make check` builds and runs existing tests.
3. Preview `README.md` and new docs for clarity.
4. If CI is added, confirm the workflow file would run the build/tests successfully.

**Decisions**
- Keep the existing research content, but separate code, docs, and outputs more clearly.
- Do not remove data files that are intentionally tracked without explicit user approval.
- Prefer incremental polish over a full build-system rewrite.
