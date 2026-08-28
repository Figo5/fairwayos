# FairwayOS Branding and Public Repository Setup Plan

> **For Hermes:** Execute this plan only after the public-repository safety review passes. Preserve the internal `ghostcaddie` package and all schema identifiers during the branding transition.

**Goal:** Rebrand the public-facing project as FairwayOS, create and publish the first public GitHub repository `fairwayos`, and preserve all existing behavior, safety gates, research-only dataset boundaries, and the 265-test baseline.

**Architecture:** FairwayOS is the public product/project name; `ghostcaddie` remains the internal Python package and import namespace for compatibility. Branding changes are limited to public display names, descriptions, documentation, report labels, and CLI help unless a separate tested migration is approved. The public repository contains source, tests, documentation, plans, safe synthetic fixtures, license/provenance notices, and reproducible acquisition instructions—but never downloaded media, weights, environments, generated artifacts, private annotations, credentials, or caches.

**Tech Stack:** Git, GitHub CLI (`gh`), Python 3.9-compatible project code, `python3 -m unittest`, `compileall`, existing local-only video tooling, and bounded yt-dlp acquisition instructions.

---

## Confirmed current state

- Working directory: `/Users/giofiore/ghostcaddie-tour`.
- No Git repository exists yet.
- GitHub CLI authentication is available for account `Figo5`; the token value must never be printed.
- The explicit final repository requirement is public: `Figo5/fairwayos`.
- Existing package namespace: `ghostcaddie`.
- Existing schema identifiers and CLI commands must remain unchanged.
- Existing full suite baseline: 265 tests.
- Existing `.gitignore` covers only `__pycache__/`, `*.pyc`, and `out/`.
- The working tree contains large/sensitive-to-publish material:
  - `.venv-video/`;
  - `.venv-video-ai/`;
  - `.venv-video-modern/`;
  - `out/` generated reports, media, and frames;
  - `yolo11n.pt` and `yolo11n-pose.pt` model weights;
  - downloaded `.mp4`, `.pkl`, `.pth`, `.pt`, image, cache, and compiled files;
  - private/local evaluation material under `out/golfdb_evaluation/`.
- Branding occurrences were inventoried across README, package metadata, package docstrings, CLI description, requirements, documentation, generated evaluation scripts, and internal compatibility paths. Internal import paths and code symbols are intentionally not public-branding replacements.

---

## Hard safety rules

1. Initialize Git only in `/Users/giofiore/ghostcaddie-tour`.
2. Never initialize Git inside any virtual environment or output directory.
3. Never commit `.env`, secrets, tokens, cookies, credentials, API keys, private URLs, personal data, private annotations, downloaded videos, generated `out/` artifacts, model weights, caches, virtual environments, or compiled artifacts.
4. Never print `gh` tokens or inspect credential contents.
5. Do not use a token-embedded remote URL.
6. Use `gh repo create fairwayos --public --source . --remote origin` only after staged-file review, or equivalent explicit API operation without exposing credentials.
7. Do not push until the complete staged file list has been reviewed and a repository-wide secret/media scan passes.
8. Do not publish downloaded GolfDB/PGA footage or unlicensed weights.
9. Preserve third-party attribution, license, and provenance notices in safe text manifests.
10. Keep source URLs and video IDs only where acquisition/provenance instructions require them; never serialize credentials, cookies, or absolute local paths.

---

## Public repository contents

### Include

- `ghostcaddie/` source package;
- `tests/` tests and safe synthetic fixtures;
- `data/` only safe synthetic/project fixtures, excluding private annotations and downloaded media;
- `docs/` public documentation, safety constraints, compatibility notes, and research plans;
- `.hermes/plans/` approved plans, excluding private session material;
- `README.md` rebranded as FairwayOS;
- `pyproject.toml` with public project metadata updated while preserving the `ghostcaddie` package name;
- `requirements.txt` with public-facing description updated while preserving dependency behavior;
- `LICENSE` for FairwayOS project source, after confirming the intended project license;
- `THIRD_PARTY_NOTICES.md` for GolfDB/SwingNet and other external attribution/provenance;
- reproducible acquisition instructions and safe model/dataset manifests containing hashes and licensing status, not the binaries.

### Exclude

- `.venv-video/`, `.venv-video-ai/`, `.venv-video-modern/`;
- `out/` and all generated reports, frames, contact sheets, videos, overlays, and evaluation outputs;
- `*.mp4`, `*.mov`, `*.avi`, `*.mkv`, `*.webm`;
- `*.pt`, `*.pth`, `*.pth.tar`, `*.onnx`, `*.safetensors`;
- `*.pkl`, `*.pickle`, `*.mat`, `*.npy`, `*.npz` when they contain downloaded/private evaluation data;
- `__pycache__/`, `*.pyc`, `.pytest_cache/`, tool caches, OS metadata, and temporary files;
- `.env`, `.env.*`, credential files, cookie jars, token files, SSH/private-key material;
- private annotations, consent records, personal data, and local source manifests that reveal sensitive paths;
- `tmp/` smoke scripts if they contain machine-specific paths or are not made safe for public publication.

Use explicit force-add only for safe text manifests or docs that have passed review.

---

## Branding policy

### Public replacements

Replace public-facing `GhostCaddie Tour` / `GhostCaddie` prose with `FairwayOS` where it refers to the project/product name:

- README title and prose;
- `pyproject.toml` project name/description where compatible;
- `requirements.txt` comments;
- CLI program description and help text;
- report display names and documentation headings;
- generated artifact display labels;
- project/repository descriptions;
- compatibility and migration documentation.

### Preserve

Do not rename:

- the `ghostcaddie` Python package directory;
- `python3 -m ghostcaddie` command paths;
- imports, module paths, schema identifiers, provenance field values, test module names, or serialized contracts;
- internal compatibility strings required by existing tests or provider detection unless covered by a separate migration plan.

Add a compatibility note:

> FairwayOS is the public project name. The internal `ghostcaddie` Python package, module paths, CLI entrypoint, and existing schema identifiers remain unchanged for compatibility.

### Required migration verification

Before editing, save the branding inventory. After editing:

- re-run the inventory;
- classify every remaining old-name occurrence as internal namespace, compatibility path, historical artifact, or missed public branding;
- inspect all changed files;
- verify no analytics behavior changed.

---

## Implementation sequence

### Task 1: Create public-repository safety inventory

Record a machine-readable local inventory outside the eventual commit or in a safe review artifact listing:

- candidate files by category;
- excluded path counts;
- branding occurrences;
- binary/media/weight paths;
- secret-pattern scan results;
- files selected for publication.

Do not include the inventory if it reveals private paths, personal data, or sensitive filenames.

### Task 2: Add and test the public `.gitignore`

Expand `.gitignore` to exclude all environment, media, output, weight, cache, secret, credential, and generated-file classes above. Include exceptions only for explicitly safe synthetic fixtures and text provenance manifests.

Before commit, validate with:

```bash
git check-ignore -v .venv-video-ai out/example.mp4 yolo11n.pt .env
```

Expected: every unsafe example is ignored.

### Task 3: Apply FairwayOS public branding

Use targeted edits only in public-facing files. Preserve internal package identifiers and serialized contracts. Add `docs/fairwayos-compatibility.md` documenting the transition and the unchanged internal namespace.

Potential files:

- `README.md`;
- `pyproject.toml`;
- `requirements.txt`;
- `ghostcaddie/__init__.py` docstring;
- `ghostcaddie/cli.py` public description/help strings;
- report display-name code and public documentation where safe;
- `docs/` public reports and plans.

Add focused tests only if a display name/help contract changes; do not weaken existing assertions merely to make a new brand pass.

### Task 4: Add public license and attribution files

Create or confirm:

- `LICENSE` for FairwayOS project source, pending the project’s chosen license;
- `THIRD_PARTY_NOTICES.md` documenting GolfDB/SwingNet attribution, repository code license statement, checkpoint license ambiguity, and research-only status;
- safe provenance instructions without committing model weights or downloaded footage.

If the project license is not explicitly approved, use a clearly marked placeholder only if it does not falsely grant rights; otherwise stop and ask for the license decision before publishing.

### Task 5: Public-repository secret and artifact review

Before Git initialization or staging, scan the entire project for:

- API-key/token patterns;
- cookies and authorization headers;
- `.env` and credential files;
- private URLs and personal data;
- media, model weights, virtual environments, caches, generated output, and private annotations.

Review false positives manually. Do not print secret contents. Use filenames, hashes, and redacted findings only.

### Task 6: Initialize Git and stage safe contents

Only after Tasks 1–5 pass:

```bash
cd /Users/giofiore/ghostcaddie-tour
git init -b main
git add --all
git status --short
git diff --cached --stat
git diff --cached --name-only
```

Inspect the complete staged file list. Confirm no prohibited path or binary is staged. If any prohibited file appears, unstage it, fix `.gitignore`, and repeat the review.

Do not commit before this review passes.

### Task 7: Create the public GitHub repository

Verify authentication immediately before creation without printing credentials:

```bash
gh auth status
```

Confirm the authenticated owner is `Figo5`, then create:

```bash
gh repo create fairwayos \
  --public \
  --source . \
  --remote origin \
  --description "FairwayOS: local-first golf video perception and shot analytics research platform"
```

If the repository already exists, do not overwrite it; inspect it and stop for an explicit recovery decision.

### Task 8: Commit and push after review

After staged-file approval:

```bash
git config user.name
 git config user.email
git commit -m "Initial FairwayOS project repository"
git branch --show-current
git rev-parse HEAD
git push -u origin main
```

Use the authenticated GitHub remote configured by `gh`; never embed a token. Verify the remote URL contains only the public HTTPS repository URL and no credentials.

### Task 9: Verify the published repository

Read back the exact remote state:

```bash
gh repo view Figo5/fairwayos --json name,owner,isPrivate,defaultBranchRef,url,description
 git ls-tree -r --name-only HEAD
 git show --stat --oneline --summary HEAD
```

Clone into a temporary clean-checkout directory outside the project, without altering the working tree:

```bash
git clone https://github.com/Figo5/fairwayos.git /tmp/fairwayos-clean-checkout
cd /tmp/fairwayos-clean-checkout
python3 -m unittest discover -s tests
python3 -m compileall -q ghostcaddie tests
```

Confirm the clean checkout contains no ignored local artifacts and that all safe source/tests/docs load without the local virtual environments.

### Task 10: Verify dataset boundaries and existing workflows

Confirm the public repository contains:

- no downloaded GolfDB/PGA videos;
- no SwingNet or YOLO weight binaries;
- no private annotations or consent records;
- only safe acquisition instructions, hashes, license status, and provenance notes;
- unchanged GolfDB acquisition plan;
- unchanged separate domains for GolfDB research data, public PGA stress clips, and FairwayOS-owned data;
- SwingNet explicitly research-only;
- human fallback and automatic-perception gates preserved.

Run all existing CLI scenarios and verify outputs remain behaviorally identical apart from approved public display branding.

---

## Rollback strategy

### Before push

- Abort the commit and remove the local Git metadata with `rm -rf .git` only if initialization was accidental and no repository history should remain.
- Revert branding edits using the recorded pre-change inventory and targeted patches.
- Remove newly created public license/branding files if not approved.
- Do not delete source, data, or evaluation artifacts outside the intended Git operation.

### After repository creation but before push

- If safety review fails, leave the remote empty and report the exact blocker.
- Do not push partial content.
- If the repository was accidentally created with the wrong visibility/name, use `gh repo edit` or delete only after explicit confirmation; never silently replace a remote.

### After push

- Do not rewrite public history casually.
- If a prohibited file was published, immediately stop further work, identify the exact path, remove it in a follow-up commit, invalidate exposed credentials if any, and report the incident. A history rewrite or repository deletion requires explicit approval.
- Preserve the initial commit hash and report any corrective commit separately.

---

## Verification gates

### Branding gate

- No missed public `GhostCaddie Tour` branding remains.
- Internal `ghostcaddie` namespace and schema identifiers remain unchanged.
- Compatibility note is present.
- Existing tests and CLI commands remain green.

### Public-repository safety gate

- `gh auth status` passes without token exposure.
- Complete staged file list was reviewed.
- No secrets, media, weights, caches, environments, outputs, or private annotations are staged.
- `LICENSE` and third-party attribution are present and accurate.
- Public repository visibility is verified as `false` for `isPrivate`.
- Remote owner/name are verified as `Figo5/fairwayos`.

### Regression gate

```bash
python3 -m unittest discover -s tests
python3 -m compileall -q ghostcaddie tests
```

Expected: the existing 265-test baseline remains green. Also run the project’s existing CLI scenarios and clean-checkout verification before reporting completion.

### Dataset/research gate

- GolfDB acquisition plan unchanged.
- SwingNet research-only.
- Public PGA clips remain qualitative stress material only.
- FairwayOS-owned dataset design remains a future consented-data workflow.
- No public video or restricted model asset is published.
- No production accuracy or recommendation claim is added.

## Final report requirements

Report only verified values:

- public repository URL;
- owner and visibility;
- branch;
- initial commit hash;
- `gh auth status` result without token data;
- test and compileall results;
- clean-checkout result;
- exact excluded path classes;
- branding compatibility summary;
- any remaining unresolved license or dataset blockers.
