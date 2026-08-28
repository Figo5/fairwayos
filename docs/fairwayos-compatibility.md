# FairwayOS Compatibility Notes

FairwayOS is the public project name for this repository. The internal
`ghostcaddie` Python package and import namespace remain unchanged for
compatibility with existing applications, tests, CLI commands, and serialized
contracts.

The following are intentionally preserved:

- `python3 -m ghostcaddie` and all existing subcommands;
- the `ghostcaddie/` module tree and import paths;
- existing schema identifiers such as `video-observations.v1` and
  `automatic-perception.v1`;
- existing analytics behavior, human fallback, and automatic-perception gates;
- GolfDB research-data, public PGA stress-data, and FairwayOS-owned-data domain
  separation;
- SwingNet research-only status until applicable licensing and labeled
  evaluation gates pass.

Public-facing README, CLI descriptions, documentation headings, report display
names, and project metadata use **FairwayOS**. Internal identifiers are not
renamed as part of this transition. A future package rename, if desired,
must be separately planned, tested, and released as a compatibility migration.
