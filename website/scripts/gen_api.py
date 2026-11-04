"""Generate the API reference pages (website/docs/api/*.md) from the docstrings in src/.

Uses griffe, which reads the source without importing it, so this works
without the project's heavy dependencies (PyTorch, transformers, ...).

    poetry run python website/scripts/gen_api.py     # from the repo root
    poetry run npm run api --prefix website          # same, through npm

The output is committed so the site builds without Python; CI regenerates it
and fails if it's out of date. Edit this file (PAGES), not the generated pages.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import griffe

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "website" / "docs" / "api"
REPO = "https://github.com/Qualcomm-MDP/PSM"
WRAP_AT = 88  # signatures longer than this get one parameter per line


@dataclass
class Section:
    title: str
    intro: str = ""
    # Dotted paths: a class or function, or a module to list all of its
    # public classes and functions.
    objects: list[str] = field(default_factory=list)


@dataclass
class Page:
    file: str
    title: str
    intro: str
    sections: list[Section]
    position: int


PAGES = [
    Page(
        "pipeline.md",
        "Pipeline & State",
        "The pipeline framework (`src/common/pipeline_chain.py`) and the PSM "
        "pipeline built on it (`src/pipeline.py`). See "
        "[Architecture](../architecture.md#pipeline-framework) for how they fit together.",
        [
            Section("PipelineChain", objects=["common.pipeline_chain.PipelineChain"]),
            Section("PipelineState", objects=["common.pipeline_chain.PipelineState"]),
            Section(
                "The PSM pipeline",
                "`build_pipeline()` defines the stages shared by the command line "
                "and the web app; `export_mesh()` is its last stage.",
                ["pipeline.build_pipeline", "pipeline.export_mesh"],
            ),
            Section("BoundingBox", objects=["common.constants.BoundingBox"]),
        ],
        1,
    ),
    Page(
        "providers.md",
        "Data Providers",
        "Clients for the external data sources, in `src/common/providers/`, and "
        "the fetch stage that uses them. See [Data Processing](../pipeline/data-processing.md).",
        [
            Section(
                "OpenStreetMap",
                objects=[
                    "common.providers.osm.OSMClient",
                    "common.providers.osm.OSM_MAP_FEATURES",
                    "common.providers.overpass.OverpassClient",
                ],
            ),
            Section(
                "Mapillary", objects=["common.providers.mapillary.MapillaryClient"]
            ),
            Section("Fetch stage", objects=["data_ingest.ingest.ingest_data"]),
        ],
        2,
    ),
    Page(
        "utilities.md",
        "Utilities",
        "The other pipeline stages and the helpers they use.",
        [
            Section("Mesh building", objects=["mesh_builder.extrude"]),
            Section("Terrain", objects=["terrain.terrain"]),
            Section("Aerial photos (NAIP)", objects=["naip.naip"]),
            Section("Texturing", objects=["texturing.tex_projection"]),
            Section("Obstruction removal", objects=["segmentation.obstruction"]),
            Section("Mesh helpers", objects=["common.MeshUtils"]),
            Section("Mesh export", objects=["common.MeshExport"]),
            Section(
                "Coordinate conversions",
                objects=["common.conversions.basic", "common.conversions.utm"],
            ),
            Section("Progress display", objects=["common.ProgressMonitor"]),
        ],
        3,
    ),
    Page(
        "reporter.md",
        "Reporter",
        "How pipeline code reports what it's doing. The command line uses the "
        "console `Reporter`; the web app passes a `JobReporter` that saves the "
        "job's log and streams it to the browser. See "
        "[Architecture](../architecture.md#reporting-output).",
        [
            Section(
                "Reporter",
                objects=["common.reporter.Reporter", "common.reporter.get_reporter"],
            ),
            Section("JobReporter (web app)", objects=["api.joblog.JobReporter"]),
            Section("Keeping secrets out of logs", objects=["common.redact.redact"]),
        ],
        4,
    ),
    Page(
        "profiler.md",
        "Profiler",
        "Per-stage timing and resource measurements, turned on with `--profile`. "
        "See [Evaluation](../pipeline/evaluation.md).",
        [
            Section("PipelineProfiler", objects=["common.profiler.PipelineProfiler"]),
            Section("StageMetrics", objects=["common.profiler.StageMetrics"]),
        ],
        5,
    ),
]


# ---- loading ----------------------------------------------------------------

_loader = griffe.GriffeLoader(
    search_paths=[str(SRC)], docstring_parser=griffe.Parser.google
)


def get(path: str) -> griffe.Object:
    top = path.split(".")[0]
    if top not in _loader.modules_collection:
        _loader.load(top)
    _loader.resolve_aliases(implicit=False, external=False)
    return _loader.modules_collection[path]


# ---- rendering --------------------------------------------------------------


def text(s: str) -> str:
    """Docstring prose as Markdown: keep it, but don't let '<' start HTML."""
    return s.replace("<", "&lt;").strip()


def cell(s: str) -> str:
    return text(s).replace("\n", " ").replace("|", "\\|")


def source_link(obj: griffe.Object) -> str:
    filepath = obj.filepath[0] if isinstance(obj.filepath, list) else obj.filepath
    rel = Path(filepath).resolve().relative_to(ROOT).as_posix()
    line = f"#L{obj.lineno}" if obj.lineno and not obj.is_module else ""
    return f"[source]({REPO}/blob/main/{rel}{line})"


def signature(obj: griffe.Function, name: str, keyword: str = "def") -> str:
    params = []
    for p in obj.parameters:
        if p.name in ("self", "cls"):
            continue
        s = p.name
        if p.kind is griffe.ParameterKind.var_positional:
            s = "*" + s
        elif p.kind is griffe.ParameterKind.var_keyword:
            s = "**" + s
        if p.annotation is not None:
            s += f": {p.annotation}"
        if p.default is not None:
            s += f" = {p.default}" if p.annotation is not None else f"={p.default}"
        params.append(s)
    ret = f" -> {obj.returns}" if keyword == "def" and obj.returns is not None else ""
    one_line = f"{keyword} {name}({', '.join(params)}){ret}"
    if len(one_line) <= WRAP_AT or not params:
        return one_line
    inner = "".join(f"    {p},\n" for p in params)
    return f"{keyword} {name}(\n{inner}){ret}"


def docstring(obj: griffe.Object) -> list[str]:
    if not obj.docstring:
        return []
    out = []
    for sec in obj.docstring.parse(griffe.Parser.google, warnings=False):
        kind = sec.kind.value
        if kind == "text":
            out.append(text(sec.value))
        elif kind in ("parameters", "other parameters", "attributes"):
            title = "Attributes" if kind == "attributes" else "Parameters"
            rows = ["| Name | Type | Description |", "| --- | --- | --- |"]
            for p in sec.value:
                kind_ = f"`{p.annotation}`" if p.annotation is not None else ""
                rows.append(f"| `{p.name}` | {cell(kind_)} | {cell(p.description)} |")
            out.append(f"**{title}**\n\n" + "\n".join(rows))
        elif kind in ("returns", "yields"):
            items = []
            for r in sec.value:
                ann = f"`{r.annotation}`: " if r.annotation is not None else ""
                items.append(f"- {ann}{text(r.description)}")
            out.append(f"**{kind.capitalize()}**\n\n" + "\n".join(items))
        elif kind in ("raises", "warns"):
            items = [f"- `{e.annotation}`: {text(e.description)}" for e in sec.value]
            out.append(f"**{kind.capitalize()}**\n\n" + "\n".join(items))
        elif kind == "examples":
            for sub_kind, value in sec.value:
                out.append(
                    text(value)
                    if sub_kind.value == "text"
                    else f"```python\n{value}\n```"
                )
        elif kind == "admonition":
            out.append(f"**{sec.title}**\n\n{text(sec.value.description)}")
    return out


def public(obj: griffe.Object) -> bool:
    return not obj.name.startswith("_")


def render_function(fn: griffe.Function, level: int, owner: str = "") -> list[str]:
    name = f"{owner}.{fn.name}" if owner else fn.name
    head = "#" * level
    lines = [f"{head} {name}()", ""]
    lines += [f"```python\n{signature(fn, fn.name)}\n```", ""]
    for block in docstring(fn):
        lines += [block, ""]
    lines += [source_link(fn), ""]
    return lines


def render_class(cls: griffe.Class, level: int) -> list[str]:
    head = "#" * level
    bases = f"({', '.join(str(b) for b in cls.bases)})" if cls.bases else ""
    lines = [f"{head} {cls.name}", ""]
    init = cls.members.get("__init__")
    if isinstance(init, griffe.Function) and not init.is_alias:
        sig = signature(init, cls.name, keyword="class")
        lines += [f"```python\n{sig}\n```", ""]
    else:
        lines += [f"```python\nclass {cls.name}{bases}\n```", ""]
    for block in docstring(cls):
        lines += [block, ""]
    if isinstance(init, griffe.Function) and not init.is_alias and init.docstring:
        for block in docstring(init):
            lines += [block, ""]

    # Dataclass fields and class constants without their own docstring.
    attrs = [
        m
        for m in members_in_order(cls)
        if isinstance(m, griffe.Attribute) and public(m)
    ]
    if attrs and "dataclass" in {d.value.__str__() for d in cls.decorators}:
        rows = ["| Field | Type | Default |", "| --- | --- | --- |"]
        for a in attrs:
            ann = f"`{a.annotation}`" if a.annotation is not None else ""
            default = f"`{a.value}`" if a.value is not None else ""
            rows.append(f"| `{a.name}` | {cell(ann)} | {cell(default)} |")
        lines += ["\n".join(rows), ""]
    elif attrs and any(str(b).endswith("Enum") for b in cls.bases):
        lines += ["\n".join(f"- `{a.name} = {a.value}`" for a in attrs), ""]

    lines += [source_link(cls), ""]
    for m in members_in_order(cls):
        if isinstance(m, griffe.Function) and public(m):
            lines += render_function(m, level + 1, owner=cls.name)
    return lines


def members_in_order(obj: griffe.Object) -> list[griffe.Object]:
    """Members defined here (not imported names), in source order."""
    members = [m for m in obj.members.values() if isinstance(m, griffe.Object)]
    return sorted(members, key=lambda m: (m.lineno or 0, m.name))


def render_object(path: str, level: int) -> list[str]:
    obj = get(path)
    if obj.is_module:
        lines = []
        for block in docstring(obj):
            lines += [block, ""]
        lines += [f"Module `{path}` · {source_link(obj)}", ""]
        for m in members_in_order(obj):
            if not public(m):
                continue
            if isinstance(m, griffe.Class):
                lines += render_class(m, level)
            elif isinstance(m, griffe.Function):
                lines += render_function(m, level)
        return lines
    if isinstance(obj, griffe.Class):
        return render_class(obj, level)
    if isinstance(obj, griffe.Function):
        return render_function(obj, level)
    raise TypeError(f"{path}: can't document a {obj.kind.value}")


def render_page(page: Page) -> str:
    lines = [
        "---",
        f"title: {page.title}",
        f"sidebar_position: {page.position}",
        "---",
        "",
        "<!-- Generated by website/scripts/gen_api.py from the docstrings in src/. "
        "Don't edit by hand. -->",
        "",
        f"# {page.title}",
        "",
        page.intro,
        "",
    ]
    for section in page.sections:
        only = section.objects[0] if len(section.objects) == 1 else ""
        if only.rsplit(".", 1)[-1] == section.title and not section.intro:
            # A section that is just one class: no separate section heading.
            lines += render_object(only, 2)
            continue
        lines += [f"## {section.title}", ""]
        if section.intro:
            lines += [section.intro, ""]
        for path in section.objects:
            lines += render_object(path, 3)
    text_out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text_out).rstrip() + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for page in PAGES:
        (OUT / page.file).write_text(render_page(page), encoding="utf-8")
        print(f"wrote {(OUT / page.file).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
