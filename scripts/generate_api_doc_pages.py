"""Generate the code reference pages.
(adapted from https://mkdocstrings.github.io/recipes/#bind-pages-to-sections-themselves)
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.nav.Nav()

root = Path(__file__).parent.parent
src = root / "src"

# Track flat nav entries we have added
added_nav_labels: set[str] = set()

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    nav_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api", nav_path)

    parts = tuple(module_path.parts)

    # Skip if this is just the bioimageio namespace package
    if parts == ("bioimageio",):
        continue

    # Skip private submodules prefixed with '_'
    if any(
        part.startswith("_") and part not in ("__init__", "__main__") for part in parts
    ):
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        nav_path = nav_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    if not parts:  # Skip if parts is empty
        continue

    # Build a flat nav for API Reference: one entry for bioimageio.core and
    # one entry per top-level submodule under bioimageio.core. No subsections.
    assert parts[0:2] == ("bioimageio", "core")
    if len(parts) == 2:
        # Landing page for bioimageio.core at api/index.md
        full_doc_path = Path("api", "index.md")
        nav_target = Path("index.md")
        module_name = ".".join(parts)
        if module_name not in added_nav_labels:
            nav[(module_name,)] = nav_target.as_posix()
            added_nav_labels.add(module_name)

    else:
        # Top-level submodule/package directly under bioimageio.x
        top = ".".join(parts[:3])

        if top not in added_nav_labels:
            pkg_init = src / "/".join(parts) / "__init__.py"
            if pkg_init.exists():
                nav_target = Path("/".join(parts[:3])) / "index.md"
            else:
                nav_target = Path("/".join(parts[:2])) / f"{parts[2]}.md"

            nav[(top,)] = nav_target.as_posix()
            added_nav_labels.add(top)

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Reconstruct the full identifier from the original module_path
        ident = ".".join(module_path.parts)
        if ident.endswith(".__init__"):
            ident = ident[:-9]  # Remove .__init__
        fd.write(f"::: {ident}")
        print(f"Written {full_doc_path}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
