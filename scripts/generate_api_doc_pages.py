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
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    # Skip if this is just the bioimageio namespace package
    if parts == ("bioimageio",):
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    if not parts:  # Skip if parts is empty
        continue

    # Build a flat nav for API Reference: one entry for bioimageio.core and
    # one entry per top-level submodule under bioimageio.core. No subsections.
    if parts[0:2] == ("bioimageio", "core"):
        if len(parts) == 2:
            # Landing page for bioimageio.core at reference/index.md
            full_doc_path = Path("reference", "index.md")
            doc_path = Path("index.md")
            if "bioimageio.core" not in added_nav_labels:
                nav[("bioimageio.core",)] = doc_path.as_posix()
                added_nav_labels.add("bioimageio.core")
        else:
            # Top-level submodule/package directly under bioimageio.core
            top = parts[2]
            if top not in added_nav_labels:
                pkg_init = src / "bioimageio" / "core" / top / "__init__.py"
                if pkg_init.exists():
                    nav_target = Path("bioimageio") / "core" / top / "index.md"
                else:
                    nav_target = Path("bioimageio") / "core" / f"{top}.md"

                nav[(top,)] = nav_target.as_posix()
                added_nav_labels.add(top)

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Reconstruct the full identifier from the original module_path
        ident = ".".join(module_path.parts)
        if ident.endswith(".__init__"):
            ident = ident[:-9]  # Remove .__init__
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
