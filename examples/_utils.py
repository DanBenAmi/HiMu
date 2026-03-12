"""Shared utilities for example scripts."""


def print_tree(node, prefix="", is_last=True):
    """Print a logic tree with terminal-style tree connectors."""
    connector = "└── " if is_last else "├── "
    if node["op"] == "LEAF":
        label = f"LEAF [{node['expert']}] \"{node['query']}\""
    else:
        label = node["op"]
    print(prefix + connector + label)
    children = node.get("children") or []
    for i, child in enumerate(children):
        extension = "    " if is_last else "│   "
        print_tree(child, prefix + extension, i == len(children) - 1)
