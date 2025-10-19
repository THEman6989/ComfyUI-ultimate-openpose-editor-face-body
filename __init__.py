from .openpose_editor_nodes import OpenposeEditorNode
from .appendage_editor_nodes import AppendageEditorNode


WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "OpenposeEditorNode10": OpenposeEditorNode,
    "AppendageEditorNode10": AppendageEditorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenposeEditorNode10": "OpenposeEditorNode10",
    "AppendageEditorNode10": "AppendageEditorNode10",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
