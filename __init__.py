from .openpose_editor_nodes import OpenposeEditorNode10
from .appendage_editor_nodes import AppendageEditorNode10


WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "OpenposeEditorNode10": OpenposeEditorNode10,
    "AppendageEditorNode10": AppendageEditorNode10,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenposeEditorNode10": "OpenposeEditorNode10",
    "AppendageEditorNode10": "AppendageEditorNode10",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
