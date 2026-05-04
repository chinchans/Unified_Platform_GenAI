"""
UI Integration Helper for Embedding Updates
This module provides integration between UI_v3.py and the embedding updater
"""

from .update_embeddings import update_embedding_after_commit, sync_git_commit_embeddings_for_rca


def sync_git_commit_embeddings_at_rca_start(
    openair_codebase_file_name="openairinterface5g-develop",
    code_dir=None,
    progress_callback=None,
):
    """
    Run once when RCA analysis starts: embed current HEAD if it is newer than
    the last embedded commit for this repository (no duplicate work per commit).
    """
    return sync_git_commit_embeddings_for_rca(
        openair_codebase_file_name=openair_codebase_file_name,
        code_dir=code_dir,
        progress_callback=progress_callback,
    )


def update_embeddings_from_ui(commit_hash, selected_code_patches, selected_config_patches, 
                              progress_callback=None, openair_codebase_file_name='openairinterface5g-develop',
                              code_dir=None):
    """
    Update embeddings after a UI commit
    
    Args:
        commit_hash: Git commit hash (e.g., "abc123...")
        selected_code_patches: List of patch names or dicts
        selected_config_patches: List of patch names or dicts
        progress_callback: Optional callback(message) for progress updates
        openair_codebase_file_name: Name of the OpenAirInterface5G codebase folder
        code_dir: Optional absolute path to the Git repository (if provided, will be used instead of searching)
    
    Returns:
        (success: bool, message: str)
    """
    
    # Format patches for embedding updater
    code_patches = []
    for patch in selected_code_patches:
        if isinstance(patch, dict):
            # Already a dict
            code_patches.append({
                'function': patch.get('function_name', patch.get('function', patch.get('title', 'unknown'))),
                'file': patch.get('file_path', patch.get('file', 'unknown'))
            })
        elif isinstance(patch, str):
            # Just a string (patch name)
            # Try to extract function and file from string
            # Format might be: "function_name (file.c)" or just "function_name"
            if ' (' in patch and patch.endswith(')'):
                parts = patch.rsplit(' (', 1)
                function = parts[0]
                file = parts[1].rstrip(')')
            else:
                function = patch
                file = 'unknown'
            
            code_patches.append({
                'function': function,
                'file': file
            })
    
    config_patches = []
    for patch in selected_config_patches:
        if isinstance(patch, dict):
            # Already a dict
            config_patches.append({
                'parameter': patch.get('parameter_name', patch.get('parameter', patch.get('title', 'unknown'))),
                'file': patch.get('file_path', patch.get('file', 'unknown'))
            })
        elif isinstance(patch, str):
            # Just a string (patch name)
            # Format might be: "parameter_name (file.conf)" or just "parameter_name"
            if ' (' in patch and patch.endswith(')'):
                parts = patch.rsplit(' (', 1)
                parameter = parts[0]
                file = parts[1].rstrip(')')
            else:
                parameter = patch
                file = 'unknown'
            
            config_patches.append({
                'parameter': parameter,
                'file': file
            })
    
    # Update embeddings
    return update_embedding_after_commit(
        commit_hash=commit_hash,
        code_patches=code_patches,
        config_patches=config_patches,
        progress_callback=progress_callback,
        openair_codebase_file_name=openair_codebase_file_name,
        code_dir=code_dir
    )

