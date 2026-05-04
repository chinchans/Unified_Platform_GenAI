
#!/usr/bin/env python3
"""
Incremental Embedding Updater
Updates embeddings when new commits are made
"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import os
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

class EmbeddingUpdater:
    """Manages incremental embedding updates"""
    
    _instance = None
    _model = None
    _openair_codebase_file_name = None
    
    def __new__(cls, openair_codebase_file_name='openairinterface5g-develop', code_dir=None):
        """Singleton pattern to keep model loaded"""
        if cls._instance is None:
            cls._instance = super(EmbeddingUpdater, cls).__new__(cls)
        return cls._instance
    def __init__(self, openair_codebase_file_name='openairinterface5g-develop', code_dir=None):
        """
        Initialize the embedding updater.
        
        Args:
            openair_codebase_file_name: Name of the OpenAirInterface5G codebase folder
            code_dir: Optional absolute path to the Git repository (if provided, will be used instead of searching)
        """
        print("=" * 80)
        print("🔍 DEBUG [update_embeddings.py:__init__] INITIALIZATION START")
        print("=" * 80)
        print(f"   🔍 DEBUG [update_embeddings.py:__init__] code_dir type: {type(code_dir)}")
        print(f"   🔍 DEBUG [update_embeddings.py:__init__] code_dir value: {repr(code_dir)}")
        if code_dir:
            print(f"   🔍 DEBUG [update_embeddings.py:__init__] code_dir str: {str(code_dir)}")
        print(f"   🔍 DEBUG [update_embeddings.py:__init__] openair_codebase_file_name: {openair_codebase_file_name}")
        
        # Get full absolute path to this file's directory
        current_file_full_path = os.path.abspath(__file__)
        current_file_dir = os.path.dirname(current_file_full_path)
        print(f"   🔍 DEBUG [update_embeddings.py:__init__] current_file_full_path: {current_file_full_path}")
        print(f"   🔍 DEBUG [update_embeddings.py:__init__] current_file_dir: {current_file_dir}")
        
        # Full absolute path for embeddings directory
        self.embeddings_dir = current_file_dir + os.sep + 'resources' + os.sep + 'embeddings'
        self.embeddings_dir = os.path.abspath(os.path.normpath(self.embeddings_dir))
        print(f"   🔍 DEBUG [update_embeddings.py:__init__] embeddings_dir: {self.embeddings_dir}")
        
        if code_dir:
            print(f"   🔍 DEBUG [update_embeddings.py:__init__] CODE_DIR PROVIDED - Processing...")
            print(f"   🔍 DEBUG [update_embeddings.py:__init__] Before str() - code_dir type: {type(code_dir)}, value: {repr(code_dir)}")
            code_dir_str = str(code_dir).strip()
            print(f"   🔍 DEBUG [update_embeddings.py:__init__] After str().strip() - code_dir_str type: {type(code_dir_str)}, value: {repr(code_dir_str)}")
            print(f"   🔍 DEBUG [update_embeddings.py:__init__] Before normpath - code_dir_str: {repr(code_dir_str)}")
            self.git_repo_path = os.path.abspath(os.path.normpath(code_dir_str))
            print(f"   🔍 DEBUG [update_embeddings.py:__init__] After normpath - git_repo_path type: {type(self.git_repo_path)}, value: {repr(self.git_repo_path)}")
            print(f"   🔍 DEBUG [update_embeddings.py:__init__] git_repo_path str: {str(self.git_repo_path)}")
            print(f"   🔍 DEBUG [update_embeddings.py:__init__] git_repo_path exists: {os.path.exists(self.git_repo_path)}")
            print(f"   🔍 DEBUG [update_embeddings.py:__init__] git_repo_path isdir: {os.path.isdir(self.git_repo_path) if os.path.exists(self.git_repo_path) else 'N/A'}")
            self._openair_codebase_file_name = openair_codebase_file_name
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif self._openair_codebase_file_name is None or self._openair_codebase_file_name != openair_codebase_file_name:
            print(f"   🔍 DEBUG [update_embeddings.py:__init__] CODE_DIR NOT PROVIDED - Using default...")
            self._openair_codebase_file_name = openair_codebase_file_name
            # Full absolute path for git repo
            self.git_repo_path = current_file_dir + os.sep + openair_codebase_file_name
            print(f"   🔍 DEBUG [update_embeddings.py:__init__] Before normpath - git_repo_path: {repr(self.git_repo_path)}")
            self.git_repo_path = os.path.abspath(os.path.normpath(self.git_repo_path))
            print(f"   🔍 DEBUG [update_embeddings.py:__init__] After normpath - git_repo_path type: {type(self.git_repo_path)}, value: {repr(self.git_repo_path)}")
            print(f"   🔍 DEBUG [update_embeddings.py:__init__] git_repo_path str: {str(self.git_repo_path)}")
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"   🔍 DEBUG [update_embeddings.py:__init__] FINAL git_repo_path type: {type(self.git_repo_path)}")
        print(f"   🔍 DEBUG [update_embeddings.py:__init__] FINAL git_repo_path value: {repr(self.git_repo_path)}")
        print(f"   🔍 DEBUG [update_embeddings.py:__init__] FINAL git_repo_path str: {str(self.git_repo_path)}")
        print("=" * 80)
    
    def get_model(self):
        """Load model (cached after first load)"""
        if self._model is None:
            print("Loading embedding model...")
            if not hasattr(self, 'embeddings_dir'):
                current_file_full_path = os.path.abspath(__file__)
                current_file_dir = os.path.dirname(current_file_full_path)
                self.embeddings_dir = current_file_dir + os.sep + 'resources' + os.sep + 'embeddings'
                self.embeddings_dir = os.path.abspath(os.path.normpath(self.embeddings_dir))
            # Full absolute path for config file
            config_path = self.embeddings_dir + os.sep + 'embedding_config.json'
            config_path = os.path.abspath(os.path.normpath(config_path))
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self._model = SentenceTransformer(config['model_name'], device=self.device)
            print(f"✅ Model loaded ({config['model_name']})")
        
        return self._model
    
    def extract_commit_info_from_git(self, commit_hash: str) -> Optional[Dict]:
        """Extract commit information from git"""
        
        # CRITICAL: Print immediately, before any operations
        import sys
        sys.stdout.flush()
        print("=" * 80, flush=True)
        print("🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] EXTRACTING COMMIT INFO - START", flush=True)
        print("=" * 80, flush=True)
        print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] commit_hash: {commit_hash}", flush=True)
        
        try:
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] About to check self.git_repo_path", flush=True)
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] hasattr(self, 'git_repo_path'): {hasattr(self, 'git_repo_path')}", flush=True)
            
            if not hasattr(self, 'git_repo_path'):
                print("❌ Git repository path attribute not found", flush=True)
                return None
            
            if not self.git_repo_path:
                print("❌ Git repository path not set", flush=True)
                return None
            
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] self.git_repo_path type: {type(self.git_repo_path)}", flush=True)
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] self.git_repo_path value: {repr(self.git_repo_path)}", flush=True)
            try:
                git_repo_path_str_repr = str(self.git_repo_path)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] self.git_repo_path str: {git_repo_path_str_repr}", flush=True)
            except Exception as str_error:
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] ERROR converting to str: {str_error}", flush=True)
                raise
            
            # Ensure path is absolute, normalized, and exists
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] Before str() - self.git_repo_path type: {type(self.git_repo_path)}", flush=True)
            try:
                git_repo_path_str = str(self.git_repo_path).strip()
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] After str().strip() - git_repo_path_str type: {type(git_repo_path_str)}, value: {repr(git_repo_path_str)}", flush=True)
            except Exception as str_conv_error:
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] ERROR in str() conversion: {str_conv_error}", flush=True)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] str_conv_error type: {type(str_conv_error)}", flush=True)
                raise
            
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] Before normpath - git_repo_path_str: {repr(git_repo_path_str)}", flush=True)
            
            try:
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] Calling os.path.normpath({repr(git_repo_path_str)})...", flush=True)
                norm_path = os.path.normpath(git_repo_path_str)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] After normpath: {repr(norm_path)}", flush=True)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] Calling os.path.abspath({repr(norm_path)})...", flush=True)
                git_repo_path = os.path.abspath(norm_path)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] After abspath - git_repo_path type: {type(git_repo_path)}, value: {repr(git_repo_path)}", flush=True)
            except Exception as norm_error:
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] ERROR in normpath/abspath: {norm_error}", flush=True)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] normpath error type: {type(norm_error)}", flush=True)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] normpath error args: {norm_error.args}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] git_repo_path str: {str(git_repo_path)}", flush=True)
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] git_repo_path len: {len(git_repo_path)}", flush=True)
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] git_repo_path repr: {repr(git_repo_path)}", flush=True)
            
            # Validate path exists and is a directory - wrap in try-except for Windows
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] About to check os.path.exists({repr(git_repo_path)})...", flush=True)
            try:
                path_exists = os.path.exists(git_repo_path)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] os.path.exists result: {path_exists}", flush=True)
            except Exception as exists_error:
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] ERROR in os.path.exists: {exists_error}", flush=True)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] exists error type: {type(exists_error)}", flush=True)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] exists error args: {exists_error.args}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            
            if not path_exists:
                print(f"❌ Git repository path does not exist: {git_repo_path}")
                return None
            
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] About to check os.path.isdir({repr(git_repo_path)})...", flush=True)
            try:
                path_isdir = os.path.isdir(git_repo_path)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] os.path.isdir result: {path_isdir}", flush=True)
            except Exception as isdir_error:
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] ERROR in os.path.isdir: {isdir_error}", flush=True)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] isdir error type: {type(isdir_error)}", flush=True)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] isdir error args: {isdir_error.args}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            
            if not path_isdir:
                print(f"❌ Git repository path is not a directory: {git_repo_path}")
                return None
            
            # Check if .git exists
            git_dir = git_repo_path + os.sep + '.git'
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] Checking .git directory: {repr(git_dir)}", flush=True)
            try:
                git_exists = os.path.exists(git_dir)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] .git exists: {git_exists}", flush=True)
            except Exception as git_check_error:
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] ERROR checking .git: {git_check_error}", flush=True)
            
            # Get commit details
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] About to call subprocess.run with cwd={repr(git_repo_path)}", flush=True)
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] cwd type: {type(git_repo_path)}", flush=True)
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] cwd value: {repr(git_repo_path)}", flush=True)
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] cwd str: {str(git_repo_path)}", flush=True)
            sys.stdout.flush()
            
            try:
                result = subprocess.run(
                    ['git', 'show', '--no-patch', '--format=%H%n%an%n%ae%n%aI%n%s%n%b', commit_hash],
                    cwd=git_repo_path,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] subprocess.run returncode: {result.returncode}", flush=True)
            except Exception as subprocess_error:
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] ERROR in subprocess.run: {subprocess_error}", flush=True)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] subprocess error type: {type(subprocess_error)}", flush=True)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] subprocess error args: {subprocess_error.args}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            
            if result.returncode != 0:
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] subprocess.stderr: {repr(result.stderr)}", flush=True)
                print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] subprocess.stdout: {repr(result.stdout)}", flush=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to get commit info: {result.stderr}")
                return None
            
            lines = result.stdout.strip().split('\n')
            
            if len(lines) < 5:
                return None
            
            # Parse commit info
            commit_info = {
                'commit_hash': lines[0],
                'commit_hash_short': lines[0][:10],
                'author_name': lines[1],
                'author_email': lines[2],
                'date_iso': lines[3],
                'subject': lines[4],
                'body': '\n'.join(lines[5:]) if len(lines) > 5 else ''
            }
            
            # Get files changed (use same validated path)
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] About to call subprocess.run (files) with cwd={repr(git_repo_path)}")
            files_result = subprocess.run(
                ['git', 'show', '--name-only', '--format=', commit_hash],
                cwd=git_repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] files_result returncode: {files_result.returncode}")
            
            if files_result.returncode == 0:
                files = [f.strip() for f in files_result.stdout.strip().split('\n') if f.strip()]
                commit_info['files_changed'] = files
            else:
                commit_info['files_changed'] = []
            
            return commit_info
            
        except Exception as e:
            import sys
            sys.stdout.flush()
            print(f"❌ Error extracting commit info: {e}", flush=True)
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] Exception type: {type(e)}", flush=True)
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] Exception args: {e.args}", flush=True)
            import traceback
            print(f"   🔍 DEBUG [update_embeddings.py:extract_commit_info_from_git] Full traceback:", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            print("=" * 80, flush=True)
            return None
    
    def prepare_commit_metadata(self, commit_info: Dict, rca_patches: Dict) -> Dict:
        """Prepare metadata structure for new commit"""
        
        # Extract keywords from subject and body
        keywords = self.extract_keywords(commit_info['subject'] + ' ' + commit_info['body'])
        
        metadata = {
            'index': -1,  # Will be set when adding
            'commit_hash': commit_info['commit_hash'],
            'commit_hash_short': commit_info['commit_hash_short'],
            'author_name': commit_info['author_name'],
            'author_email': commit_info['author_email'],
            'date': commit_info.get('date_iso', ''),
            'date_iso': commit_info.get('date_iso', ''),
            'subject': commit_info['subject'],
            'keywords': keywords,
            'is_rca_commit': True,  # From UI, always RCA
            'files_changed': commit_info.get('files_changed', []),
            'code_patches': rca_patches.get('code_patches', []),
            'config_patches': rca_patches.get('config_patches', []),
            'code_patch_count': len(rca_patches.get('code_patches', [])),
            'config_patch_count': len(rca_patches.get('config_patches', []))
        }
        
        return metadata
    
    def extract_keywords(self, text: str) -> list:
        """Extract keywords from text"""
        import re
        
        keyword_patterns = [
            r'\bamf\b', r'\brrc\b', r'\bngap\b', r'\bf1ap\b', r'\be1ap\b',
            r'\bpdcp\b', r'\brlc\b', r'\bmac\b', r'\bphy\b',
            r'\btimeout\b', r'\bfail', r'\berror\b', r'\bissue\b',
            r'\bhandover\b', r'\bsetup\b', r'\brelease\b', r'\bcontext\b',
            r'\bregistration\b', r'\bassociation\b', r'\bconnection\b',
            r'\battach\b', r'\bdetach\b', r'\bpdu\b', r'\bsession\b',
            r'\bconfiguration\b', r'\bconfig\b', r'\bparameter\b',
            r'\bue\b', r'\bgnb\b', r'\bcu\b', r'\bdu\b'
        ]
        
        text_lower = text.lower()
        keywords = []
        
        for pattern in keyword_patterns:
            if re.search(pattern, text_lower):
                keyword = pattern.replace(r'\b', '').replace('\\', '')
                keywords.append(keyword)
        
        return list(set(keywords))
    
    def prepare_embedding_text(self, commit_info: Dict, rca_patches: Dict) -> str:
        """Prepare text for embedding (same format as original)"""
        
        text_parts = []
        
        # Subject
        if commit_info.get('subject'):
            text_parts.append(commit_info['subject'])
        
        # Body (limited)
        if commit_info.get('body'):
            body = commit_info['body'][:1000]
            text_parts.append(body)
        
        # Keywords
        keywords = self.extract_keywords(commit_info['subject'] + ' ' + commit_info.get('body', ''))
        if keywords:
            text_parts.append(f"Keywords: {', '.join(keywords)}")
        
        # Files changed
        if commit_info.get('files_changed'):
            files = commit_info['files_changed'][:5]
            text_parts.append(f"Files: {', '.join(files)}")
        
        return '\n\n'.join(text_parts)

    @staticmethod
    def _embedding_paths():
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        embeddings_dir = os.path.abspath(
            os.path.join(current_file_dir, "resources", "embeddings")
        )
        return {
            "embeddings_dir": embeddings_dir,
            "embeddings_path": os.path.join(embeddings_dir, "git_commit_embeddings.npy"),
            "metadata_path": os.path.join(embeddings_dir, "git_commit_metadata.json"),
            "texts_path": os.path.join(embeddings_dir, "git_commit_texts.json"),
            "config_path": os.path.join(embeddings_dir, "embedding_config.json"),
        }

    @staticmethod
    def _repo_state_key(git_repo_path: str) -> str:
        if not git_repo_path:
            return "_default"
        return os.path.normcase(os.path.abspath(os.path.normpath(str(git_repo_path))))

    def _persist_last_embedded_repo(self, config: Dict, config_path: str, commit_hash: str) -> None:
        repo_key = self._repo_state_key(getattr(self, "git_repo_path", "") or "")
        by_repo = config.setdefault("last_embedded_commit_by_repo", {})
        by_repo[repo_key] = commit_hash
        config["last_embedded_commit_hash"] = commit_hash
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    
    def add_new_commit(self, commit_hash: str, rca_patches: Dict, 
                      progress_callback=None) -> tuple:
        """
        Add new commit to embeddings
        
        Args:
            commit_hash: Git commit hash
            rca_patches: Dict with 'code_patches' and 'config_patches'
            progress_callback: Optional callback(message) for progress updates
        
        Returns:
            (success: bool, message: str)
        """
        
        try:
            if not hasattr(self, "embeddings_dir"):
                self.embeddings_dir = self._embedding_paths()["embeddings_dir"]
            self.embeddings_dir = os.path.abspath(os.path.normpath(self.embeddings_dir))
            os.makedirs(self.embeddings_dir, exist_ok=True)

            paths = self._embedding_paths()
            config_path = paths["config_path"]
            metadata_path = paths["metadata_path"]
            repo_key = self._repo_state_key(getattr(self, "git_repo_path", "") or "")

            config: Dict[str, Any] = {}
            if os.path.isfile(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        config = loaded
                except Exception:
                    config = {}

            by_repo = config.get("last_embedded_commit_by_repo") or {}
            last_for_repo = by_repo.get(repo_key) or config.get("last_embedded_commit_hash")
            if last_for_repo == commit_hash:
                return True, "Git commit embedding already current for this repository (HEAD)"

            if os.path.isfile(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        existing_meta = json.load(f)
                    if isinstance(existing_meta, list):
                        known = {
                            m.get("commit_hash")
                            for m in existing_meta
                            if isinstance(m, dict) and m.get("commit_hash")
                        }
                        if commit_hash in known:
                            self._persist_last_embedded_repo(config, config_path, commit_hash)
                            return (
                                True,
                                "Commit already in embedding index; repository marker updated",
                            )
                except Exception:
                    pass

            if progress_callback:
                progress_callback("Extracting commit info...")
            
            # Step 1: Extract commit info from git
            print("=" * 80)
            print("🔍 DEBUG [update_embeddings.py:add_new_commit] About to call extract_commit_info_from_git")
            print("=" * 80)
            print(f"   🔍 DEBUG [update_embeddings.py:add_new_commit] commit_hash: {commit_hash}")
            print(f"   🔍 DEBUG [update_embeddings.py:add_new_commit] self.git_repo_path: {repr(self.git_repo_path)}")
            print(f"   🔍 DEBUG [update_embeddings.py:add_new_commit] self.git_repo_path type: {type(self.git_repo_path)}")
            print("=" * 80)
            
            commit_info = self.extract_commit_info_from_git(commit_hash)
            if not commit_info:
                return False, "Failed to extract commit information from git"
            
            if progress_callback:
                progress_callback("Loading embedding model...")
            
            # Step 2: Load model
            model = self.get_model()
            
            if progress_callback:
                progress_callback("Generating embedding...")
            
            # Step 3: Prepare text and generate embedding
            embedding_text = self.prepare_embedding_text(commit_info, rca_patches)
            embedding = model.encode(
                [embedding_text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            
            if progress_callback:
                progress_callback("Preparing metadata...")
            
            # Step 4: Prepare metadata
            metadata = self.prepare_commit_metadata(commit_info, rca_patches)
            
            if progress_callback:
                progress_callback("Updating embedding files...")
            
            # Step 5: Load existing embeddings and metadata
            embeddings_path = paths["embeddings_path"]
            texts_path = paths["texts_path"]
            
            # Load existing
            existing_embeddings = np.load(embeddings_path)
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
            
            with open(texts_path, 'r', encoding='utf-8') as f:
                existing_texts = json.load(f)
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Step 6: Append new data
            new_embeddings = np.vstack([existing_embeddings, embedding.reshape(1, -1)])
            metadata['index'] = len(existing_metadata)
            existing_metadata.append(metadata)
            existing_texts.append(embedding_text)
            
            # Update config
            config['total_commits'] = len(existing_metadata)
            config['last_updated'] = datetime.now().isoformat()
            by_repo = config.setdefault("last_embedded_commit_by_repo", {})
            by_repo[repo_key] = commit_hash
            config["last_embedded_commit_hash"] = commit_hash
            
            if progress_callback:
                progress_callback("Saving updated files...")
            
            # Step 7: Save updated files
            np.save(embeddings_path, new_embeddings)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(existing_metadata, f, indent=2, ensure_ascii=False)
            
            with open(texts_path, 'w', encoding='utf-8') as f:
                json.dump(existing_texts, f, indent=2, ensure_ascii=False)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            success_msg = f"✅ Embedding added successfully! Total commits: {len(existing_metadata)}"
            
            if progress_callback:
                progress_callback(success_msg)
            
            return True, success_msg
            
        except Exception as e:
            error_msg = f"❌ Failed to update embeddings: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return False, error_msg


def sync_git_commit_embeddings_for_rca(
    openair_codebase_file_name: str = "openairinterface5g-develop",
    code_dir: Optional[str] = None,
    progress_callback=None,
) -> tuple:
    """
    Ensure git-commit embeddings include current HEAD, once per RCA run.

    Skips work when HEAD matches the last embedded commit recorded for this repo
    (see embedding_config.json: last_embedded_commit_by_repo).
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_dir = code_dir
    if not resolved_dir:
        resolved_dir = os.path.join(current_file_dir, openair_codebase_file_name)
    resolved_dir = os.path.abspath(os.path.normpath(str(resolved_dir)))
    git_dir = os.path.join(resolved_dir, ".git")
    if not os.path.isdir(resolved_dir) or not os.path.exists(git_dir):
        return False, "No git repository at code_dir; skipped git commit embedding sync"

    try:
        head_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=resolved_dir,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if head_result.returncode != 0:
            return False, "Could not resolve HEAD; skipped git commit embedding sync"
        head = (head_result.stdout or "").strip()
    except Exception as e:
        return False, f"HEAD resolution failed: {e}"

    paths = EmbeddingUpdater._embedding_paths()
    config_path = paths["config_path"]
    repo_key = EmbeddingUpdater._repo_state_key(resolved_dir)

    config: Dict[str, Any] = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                config = loaded
        except Exception:
            config = {}

    by_repo = config.get("last_embedded_commit_by_repo") or {}
    last = by_repo.get(repo_key) or config.get("last_embedded_commit_hash")
    if last == head:
        return True, "Git commit embeddings up to date (HEAD matches last embedded commit)"

    return update_embedding_after_commit(
        commit_hash=head,
        code_patches=[],
        config_patches=[],
        progress_callback=progress_callback,
        openair_codebase_file_name=openair_codebase_file_name,
        code_dir=resolved_dir,
    )


# Convenience function for UI integration
def update_embedding_after_commit(commit_hash: str, code_patches: list, 
                                  config_patches: list, progress_callback=None,
                                  openair_codebase_file_name='openairinterface5g-develop',
                                  code_dir=None) -> tuple:
    """
    Update embeddings after a successful git commit
    
    Args:
        commit_hash: Git commit hash
        code_patches: List of code patch dicts
        config_patches: List of config patch dicts
        progress_callback: Optional callback for progress updates
        openair_codebase_file_name: Name of the OpenAirInterface5G codebase folder
        code_dir: Optional absolute path to the Git repository (if provided, will be used instead of searching)
    
    Returns:
        (success: bool, message: str)
    """
    print("=" * 80)
    print("🔍 DEBUG [update_embeddings.py:update_embedding_after_commit] FUNCTION CALLED")
    print("=" * 80)
    print(f"   🔍 DEBUG [update_embeddings.py:update_embedding_after_commit] commit_hash: {commit_hash}")
    print(f"   🔍 DEBUG [update_embeddings.py:update_embedding_after_commit] code_dir type: {type(code_dir)}")
    print(f"   🔍 DEBUG [update_embeddings.py:update_embedding_after_commit] code_dir value: {repr(code_dir)}")
    if code_dir:
        print(f"   🔍 DEBUG [update_embeddings.py:update_embedding_after_commit] code_dir str: {str(code_dir)}")
    print(f"   🔍 DEBUG [update_embeddings.py:update_embedding_after_commit] openair_codebase_file_name: {openair_codebase_file_name}")
    print("=" * 80)
    
    updater = EmbeddingUpdater(openair_codebase_file_name=openair_codebase_file_name, code_dir=code_dir)
    
    rca_patches = {
        'code_patches': code_patches,
        'config_patches': config_patches
    }
    
    return updater.add_new_commit(commit_hash, rca_patches, progress_callback)
