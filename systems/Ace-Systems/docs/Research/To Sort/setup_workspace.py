#!/usr/bin/env python3
"""
UCF Workspace Setup
===================
Loads a UCF session zip archive and prepares the workspace for execution.

Usage:
    python setup_workspace.py [--session-zip /path/to/ucf-session-*.zip]
    python setup_workspace.py --from-skill  # Load from skill directory
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path


def setup_from_zip(zip_path: str, workspace_dir: str = '/home/claude') -> dict:
    """
    Extract UCF session zip and set up workspace.
    
    Returns:
        dict with setup status and initial state
    """
    zip_path = Path(zip_path)
    workspace = Path(workspace_dir)
    
    if not zip_path.exists():
        raise FileNotFoundError(f"Session zip not found: {zip_path}")
    
    # Extract session ID from filename
    session_id = zip_path.stem  # e.g., "ucf-session-20251215_175842"
    extract_dir = workspace / f"{session_id}-extracted"
    
    print(f"[Setup] Loading session: {session_id}")
    print(f"[Setup] Extracting to: {extract_dir}")
    
    # Extract archive
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    
    # Load manifest for initial state
    manifest_path = extract_dir / 'manifest.json'
    initial_state = None
    
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        initial_state = manifest.get('final_state', {})
        print(f"[Setup] Loaded manifest from session: {manifest.get('session_id')}")
        print(f"[Setup] Previous final z: {initial_state.get('z', 'N/A')}")
        print(f"[Setup] TRIAD unlocked: {initial_state.get('triad', {}).get('unlocked', False)}")
    
    return {
        'status': 'ready',
        'session_id': session_id,
        'extract_dir': str(extract_dir),
        'initial_state': initial_state,
        'files_extracted': list(extract_dir.rglob('*'))
    }


def setup_from_skill(skill_dir: str = '/mnt/skills/user/unified-consciousness-framework',
                     workspace_dir: str = '/home/claude') -> dict:
    """
    Set up workspace from skill directory (fresh start).
    
    Returns:
        dict with setup status
    """
    skill_path = Path(skill_dir)
    workspace = Path(workspace_dir)
    
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill directory not found: {skill_path}")
    
    # Copy ucf package
    ucf_src = skill_path / 'ucf'
    ucf_dst = workspace / 'ucf'
    
    if ucf_dst.exists():
        shutil.rmtree(ucf_dst)
    
    shutil.copytree(ucf_src, ucf_dst)
    print(f"[Setup] Copied UCF package to: {ucf_dst}")
    
    # Verify installation
    sys.path.insert(0, str(workspace))
    try:
        from ucf.constants import __version__, Z_CRITICAL, PHI
        print(f"[Setup] UCF version: {__version__}")
        print(f"[Setup] Z_CRITICAL: {Z_CRITICAL}")
        print(f"[Setup] PHI: {PHI}")
        verified = True
    except ImportError as e:
        print(f"[Setup] WARNING: Import failed: {e}")
        verified = False
    
    return {
        'status': 'ready' if verified else 'warning',
        'ucf_path': str(ucf_dst),
        'verified': verified,
        'initial_state': {
            'z': 0.800,
            'phase': 'PARADOX',
            'triad': {'unlocked': False, 'counter': 0}
        }
    }


def create_executor_script(workspace_dir: str = '/home/claude') -> str:
    """
    Create the hit_it_session.py executor in the workspace.
    
    Returns:
        Path to created script
    """
    script_path = Path(workspace_dir) / 'hit_it_session.py'
    
    # The executor script content (abbreviated - full version in separate file)
    script_content = '''#!/usr/bin/env python3
"""
UCF "hit it" Session Executor - See full implementation
"""
print("Executor placeholder - use full hit_it_session.py")
'''
    
    # Note: In practice, copy the full hit_it_session.py
    # This is a placeholder showing the setup process
    
    print(f"[Setup] Executor script location: {script_path}")
    return str(script_path)


def main():
    parser = argparse.ArgumentParser(
        description='UCF Workspace Setup',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--session-zip', type=str,
                      help='Path to UCF session zip archive')
    group.add_argument('--from-skill', action='store_true',
                      help='Set up from skill directory (fresh start)')
    
    parser.add_argument('--workspace', type=str, default='/home/claude',
                       help='Workspace directory (default: /home/claude)')
    parser.add_argument('--skill-dir', type=str,
                       default='/mnt/skills/user/unified-consciousness-framework',
                       help='Skill directory path')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("UCF WORKSPACE SETUP")
    print("=" * 60 + "\n")
    
    try:
        if args.session_zip:
            result = setup_from_zip(args.session_zip, args.workspace)
        else:
            result = setup_from_skill(args.skill_dir, args.workspace)
        
        print("\n" + "-" * 60)
        print("SETUP COMPLETE")
        print("-" * 60)
        print(f"Status: {result['status']}")
        
        if result.get('initial_state'):
            state = result['initial_state']
            print(f"\nInitial State:")
            print(f"  z: {state.get('z', 0.800)}")
            print(f"  Phase: {state.get('phase', 'PARADOX')}")
            triad = state.get('triad', {})
            print(f"  TRIAD: {'UNLOCKED' if triad.get('unlocked') else 'LOCKED'}")
        
        print("\n" + "=" * 60)
        print("Ready for 'hit it' execution")
        print("=" * 60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
