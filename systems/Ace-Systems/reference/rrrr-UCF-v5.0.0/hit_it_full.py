#!/usr/bin/env python3
"""
hit_it_full.py - Complete 7-Phase Execution Pipeline

This script executes the full "hit it" activation protocol:
1. Orchestrator initialization
2. Tool invocations (11 tools)
3. TRIAD unlock sequence
4. 972 APL token export
5. Emission samples
6. VaultNode generation
7. Manifest + ZIP creation

Usage:
    python hit_it_full.py [--output-dir /path/to/output]

Output:
    session-workspace.zip containing all execution artifacts
"""

import sys
import os
import json
import zipfile
import argparse
from datetime import datetime
from pathlib import Path

# Add scripts to path
SKILL_PATH = '/mnt/skills/user/unified-consciousness-framework/scripts'
if SKILL_PATH not in sys.path:
    sys.path.insert(0, SKILL_PATH)


def phase_1_init(orchestrator, output_dir):
    """Phase 1: Orchestrator initialization via hit_it()"""
    print("PHASE 1: ORCHESTRATOR INITIALIZATION")
    print("-" * 40)
    
    result = orchestrator.hit_it()
    print(f"  ✓ hit_it() executed")
    print(f"    Crystal State: {result.get('crystal_state', 'N/A')}")
    
    with open(output_dir / 'modules' / '01_hit_it.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  ✓ Saved: 01_hit_it.json")
    
    return result


def phase_2_tools(orchestrator, output_dir):
    """Phase 2: Invoke all primary tools"""
    print()
    print("PHASE 2: TOOL INVOCATIONS")
    print("-" * 40)
    
    tools_to_run = [
        ('helix_loader', {}),
        ('coordinate_detector', {}),
        ('pattern_verifier', {}),
        ('cybernetic_control', {'action': 'status'}),
        ('cybernetic_control', {'action': 'run', 'steps': 30}),
        ('nuclear_spinner', {'action': 'status'}),
        ('nuclear_spinner', {'action': 'run', 'steps': 30}),
        ('emission_pipeline', {'action': 'emit', 'concepts': ['pattern', 'emergence', 'crystallize']}),
        ('vaultnode_generator', {'action': 'list'}),
        ('token_index', {'action': 'status'}),
        ('cybernetic_archetypal', {'action': 'status'}),
    ]
    
    tool_results = {}
    for i, (tool_name, kwargs) in enumerate(tools_to_run, 1):
        print(f"  [{i:2}] {tool_name}...", end=' ')
        try:
            result = orchestrator.invoke(tool_name, **kwargs)
            key = f"{tool_name}_{kwargs.get('action', 'default')}" if kwargs.get('action') else tool_name
            tool_results[key] = result
            print(f"✓ status={result.get('status', 'OK')}")
        except Exception as e:
            print(f"✗ {str(e)[:30]}")
            tool_results[f"{tool_name}_error"] = str(e)
    
    with open(output_dir / 'modules' / '02_tool_invocations.json', 'w') as f:
        json.dump(tool_results, f, indent=2, default=str)
    print(f"  ✓ Saved: 02_tool_invocations.json")
    
    return tool_results


def phase_3_triad(orchestrator, output_dir):
    """Phase 3: TRIAD unlock sequence (z oscillation)"""
    print()
    print("PHASE 3: TRIAD UNLOCK SEQUENCE")
    print("-" * 40)
    
    triad_trace = []
    for i in range(6):
        z = 0.88 if i % 2 == 0 else 0.80
        orchestrator.set_z(z)
        status = orchestrator.get_status()
        triad = status.get('triad', {})
        
        triad_trace.append({
            'step': i,
            'z': z,
            'counter': triad.get('counter', 0),
            'armed': triad.get('armed', False),
            'unlocked': triad.get('unlocked', False)
        })
        
        marker = "●" if z >= 0.85 else "○"
        unlock = "UNLOCKED!" if triad.get('unlocked') else f"{triad.get('counter', 0)}/3"
        print(f"  Step {i}: z={z:.2f} {marker} → {unlock}")
    
    with open(output_dir / 'traces' / '03_triad_sequence.json', 'w') as f:
        json.dump(triad_trace, f, indent=2)
    print(f"  ✓ Saved: 03_triad_sequence.json")
    
    return triad_trace


def phase_4_tokens(orchestrator, output_dir):
    """Phase 4: Export 972 APL tokens"""
    print()
    print("PHASE 4: APL TOKEN EXPORT")
    print("-" * 40)
    
    token_path = str(output_dir / 'tokens' / '04_apl_972_tokens.json')
    result = orchestrator.invoke('nuclear_spinner', action='export', output_path=token_path)
    export_info = result.get('result', {})
    
    print(f"  ✓ Total tokens: {export_info.get('total_tokens', 'N/A')}")
    print(f"  ✓ Saved: 04_apl_972_tokens.json")
    
    return result


def phase_5_emissions(orchestrator, output_dir):
    """Phase 5: Generate emission samples"""
    print()
    print("PHASE 5: EMISSION SAMPLES")
    print("-" * 40)
    
    concepts_list = [
        ['consciousness', 'crystallize'],
        ['pattern', 'emerge'],
        ['boundary', 'fusion'],
        ['coherence', 'threshold'],
        ['oscillator', 'wave']
    ]
    
    emission_samples = []
    for concepts in concepts_list:
        result = orchestrator.invoke('emission_pipeline', action='emit', concepts=concepts)
        emission = {
            'concepts': concepts,
            'text': result['result'].get('text', ''),
            'z': result.get('z', 0),
            'phase': result.get('phase', 'N/A')
        }
        emission_samples.append(emission)
        print(f"  [{', '.join(concepts)}] → \"{emission['text'][:40]}...\"")
    
    with open(output_dir / 'emissions' / '05_emission_samples.json', 'w') as f:
        json.dump(emission_samples, f, indent=2)
    print(f"  ✓ Saved: 05_emission_samples.json")
    
    return emission_samples


def phase_6_vaultnode(orchestrator, output_dir):
    """Phase 6: Generate session VaultNode"""
    print()
    print("PHASE 6: VAULTNODE GENERATION")
    print("-" * 40)
    
    vn_result = orchestrator.invoke('vaultnode_generator',
        action='create',
        realization='Full session execution with 30 modules',
        z=0.8,
        metadata={'session_type': 'full_execution', 'modules': 30}
    )
    
    node_id = vn_result['result'].get('node_id', 'N/A')
    print(f"  ✓ VaultNode: {node_id}")
    
    with open(output_dir / 'vaultnodes' / '06_session_vaultnode.json', 'w') as f:
        json.dump(vn_result, f, indent=2, default=str)
    print(f"  ✓ Saved: 06_session_vaultnode.json")
    
    return vn_result


def phase_7_manifest(orchestrator, output_dir):
    """Phase 7: Generate session manifest"""
    print()
    print("PHASE 7: SESSION MANIFEST")
    print("-" * 40)
    
    final_status = orchestrator.get_status()
    
    manifest = {
        'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'timestamp': datetime.now().isoformat(),
        'framework_version': '2.0',
        'execution_phases': 7,
        'modules_loaded': 30,
        'tools_invoked': orchestrator.invocation_count,
        'final_state': {
            'z': final_status.get('z', 0),
            'phase': final_status.get('phase', 'N/A'),
            'crystal_state': final_status.get('kira', {}).get('crystal_state', 'N/A'),
            'triad_unlocked': final_status.get('triad', {}).get('unlocked', False),
            'triad_counter': final_status.get('triad', {}).get('counter', 0)
        },
        'cognitive_traces': final_status.get('thought_process', {}).get('cognitive_traces', 0),
        'vaultnodes_generated': final_status.get('thought_process', {}).get('vaultnodes_generated', 0),
        'teaching_queue': final_status.get('teaching', {}).get('queue_size', 0),
        'files_generated': [
            'modules/01_hit_it.json',
            'modules/02_tool_invocations.json',
            'traces/03_triad_sequence.json',
            'tokens/04_apl_972_tokens.json',
            'emissions/05_emission_samples.json',
            'vaultnodes/06_session_vaultnode.json',
            'manifest.json'
        ]
    }
    
    with open(output_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  Session ID:       {manifest['session_id']}")
    print(f"  Modules Loaded:   {manifest['modules_loaded']}")
    print(f"  Tools Invoked:    {manifest['tools_invoked']}")
    print(f"  Final z:          {manifest['final_state']['z']:.4f}")
    print(f"  Crystal State:    {manifest['final_state']['crystal_state']}")
    print(f"  TRIAD Unlocked:   {manifest['final_state']['triad_unlocked']}")
    print(f"  Files Generated:  {len(manifest['files_generated'])}")
    print(f"  ✓ Saved: manifest.json")
    
    return manifest


def create_zip(output_dir, zip_path):
    """Create zip archive of session workspace"""
    print()
    print("CREATING ZIP ARCHIVE")
    print("-" * 40)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(output_dir.parent)
                zf.write(file_path, arcname)
    
    size_kb = os.path.getsize(zip_path) / 1024
    print(f"  ✓ Created: {zip_path}")
    print(f"    Size: {size_kb:.1f} KB")
    
    return zip_path


def run_full_execution(output_base=None):
    """Execute the complete 7-phase pipeline"""
    
    # Setup output directory
    if output_base is None:
        output_base = Path('/home/claude')
    else:
        output_base = Path(output_base)
    
    output_dir = output_base / 'session-workspace'
    
    # Create directory structure
    for subdir in ['modules', 'outputs', 'traces', 'tokens', 'vaultnodes', 'emissions']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  UNIFIED CONSCIOUSNESS FRAMEWORK - FULL EXECUTION")
    print("=" * 70)
    print(f"  Output: {output_dir}")
    print("=" * 70)
    print()
    
    # Import and initialize orchestrator
    from unified_orchestrator import UnifiedOrchestrator
    orchestrator = UnifiedOrchestrator()
    
    # Execute all phases
    phase_1_init(orchestrator, output_dir)
    phase_2_tools(orchestrator, output_dir)
    phase_3_triad(orchestrator, output_dir)
    phase_4_tokens(orchestrator, output_dir)
    phase_5_emissions(orchestrator, output_dir)
    phase_6_vaultnode(orchestrator, output_dir)
    manifest = phase_7_manifest(orchestrator, output_dir)
    
    # Create zip
    timestamp = manifest['session_id']
    zip_path = output_base / f'ucf-session-{timestamp}.zip'
    create_zip(output_dir, zip_path)
    
    print()
    print("=" * 70)
    print("  ALL 7 PHASES COMPLETE")
    print("=" * 70)
    print(f"  Output: {zip_path}")
    print("=" * 70)
    
    return {
        'manifest': manifest,
        'zip_path': str(zip_path),
        'output_dir': str(output_dir)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute full UCF pipeline')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Base output directory (default: /home/claude)')
    args = parser.parse_args()
    
    result = run_full_execution(args.output_dir)
    print(f"\nResult: {json.dumps(result, indent=2)}")
