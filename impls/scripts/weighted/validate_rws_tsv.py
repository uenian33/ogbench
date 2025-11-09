#!/usr/bin/env python3
"""
validate_rws_tsv.py - Validate and fix common issues in RWS-weighted TSV files
"""

import sys
import argparse
from collections import defaultdict

def validate_tsv(filename):
    """Validate TSV file and report issues."""
    print(f"Validating: {filename}")
    print("=" * 60)
    
    issues = []
    stats = defaultdict(lambda: defaultdict(int))
    lines_data = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Check header
    if not lines or not lines[0].startswith('#'):
        issues.append("Missing header line (should start with #)")
    else:
        header = lines[0].strip('# \n').split('\t')
        expected_cols = 9
        if len(header) != expected_cols:
            issues.append(f"Header has {len(header)} columns, expected {expected_cols}")
        print(f"Header columns: {header}")
    
    # Parse data lines
    data_lines = []
    for i, line in enumerate(lines[1:], 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split('\t')
        if len(parts) != 9:
            issues.append(f"Line {i+1}: Has {len(parts)} columns, expected 9")
            continue
        
        task, agent, seed, alpha, discount, subdir, epoch, weighting, extra = parts
        
        # Store for duplicate checking
        config_key = (task, agent, seed, alpha, discount, subdir, epoch, weighting, extra)
        data_lines.append(config_key)
        
        # Collect stats
        stats['tasks'][task] += 1
        stats['agents'][agent] += 1
        stats['seeds'][seed] += 1
        stats['weightings'][weighting] += 1
        stats['alphas'][f"{agent}:{alpha}"] += 1
        
        # Check alpha values
        if agent == 'gciql':
            if 'explore' in task and alpha != '0.01':
                issues.append(f"Line {i+1}: GCIQL explore task should have alpha=0.01, found {alpha}")
            elif 'humanoidmaze' in task and alpha != '0.1':
                issues.append(f"Line {i+1}: GCIQL humanoidmaze should have alpha=0.1, found {alpha}")
            elif 'antmaze' in task and 'explore' not in task and alpha != '0.3':
                issues.append(f"Line {i+1}: GCIQL antmaze should have alpha=0.3, found {alpha}")
            elif 'pointmaze' in task and alpha != '0.003':
                issues.append(f"Line {i+1}: GCIQL pointmaze should have alpha=0.003, found {alpha}")
    
    # Check for duplicates
    seen = set()
    duplicates = []
    for i, config in enumerate(data_lines):
        if config in seen:
            duplicates.append(f"Line {i+2}: Duplicate configuration")
        seen.add(config)
    
    if duplicates:
        issues.extend(duplicates[:10])  # Show first 10 duplicates
        if len(duplicates) > 10:
            issues.append(f"... and {len(duplicates)-10} more duplicates")
    
    # Print statistics
    print("\nStatistics:")
    print(f"  Total lines: {len(data_lines)}")
    print(f"  Unique configs: {len(seen)}")
    print(f"  Duplicates: {len(duplicates)}")
    print(f"\n  Tasks: {len(stats['tasks'])}")
    for task, count in sorted(stats['tasks'].items())[:5]:
        print(f"    {task}: {count}")
    if len(stats['tasks']) > 5:
        print(f"    ... and {len(stats['tasks'])-5} more")
    
    print(f"\n  Agents: {dict(stats['agents'])}")
    print(f"  Weightings: {dict(stats['weightings'])}")
    print(f"  Seeds: {dict(stats['seeds'])}")
    
    # Check seed distribution
    expected_seeds = ['0', '1', '2', '3', '4']
    missing_seeds = set(expected_seeds) - set(stats['seeds'].keys())
    if missing_seeds:
        issues.append(f"Missing seeds: {missing_seeds}")
    
    # Report issues
    if issues:
        print(f"\n❌ Found {len(issues)} issues:")
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues)-20} more")
    else:
        print("\n✓ No issues found!")
    
    return len(issues) == 0

def main():
    parser = argparse.ArgumentParser(description='Validate RWS-weighted TSV file')
    parser.add_argument('tsv_file', help='TSV file to validate')
    parser.add_argument('--fix', action='store_true', 
                       help='Generate fixed version')
    
    args = parser.parse_args()
    
    is_valid = validate_tsv(args.tsv_file)
    
    if not is_valid and args.fix:
        print("\n" + "="*60)
        print("To generate a fixed TSV, run:")
        print("  python generate_rws_weighted_tsv.py --output fixed_runs.tsv")
        print("\nThis will create a properly formatted TSV with:")
        print("  - 5 seeds (0-4) per configuration")
        print("  - Correct task-specific alpha values")
        print("  - No duplicates")
    
    return 0 if is_valid else 1

if __name__ == '__main__':
    sys.exit(main())