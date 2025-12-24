import argparse
import random
import sys
import typing as t
from pathlib import Path

import ampal
import numpy as np
import pandas as pd
from ampal import AmpalContainer

# --- PROJECT IMPORTS ---
try:
    from src.fragments.classification_config import fragment_lengths as FRAGMENT_LENGTHS
except ImportError:
    print("Warning: Could not import 'fragment_lengths'. Ensure you are in the project root.")
    sys.exit(1)


def get_available_pdbs(pdb_dir: Path) -> t.Dict[str, Path]:
    if not pdb_dir.exists():
        raise FileNotFoundError(f"PDB Directory not found: {pdb_dir}")

    pdb_map = {}
    for file_path in pdb_dir.iterdir():
        if file_path.is_file() and file_path.suffix in ['.pdb', '.pdb1', '.ent']:
            pdb_id = file_path.stem.upper()
            pdb_map[pdb_id] = file_path
    return pdb_map


def sample_non_overlapping(seq_len: int, lengths: t.List[int], ids: t.List[int], max_retries: int = 500) -> t.Optional[
    t.List[t.Tuple[t.Tuple[int, int], int]]]:
    """Returns [ ((start, end), original_frag_id), ... ]"""
    items = list(zip(lengths, ids))

    for _ in range(max_retries):
        occupied = np.zeros(seq_len, dtype=bool)
        current_solution = []
        random.shuffle(items)

        success = True
        for length, fid in items:
            if length > seq_len:
                success = False;
                break

            # Find valid start indices (0-based)
            valid_starts = [i for i in range(seq_len - length + 1) if not occupied[i: i + length].any()]

            if not valid_starts:
                success = False;
                break

            start = random.choice(valid_starts)
            end = start + length

            occupied[start: end] = True
            current_solution.append(((start, end), fid))

        if success:
            current_solution.sort(key=lambda x: x[0][0])
            return current_solution
    return None


def safe_get_sequence(ampal_obj):
    try:
        return ampal_obj.sequence
    except AttributeError:
        try:
            return "".join([m.mol_letter for m in ampal_obj.monomers])
        except:
            return "X" * len(ampal_obj)


def generate_single_slice(pdb_path, chain_id, target_len, original_fid):
    """
    Tries to grab ONE random slice of length `target_len` from the PDB.
    Returns dictionary with structure and metadata, or None if failed.
    """
    pdb_id = pdb_path.stem.upper()
    try:
        structure = ampal.load_pdb(str(pdb_path))
        if isinstance(structure, AmpalContainer):
            structure = structure[0]

        if chain_id in structure.id:
            chain = structure[chain_id]
        else:
            chain = structure[0]

            # Attempt to find a spot
        placements = sample_non_overlapping(len(chain), [target_len], [original_fid])

        if not placements: return None

        (start_idx, end_idx), _ = placements[0]
        fragment_structure = chain[start_idx:end_idx]

        start_res_id = fragment_structure[0].id
        end_res_id = fragment_structure[-1].id
        sequence = safe_get_sequence(fragment_structure)

        return {'structure': fragment_structure,
            'meta': {'Identifier': pdb_id, 'Chain': chain_id, 'Start Residue': start_res_id, 'End Residue': end_res_id,
                'Classification': 'Naive_Control', 'Amino Acid Sequence': sequence, 'Length': len(sequence),
                'Original_Source_Fragment': original_fid}}
    except Exception as e:
        # print(f"Error processing {pdb_id}: {e}") # Quiet fail allows retrying next in list
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate naive fragment sets with exact vocabulary size.")
    parser.add_argument("--input_csv", type=Path, required=True, help="Path to original fragments.csv")
    parser.add_argument("--pdb_dir", type=Path, required=True, help="Directory containing local PDBs")
    parser.add_argument("--output_dir", type=Path, default=Path("./data/naive_multiset"), help="Where to save results")
    parser.add_argument("--num_sets", type=int, default=5, help="Number of random sets to generate")
    parser.add_argument("--vocab_size", type=int, required=True,
                        help="Exact number of fragments to generate (e.g. 40 or 265). One folder per fragment.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")

    args = parser.parse_args()

    fragment_root = args.output_dir / "fragments"
    fragment_root.mkdir(parents=True, exist_ok=True)

    available_pdbs = get_available_pdbs(args.pdb_dir)
    if not available_pdbs:
        print("CRITICAL ERROR: No PDB files found.")
        sys.exit(1)

    print(f"Reading specs from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    # Type cleaning
    df['Fragment Number'] = pd.to_numeric(df['Fragment Number'], errors='coerce')
    df = df.dropna(subset=['Fragment Number'])
    df['Fragment Number'] = df['Fragment Number'].astype(int)

    # Valid PDB check
    df['temp_id'] = df['Identifier'].str.upper()
    df_valid = df[df['temp_id'].isin(available_pdbs.keys())].copy()

    # Create the "Task Pool": A list of all available slots in the dataset
    # [(pdb_id, chain_id, length, original_fid), ...]
    task_pool = []
    for _, row in df_valid.iterrows():
        fid = row['Fragment Number']
        if fid in FRAGMENT_LENGTHS:
            length = FRAGMENT_LENGTHS[fid]
            task_pool.append({'pdb': row['temp_id'], 'chain': row['Chain'], 'len': length, 'fid': fid})

    print(f"Pool Size: {len(task_pool)} potential fragment slots found in dataset.")

    if args.vocab_size > len(task_pool):
        print(f"WARNING: vocab_size ({args.vocab_size}) is larger than dataset size ({len(task_pool)}).")
        print("We will generate as many as possible.")

    all_rows = []

    # --- MAIN LOOP ---
    for set_i in range(args.num_sets):
        current_seed = args.seed + set_i
        random.seed(current_seed)
        np.random.seed(current_seed)

        print(f"\n--- Generating Set {set_i} (Target: {args.vocab_size} fragments) ---")

        # Shuffle the full pool of available tasks
        # This ensures we pick a RANDOM selection of proteins/lengths to fill our quota
        current_pool = task_pool.copy()
        random.shuffle(current_pool)

        generated_count = 0
        set_rows = []

        # Iterate through the random pool until we hit vocab_size
        for task in current_pool:
            if generated_count >= args.vocab_size:
                break

            pdb_path = available_pdbs[task['pdb']]

            # Try to generate a random slice
            result = generate_single_slice(pdb_path, task['chain'], task['len'], task['fid'])

            if result:
                generated_count += 1
                unique_id = generated_count  # 1, 2, 3 ... vocab_size

                # Folder: set_X / 1 / file.pdb1
                out_dir = fragment_root / f"set_{set_i}" / str(unique_id)
                out_dir.mkdir(parents=True, exist_ok=True)

                pdb_name = f"{result['meta']['Identifier']}.pdb1"
                out_path = out_dir / pdb_name

                with open(out_path, 'w') as f:
                    f.write(result['structure'].make_pdb())

                row = result['meta'].copy()
                row['Set_ID'] = set_i
                row['Fragment Number'] = unique_id  # The new Unique ID
                set_rows.append(row)

        print(f"Set {set_i}: Successfully generated {generated_count} fragments.")
        all_rows.extend(set_rows)

    out_csv = args.output_dir / f"naive_fragments_vocab{args.vocab_size}.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)

    print(f"\nSuccess! Saved to {out_csv}")
    print(f"Folder structure: {fragment_root}/set_0/1/, /set_0/2/ ... /set_0/{args.vocab_size}/")


if __name__ == "__main__":
    main()