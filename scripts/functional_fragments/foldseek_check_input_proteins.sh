# Checks input PDB in foldseek to see if we get similar functions
python src/scripts/functional_fragments/foldseek_check_proteins.py --input_pdb src/scripts/functional_fragments/pdb_files --slice 2:3
python src/scripts/functional_fragments/foldseek_check_proteins.py --input_pdb data/ --slice 2:3
