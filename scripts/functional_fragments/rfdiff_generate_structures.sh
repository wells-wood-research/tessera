# Generate random structures using RFDiffusion
python scripts/run_inference.py inference.output_prefix=fragment_designs/random 'contigmap.contigs=[140-150]' inference.num_designs=10
python scripts/run_inference.py inference.output_prefix=fragment_designs/dna_binding inference.input_pdb=fragments/1/1A9X.pdb1 'contigmap.contigs=[40-50/A499-518/40-50]'  inference.num_designs=10
python scripts/run_inference.py inference.output_prefix=fragment_designs/metal_binding inference.input_pdb=fragments/22/2FZF.pdb1 'contigmap.contigs=[40-50/A117-151/40-50]'  inference.num_designs=10
python scripts/run_inference.py inference.output_prefix=fragment_designs/metal_binding inference.input_pdb=fragments/22/2FZF.pdb1 'contigmap.contigs=[40-50/A117-151/40-50]'  inference.num_designs=10
python scripts/run_inference.py inference.output_prefix=fragment_designs/dna_metal_binding inference.input_pdb=f1_f22.pdb 'contigmap.contigs=[40-50/A499-518/1000/B117-151/40-50]'  inference.num_designs=10