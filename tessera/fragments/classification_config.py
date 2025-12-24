from collections import namedtuple

import numpy as np

fragment_lengths = {
    1: 20,
    2: 22,
    3: 30,
    4: 20,
    5: 24,
    6: 26,
    7: 23,
    8: 30,
    9: 26,
    10: 22,
    11: 26,
    12: 18,
    13: 37,
    14: 23,
    15: 15,
    16: 23,
    17: 9,
    18: 15,
    19: 31,
    20: 15,
    21: 20,
    22: 35,
    23: 18,
    24: 25,
    25: 19,
    26: 18,
    27: 32,
    28: 32,
    29: 27,
    30: 31,
    31: 24,
    32: 30,
    33: 23,
    34: 25,
    35: 29,
    36: 26,
    37: 15,
    38: 30,
    39: 33,
    40: 19,
}

min_fragment_length = int(min(fragment_lengths.values()))
max_fragment_length = int(max(fragment_lengths.values()))
median_fragment_length = int(np.median(list(fragment_lengths.values())))

# Define dictionaries for each metric
optimal_thresholds = {
    "inverse_median": 39.8061881188795,  # TP: 0.79, FP: 0.47
    "max": 0.03652618957076424,  # TP: 0.73, FP: 0.10
    "inverse_entropy": 0.2721902815005981,  # TP: 0.73, FP: 0.60
}

tpr_85_thresholds = {
    "inverse_median": 39.32125858274231,
    "max": 0.034630304805826705,
    "inverse_entropy": 0.272067044378336,
}

tpr_90_thresholds = {
    "inverse_median": 39.09191562516131,
    "max": 0.03417436227515402,
    "inverse_entropy": 0.2719848247394283,
}

tpr_95_thresholds = {
    "inverse_median": 38.696484251900344,
    "max": 0.032662020613613583,
    "inverse_entropy": 0.2717644678174031,
}


def get_threshold(metric: str, threshold_type: str) -> float:
    """
    Returns the threshold_distance value for a given metric and threshold_distance type.
    """
    assert metric in optimal_thresholds, f"Invalid metric: {metric}"
    assert threshold_type in [
        "optimal",
        "tpr_85",
        "tpr_90",
        "tpr_95",
    ], f"Invalid threshold_distance type: {threshold_type}"

    if threshold_type == "optimal":
        return optimal_thresholds[metric]
    elif threshold_type == "tpr_85":
        return tpr_85_thresholds[metric]
    elif threshold_type == "tpr_90":
        return tpr_90_thresholds[metric]
    elif threshold_type == "tpr_95":
        return tpr_95_thresholds[metric]


categorical_node_attrs = {"fragment_class": len(fragment_lengths) + 1}  # +1 for unknown
categorical_edge_attrs = {"peptide_bond": 2}

BIOLOGICAL_PROCESS = "GO:0008150"
MOLECULAR_FUNCTION = "GO:0003674"
CELLULAR_COMPONENT = "GO:0005575"
FUNC_DICT = {
    "cc": CELLULAR_COMPONENT,
    "mf": MOLECULAR_FUNCTION,
    "bp": BIOLOGICAL_PROCESS,
}
NAMESPACES_REVERT = {
    "cellular_component": "cc",
    "molecular_function": "mf",
    "biological_process": "bp",
}

go_to_prosite = {
    "GO:0003677": {"description": "DNA binding", "prosite": dict([("PRU00655", "Alpha box DNA-binding domain"),("PRU00366", "AP2/ERF DNA-binding domain"),("PRU00764", "AP endonucleases family 1"),("PRU00763", "AP endonucleases family 2"),("PRU00630", "APSES-type HTH DNA_binding domain"),("PRU00593", "araC/xylS-type HTH domain"),("PRU00355", "ARID domain"),("PRU00340", "ArsR-type HTH domain"),("PRU00319", "AsnC-type HTH domain"),("PRU00326", "B3 domain"),("PRU01085", "Cas9-type HNH domain"),("PRU00583", "CENPB-type HTH domain"),("PRU00055", "Copper-fist DNA-binding domain"),("PRU01248", "Core-binding (CB) domain"),("PRU01364", "CRESS-DNA virus replication initiator protein (Rep) endonuclease domain"),("PRU00257", "Cro/C1-type HTH domain"),("PRU00387", "Crp-type HTH domain"),("PRU00436", "CTF/NF-I DNA-binding domain"),("PRU00746", "DBINO domain"),("PRU00349", "DeoR-type HTH domain"),("PRU00070", "DM DNA-binding domain"),("PRU01362", "DNA ADP-ribosyl transferase (DarT) domain"),("PRU01073", "DNA-binding recombinase domain"),("PRU00296", "DtxR-type HTH domain"),("PRU00237", "ETS domain"),("PRU00089", "Fork-head DNA-binding domain"),("PRU00392", "Formamidopyrimidine-DNA glycosylase catalytic domain"),("PRU00227", "Fungal Zn(2)-Cys(6) binuclear cluster domain"),("PRU00245", "GCM domain"),("PRU00307", "GntR-type HTH domain"),("PRU00484", "GTF2I-like repeat"),("PRU00267", "HMG boxes A and B DNA-binding domains"),("PRU00108", "Homeobox DNA-binding domain"),("PRU01162", "Homeo-Prospero (HPD) domain"),("PRU00549", "HSA domain"),("PRU00393", "IclR-type HTH domain"),("PRU00840", "IRF tryptophan pentad repeat DNA-binding domain"),("PRU00615", "IS21 transposase-type HTH domain"),("PRU00616", "IS408/IS1162 transposase-type HTH domain"),("PRU00312", "KID domain"),("PRU00111", "LacI-type HTH domain"),("PRU00620", "Large T-antigen (T-ag) origin-binding domain (OBD)"),("PRU00837", "Linker histone H1/H5 globular (H15) domain"),("PRU00411", "LuxR-type HTH domain"),("PRU00253", "LysR-type HTH domain"),("PRU00112", "LytTR-type HTH domain"),("PRU00372", "MADF domain"),("PRU00251", "MADS-box domain"),("PRU00345", "MarR-type HTH domain"),("PRU00254", "MerR-type HTH domain"),("PRU01039", "Mu-type HTH domain"),("PRU00625", "Myb-type HTH DNA-binding domain"),("PRU00353", "NAC domain"),("PRU00966", "NF-YA/HAP2 family"),("PRU00407", "Nuclear hormone receptors DNA-binding domain"),("PRU00138", "OAR domain"),("PRU01091", "OmpR/PhoB-type DNA-binding domain"),("PRU00381", "Paired DNA-binding domain"),("PRU01366", "Parvovirus (PV) NS1 nuclease (NS1-Nuc) domain profile"),("PRU00530", "POU-specific (POUs) domain"),("PRU00320", "Psq-type HTH domain"),("PRU00858", "RFX-type winged-helix DNA-binding domain"),("PRU00390", "RpiR-type HTH domain"),("PRU00540", "Rrf2-type HTH domain"),("PRU00399", "Runt domain"),("PRU00852", "RWP-RK domain"),("PRU00185", "SAND domain"),("PRU00596", "SF4 helicase domain"),("PRU01076", "SpoVT-AbrB domain"),("PRU00252", "SSB domain"),("PRU00201", "T-box domain"),("PRU00701", "TCP domain"),("PRU00335", "TetR-type HTH domain"),("PRU00682", "TFIIE beta central core DNA-binding domain"),("PRU01383", "Topoisomerase (Topo) IA-type catalytic domain"),("PRU01382", "Topoisomerase (Topo) IB-type catalytic domain"),("PRU01384", "Topoisomerase (Topo) IIA-type catalytic domain"),("PRU01385", "Topoisomerase (Topo) IIB-type catalytic domain"),("PRU00995", "Toprim domain"),("PRU01246", "Tyrosine recombinase domain"),("PRU00223", "WRKY DNA-binding domain"),("PRU00071", "Zinc finger Dof-type"),("PRU00264", "Zinc finger poly(ADP-ribose) polymerase (PARP)-type"),("PRU00309", "Zinc finger THAP-type"),])},
    "GO:0003723": {"description": "RNA binding", "prosite": dict([("PRU01085", "Cas9-type HNH domain"),("PRU01293", "Coronavirus Nsp12 RNA-dependent RNA polymerase (RdRp) domain"),("PRU01277", "Coronavirus nucleocapsid (CoV N) protein C-terminal domain"),("PRU01276", "Coronavirus nucleocapsid (CoV N) protein N-terminal domain (NTD)"),("PRU00626", "CRM domain"),("PRU00657", "Dicer double-stranded RNA-binding fold domain"),("PRU00266", "Double stranded RNA-binding domain"),("PRU01304", "Eukaryotic uridylate-specific endoribonuclease (EndoU) domain"),("PRU00117", "KH domain"),("PRU00118", "KH type-2 domain"),("PRU00332", "La-type HTH domain"),("PRU00924", "mRNA cap 0 and cap 1 methyltransferase (EC 2.1.1.56 and EC 2.1.1.57)"),("PRU00895", "mRNA (guanine-N(7))-methyltransferase (EC 2.1.1.56)"),("PRU00896", "N6-adenosine-methyltransferase catalytic subunit (EC=2.1.1.348)"),("PRU00539", "RdRp"),("PRU01203", "Rho RNA-binding (RNA-BD) domain"),("PRU00956", "RNA methyltransferase TRMH"),("PRU00176", "RNA recognition motif (RRM) domain"),("PRU01319", "RNase H type-2 domain"),("PRU01026", "rRNA adenine N(6)-methyltransferase family"),("PRU01023", "SAM-dependent MTase RsmB/NOP-type domain"),("PRU01346", "Sm domain"),("PRU00529", "THUMP domain"),("PRU00209", "tRNA-binding domain"),("PRU00955", "tRNA guanosine-2'-O-methyltransferase"),("PRU00855", "Zinc finger nanos-type"),])},
    "GO:0005525": {"description": "GTP binding", "prosite": dict([("PRU01057", "AIG1-type G domain"),("PRU01051", "Bms1-type guanine nucleotide-binding (G) domain"),("PRU01058", "Circularly permuted (CP)-type guanine nucleotide-binding (G) domain"),("PRU01055", "Dynamin-type guanine nucleotide-binding (G) domain"),("PRU01049", "EngA-type guanine nucleotide-binding (G) domain"),("PRU01043", "EngB-type guanine nucleotide-binding (G) domain"),("PRU01050", "Era-type guanine nucleotide-binding (G) domain profile"),("PRU01048", "FeoB-type guanine nucleotide-binding (G) domain"),("PRU01230", "G-alpha domain"),("PRU01052", "GB1/RHD3-type guanine nucleotide-binding (G) domain"),("PRU01042", "HflX-type guanine nucleotide-binding (G) domain"),("PRU01053", "IRG-type guanine nucleotide-binding (G) domain"),("PRU00757", "Miro domain"),("PRU01047", "OBG-type guanine nucleotide-binding (G) domain"),("PRU01056", "Septin-type guanine nucleotide-binding (G) domain"),("PRU00758", "small GTPase superfamily. Roc family"),("PRU01059", "Translational (tr)-type guanine nucleotide-binding (G) domain"),("PRU01046", "TrmE-type guanine nucleotide-binding (G) domain"),("PRU01054", "Very large inducible GTPASE (VLIG)-type guanine nucleotide-binding (G) domain"),])},
    "GO:0005524": {"description": "ATP binding", "prosite": dict([("PRU00434", "ABC transporter family domain"),("PRU00492", "ATP-cone domain"),("PRU00409", "ATP-grasp domain"),("PRU00783", "DAG-kinase catalytic (DAGKc) domain"),("PRU00289", "FtsK domain"),("PRU00886", "GMP synthetase ATP pyrophosphatase (GMPS ATP-PPase) domain"),("PRU00100", "Guanylate kinase-like domain"),("PRU00541", "Helicase ATP-binding domain"),("PRU01084", "Hexokinase domain"),("PRU00487", "KaiC domain"),("PRU00283", "Kinesin motor domain"),("PRU00782", "Myosin motor domain"),("PRU00136", "NACHT-NTPase domain"),("PRU00706", "Nucleoside diphosphate kinase (NDK)-like domain"),("PRU01047", "OBG-type guanine nucleotide-binding (G) domain"),("PRU00843", "Phosphagen kinase C-terminal domain"),("PRU00781", "Phosphatidylinositol phosphate kinase (PIPK) domain"),("PRU00159", "Protein kinase domain"),("PRU10141", "Protein kinases ATP-binding region signature"),("PRU00990", "(+)RNA virus helicase core domain"),("PRU00596", "SF4 helicase domain"),("PRU00193", "Sigma-54 factor interaction domain"),("PRU00560", "UvrD helicase ATP-binding domain")])},
    "GO:0046872": {"description": "metal ion binding", "prosite": dict([("PRU01192",  "3'5'-cyclic nucleotide phosphodiesterase (PDEase) domain"), ("PRU00989",  "4Fe-4S domain"), ("PRU00711",  "4Fe-4S ferredoxin-type domain"), ("PRU01010",  "4Fe-4S WhiB-like (Wbl)-type iron-sulfur binding domain"), ("PRU00865",  "ADD domain"), ("PRU01134",  "Alpha-carbonic anhydrase domain"), ("PRU01339",  "Anthrax toxin lethal factor (ATLF)-like domain"), ("PRU00763",  "AP endonucleases family 2"), ("PRU00742",  "Arginase family"), ("PRU00340",  "ArsR-type HTH domain"), ("PRU00985",  "Arteriviridae zinc-binding (AV ZBD) domain"), ("PRU00029",  "BIR"), ("PRU01356",  "CFEM) domain"), ("PRU00055",  "Copper-fist DNA-binding domain"), ("PRU00986",  "Coronaviridae zinc-binding (CV ZBD) domain"), ("PRU01297",  "Coronavirus (CoV) ExoN/MTase coactivator domain"), ("PRU01333",  "Coronavirus (CoV) Nsp2 N-terminal domain"), ("PRU01364",  "CRESS-DNA virus replication initiator protein (Rep) endonuclease domain"), ("PRU00509",  "CXXC-type zinc finger"), ("PRU00242",  "Cytochrome b561 domain"), ("PRU00968",  "Cytochrome b/b6 N-terminal region"), ("PRU00692",  "Cytochrome c oxidase subunit Vb, zinc binding domain"), ("PRU00600",  "DBF4-type zinc finger"), ("PRU01043",  "EngB-type guanine nucleotide-binding (G) domain"), ("PRU01304",  "Eukaryotic uridylate-specific endoribonuclease (EndoU) domain"), ("PRU01146",  "Extended PHD (ePHD) domain"), ("PRU01048",  "FeoB-type guanine nucleotide-binding (G) domain"), ("PRU01230",  "G-alpha domain"), ("PRU00245",  "GCM domain"), ("PRU01042",  "HflX-type guanine nucleotide-binding (G) domain"), ("PRU00705",  "High potential iron-sulfur proteins family"), ("PRU00538",  "JmjC domain"), ("PRU01198",  "KARI C-terminal domain"), ("PRU00726",  "Lipoxygenase iron-binding catalytic domain"), ("PRU00658",  "L-type lectin-like (leguminous) domain"), ("PRU01063",  "MYST-type histone acetyltransferase (HAT) domain"), ("PRU01298",  "Nidovirus 3'-5' exoribonuclease (ExoN) domain"), ("PRU00407",  "Nuclear hormone receptors DNA-binding domain"), ("PRU00706",  "Nucleoside diphosphate kinase (NDK)-like domain"), ("PRU00794",  "Nudix hydrolase domain"), ("PRU01047",  "OBG-type guanine nucleotide-binding (G) domain"), ("PRU01366",  "Parvovirus (PV) NS1 nuclease (NS1-Nuc) domain profile"), ("PRU01031",  "Peptidase family M66 domain"), ("PRU00679",  "Phosphotriesterase family"), ("PRU01082",  "PPM-type phosphatase domain"), ("PRU00678",  "Prokaryotic zinc-dependent phospholipase C domain"), ("PRU01266",  "Radical SAM core domain"), ("PRU10073",  "Renal dipeptidase family"), ("PRU00540",  "Rrf2-type HTH domain"), ("PRU00812",  "RTR1-type zinc"), ("PRU01237",  "Rubella virus (RUBV) nonstructural (NS) protease domain"), ("PRU00241",  "Rubredoxin-like domain"), ("PRU01032",  "Sedolisin domain"), ("PRU00741",  "Transferrin-like domain"), ("PRU01221",  "TRIAD supradomain"), ("PRU01046",  "TrmE-type guanine nucleotide-binding (G) domain"), ("PRU00502",  "UBP-type zinc finger"), ("PRU01388",  "UBR4 E3 catalytic module"), ("PRU01341",  "UPF1 cysteine-histidine-rich (CH-rich) domain"), ("PRU00700",  "Urease domain"), ("PRU00748",  "Xylose isomerase family"), ("PRU01148",  "Zinc finger C2HC baculovirus (BV)-type"), ("PRU01371",  "Zinc finger C2HC/C3H-type"), ("PRU01145",  "Zinc finger C2HC LYAR-type"), ("PRU01144",  "Zinc finger C2HC RNF-type"), ("PRU01244",  "Zinc finger C4H2-type"), ("PRU01357",  "Zinc finger C6H2-type"), ("PRU01153",  "Zinc finger CCHC FOG-type"), ("PRU01154",  "Zinc finger CCHC HIVEP-type"), ("PRU01142",  "Zinc finger CCHC NOA-type"), ("PRU01143",  "Zinc finger CCHHC-type"), ("PRU01141",  "Zinc finger CHHC U11-48K-type"), ("PRU00834",  "Zinc finger DNL-type"), ("PRU00671",  "Zinc finger large T-antigen (T-ag) D1-type"), ("PRU01101",  "Zinc finger RAG1-type"), ("PRU01381",  "Zinc finger reverse gyrase C-terminal-type"), ("PRU01380",  "Zinc finger reverse gyrase N-terminal-type"), ("PRU00472",  "Zinc finger TFIIS-type"), ("PRU00309",  "Zinc finger THAP-type"), ("PRU01254",  "Zinc finger UBZ2-type"), ("PRU01255",  "Zinc finger UBZ3-type"), ("PRU01256",  "Zinc finger UBZ4-type"), ("PRU01220",  "Zinc finger ZBR-type"), ("PRU00471",  "Zinc-hook domain")])},
}

UniprotResults = namedtuple("UniprotResults", ["uniprot_id", "sequence_length", "go_codes"])

selected_pdbs = set(["1A0B","1A17","1D1N","1H99","1ID0","1IR3","1JAD","1NSQ","1ODF","1Q2L","1Q31","1Q57","1V47","1WQS","1WXL","1ZAE","1ZUN","2A1A","2AFF","2B0L","2BCW","2CJQ","2CJR","2CS1","2DK1","2DWQ","2FH5","2G2K","2GX5","2K85","2KUE","2LKC","2LKZ","2LW7","2MBV","2MFR","2N51","2NA2","2NRR","2O2V","2OPU","2PMY","2Q2E","2Q7U","2QMH","2QY2","2RHK","2VDW","2WX4","2Y9Y","2YI9","2YKH","3A1G","3AI4","3AQ4","3BLE","3BLQ","3C5H","3CRV","3CXH","3EPK","3FZM","3H63","3HRT","3HYR","3IBP","3KMP","3KVT","3MDO","3MGZ","3MP2","3O47","3RGH","3RQI","3T12","3TIX","3TWL","3U1I","3UAI","3VKP","3VPY","4A69","4ACB","4ARZ","4ASN","4B47","4BZQ","4C0D","4CBL","4DW4","4E6N","4FWT","4GEH","4GP7","4IAO","4IJX","4LAW","4MN4","4N3S","4OGA","4OHZ","4OXP","4QS7","4R4M","4R71","4RD6","4U4P","4U12","4UUD","4WBZ","4WNR","4XJ1","4ZGQ","4ZU9","5C1T","5CA9","5CSA","5DIS","5DV7","5EYA","5FWH","5HD9","5HNO","5HZH","5I4Q","5IRC","5IRR","5IZL","5JPX","5LBD","5LM7","5LOJ","5LUT","5MLC","5O3N","5O5S","5UJE","5X3S","5YWW","5ZZ7","6ADM","6BKG","6DI7","6DS6","6E0M","6EHR","6F5D","6FGZ","6G0Y","6H2X","6H9M","6HZ4","6IY6","6J72","6K71","6KWR","6LUR","6MD3","6O8H","6O56","6RIE","6VKJ","6WG6","6X1M","6XR4","6ZHU","6ZN8","7B7Z","7C7L","7CRW","7D8U","7DPQ","7E2M","7EGY","7EPD","7FID","7GNZ","7JQQ","7KSL","7LJN","7MSS","7O1Q","7O9G","7OQC","7RY1","7TAG","7TM8","7V3W","7VRC","7VUF","7W02","7WU8","7XC4","7Y04","7YZ8","7Z6L","7ZJ1","7ZR1","8ATD","8B0J","8BAH","8BI0","8CYL","8D9S","8FX4","8H5V","8JF7","8OWO","8PAG","8PK5","8Q6Q","8S9X","8U5H","8W4J","8YB5","9ASP","9BE2","9F5B","9FOF"])
