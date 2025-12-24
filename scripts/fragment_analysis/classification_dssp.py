fragments = {
    "32": "--HHHHHHHHHHHHHHHH---HHH------",
    "35": "-HHHHHHHHH-------EE----EE---",
    "34": "------------HHHHHHHHHHHH-",
    "33": "----------HHHHHHHHHHH--",
    "20": "---------------",
    "18": "-------HHHHH---",
    "27": "-HHHHHHHHHH--HHHHHH----HHHHHHH-",
    "9": "------HHHH---------HHHHHH-",
    "11": "-EEEHHH--------------E-EE-",
    "7": "-----HHHHH-------------",
    "29": "-HHHHHHHHHHHH-----HHHHHHH-",
    "16": "------------HHHHHHHHHH-",
    "6": "-HHHH-----HHHHHHHHHH------",
    "28": "-HHHHHHHHHHHH---HHHHHHHHHH-----",
    "17": "---------",
    "1": "----HHHHHH----------",
    "10": "-EE--EE------------EE-",
    "19": "-E----E--------------HHH-------",
    "26": "-------EE------EE-",
    "8": "-EEEEE---HHHHHHHHHHHH---EEEEE-",
    "21": "---------HHHHHHHHHH-",
    "38": "-HHHHHHHHHHHH---HHHHHHHHHHHH-",
    "36": "-EEEE----HHHHHHHHH---EEEE-",
    "31": "--HHHHHHHH----HHHHHHHHH-",
    "30": "-HHHHHHHHHHHH----HHHHHHHHHHHH-",
    "37": "------HHHHHHH-",
    "39": "-----HHHHHHHHHH----HHHHHHHHHHHHH-",
    "24": "-HHHHHHHHHHH---HHHHHHHHH-",
    "23": "-HHHH-----------",
    "4": "-HHHHH---HHHHHHHHH-",
    "15": "-EEE-----E-EE--",
    "3": "-HHHHHHHHHHHH--------HHHHHHH-",
    "12": "-E--------------E-",
    "40": "--------HHHHHHHHHH-",
    "2": "-HHHH------HHHHHHHHHH-",
    "13": "-E--EEE-----EEEEE-----EEEE------EEE--",
    "5": "--HHHHHHHH-HHHHHHHHHHHH-",
    "14": "-E------E---HHHHHHHHHH-",
    "22": "-HHHHHHHHHHH----HHHHHHHHHHHHHHHHH-",
    "25": "-EEEEEEE---EEEEEEE-",
}

classification = {"alpha": [], "beta": [], "alpha_beta": []}

for fragment, sequence in fragments.items():
    has_alpha = "H" in sequence
    has_beta = "E" in sequence

    if has_alpha and not has_beta:
        classification["alpha"].append(fragment)
    elif has_beta and not has_alpha:
        classification["beta"].append(fragment)
    elif has_alpha and has_beta:
        classification["alpha_beta"].append(fragment)

# Display the classification
print(classification)
