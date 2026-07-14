"""
Mini jeux de cas intégrés (2 cas par benchmark) pour que le squelette tourne
SANS télécharger les vrais datasets.

Dans le vrai projet, on remplacerait chaque liste par un vrai loader :
  - AgentClinic : agentclinic_medqa.jsonl
  - MEDIQ       : MedQA / CRAFT-MD convertis en format interactif
  - MEDDxAgent  : ddxplus / rarebench / icraftmd
  - PatientSim  : MIMIC-IV / MIMIC-ED (PhysioNet, accès crédentialé)
Le point important : le format des Case reste identique, donc seul le loader change.
"""

# --- AgentClinic : dossier riche avec examens ---
AGENTCLINIC_CASES = [
    {
        "id": "ac_01",
        "diagnosis": "Myasthénie grave",
        "presentation": "Femme de 35 ans, diplopie et faiblesse musculaire fluctuantes.",
        "patient_info": {
            "demographics": "Femme, 35 ans",
            "history": "Diplopie depuis 1 mois, difficulté à monter les escaliers, "
                       "symptômes aggravés par l'effort et améliorés par le repos.",
            "symptoms": ["diplopie", "faiblesse des membres supérieurs", "fatigabilité"],
        },
        "test_results": {
            "test_anticorps_RACh": "positif",
            "test_tensilon": "amélioration transitoire",
            "NFS": "normale",
        },
    },
    {
        "id": "ac_02",
        "diagnosis": "Embolie pulmonaire",
        "presentation": "Homme de 60 ans, dyspnée aiguë et douleur thoracique.",
        "patient_info": {
            "demographics": "Homme, 60 ans",
            "history": "Dyspnée brutale, douleur thoracique pleurétique, "
                       "voyage long-courrier récent, mollet droit douloureux.",
            "symptoms": ["dyspnée", "douleur thoracique", "tachycardie"],
        },
        "test_results": {
            "D-dimeres": "élevés",
            "angioscanner": "défect de perfusion lobaire droit",
            "ECG": "tachycardie sinusale",
        },
    },
]

# --- MEDIQ : QCM MedQA-like avec dossier partiel ---
MEDIQ_CASES = [
    {
        "id": "mq_01",
        "question": "Quel est le diagnostic le plus probable ?",
        "options": {"A": "Asthme", "B": "Insuffisance cardiaque",
                    "C": "Pneumonie", "D": "BPCO"},
        "answer": "C",
        "initial_info": "Homme de 68 ans, toux et fièvre depuis 3 jours.",
        "full_record": {
            "age": 68, "fièvre": "39.1°C", "toux": "productive, expectorations rouille",
            "auscultation": "crépitants base droite", "radio": "opacité lobaire droite",
            "GB": "élevés",
        },
    },
    {
        "id": "mq_02",
        "question": "Quel est le diagnostic le plus probable ?",
        "options": {"A": "Migraine", "B": "Hémorragie méningée",
                    "C": "Céphalée de tension", "D": "Sinusite"},
        "answer": "B",
        "initial_info": "Femme de 45 ans, céphalée brutale intense.",
        "full_record": {
            "âge": 45, "début": "en coup de tonnerre, intensité maximale d'emblée",
            "raideur_nuque": "présente", "photophobie": "présente",
            "scanner": "hyperdensité dans les espaces sous-arachnoïdiens",
        },
    },
]

# --- MEDDxAgent : cas orientés diagnostic différentiel ---
MEDDX_CASES = [
    {
        "id": "dx_01",
        "diagnosis": "Appendicite aiguë",
        "presentation": "Homme de 22 ans, douleur abdominale migrant en fosse iliaque droite.",
        "patient_record": {
            "douleur": "péri-ombilicale puis FID", "fièvre": "38.3°C",
            "signe_de_McBurney": "positif", "nausées": "présentes",
            "GB": "14000",
        },
    },
    {
        "id": "dx_02",
        "diagnosis": "Diabète de type 1",
        "presentation": "Adolescent de 14 ans, polyurie, polydipsie et amaigrissement.",
        "patient_record": {
            "symptômes": "polyurie, polydipsie, perte de poids 5 kg en 3 semaines",
            "glycémie": "3.2 g/L", "cétonurie": "présente",
            "haleine": "cétonique",
        },
    },
]

# --- PatientSim : cas de consultation aux urgences ---
PATIENTSIM_CASES = [
    {
        "id": "ps_01",
        "diagnosis": "Infarctus du myocarde",
        "presentation": "Patient de 58 ans se présentant aux urgences pour douleur thoracique.",
        "patient_info": {
            "douleur": "rétrosternale constrictive irradiant au bras gauche",
            "durée": "45 min", "sueurs": "profuses", "antécédents": "HTA, tabac",
        },
    },
    {
        "id": "ps_02",
        "diagnosis": "Accident vasculaire cérébral",
        "presentation": "Patiente de 72 ans amenée pour déficit neurologique brutal.",
        "patient_info": {
            "déficit": "hémiparésie droite et trouble du langage d'apparition brutale",
            "heure_début": "il y a 1 heure", "antécédents": "fibrillation auriculaire",
        },
    },
]
