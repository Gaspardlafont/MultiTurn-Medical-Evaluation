# med_eval — squelette d'évaluation multi-benchmarks cliniques

Code de base minimal pour évaluer **un même modèle OpenAI** sur quatre benchmarks
cliniques agentiques : **AgentClinic**, **MEDIQ/iMedQA**, **MEDDxAgent**, **PatientSim**.

But : comprendre **comment structurer** ce type d'évaluation, pas fournir des
chiffres publiables. Chaque benchmark est implémenté de façon *fidèle au protocole
réel* (lu dans les repos originaux) mais *simplifié* pour rester lisible.

## Démarrage

```bash
pip install openai
export OPENAI_API_KEY=sk-...
python run.py --benchmark all --limit 1        # 1 cas par benchmark
python run.py --benchmark agentclinic --limit 2
```

Sans clé API, on peut quand même vérifier la structure (chargement des cas,
enchaînement des agents) — voir les tests dans l'historique.

## Architecture

Tout repose sur une observation : **les 4 benchmarks partagent la même ossature**.

```
run.py                    ← runner : ne connaît que REGISTRY + .evaluate()
  │
  └── med_eval/
        __init__.py       ← REGISTRY : {nom -> classe Benchmark}
        llm.py            ← LLM : POINT D'ENTRÉE UNIQUE vers OpenAI
        base.py           ← Case, Agent, Benchmark, Result, judge_diagnosis
        _sample_data.py   ← mini-jeux de cas intégrés (à remplacer par vrais loaders)
        benchmarks/
          agentclinic.py  ← Doctor + Patient + Measurement, stop "DIAGNOSIS READY"
          mediq.py        ← Expert (question|choix) + Patient, QCM interactive
          meddxagent.py   ← orchestrateur : HistoryTaking + RAG + Diagnosis
          patientsim.py   ← Doctor + Patient À PERSONA (4 axes)
          benchmarks_shared.py ← agents réutilisables (SimpleDoctorAgent)
```

Les deux abstractions qui font tout tenir :

- **`LLM`** (llm.py) : un seul endroit appelle l'API. Remplacer OpenAI par vLLM/HF
  = modifier ce seul fichier. C'est le rôle de `lm_eval/models/` dans lm-eval-harness.
- **`Benchmark`** (base.py) : interface commune `load_cases()` + `run_case()`.
  Le runner appelle `.evaluate()` sans rien savoir du protocole interne.

## Correspondance avec le code réel

| Fichier ici | Repo original | Éléments repris |
|---|---|---|
| `agentclinic.py` | `SamuelSchmidgall/AgentClinic` → `agentclinic.py` | boucle `main()`, `DoctorAgent`/`PatientAgent`/`MeasurementAgent`, marqueurs `DIAGNOSIS READY:` / `REQUEST TEST:`, scoring par moderator |
| `mediq.py` | `stellalisy/mediQ` → `mediQ_benchmark.py` | `run_patient_interaction`, `Expert.respond` renvoyant question ou choix, `Patient.get_state` |
| `meddxagent.py` | `nec-research/meddxagent` → `ddxdriver/` | orchestrateur `DDxDriver`, sous-agents `history_taking`/`rag`/`diagnosis`, sortie = différentiel |
| `patientsim.py` | PhysioNet `persona-patientsim` (papier) | patient à persona 4 axes ; données MIMIC non incluses (accès crédentialé) |

## Ce qui est simplifié (à faire ensuite)

- **Données** : `_sample_data.py` contient 2 cas par benchmark, écrits à la main.
  Prochaine étape = brancher les vrais loaders (jsonl AgentClinic, ddxplus, etc.).
- **RAG de MEDDxAgent** : ici un simple appel LLM ; le vrai utilise MedRAG/recherche.
- **PatientSim** : les personas sont codés en dur ; le vrai dataset vit sur PhysioNet.
- **Métriques** : accuracy + avg_turns + recall@k. Les papiers ajoutent robustesse
  au biais (AgentClinic), calibration/abstention (MEDIQ), métriques par persona.
- **Coût** : chaque cas = plusieurs dizaines d'appels LLM (boucle multi-tours).
  Garder `--limit` petit pour les tests.

## Lien avec lm-evaluation-harness

Ces benchmarks étant **interactifs multi-agents**, lm-eval-harness ne les orchestre
pas nativement (il fait de l'éval statique un-prompt-une-réponse). Ce squelette est
la *couche d'orchestration au-dessus*. On peut y rebrancher lm-eval-harness à deux
endroits : (1) comme backend de modèle unifié à la place de `llm.py`, (2) pour
évaluer les sous-tâches statiques (ex : la QCM finale de MEDIQ) via une tâche YAML.
