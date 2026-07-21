# Faire tourner MEDDxAgent (natif, sans patch) sur RCP — Apertus vs Meditron

Ce document décrit comment on a branché MEDDxAgent
([nec-research/meddxagent](https://github.com/nec-research/meddxagent)) dans
PtitWrap et comment lancer la comparaison **Apertus-8B de base vs
Apertus-8B-MeditronFO** comme docteur, sur le cluster RCP. L'infra RCP (pod,
venv, deux serveurs vLLM) est la même que pour
[`agentclinic_rcp_run.md`](agentclinic_rcp_run.md) — seule la façon dont le
modèle parle au benchmark diffère.

## 1. Pourquoi aucun patch n'était nécessaire

MediQ et AgentClinic font passer **tous** leurs appels modèle par une seule
fonction (`get_response` / `query_model`), qu'on remplace par un shim
(monkeypatch). MEDDxAgent, lui, choisit déjà son backend LLM **par chaîne de
config** via `init_model(class_name, **config)`. On injecte donc sans toucher au
code amont :

1. On enregistre un backend `Model` supplémentaire — `PtitWrapModel`, qui
   emballe notre `LM` — dans `sys.modules` sous un chemin synthétique
   (`ddxdriver.models.ptitwrap_model`).
2. On pointe le `model.class_name` de **chaque** agent (driver, history-taking,
   diagnosis, patient) vers ce backend, en passant l'objet `LM` vivant dans la
   config de l'agent.
3. On déroule la boucle multi-turn native de MEDDxAgent (`DDxDriver.__call__`)
   patient par patient.

Rien n'est modifié dans `PtitWrap/external/meddxagent/` — même garantie
« kept pristine » que les deux autres benchmarks. (`meddxagent` reste
importable tel quel : `sys.path.insert` sur le dossier, puis `import ddxdriver`.)

## 2. Séparation des rôles (comme AgentClinic)

- **Docteur = modèle testé** (`--model`) : il joue **le driver (orchestrateur),
  l'agent de prise d'historique ET l'agent de diagnostic**. C'est tout le côté
  « médecin agentique » qu'on évalue.
- **Patient = modèle fixe séparé** (`--judge_model`, ici Qwen) : simule le
  patient. On le garde identique entre les deux runs pour que **la seule
  variable soit le docteur**.

MEDDxAgent note le diagnostic contre la vérité terrain avec son propre
`metrics.py` (matching de chaînes, **pas** de LLM-juge). Il n'y a donc pas de
confound d'auto-évaluation à retirer au-delà de garder le patient sur un modèle
séparé.

## 3. Portée V1 (fidèle à MEDDxAgent, et alignée sur le wrapper Inspect)

iCraftMD uniquement (140 patients, données in-tree — rien à télécharger),
multi-turn : `history_taking` + `diagnosis`, **RAG et few-shot désactivés**.
Avec la prise d'historique active, le driver masque le profil complet du
patient : le docteur doit donc interroger le patient pour reconstituer le
différentiel. `diagnosis_class=standard`, `max_turns=6`, `max_questions=10`,
`enforce_diagnosis_options=True`, matching **strict** (exact) par défaut. Ces
réglages reproduisent le wrapper Inspect utilisé pour la comparaison
inter-harness.

## 4. Prérequis sur RCP

**a) Récupérer le code MEDDxAgent avec le repo.** MEDDxAgent est vendored sous
`PtitWrap/external/meddxagent/` (comme mediQ / AgentClinic). ⚠️ S'il a été ajouté
via un `git clone`, il contient un **`.git` imbriqué** : il faut le retirer avant
de committer, sinon les fichiers ne suivront pas au `git pull` sur RCP (git le
verrait comme un sous-module cassé) :
```bash
rm -rf PtitWrap/external/meddxagent/.git   # vendoring, comme mediQ/AgentClinic
git add PtitWrap/external/meddxagent
```
> Licence : MEDDxAgent est sous licence NEC **recherche non-commerciale**.
> Committer/pousser le code = redistribution — à garder pour un repo de labo
> (usage académique), pas une diffusion publique large.

**b) Installer les dépendances MEDDxAgent** (une fois, dans le venv du pod) :
```bash
pip install -e PtitWrap/external/meddxagent
```
Ça tire `datasets`, `faiss-cpu`, `transformers`, `colorama`, etc. — importés au
chargement de `ddxdriver.benchmarks` **même si** RAG/few-shot sont off. `torch`
et `vllm` sont déjà là (venv MediQ/AgentClinic).

## 5. Lancer les deux runs

Deux configs prêtes à l'emploi dans [`PtitWrap/configs/`](../configs/), une par
docteur. Chacune lance **elle-même** deux serveurs vLLM (docteur port 8000,
patient Qwen port 8001) via le backend `vllm-server` — rien à démarrer à la main.

```bash
export GASPAR=zbourlar
source /lightscratch/users/$GASPAR/mediq_venv/bin/activate
cd /lightscratch/users/$GASPAR/MultiTurn-Medical-Evaluation

# Apertus de base (docteur) vs Qwen (patient)
python -m PtitWrap.cli --config PtitWrap/configs/meddxagent_apertus_base_vs_qwen.yaml

# Meditron (docteur) vs Qwen (patient)
python -m PtitWrap.cli --config PtitWrap/configs/meddxagent_meditron_vs_qwen.yaml
```

Conseils :
- Lancer **dans `tmux`** (les runs sont longs et le pod peut redémarrer).
- Smoke rapide d'abord : `--task_args limit=3` sur une des deux configs.
- Suivi de l'avancée : le harness imprime une ligne par patient
  (`[meddxagent] i/N id=… ok/reached/FAILED pred=… gt=…`), et
  `tail -f /tmp/vllm_server_8000.log` montre le démarrage/les requêtes vLLM.

### Réglages notables dans les configs (déjà posés)
- `enforce_eager: true` sur le serveur docteur : Apertus (activation xIELU en
  fallback Python) peut sinon passer 10-15 min à compiler au chargement et
  dépasser le timeout de démarrage. Léger coût d'inférence, mais démarrage
  fiable et rapide. (Qwen n'en a pas besoin.)
- `max_tokens: 2048` côté docteur : de la marge pour un différentiel de 10
  maladies + rationale (MEDDxAgent ne passe pas de `max_tokens`, donc c'est le
  défaut du modèle qui s'applique — sans ça, le défaut 256 tronquerait le
  différentiel et le parsing échouerait).
- `max_model_len: 8192` : la liste des 392 options de diagnostic d'iCraftMD +
  profil + dialogue consomme ~4-6k tokens de contexte ; 8192 laisse la place à
  la sortie. Un patient dont le contexte déborde est simplement sauté (capté),
  pas fatal pour le run.

## 6. Résultats produits

Deux sorties par run (chemins dans la config) :
- **JSON** (`--output`) : bloc `metrics` + un enregistrement par patient
  (`transcript` docteur/patient, `final_ddx`, `diagnosis`, `correct`,
  `gtpa`, `num_questions`, `error` éventuel).
- **Log Inspect `.eval`** (`--inspect_log`) : visualisable avec
  `inspect view --log-dir PtitWrap/results/inspect`. Le dialogue s'affiche
  (docteur → assistant, patient → user), avec le verdict et la vérité terrain.

`metrics` contient :
- `accuracy` — GTPA@1 sur **tous** les patients tentés (les échecs comptent
  comme faux).
- `accuracy_over_reached` — GTPA@1 sur les seuls patients ayant produit un
  différentiel.
- `n_reached_diagnosis`, `num_patients`.
- Les métriques natives de MEDDxAgent (`get_metrics`) : `GTPA@1/3/5/10`,
  `Average Rank` — calculées exactement comme un run MEDDxAgent natif (sur les
  patients ayant abouti), donc directement comparables.

La comparaison Apertus vs Meditron se lit en mettant côte à côte les deux
fichiers de résultats (ou les deux `.eval` dans `inspect view`).
