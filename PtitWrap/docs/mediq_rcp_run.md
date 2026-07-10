# Faire tourner MediQ (natif, sans adapter) sur RCP avec Meditron

Ce document décrit comment on a obtenu un run MediQ fonctionnel sur le cluster RCP,
avec Apertus-8B-MeditronFO comme expert (docteur). Il sert de référence pour
refaire l'opération ou pour l'étendre à d'autres benchmarks.

## 1. Prérequis côté cluster

- Un GASPAR avec accès au projet `light-$GASPAR` sur RCP (voir la doc "Connecting to
  the RCP" du LiGHT lab).
- Une image Docker buildée et pushée sur `registry.rcp.epfl.ch` (voir
  `LiGHT-cluster-template`, fichier `.env` pour retrouver le chemin exact de ton
  image : `${LAB_NAME}/${USR}/${PROJECT_NAME}:amd64-cuda-${USR}-latest`).
- Le repo cloné/pull sur `/lightscratch/users/$GASPAR/...` (persistant — le
  système de fichiers du pod ne l'est pas).

## 2. Lancer un pod GPU

```bash
export GASPAR=<ton-gaspar>
runai submit \
  --name mediq-meditron-smoke \
  --image registry.rcp.epfl.ch/<lab_name>/<usr>/<project_name>:amd64-cuda-<usr>-latest \
  --pvc light-scratch:/lightscratch \
  --large-shm \
  -e NAS_HOME=/lightscratch/users/$GASPAR \
  -e HF_API_KEY_FILE_AT=/lightscratch/users/$GASPAR/keys/hf_key.txt \
  --backoff-limit 0 \
  --run-as-gid 84257 \
  --node-pool h100 \
  --gpu 1 \
  -- sleep infinity

runai bash mediq-meditron-smoke
```

Diagnostic utile si ça ne démarre pas :
```bash
runai describe job mediq-meditron-smoke
kubectl get pods -n runai-light-$GASPAR
kubectl describe pod -n runai-light-$GASPAR -l job-name=mediq-meditron-smoke
```

## 3. Créer un environnement Python isolé (le point clé)

L'image du pod contient déjà une stack Python système (torch/transformers/...)
pensée pour un autre projet. Installer vLLM par-dessus sans isolation crée un
conflit de versions (ex: `torch.int1` introuvable). On crée donc un **venv dédié**,
qui ne voit *pas* les paquets système :

```bash
python -m venv /lightscratch/users/$GASPAR/mediq_venv
source /lightscratch/users/$GASPAR/mediq_venv/bin/activate
pip install --upgrade pip
pip install -U vllm   # la version pinée dans environment.yml (0.6.6.post1) ne
                      # connaît pas l'architecture ApertusForCausalLM, il faut
                      # une version plus récente de vLLM
```

Vérifier :
```bash
python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
```

Ce venv persiste sur `/lightscratch`, donc pas besoin de le recréer à chaque
nouveau pod — juste le réactiver (`source .../mediq_venv/bin/activate`).

## 4. (Optionnel) Authentification HuggingFace

Nécessaire seulement pour des modèles gated (ex: Llama). Meditron et Qwen sont
publics, donc pas obligatoire ici.

```bash
cat /lightscratch/users/$GASPAR/keys/hf_key.txt   # doit contenir un token HF
export HF_TOKEN=$(cat /lightscratch/users/$GASPAR/keys/hf_key.txt)
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

## 5. Lancer MediQ

Le venv activé, depuis `PtitWrap/external/mediQ/src` :

```bash
# petit sous-ensemble du dataset pour un smoke test rapide
# (mediQ_benchmark.py n'a pas de flag --limit)
head -n 3 ../data/all_dev_good.jsonl > ../data/smoke_3.jsonl

python mediQ_benchmark.py \
  --expert_module expert --expert_class FixedExpert \
  --expert_model EPFLiGHT/Apertus-8B-MeditronFO \
  --expert_model_question_generator Qwen/Qwen2.5-7B-Instruct \
  --use_vllm \
  --patient_module patient --patient_class RandomPatient \
  --data_dir ../data --dev_filename smoke_3.jsonl \
  --output_filename ../../../results/mediq_meditron_smoke.jsonl \
  --log_filename ../../../results/logs/mediq_meditron_smoke.log \
  --history_log_filename ../../../results/logs/mediq_meditron_smoke_history.log \
  --detail_log_filename ../../../results/logs/mediq_meditron_smoke_detail.log \
  --message_log_filename ../../../results/logs/mediq_meditron_smoke_message.log \
  --max_questions 5
```

Notes :
- `--patient_class RandomPatient` : aucun modèle côté patient, pour isoler la
  variable "est-ce que Meditron tourne" du reste.
- Les 4 flags de log (`--log_filename`, `--history_log_filename`,
  `--detail_log_filename`, `--message_log_filename`) sont **tous obligatoires**
  en pratique : sans eux, `expert_functions.log_info()` plante
  (`'str' object has no attribute 'info'`) car le logger correspondant n'a
  jamais été enregistré via `logging.getLogger()`.

## 6. Problèmes rencontrés et corrigés (dans l'ordre)

| Symptôme | Cause | Fix |
|---|---|---|
| `ImagePullBackOff` / image "not found" | Mauvais chemin d'image (celui d'un autre projet/utilisateur) | Retrouver le vrai chemin via le `.env` du `LiGHT-cluster-template` |
| `AttributeError: 'str' object has no attribute 'info'` | Loggers `detail_logger`/`message_logger` jamais enregistrés (flags de log non fournis) | Passer les 4 flags de log |
| `No module named 'vllm'` | vLLM jamais installé dans le venv | `pip install vllm==0.6.6.post1` (ou plus récent) |
| `AttributeError: module 'torch' has no attribute 'int1'` | Conflit entre le torch système (image) et le torch installé par pip en local utilisateur | Installer dans un venv isolé, sans `--system-site-packages` |
| `Model architectures ['ApertusForCausalLM'] are not supported` | vLLM 0.6.6.post1 ne connaît pas encore Apertus | `pip install -U vllm` |
| `GatedRepoError` sur `meta-llama/Llama-3.1-8B-Instruct` | `--expert_model_question_generator` non précisé, tombe sur le défaut gated | Préciser un modèle ouvert (`Qwen/Qwen2.5-7B-Instruct`) ou authentifier `HF_TOKEN` |

## 7. Ce qui marche, résultat attendu

Le script traite chaque cas du `--dev_filename` et affiche :
```
[...] Processed 1/3 patients | Accuracy: ... | Timeout Rate: ... | Avg. Turns: ...
```
et écrit une ligne JSON par cas dans `--output_filename`.
