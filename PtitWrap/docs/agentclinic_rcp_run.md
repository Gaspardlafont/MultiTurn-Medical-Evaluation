# Faire tourner AgentClinic (natif, avec un patch minimal) sur RCP avec Meditron

Ce document décrit comment on a branché un modèle vLLM local (Meditron) sur
AgentClinic, et comment lancer un run sur le cluster RCP. Il complète
[`mediq_rcp_run.md`](mediq_rcp_run.md) — l'infra RCP (pod, venv) est la même,
seule la partie "comment le modèle parle au harness" diffère.

## 1. Pourquoi un patch était nécessaire

`agentclinic.py` route tous les appels modèle via `query_model(model_str, ...)`,
qui dispatche selon la valeur de `model_str` :
- `gpt4`, `gpt3.5`, `gpt4o`, `claude3.5sonnet`, `o1-preview` → APIs propriétaires
  (OpenAI, Anthropic)
- `llama-2-70b-chat`, `mixtral-8x7b`, `llama-3-70b-instruct` → Replicate
- `HF_<model>` → censé charger un modèle HuggingFace local via
  `transformers.pipeline`, mais **cassé** : la branche lève juste
  `raise Exception("Sorry, fixing TODO :3")`, sans jamais faire d'inférence.

Il n'y avait donc aucun chemin fonctionnel pour un modèle local/open-weight
comme Meditron. On a ajouté une nouvelle branche `VLLM_` plutôt que de réparer
`HF_` (qui reste cassé, intact, non prioritaire pour nous).

## 2. Le patch : branche `VLLM_`

Deux changements dans `agentclinic.py` :

**a) Check de validité (ligne ~21)** — ajout de `"VLLM_" not in model_str`
pour que les modèles préfixés `VLLM_` ne soient pas rejetés.

**b) Nouvelle branche, après la branche `HF_` cassée** — appelle un serveur
vLLM local, compatible OpenAI, via le SDK `openai` déjà importé dans le
fichier :
```python
elif "VLLM_" in model_str:
    # Forme "VLLM_<port>:<model>" pour cibler un serveur précis ; sinon
    # défaut au port 8000 (rétro-compatible avec "VLLM_<model>" seul).
    vllm_spec = model_str.replace("VLLM_", "", 1)
    if ":" in vllm_spec and vllm_spec.split(":", 1)[0].isdigit():
        vllm_port, vllm_model_name = vllm_spec.split(":", 1)
    else:
        vllm_port, vllm_model_name = "8000", vllm_spec
    client = openai.OpenAI(base_url=f"http://localhost:{vllm_port}/v1", api_key="none")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}]
    response = client.chat.completions.create(
            model=vllm_model_name,
            messages=messages,
            temperature=0.05,
            max_tokens=200,
        )
    answer = response.choices[0].message.content
```

**Usage du préfixe :**
- `VLLM_EPFLiGHT/Apertus-8B-MeditronFO` → vise `localhost:8000` (défaut)
- `VLLM_8001:Qwen/Qwen2.5-7B-Instruct` → vise `localhost:8001`

Le préfixe par port permet de faire tourner **deux modèles différents sur
deux serveurs vLLM différents en même temps** (voir §4, confound
d'auto-évaluation).

Contrairement à MediQ (qui appelle `vllm.LLM(...)` en process), AgentClinic
parle à vLLM **en client HTTP** — il faut donc démarrer le serveur vLLM
(`python -m vllm.entrypoints.openai.api_server`) séparément, et il doit rester
actif pendant tout le run.

## 3. Limite connue : les images ne sont pas transmises

Le docteur peut demander un test (`"REQUEST TEST: [test]"`) ou, si
`--doctor_image_request` est actif sur le dataset NEJM, une image
(`"REQUEST IMAGES"`) — ce sont de pures conventions textuelles : le modèle
doit écrire cette chaîne exacte dans son texte, et la boucle principale fait
un simple `if "REQUEST TEST" in doctor_dialogue: ...`.

Mais la construction du message multimodal (avec `scene.image_url`) n'est
câblée en dur que pour 4 branches : `gpt4v`, `gpt-4o-mini`, `gpt4`, `gpt4o`.
**Notre branche `VLLM_` (comme `HF_`, Claude et Replicate) ignore
silencieusement `image_requested`** : pas d'erreur, mais l'image n'est jamais
réellement envoyée au modèle. Non corrigé pour l'instant — à faire si on a
besoin d'images sur NEJM avec un modèle vLLM local.

## 4. Confound d'auto-évaluation, et comment on l'évite

`compare_results()` — qui juge si le diagnostic du docteur est correct — est
un **LLM-juge** : il envoie tout le dialogue + le diagnostic correct au
`moderator_llm`, demande "Are these the same?", et compare la réponse
(en minuscule) à la chaîne exacte `"yes"`. Pas de règle clinique, pas de
matching structuré.

Si `--moderator_llm` (et `--patient_llm`/`--measurement_llm`) pointent vers le
**même** modèle que `--doctor_llm`, ce modèle juge son propre diagnostic — on
l'a vu concrètement : Meditron a jugé "CORRECT" un diagnostic
(`Patellar tendinopathy`) clairement différent du bon (`Pes anserine
bursitis`). D'où la nécessité de deux serveurs vLLM séparés : un pour le
docteur (modèle testé), un pour patient/mesure/modérateur (juge indépendant).

## 5. Lancer un run : `run_agentclinic_local_vllm.sh`

Script : `PtitWrap/scripts/run_agentclinic_local_vllm.sh`. Il automatise :
1. Active le venv (réutilise celui créé pour MediQ, `vllm` déjà installé).
2. Installe les dépendances propres à AgentClinic (`openai`, `anthropic`,
   `replicate` — absentes du venv MediQ).
3. Démarre **deux** serveurs vLLM en arrière-plan : le modèle docteur (port
   8000) et un modèle juge séparé (port 8001, Qwen2.5-7B-Instruct par défaut).
4. Attend que les deux répondent sur `/v1/models` (jusqu'à 10 min chacun).
5. Lance `agentclinic.py` avec `--doctor_llm VLLM_8000:<model>` et
   `--patient_llm`/`--measurement_llm`/`--moderator_llm` sur
   `VLLM_8001:<judge_model>`.
6. Arrête les deux serveurs à la fin (`trap cleanup EXIT`), succès ou échec.

```bash
export GASPAR=zbourlar
NUM_SCENARIOS=10 TOTAL_INFERENCES=20 ./PtitWrap/scripts/run_agentclinic_local_vllm.sh
```
Variables surchargeables : `MODEL`, `JUDGE_MODEL`, `PORT`, `JUDGE_PORT`,
`GPU_MEM_UTIL`, `MAX_MODEL_LEN`, `DATASET`, `NUM_SCENARIOS`,
`TOTAL_INFERENCES`, `VENV`, `AGENTCLINIC_DIR`.

## 6. Problèmes rencontrés et corrigés (dans l'ordre)

| Symptôme | Cause | Fix |
|---|---|---|
| Aucun chemin fonctionnel pour un modèle local | `HF_` cassé (`raise Exception("Sorry, fixing TODO :3")`), aucune autre branche ne couvre un serveur local | Nouvelle branche `VLLM_`, HTTP vers un serveur vLLM local via le SDK `openai` déjà présent dans le fichier |
| Impossible de faire tourner un 2e modèle (juge) sur un port différent | Le premier patch codait `http://localhost:8000/v1` en dur | Forme `VLLM_<port>:<model>` ajoutée, rétro-compatible (sans port → 8000 par défaut) |
| Deuxième serveur vLLM : `ImagePullBackOff`-like refus, `RuntimeError: Engine core initialization failed` | Deux serveurs vLLM sur un seul GPU réservent chacun ~90% de la mémoire par défaut | `--gpu-memory-utilization 0.45` sur les deux serveurs |
| Malgré ça : `ValueError: No available memory for the cache blocks` (`-0.32 GiB`) | Le contexte max par défaut (32k-65k tokens) sur-réserve le cache KV, alors qu'AgentClinic n'a besoin que de quelques échanges courts | `--max-model-len 4096` sur les deux serveurs — réduit fortement la mémoire KV cache requise |
| `"Scene X, CORRECT"` sur un diagnostic manifestement faux | `--moderator_llm` pointait sur le même modèle que `--doctor_llm` (auto-évaluation) | Modèle juge séparé (Qwen2.5-7B-Instruct) sur son propre serveur vLLM |
| Aucun résumé/accuracy affiché après le run | `main()` n'affiche un résultat que si `"DIAGNOSIS READY"` apparaît dans le dialogue **avant** `--total_inferences` ; pas de résumé final agrégé de toute façon (limite native du fichier, non corrigée) | Augmenter `--total_inferences` (20, la valeur par défaut native) pour laisser le docteur conclure ; vérifier avec `grep "Scene\|DIAGNOSIS READY"` |

## 7. Ce qu'AgentClinic propose nativement comme "résultats"

Très minimal, à la différence de MediQ :
- **Par scène, si le docteur conclut** : jugement binaire correct/incorrect
  (LLM-juge, voir §4), imprimé en `print()`.
- **Une précision cumulée** (`total_correct / total_presents`), affichée à
  chaque scène qui conclut.
- **Rien d'autre** : pas de résumé final agrégé, pas de fichier de sortie,
  pas de logging structuré, pas de métriques par rôle ou par tour. Tout est
  volatile dans le terminal — à améliorer si on veut des résultats
  exploitables (voir §6, dernière ligne).
