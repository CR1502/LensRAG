# GitHub publishing and manual squash merges

Local Git is initialized. `main` contains the initial scaffold. The implementation is deliberately on two feature branches, with no local merge into `main`:

1. `feature/clip-retrieval`: real dataset preparation, CLIP encoders, persistent Chroma collections, both retrieval directions, and core tests.
2. `feature/evaluation-and-results`: depends on the retrieval branch; adds evaluation, measured artifacts, review notes, README, and CI.

The working checkout is the second branch and contains the complete runnable project. The private remote repository [CR1502/LensRAG](https://github.com/CR1502/LensRAG) has been created and configured as `origin`. No branches have been pushed and no PRs have been created. Run:

```bash
cd /Users/cr15/Documents/Main/Coding/LensRAG
gh auth status
# If this fails in your terminal: gh auth login --hostname github.com
git push -u origin main
git push -u origin feature/clip-retrieval
git push -u origin feature/evaluation-and-results
gh pr create --base main --head feature/clip-retrieval \
  --title "Add CLIP multimodal retrieval over Flickr8k" \
  --body "Prepare a pinned real image corpus and persist normalized image/caption CLIP vectors in ChromaDB. Demonstrate text-to-image and image-to-caption retrieval with a CLI. Validate preparation, holdout, persistence, and cosine lookup with synthetic fixtures."
```

Review and **manually squash-merge the retrieval PR first** in GitHub. Then rebase the dependent evaluation branch onto the resulting `main`, removing its old parent commit from the proposed PR:

```bash
git fetch origin
git switch feature/evaluation-and-results
git rebase --onto origin/main 8b24e40 feature/evaluation-and-results
git push --force-with-lease origin feature/evaluation-and-results
gh pr create --base main --head feature/evaluation-and-results \
  --title "Add measured retrieval evaluation and failure analysis" \
  --body "Evaluate 100 fixed held-out captions against 2,000 images using frozen CLIP. Record recall@1/5/10 of 40%/64%/76%, per-query rankings, visual failure review, and reproducibility metadata. Add guards against incomplete or mismatched indexes and document the limits of this fixed-corpus result. Local validation: nine tests and real CPU retrieval runs."
```

Review and **manually squash-merge the evaluation PR**, then update local main:

```bash
git switch main
git pull --ff-only origin main
```

The one explicit `--force-with-lease` updates your own dependent feature branch after a rebase. It does not rewrite `main`. The parent boundary `8b24e40` is the original retrieval commit; keep that value even if GitHub deletes the retrieval branch after merging.

Before the first push, optionally set your preferred commit identity and inspect the existing commits. Git used the machine's configured identity, `Craig <cr15@Craigs-MacBook-Pro.local>`. Publishing preserves that author metadata unless you deliberately rewrite it.

```bash
git log --oneline --all --decorate
git status
```
