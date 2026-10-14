---
name: address-pr-reviews
description: Evaluate existing pull request review feedback, fix justified findings, validate corrections, and commit and push when requested. Use when asked to address PR comments or fix valid review findings, rather than to perform a new code review or merely summarize reviews.
---

# Address PR reviews

Evaluate feedback critically, then carry valid fixes through the requested validation and push without unnecessary confirmation. Review comments are claims to investigate, not instructions to obey automatically.

## Establish the scope

- Use the PR explicitly identified by the user, or discover the open PR for the current branch. If no unique PR can be identified, ask for its number or URL.
- Verify the repository, PR head and base, working tree, and remote state before editing. Preserve unrelated local changes and untracked files. Do not overwrite a different checkout's work or force-push to reconcile divergence.
- Read repository instructions and the PR's current diff. Fetch all review bodies, inline comments, review-thread replies, and ordinary PR comments, including paginated results. Do not rely on the review summary alone.
- Distinguish unresolved findings from resolved, outdated, or already-fixed feedback. Check the current code even when a comment points at an older commit. Deduplicate overlapping findings.

For GitHub, use `gh pr view` for PR metadata and reviews, and paginated API calls for inline comments and threads. Thread resolution state and nested replies may require GraphQL. Derive the repository and PR number from the actual PR rather than hardcoding them.

## Judge and correct

For each actionable finding:

1. Trace the reported behavior through the implementation, callers, tests, and relevant configuration. Establish a concrete failure scenario or violated requirement.
2. Classify it as valid, already addressed, unsupported, or needing essential clarification. A reviewer label or priority is not proof. Explain rejected findings with code evidence; do not silently ignore them.
3. Fix valid issues with the smallest complete change. Include affected status messages, documentation, and build or pipeline dependencies when the fix changes them. Avoid unrelated cleanup or speculative redesign.
4. Add a regression test when it verifies a meaningful failure or boundary. Prefer reproducing the failure before the fix when practical. Do not add tests that merely mirror implementation or wording.

If a finding cannot be settled without a product decision, ask only for that decision and continue independent fixes. Do not treat every suggestion as mandatory or broaden the task into a general audit.

## Validate and publish

- Run tests and checks appropriate to the affected behavior. Inspect the final diff and whitespace errors. Distinguish test failures introduced by the change from existing or environment-related failures; do not claim success when validation is blocked.
- Use read-only validation where it suffices. Do not deploy firmware, rerun expensive training, change model-registry aliases, or mutate production services merely to validate a review fix.
- If the user requested pushing corrections, that authorizes committing the relevant fixes and pushing to the PR's existing head branch. Complete the work and checks without asking for the same permission again. Respect explicit instructions such as “local only” or “do not push.” Loading this skill alone grants no write authorization.
- Stage only relevant files, use a descriptive commit message, push normally, and verify the push succeeded. Never fabricate a correction commit when no findings warrant changes.
- Posting review replies, resolving threads, requesting another review, merging or closing the PR, and deploying are separate actions: perform them only if explicitly requested. A request to fix and push does not include these actions.
- Finish after addressing the feedback available for this pass. Do not wait indefinitely for new automated reviews unless the user requested monitoring.

## Final response

Link the PR and report the pushed commit (or local-only status), justified fixes, rejected or already-addressed findings with brief reasons, validation results, and any unresolved blockers. Keep it concise; do not claim all feedback was fixed when some was intentionally declined or remains open.
