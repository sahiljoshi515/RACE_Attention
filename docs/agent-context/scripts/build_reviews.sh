#!/usr/bin/env bash
set -euo pipefail
# build_reviews.sh - regenerate pages/reviews/*.md from the OpenReview API.
#
# Source of truth: ICLR 2026 Submission 22728, forum RR8Lh8RHgA.
# The discussion period is closed and the decision is final (Accept, Poster), so this
# is effectively a snapshot regenerator - rerun it only to re-verify against OpenReview
# or to pick up a late metadata edit. Requires: curl, jq.
#
# Output is deterministic: notes are rendered by id/role, not by fetch order.

FORUM_ID="RR8Lh8RHgA"
API="https://api2.openreview.net/notes?forum=${FORUM_ID}&details=replies"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
REVIEWS_DIR="$DOCS_PATH/pages/reviews"
RAW="$DOCS_PATH/.openreview-raw.json"

mkdir -p "$REVIEWS_DIR"

echo "Fetching $API"
curl -fsS --proto '=https' --max-redirs 2 --max-time 60 --connect-timeout 10 "$API" -o "$RAW"
notes=$(jq '.notes | length' "$RAW")
echo "  fetched $notes notes"
if [[ "$notes" -lt 5 ]]; then
  echo "Refusing to regenerate: expected >=5 notes, got $notes (truncated response?)." >&2
  exit 1
fi

prov() {
  # Shared provenance footer.
  printf '\n---\nSource: OpenReview forum %s (ICLR 2026, Submission 22728). Regenerate: scripts/build_reviews.sh\n' "$FORUM_ID"
}

# --- 00-overview: ratings table + decision ---------------------------------------
{
  echo "# Reviews overview - ICLR 2026 Submission 22728"
  echo
  echo "Forum: https://openreview.net/forum?id=${FORUM_ID}"
  echo
  echo "## Decision"
  jq -r '.notes[] | select(.invitations[0]|test("Decision")) | "**\(.content.title.value)** — **\(.content.decision.value)**"' "$RAW"
  echo
  echo "## Reviewer scores"
  echo
  echo "| Reviewer | Rating | Soundness | Presentation | Contribution | Confidence |"
  echo "| --- | --- | --- | --- | --- | --- |"
  jq -r '.notes[] | select(.invitations[0]|test("Official_Review"))
    | "| \(.signatures[0]|split("/")|last) | \(.content.rating.value) | \(.content.soundness.value) | \(.content.presentation.value) | \(.content.contribution.value) | \(.content.confidence.value) |"' "$RAW"
  echo
  echo "Scale: ICLR ratings 1-10 (8=accept good paper, 6=marginal accept, 3=reject, etc.);"
  echo "soundness/presentation/contribution 1-4; confidence 1-5."
  echo
  echo "## Page map"
  echo "- \`reviews/reviewer-w9fa\`, \`reviews/reviewer-eqbu\`, \`reviews/reviewer-if3u\` - the three official reviews"
  echo "- \`reviews/meta-review\` - Area Chair summary; \`reviews/decision\` - Program Chair decision"
  echo "- \`reviews/rebuttal-<reviewer>\` - threaded author responses + reviewer follow-ups"
  echo "- \`reviews/author-summaries\` - cross-cutting author posts (revision summary, AC summary, all-reviewers response)"
  prov
} >"$REVIEWS_DIR/00-overview.md"
echo "  wrote 00-overview.md"

# --- one page per official review --------------------------------------------------
render_review() {
  local rid="$1" slug="$2"
  {
    local who
    who=$(jq -r --arg id "$rid" '.notes[] | select(.id==$id) | .signatures[0]|split("/")|last' "$RAW")
    echo "# Official Review - $who"
    echo
    jq -r --arg id "$rid" '.notes[] | select(.id==$id)
      | "Rating: \(.content.rating.value)  |  Soundness: \(.content.soundness.value)  |  Presentation: \(.content.presentation.value)  |  Contribution: \(.content.contribution.value)  |  Confidence: \(.content.confidence.value)"' "$RAW"
    echo
    local f title
    for f in summary strengths weaknesses questions; do
      case "$f" in
      summary) title="Summary" ;;
      strengths) title="Strengths" ;;
      weaknesses) title="Weaknesses" ;;
      questions) title="Questions" ;;
      esac
      echo "## $title"
      echo
      jq -r --arg id "$rid" --arg f "$f" '.notes[] | select(.id==$id) | .content[$f].value // "(none)"' "$RAW"
      echo
    done
    prov
  } >"$REVIEWS_DIR/$slug.md"
  echo "  wrote $slug.md"
}

# Resolve review note ids by reviewer signature (stable across re-fetch).
rid_for() {
  jq -r --arg sig "$1" '.notes[] | select((.invitations[0]|test("Official_Review")) and (.signatures[0]|test($sig))) | .id' "$RAW"
}
render_review "$(rid_for "Reviewer_W9FA")" "reviewer-w9fa"
render_review "$(rid_for "Reviewer_eQBU")" "reviewer-eqbu"
render_review "$(rid_for "Reviewer_if3U")" "reviewer-if3u"

# --- meta-review -------------------------------------------------------------------
{
  echo "# Meta-Review (Area Chair)"
  echo
  jq -r '.notes[] | select(.invitations[0]|test("Meta_Review")) | .content
    | to_entries[] | "## \(.key)\n\n\(.value.value)\n"' "$RAW"
  prov
} >"$REVIEWS_DIR/meta-review.md"
echo "  wrote meta-review.md"

# --- decision ----------------------------------------------------------------------
{
  echo "# Program Chair Decision"
  echo
  jq -r '.notes[] | select(.invitations[0]|test("Decision")) | .content
    | "**\(.title.value)**\n\nDecision: **\(.decision.value)**\n\n\(.comment.value // "")"' "$RAW"
  prov
} >"$REVIEWS_DIR/decision.md"
echo "  wrote decision.md"

# --- rebuttal threads, one page per reviewer ---------------------------------------
# Render every Official_Comment whose chain roots at the given reviewer's review or who
# is signed by that reviewer, ordered by tcdate (chronological thread order).
render_rebuttal() {
  local sig="$1" slug="$2" reviewid="$3"
  {
    echo "# Rebuttal thread - $sig"
    echo
    echo "Threaded discussion between the authors and $sig (chronological)."
    echo
    # comments that reply to the review note, plus follow-ups by the reviewer
    jq -r --arg rev "$reviewid" --arg sig "$sig" '
      def author(s): (s|split("/")|last);
      [ .notes[] | select(.invitations[0]|test("Official_Comment")) ]
      | map(. + {auth: author(.signatures[0])})
      | map(select(.replyto==$rev or (.auth==$sig)))
      | sort_by(.tcdate)
      | .[]
      | "## \(.content.title.value // "(comment)")  — \(.auth)\n\n\(.content.comment.value)\n"
    ' "$RAW"
    prov
  } >"$REVIEWS_DIR/$slug.md"
  echo "  wrote $slug.md"
}
render_rebuttal "Reviewer_W9FA" "rebuttal-w9fa" "$(rid_for "Reviewer_W9FA")"
render_rebuttal "Reviewer_eQBU" "rebuttal-eqbu" "$(rid_for "Reviewer_eQBU")"
render_rebuttal "Reviewer_if3U" "rebuttal-if3u" "$(rid_for "Reviewer_if3U")"

# --- author cross-cutting summaries (replies addressed to the forum/AC) -------------
{
  echo "# Author cross-cutting posts"
  echo
  echo "Author comments addressed to all reviewers / the Area Chair (not a single reviewer thread)."
  echo
  jq -r --arg forum "$FORUM_ID" '
    def author(s): (s|split("/")|last);
    [ .notes[] | select(.invitations[0]|test("Official_Comment")) ]
    | map(. + {auth: author(.signatures[0])})
    | map(select(.auth=="Authors" and .replyto==$forum))
    | sort_by(.tcdate)
    | .[]
    | "## \(.content.title.value // "(comment)")\n\n\(.content.comment.value)\n"
  ' "$RAW"
  prov
} >"$REVIEWS_DIR/author-summaries.md"
echo "  wrote author-summaries.md"

echo "Done. Reviews pages regenerated in $REVIEWS_DIR"
