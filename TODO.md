# Memory Lane: next decisions

The face-clustering and named-person experiment has been retired. The existing
**Find similar faces** search remains a simple, unnamed visual-similarity tool.

## Search results and similarity

- Revisit the fixed top-100 result limit used by text search, **Find similar
  photos**, and **Find similar faces**.
- Compare three approaches with real searches:
  1. keep a fixed number of results;
  2. show only results above a similarity-score threshold;
  3. use a hybrid: a sensible maximum plus a minimum score.
- Consider showing each result's similarity as an approachable percentage.
  Explain in the guide that this is a relative match score—not a probability or
  a claim that the computer is literally “90% certain.”
- Add a short guide explanation of what “similar” means for:
  - a text search: the photograph visually matches the words;
  - **Find similar photos**: the whole photograph looks related;
  - **Find similar faces**: the selected face looks visually related.

## Browse controls

- Decide whether the decade buttons are already clearer than a range slider for
  a slow or infrequent computer user. Only replace them if the slider is plainly
  easier in a real walkthrough.

## Discussion before implementation

- Inspect the score ranges from several strong, weak, and intentionally odd
  queries before choosing a threshold or percentage display. Text, whole-photo,
  and face similarities may need different cutoffs and labels.
