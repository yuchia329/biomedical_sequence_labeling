#!/usr/bin/env python3
"""
Usage:
  python evaluate.py test.answers test_output.txt

Prints out the precision, recall, and F1 on BIO entities.
"""

import sys

def read_bio_file(filename):
    """
    Reads a BIO-format file:
      token  tag
      token  tag
      [blank line separates sentences]

    Returns:
      A list of sentences, each sentence is a list of tag strings (one per token).
      We ignore the token text for evaluation. We only collect the tags.
    """
    sentences = []
    current_tags = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                # blank line => end of sentence
                if current_tags:
                    sentences.append(current_tags)
                    current_tags = []
            else:
                # line has something like "Token\tTag"
                parts = line.split()
                if len(parts) >= 2:
                    # last column is the tag, ignoring the token text
                    tag = parts[-1]
                    current_tags.append(tag)
                else:
                    # if there's a mismatch in the file format,
                    # you may need to adapt this logic
                    pass

        # handle final sentence if file doesn't end with a blank line
        if current_tags:
            sentences.append(current_tags)

    return sentences

def get_entities_bio(sequence):
    """
    Given a list of BIO tags (e.g. ["B-PER", "I-PER", "O", "B-ORG", ...]),
    return a set of (entity_type, start_index, end_index) tuples,
    where 'start_index' is inclusive and 'end_index' is exclusive.

    For example, if sequence = ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-ORG"],
    we extract:
       ("PER", 0, 2), ("ORG", 3, 6)
    """
    entities = set()
    start = None
    end = None
    current_type = None

    for i, tag in enumerate(sequence):
        if tag == 'O' or tag.startswith('B-') or tag.startswith('S-'):
            # if we were in a chunk, close it out
            if current_type is not None:
                entities.add((current_type, start, i))
                current_type = None
                start = None
                end = None

            # if tag is B-xxx, begin a new chunk
            if tag.startswith('B-'):
                current_type = tag[2:]
                start = i

        elif tag.startswith('I-'):
            # continuing a chunk. if no preceding B-, treat it as B- anyway
            ttype = tag[2:]
            if current_type is None:
                # treat as a new chunk
                current_type = ttype
                start = i
            elif ttype != current_type:
                # the current I-xxx doesn't match the old chunk type -> start new
                entities.add((current_type, start, i))
                current_type = ttype
                start = i

        elif tag.startswith('E-'):
            # E- can be used if following a B- chunk
            ttype = tag[2:]
            if current_type is None:
                # treat as a new chunk
                current_type = ttype
                start = i
            elif ttype != current_type:
                # end of a different type? close out old, start new
                entities.add((current_type, start, i))
                current_type = ttype
                start = i
            # close out at i+1
            entities.add((current_type, start, i+1))
            current_type = None
            start = None

        # If you have S- or other variations, you can handle them similarly.

    # if we ended with an open chunk, close it
    if current_type is not None:
        entities.add((current_type, start, len(sequence)))

    return entities

def precision_recall_f1(true_set, pred_set):
    """
    Computes precision, recall, and F1 given sets of gold and predicted chunks.
    """
    # intersection:
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision+recall) > 0 else 0.0

    return precision, recall, f1

def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <gold_file> <pred_file>")
        sys.exit(1)

    gold_file = sys.argv[1]
    pred_file = sys.argv[2]

    gold_sentences = read_bio_file(gold_file)  # list of lists of tags
    pred_sentences = read_bio_file(pred_file)  # same structure

    # If there's a mismatch in sentence counts or token counts, you need
    # to check your formatting – they must align 1:1 for a fair comparison.
    if len(gold_sentences) != len(pred_sentences):
        print(f"WARNING: # of sentences differ: {len(gold_sentences)} vs {len(pred_sentences)}")

    all_gold_chunks = set()
    all_pred_chunks = set()
    offset = 0  # to keep unique indices across sentences

    # We'll combine all chunks from all sentences into two big sets
    # with index offsets so they don't collide across sentences.
    # (Another approach is to compute sentence-level P/R/F and average.)
    for sent_idx, (g_tags, p_tags) in enumerate(zip(gold_sentences, pred_sentences)):
        # If lengths differ, either handle carefully or skip
        if len(g_tags) != len(p_tags):
            print(f"Sentence {sent_idx} length mismatch: gold={len(g_tags)} pred={len(p_tags)}")
            # You may want to handle partial match or skip – here we skip
            continue

        # Convert each list of tags into a set of chunk spans
        g_chunks = get_entities_bio(g_tags)
        p_chunks = get_entities_bio(p_tags)

        # shift chunk spans by offset so each sentence is uniquely indexed
        g_shifted = {(t, start+offset, end+offset) for (t, start, end) in g_chunks}
        p_shifted = {(t, start+offset, end+offset) for (t, start, end) in p_chunks}

        all_gold_chunks.update(g_shifted)
        all_pred_chunks.update(p_shifted)
        offset += len(g_tags)  # next sentence starts at a new offset

    precision, recall, f1 = precision_recall_f1(all_gold_chunks, all_pred_chunks)

    print("BIO-tagged entity-level results:")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1        = {f1:.4f}")

if __name__ == "__main__":
    main()
