#!/bin/bash
# set -e

BUILDDIR=build
CORPUS=corpus.txt
VOCAB_FILE=vocab.txt
SAVE_FILE=glove.w2v

VERBOSE=2
MEMORY=4.0

VOCAB_MIN_COUNT=5

WINDOW_SIZE=5
COOCCURRENCE_FILE=cooccurrence.bin
WEIGHT=1

COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin

VECTOR_SIZE=256
MAX_ITER=25
WINDOW_SIZE=2
BINARY=0
NUM_THREADS=8
X_MAX=10
HEADLINE=1

echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE

echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -distance-weighting $WEIGHT < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -distance-weighting $WEIGHT < $CORPUS > $COOCCURRENCE_FILE

echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE -write-header $HEADLINE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE -write-header $HEADLINE


