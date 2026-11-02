#!/bin/sh
# Builds and runs the C security regression tests. Uses clang's libFuzzer
# runtime for the request-parser fuzz target when it is available.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OUT="$ROOT/build/c-tests"
CC=${CC:-cc}
FUZZ_CC=${FUZZ_CC:-clang}

mkdir -p "$OUT"

INCLUDES="-I$ROOT -I$ROOT/src -I$ROOT/src/doom"
SAN="-fsanitize=address,undefined -fno-omit-frame-pointer"
CFLAGS="-std=c99 -O1 -g -fcommon -Wall -Wextra"

# shellcheck disable=SC2086
"$CC" $CFLAGS $SAN $INCLUDES \
    "$ROOT/tests/c/test_api_security.c" \
    "$ROOT/src/doom/api_request.c" \
    "$ROOT/src/doom/api_yuarel.c" \
    -o "$OUT/test_api_security"

ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 "$OUT/test_api_security"

if command -v "$FUZZ_CC" >/dev/null 2>&1 &&
   printf 'int LLVMFuzzerTestOneInput(const char *d, unsigned long s){return 0;}\n' |
       "$FUZZ_CC" -x c -fsanitize=fuzzer -o "$OUT/.fuzz_probe" - >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    "$FUZZ_CC" $CFLAGS $SAN -fsanitize=fuzzer $INCLUDES \
        "$ROOT/tests/c/fuzz_parse_request.c" \
        "$ROOT/src/doom/api_request.c" \
        "$ROOT/src/doom/api_yuarel.c" \
        -o "$OUT/fuzz_parse_request"

    ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
        "$OUT/fuzz_parse_request" -runs=20000 -max_len=2048 >/dev/null
    echo "api request fuzzer: ok"
else
    echo "api request fuzzer: skipped (clang libFuzzer runtime not available)"
fi
