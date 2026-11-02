// Regression tests for the REST request parser and the shared bounds helpers.
// Built with AddressSanitizer/UBSan by tests/c/run.sh.
#include <stdio.h>
#include <string.h>

#include "api.h"
#include "api_validate.h"

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static void test_bounds_helpers(void)
{
    CHECK(API_IndexInRange(0, 4));
    CHECK(API_IndexInRange(3, 4));
    CHECK(!API_IndexInRange(4, 4));
    CHECK(!API_IndexInRange(-1, 4));

    CHECK(API_WeaponNumberInRange(1, 9));
    CHECK(API_WeaponNumberInRange(9, 9));
    CHECK(!API_WeaponNumberInRange(0, 9));
    CHECK(!API_WeaponNumberInRange(10, 9));

    CHECK(API_HttpMethodIsReadOnly("GET"));
    CHECK(API_HttpMethodIsReadOnly("HEAD"));
    CHECK(!API_HttpMethodIsReadOnly("PATCH"));
    CHECK(!API_HttpMethodIsReadOnly("POST"));
    CHECK(!API_HttpMethodIsReadOnly("DELETE"));

    CHECK(API_ConstantTimeEqual("s3cret", "s3cret"));
    CHECK(!API_ConstantTimeEqual("s3cret", "s3cres"));
    CHECK(!API_ConstantTimeEqual("s3cret", "s3cret-longer"));
    CHECK(!API_ConstantTimeEqual("", "s3cret"));
}

static int parse(const char *text)
{
    char buffer[2048];
    api_request_t request;
    size_t len = strlen(text);

    memcpy(buffer, text, len + 1);
    memset(&request, 0, sizeof(request));
    return API_ParseRequest(buffer, len, &request);
}

static void test_request_line(void)
{
    char buffer[2048];
    api_request_t request;
    size_t len;
    char long_method[64];
    char long_path[400];
    char long_protocol[64];

    memset(buffer, 0, sizeof(buffer));
    snprintf(buffer, sizeof(buffer), "GET /api/world HTTP/1.0\r\nHost: x\r\n\r\n");
    memset(&request, 0, sizeof(request));
    CHECK(API_ParseRequest(buffer, strlen(buffer), &request));
    CHECK(strcmp(request.method, "GET") == 0);
    CHECK(strcmp(request.full_path, "/api/world") == 0);
    CHECK(strcmp(request.url.path, "api/world") == 0);

    CHECK(parse("GET /api/world HTTP/1.0\r\n\r\n"));

    // Over-long request-line tokens must be rejected, not truncated into an
    // adjacent buffer (the historical unbounded sscanf("%s%s%s") bug).
    memset(long_method, 'A', sizeof(long_method) - 1);
    long_method[sizeof(long_method) - 1] = '\0';
    snprintf(buffer, sizeof(buffer), "%s /api/world HTTP/1.0\r\n\r\n", long_method);
    CHECK(!API_ParseRequest(buffer, strlen(buffer), &request));

    memset(long_path, 'B', sizeof(long_path) - 1);
    long_path[sizeof(long_path) - 1] = '\0';
    snprintf(buffer, sizeof(buffer), "GET %s HTTP/1.0\r\n\r\n", long_path);
    CHECK(!API_ParseRequest(buffer, strlen(buffer), &request));

    memset(long_protocol, 'C', sizeof(long_protocol) - 1);
    long_protocol[sizeof(long_protocol) - 1] = '\0';
    snprintf(buffer, sizeof(buffer), "GET /api/world %s\r\n\r\n", long_protocol);
    CHECK(!API_ParseRequest(buffer, strlen(buffer), &request));

    // A full 1024-byte read has no room for a terminator; callers reject it
    // before parsing (413). Parsing it directly must still fail safely.
    memset(buffer, 'D', 1024);
    CHECK(!API_ParseRequest(buffer, 1024, &request));

    // No blank line => no body separator => reject.
    CHECK(!parse("GET /api/world HTTP/1.0\r\n"));
    CHECK(!parse("GET\r\n\r\n"));
    CHECK(!parse(""));

    // Exactly-at-the-limit path is still accepted.
    memset(buffer, 0, sizeof(buffer));
    len = 0;
    len += (size_t)snprintf(buffer + len, sizeof(buffer) - len, "GET /");
    memset(buffer + len, 'E', 97);
    len += 97;
    snprintf(buffer + len, sizeof(buffer) - len, " HTTP/1.0\r\n\r\n");
    CHECK(API_ParseRequest(buffer, strlen(buffer), &request));
}

// Deterministic mutation fuzzing so the memory-safety gate also runs where
// clang's libFuzzer runtime is unavailable.
static void test_random_inputs(void)
{
    char buffer[2048];
    api_request_t request;
    unsigned long state = 0x9e3779b9UL;
    int i;
    int j;

    for (i = 0; i < 50000; i++)
    {
        size_t len;

        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        len = (size_t)(state % (sizeof(buffer) - 1));

        for (j = 0; j < (int)len; j++)
        {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            buffer[j] = (char)(state & 0xff);
        }

        // Bias half the cases towards plausible request lines.
        if ((i & 1) == 0 && len > 20)
        {
            const char *prefix = "GET /api/world HTTP/1.0\r\n\r\n";
            size_t prefix_len = strlen(prefix);
            if (prefix_len < len)
            {
                memcpy(buffer, prefix, prefix_len);
            }
        }

        memset(&request, 0, sizeof(request));
        (void)API_ParseRequest(buffer, len, &request);
    }
}

int main(void)
{
    test_bounds_helpers();
    test_request_line();
    test_random_inputs();

    if (failures != 0)
    {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    puts("api security tests: ok");
    return 0;
}
