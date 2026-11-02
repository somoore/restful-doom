// libFuzzer entry point for the REST request parser. Built by tests/c/run.sh
// when clang's libFuzzer runtime is available.
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "api.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    static char buffer[2048];
    api_request_t request;

    if (size >= sizeof(buffer))
    {
        return 0;
    }

    memcpy(buffer, data, size);
    memset(&request, 0, sizeof(request));
    (void)API_ParseRequest(buffer, size, &request);
    return 0;
}
