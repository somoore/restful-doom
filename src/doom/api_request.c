#include <stddef.h>
#include <string.h>

#include "api.h"

// Scratch buffer for the parsed request path. Kept for compatibility with
// callers that referenced the global directly.
char path[100];

// Copy the next whitespace-delimited token into dst, rejecting tokens that do
// not fit. Never reads past the NUL terminator of the input buffer.
static boolean API_NextToken(const char **cursor, char *dst, size_t dst_size)
{
    const char *start;
    const char *p;
    size_t len;

    p = *cursor;
    while (*p == ' ' || *p == '\t')
    {
        p++;
    }

    start = p;
    while (*p != '\0' && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n')
    {
        p++;
    }

    len = (size_t)(p - start);
    if (len == 0 || len >= dst_size)
    {
        return false;
    }

    memcpy(dst, start, len);
    dst[len] = '\0';
    *cursor = p;
    return true;
}

boolean API_ParseRequest(char *buffer, size_t buffer_len, api_request_t *request)
{
    char method[10];
    char protocol[10];
    const char *cursor;
    struct yuarel url;
    char *http_entity_body;

    if (buffer == NULL || request == NULL || buffer_len == 0)
    {
        return false;
    }

    // The read loop guarantees buffer_len < sizeof(buffer); the terminator
    // always lands inside the caller's buffer.
    buffer[buffer_len] = '\0';

    cursor = buffer;
    if (!API_NextToken(&cursor, method, sizeof(method)))
    {
        return false;
    }
    if (!API_NextToken(&cursor, path, sizeof(path)))
    {
        return false;
    }
    if (!API_NextToken(&cursor, protocol, sizeof(protocol)))
    {
        return false;
    }

    // Anything other than the end of the request line after the three tokens
    // means a token exceeded its documented width or the line is malformed.
    while (*cursor == ' ' || *cursor == '\t')
    {
        cursor++;
    }
    if (*cursor != '\0' && *cursor != '\r' && *cursor != '\n')
    {
        return false;
    }

    snprintf(request->full_path, sizeof(request->full_path), "%s", path);

    if (-1 == yuarel_parse(&url, path))
    {
        return false;
    }

    http_entity_body = strstr(buffer, "\r\n\r\n");
    if (http_entity_body == NULL)
    {
        return false;
    }

    request->url = url;
    snprintf(request->method, sizeof(request->method), "%s", method);
    request->body = http_entity_body;
    return true;
}
