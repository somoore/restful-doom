#ifndef __API_VALIDATE_H__
#define __API_VALIDATE_H__

#include <stddef.h>
#include <string.h>

// Small, dependency-free validation helpers shared by the REST controllers.
// They live in a header so the C regression tests and fuzz targets can
// exercise the exact code the server uses.

static inline int API_IndexInRange(int value, int count)
{
    return value >= 0 && value < count;
}

// The player API uses 1-based weapon numbers (weapon 1 == weaponowned[0]).
static inline int API_WeaponNumberInRange(int value, int numweapons)
{
    return value >= 1 && value <= numweapons;
}

static inline int API_HttpMethodIsReadOnly(const char *method)
{
    if (method == NULL)
        return 0;
    return strcmp(method, "GET") == 0 ||
           strcmp(method, "HEAD") == 0 ||
           strcmp(method, "OPTIONS") == 0;
}

// Length-independent comparison; used for bearer tokens so a timing
// side channel cannot be used to guess the configured value.
static inline int API_ConstantTimeEqual(const char *a, const char *b)
{
    size_t alen;
    size_t blen;
    size_t len;
    unsigned char diff;
    size_t i;

    if (a == NULL || b == NULL)
        return 0;

    alen = strlen(a);
    blen = strlen(b);
    len = alen > blen ? alen : blen;
    diff = (unsigned char)((alen ^ blen) != 0);

    for (i = 0; i < len; i++)
    {
        unsigned char ca = i < alen ? (unsigned char)a[i] : 0;
        unsigned char cb = i < blen ? (unsigned char)b[i] : 0;
        diff |= (unsigned char)(ca ^ cb);
    }

    return diff == 0;
}

#endif
