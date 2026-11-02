#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <SDL_net.h>
#include <math.h>

#include "api.h"
#include "d_player.h"
#include "m_menu.h"
#include "../d_event.h"
#include "../doomkeys.h"
#include "p_local.h"
#include "api_player_controller.h"
#include "api_world_controller.h"
#include "api_door_controller.h"
#include "api_object_controller.h"

/* Shared control state declared in api.h. Defined once here so every
   translation unit sees the same object. */
int keys_down[NUMKEYS];
int target_angle;


#define API_TOKEN_MAX 256

extern api_obj_description_t api_descriptors[];
char hud_message[512];

TCPsocket server_sd;
TCPsocket client_sd;
SDLNet_SocketSet set;

// Control-plane hardening. The API listens on loopback unless the operator
// explicitly opts into another interface, and mutations require a bearer
// token whenever the listener is not loopback-only.
static const char *api_bind_host = "127.0.0.1";
static char api_token[API_TOKEN_MAX];
static boolean api_token_configured = false;

void API_SendResponse(api_response_t resp);
void API_AfterTic();
api_response_t API_RouteRequest(api_request_t request);

// externally-defined game variables
extern player_t players[MAXPLAYERS];
extern int consoleplayer;

static boolean API_HostIsLoopback(const char *host)
{
    if (host == NULL)
    {
        return false;
    }
    return strcmp(host, "127.0.0.1") == 0 ||
           strcmp(host, "localhost") == 0 ||
           strcmp(host, "::1") == 0 ||
           strcmp(host, "[::1]") == 0;
}

static int API_StrnCaseCmp(const char *a, const char *b, size_t len)
{
    size_t i;
    for (i = 0; i < len; i++)
    {
        int ca = tolower((unsigned char)a[i]);
        int cb = tolower((unsigned char)b[i]);
        if (ca != cb || ca == 0)
        {
            return ca - cb;
        }
    }
    return 0;
}

// Copy a single HTTP header value out of the request head. Stops at the
// blank line so body content can never be mistaken for a header.
static boolean API_FindHeader(const char *raw, const char *name, char *out, size_t out_size)
{
    const char *end;
    const char *p;
    size_t name_len;

    if (raw == NULL || out == NULL || out_size == 0)
    {
        return false;
    }

    end = strstr(raw, "\r\n\r\n");
    p = raw;
    name_len = strlen(name);

    while (*p != '\0' && (end == NULL || p < end))
    {
        const char *eol;
        size_t line_len;

        eol = strstr(p, "\r\n");
        if (eol == NULL || (end != NULL && eol > end))
        {
            eol = end;
        }
        if (eol == NULL)
        {
            eol = p + strlen(p);
        }

        line_len = (size_t)(eol - p);
        if (line_len > name_len &&
            p[name_len] == ':' &&
            API_StrnCaseCmp(p, name, name_len) == 0)
        {
            const char *value = p + name_len + 1;
            size_t value_len;

            while (*value == ' ' || *value == '\t')
            {
                value++;
            }
            value_len = (size_t)(eol - value);
            while (value_len > 0 &&
                   (value[value_len - 1] == ' ' || value[value_len - 1] == '\t'))
            {
                value_len--;
            }
            if (value_len >= out_size)
            {
                value_len = out_size - 1;
            }
            memcpy(out, value, value_len);
            out[value_len] = '\0';
            return true;
        }

        if (eol == end || *eol == '\0')
        {
            break;
        }
        p = eol + 2;
    }

    return false;
}

static boolean API_RequestHasValidToken(const char *raw)
{
    char header[API_TOKEN_MAX];
    const char *value = header;

    if (!api_token_configured)
    {
        return false;
    }

    if (!API_FindHeader(raw, "Authorization", header, sizeof(header)) &&
        !API_FindHeader(raw, "X-Doom-Token", header, sizeof(header)))
    {
        return false;
    }

    if (strlen(header) > 7 && API_StrnCaseCmp(header, "Bearer ", 7) == 0)
    {
        value = header + 7;
    }

    return API_ConstantTimeEqual(value, api_token);
}

// Read-only requests are always allowed. Mutations require a configured
// token, and are refused outright when the listener is reachable off-host
// but no token was configured.
static boolean API_TokenRequiredFor(const char *method)
{
    if (API_HttpMethodIsReadOnly(method))
    {
        return false;
    }
    if (api_token_configured)
    {
        return true;
    }
    return !API_HostIsLoopback(api_bind_host);
}

static void API_ConfigureListener(void)
{
    const char *value;
    int p;

    p = M_CheckParmWithArgs("-apibind", 1);
    if (p > 0)
    {
        api_bind_host = myargv[p + 1];
    }
    else if ((value = getenv("DOOM_API_BIND")) != NULL && *value != '\0')
    {
        api_bind_host = value;
    }

    p = M_CheckParmWithArgs("-apitoken", 1);
    value = p > 0 ? myargv[p + 1] : getenv("DOOM_API_TOKEN");
    if (value != NULL && *value != '\0')
    {
        if (strlen(value) >= sizeof(api_token))
        {
            fprintf(stderr, "Error: API token must be shorter than %d bytes\n",
                    (int)sizeof(api_token));
            exit(EXIT_FAILURE);
        }
        snprintf(api_token, sizeof(api_token), "%s", value);
        api_token_configured = true;
    }
}

// SDL_net 2.x's SDLNet_TCP_Open() can only listen on INADDR_ANY; a concrete
// host makes it act as a client. Enforce the loopback default at accept time
// instead, so non-local peers are dropped before any bytes are read.
static boolean API_PeerAllowed(TCPsocket csd)
{
    IPaddress *peer;
    uint32_t host;

    if (!API_HostIsLoopback(api_bind_host))
    {
        return true;
    }

    peer = SDLNet_TCP_GetPeerAddress(csd);
    if (peer == NULL)
    {
        return false;
    }

    host = ntohl((uint32_t)peer->host);
    return (host >> 24) == 127;
}

void API_Init(int port)
{
    IPaddress ip;
    const char *host;

    target_angle = -1;

    API_ConfigureListener();

    if (SDLNet_Init() < 0)
    {
        fprintf(stderr, "Error: SDLNet_Init: %s\n", SDLNet_GetError());
        exit(EXIT_FAILURE);
    }
    if (SDLNet_ResolveHost(&ip, NULL, port) < 0)
    {
        fprintf(stderr, "Error: SDLNet_ResolveHost: %s\n", SDLNet_GetError());
        exit(EXIT_FAILURE);
    }

    if (!(server_sd = SDLNet_TCP_Open(&ip)))
    {
        fprintf(stderr, "Error: SDLNet_TCP_Open: %s\n", SDLNet_GetError());
        exit(EXIT_FAILURE);
    }
    if (!(set = SDLNet_AllocSocketSet(10)))
    {
        fprintf(stderr, "Error: SDLNet_AllocSocketSet: %s\n", SDLNet_GetError());
        exit(EXIT_FAILURE); 
    }

    for (int i = 0; i < NUMKEYS; i++) {
        keys_down[i] = -1;
    }

    host = SDLNet_ResolveIP(&ip);
    if (API_HostIsLoopback(api_bind_host))
    {
        printf("API_Init: Listening on port %d (loopback peers only)\n", port);
    }
    else
    {
        printf("API_Init: Listening for connections on %s:%d\n", host, port);
        if (api_token_configured)
        {
            printf("API_Init: mutating requests require the configured bearer token\n");
        }
        else
        {
            printf("API_Init: WARNING: non-loopback bind without DOOM_API_TOKEN; "
                   "mutating requests will be rejected\n");
        }
    }
}

void API_RunIO()
{
    TCPsocket csd;
    int recv_len;
    char buffer[1024];
    api_request_t request;
    api_response_t response;
    boolean parsed = false;
    IPaddress *remote_ip;
    const char *ip_str;

    if ((csd = SDLNet_TCP_Accept(server_sd)))
    {
        if (!API_PeerAllowed(csd))
        {
            // Off-host peer while the listener is in loopback-only mode.
            SDLNet_TCP_Close(csd);
        }
        else
        {
            // Only one client is tracked at a time; release the previous socket
            // before adopting the new one so it cannot be reused after close.
            if (client_sd != NULL)
            {
                SDLNet_TCP_DelSocket(set, client_sd);
                SDLNet_TCP_Close(client_sd);
                client_sd = NULL;
            }

            client_sd = csd;
            if (SDLNet_TCP_AddSocket(set, client_sd) < 0)
            {
                SDLNet_TCP_Close(client_sd);
                client_sd = NULL;
            }
        }
    }

    if (client_sd != NULL && SDLNet_CheckSockets(set, 0) > 0)
    {
        recv_len = SDLNet_TCP_Recv(client_sd, buffer, (int)sizeof(buffer));

        if (recv_len <= 0)
        {
            // Peer connected and closed without sending (or the connection
            // failed). Drop the dead socket instead of leaving it in the set.
            SDLNet_TCP_DelSocket(set, client_sd);
            SDLNet_TCP_Close(client_sd);
            client_sd = NULL;
        }
        else
        {
            memset(&request, 0, sizeof(request));

            if (recv_len >= (int)sizeof(buffer))
            {
                // No room for a terminator; reject rather than truncate.
                response = API_CreateErrorResponse(413, "request too large");
            }
            else if (API_ParseRequest(buffer, (size_t)recv_len, &request))
            {
                parsed = true;
                if (API_TokenRequiredFor(request.method) &&
                    !API_RequestHasValidToken(buffer))
                {
                    response = API_CreateErrorResponse(401, "authentication required");
                }
                else
                {
                    char msg[512];
                    snprintf(msg, sizeof(msg), "%s %s", request.method, request.full_path);
                    API_SetHUDMessage(msg);
                    response = API_RouteRequest(request);
                }
            }
            else
            {
                response = API_CreateErrorResponse(400, "invalid request");
            }

            API_SendResponse(response);

            // Access log only for successfully parsed requests.
            if (parsed)
            {
                remote_ip = SDLNet_TCP_GetPeerAddress(client_sd);
                ip_str = remote_ip != NULL ? SDLNet_ResolveIP(remote_ip) : "unknown";
                printf("access_log: %s - - - \"%s %s\" %d\n", ip_str,
                       request.method, request.full_path, response.status_code);
            }

            SDLNet_TCP_DelSocket(set, client_sd);
            SDLNet_TCP_Close(client_sd);
            client_sd = NULL;
        }
    }

    API_AfterTic();
}


static boolean API_BodyIsEmpty(const char *body)
{
    if (body == NULL)
    {
        return true;
    }
    while (*body != '\0')
    {
        if (!isspace((unsigned char)*body))
        {
            return false;
        }
        body++;
    }
    return true;
}


// ----
//  Route an http path + method to a controller action
// ----
api_response_t API_RouteRequest(api_request_t req)
{
    int id = 0;
    int id2 = 0;
    int distance = 0;
    int p = 0;
    float x = 0.0;
    float y = 0.0;
    struct yuarel_param params[3];
    char *method = req.method;
    char *path = req.url.path;
    cJSON *json = req.body != NULL ? cJSON_Parse(req.body) : NULL;

    if (json == NULL && !API_BodyIsEmpty(req.body))
    {
        return API_CreateErrorResponse(400, "invalid JSON body");
    }
    
    if (strcmp(path, "api/message") == 0)
    {
        if (strcmp(method, "POST") == 0)
        {
            return API_PostMessage(json);
        } 
        return API_CreateErrorResponse(405, "Method not allowed");
    }
    else if (strcmp(path, "api/player") == 0)
    {
        if (strcmp(method, "PATCH") == 0) 
        {
            return API_PatchPlayer(json);
        }
        else if (strcmp(method, "GET") == 0)
        {
            return API_GetPlayer();
        }
        else if (strcmp(method, "DELETE") == 0) {
            return API_DeletePlayer();
        }
        return API_CreateErrorResponse(405, "Method not allowed");
    }
    else if (strcmp(path, "api/player/actions") == 0)
    {
        if (strcmp(method, "POST") == 0)
        {
            return API_PostPlayerAction(json);
        }
        return API_CreateErrorResponse(405, "Method not allowed");
    }
    else if (strcmp(path, "api/player/turn") == 0)
    {
        if (strcmp(method, "POST") == 0)
        {
            return API_PostTurnDegrees(json);
        }
        return API_CreateErrorResponse(405, "Method not allowed");
    }
    else if (strcmp(path, "api/players") == 0)
    {
        if (strcmp(method, "GET") == 0)
        {
            return API_GetPlayers();
        }
        return API_CreateErrorResponse(405, "Method not allowed");
    }
    else if (strncmp(path, "api/players/", 12) == 0) {
        if (sscanf(path, "api/players/%d", &id) != 1) {
            return API_CreateErrorResponse(404, "path not found");
        }
        else if (strcmp(method, "GET") == 0)
        {
            return API_GetPlayerById(id);
        }
        else if (strcmp(method, "PATCH") == 0) 
        {
            return API_PatchPlayerById(json, id);
        }
        else if (strcmp(method, "DELETE") == 0) 
        {
            return API_DeletePlayerById(id);
        }
        return API_CreateErrorResponse(405, "Method not allowed");
    }
    else if (strcmp(path, "api/world") == 0) {
        if (strcmp(method, "GET") == 0) {
            return API_GetWorld();
        }
        else if (strcmp(method, "PATCH") == 0) {
            return API_PatchWorld(json);
        }
        return API_CreateErrorResponse(405, "Method not allowed");
    }
    else if (strcmp(path, "api/world/screenshot") == 0) {
        if (strcmp(method, "GET") == 0) {
            return API_GetWorldScreenshot();
        }
        return API_CreateErrorResponse(405, "Method not allowed");
    }
    else if (strcmp(path, "api/world/objects") == 0)
    {
        if (strcmp(method, "POST") == 0) 
        {
            return API_PostObject(json);
        }
        else if (strcmp(method, "GET") == 0)
        {
            distance = 0;
            p = yuarel_parse_query(req.url.query, '&', params, 1);
            while (p-- > 0) {
                if (strcmp("distance", params[p].key) == 0) {
                    distance = atoi(params[p].val);
                }
            }
            return API_GetObjects(distance);
        }
        return API_CreateErrorResponse(405, "Method not allowed");
    }
    else if (strncmp(path, "api/world/objects/", 18) == 0) {
        if (sscanf(path, "api/world/objects/%d", &id) != 1) {
            return API_CreateErrorResponse(404, "path not found");
        }
        if (strcmp(method, "DELETE") == 0)
        {
            return API_DeleteObject(id);
        }
        else if (strcmp(method, "GET") == 0)
        {
            return API_GetObject(id);
        }
        else if (strcmp(method, "PATCH") == 0)
        {
            return API_PatchObject(id, json);
        }
        return API_CreateErrorResponse(405, "Method not allowed");
    }
    else if (strncmp(path, "api/world/los/", 14) == 0) {
        if (sscanf(path, "api/world/los/%d/%d", &id, &id2) != 2) {
            return API_CreateErrorResponse(404, "path not found");
        }
        if (strcmp(method, "GET") == 0)
        {
            return API_GetLineOfSightToObject(id,id2);
        }
       
        return API_CreateErrorResponse(405, "Method not allowed");
    }

    else if (strcmp(path, "api/world/movetest") == 0) {
        if (strcmp(method, "GET") == 0)
        {
            p = yuarel_parse_query(req.url.query, '&', params, 3);
            while (p-- > 0) {
                if (strcmp("id", params[p].key) == 0) {
                    id = atoi(params[p].val); 
                }
                if (strcmp("x", params[p].key) == 0) {
                    x = atoi(params[p].val);
                }
                if (strcmp("y", params[p].key) == 0) {
                    y = atoi(params[p].val);
                }
            }
            return API_GetCheckTraverse(id,x,y);
        }
        return API_CreateErrorResponse(405, "Method not allowed");
    }

    else if (strcmp(path, "api/world/doors") == 0) {
        if (strcmp(method, "GET") == 0)
        {
            distance = 0;
            p = yuarel_parse_query(req.url.query, '&', params, 1);
            while (p-- > 0) {
                if (strcmp("distance", params[p].key) == 0) {
                    distance = atoi(params[p].val);
                }
            }
            return API_GetDoors(distance);
        }
        return API_CreateErrorResponse(405, "Method not allowed");
    }
    else if (strncmp(path, "api/world/doors/", 16) == 0)
    {
        if (sscanf(path, "api/world/doors/%d", &id) != 1) {
            return API_CreateErrorResponse(404, "path not found");
        }
        if (strcmp(method, "PATCH") == 0)
        {
            return API_PatchDoor(id, json);
        }
        else if (strcmp(method, "GET") == 0)
        {
            return API_GetDoor(id);
        }
        return API_CreateErrorResponse(405, "Method not allowed");
    }
    return API_CreateErrorResponse(404, "Not found");
}

void API_SendResponse(api_response_t resp) {
    char buffer[512];
    int len;
    const char *cors_origin;

    len = snprintf(buffer, sizeof(buffer),
                   "HTTP/1.0 %d\r\nConnection: close\r\n"
                   "Content-Type: application/json\r\n",
                   resp.status_code);
    if (len < 0)
    {
        return;
    }
    if ((size_t)len >= sizeof(buffer))
    {
        len = (int)sizeof(buffer) - 1;
    }

    // Cross-origin reads are opt-in. The previous unconditional
    // "Access-Control-Allow-Origin: *" let any web page the developer visited
    // read the local game state.
    cors_origin = getenv("DOOM_API_CORS_ORIGIN");
    if (cors_origin != NULL && *cors_origin != '\0')
    {
        int written = snprintf(buffer + len, sizeof(buffer) - (size_t)len,
                               "Access-Control-Allow-Origin: %s\r\n", cors_origin);
        if (written > 0)
        {
            len += written;
            if ((size_t)len >= sizeof(buffer))
            {
                len = (int)sizeof(buffer) - 1;
            }
        }
    }
    if ((size_t)len < sizeof(buffer) - 2)
    {
        buffer[len++] = '\r';
        buffer[len++] = '\n';
        buffer[len] = '\0';
    }

    len = (int)strlen(buffer);
    if (SDLNet_TCP_Send(client_sd, (void *)buffer, len) < len) {
        printf("failed to send all bytes\n");
    }
    if (resp.json)
    {
        char *jsonToString = cJSON_Print(resp.json);
        len = strlen(jsonToString);
        if (SDLNet_TCP_Send(client_sd, (void *)jsonToString, len) < len) {
            printf("failed to send all bytes\n");
        }
        free(jsonToString);
        cJSON_Delete(resp.json);
    }
}

api_response_t API_CreateErrorResponse(int status, char *message) {
  cJSON *body = cJSON_CreateObject();
  cJSON_AddStringToObject(body, "error", message);
  return (api_response_t) { status, body };
}

void postTurnEvent(int amount)
{
    event_t event;
    event.type = ev_mouse;
    event.data1 = 0;  // buttons held down
    event.data2 = amount;  // turn (positive clockwise)
    event.data3 = 0;  // move (max 16 units per tic)
    D_PostEvent(&event);
}

int angleToDegrees(angle_t angle)
{
    return ((double)angle / ANG_MAX) * 360;
}

int turnAmount(int remainingAngle)
{
    int amount = pow(remainingAngle, 2);
    if (amount > 500)
        return 500;
    else
        return amount;
}

void turnPlayer()
{
    int direction;
    int remaining;
    int playerAngle;

    if (target_angle >= 0)
    {
        player_t *player = &players[consoleplayer];
        playerAngle = angleToDegrees(player->mo->angle);
        remaining = target_angle - playerAngle;
        if (remaining < 0)
            remaining = remaining + 360;

        if (remaining < 180)
            direction = -1;
        else
            direction = 1;

        if (remaining == 0) {
            target_angle =- 1;
            return;
        }
        else
        {
            postTurnEvent(turnAmount(remaining) * direction);
        }
    }
}

void API_AfterTic()
{
    for (int i = 0; i < NUMKEYS; i++) {
        if (keys_down[i] >= 0) {
            keys_down[i]--;
        }
        if (keys_down[i] == 0) {
            event_t event;
            event.type = ev_keyup;
            event.data1 = i;
            event.data2 = 0;
            D_PostEvent(&event);
        }
    }

    turnPlayer();
}

// Helper methods

float API_FixedToFloat(fixed_t fixed) {
    double val = fixed;
    double ex = 1<<16;
    return val / ex;
}

fixed_t API_FloatToFixed(float val) {
    return (int)(val * (1<<16));
}

cJSON* DescribeMObj(mobj_t *obj)
{
    cJSON *pos;
    cJSON *flags;
    cJSON *root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "id", obj->id);
    cJSON_AddItemToObject(root, "position", pos = cJSON_CreateObject());
    cJSON_AddNumberToObject(pos, "x", API_FixedToFloat(obj->x));
    cJSON_AddNumberToObject(pos, "y", API_FixedToFloat(obj->y));
    cJSON_AddNumberToObject(pos, "z", API_FixedToFloat(obj->z));
    cJSON_AddNumberToObject(root, "angle", angleToDegrees(obj->angle));
    cJSON_AddNumberToObject(root, "height", API_FixedToFloat(obj->height));
    cJSON_AddNumberToObject(root, "health", obj->health);
    cJSON_AddNumberToObject(root, "typeId", mobjinfo[obj->type].doomednum);

    // this is... inefficient :(
    for (int i = 0; i < NUMDESCRIPTIONS; i++)
    {
        if (api_descriptors[i].id == mobjinfo[obj->type].doomednum) {
            cJSON_AddStringToObject(root, "type", api_descriptors[i].text);
            break;
        }
    }

    if (obj->target) {
        cJSON_AddNumberToObject(root, "attacking", obj->target->id);
    }

    flags = cJSON_CreateObject();
    if (obj->flags & MF_SPECIAL) cJSON_AddTrueToObject(flags, "MF_SPECIAL");
    if (obj->flags & MF_SOLID) cJSON_AddTrueToObject(flags, "MF_SOLID");
    if (obj->flags & MF_SHOOTABLE) cJSON_AddTrueToObject(flags, "MF_SHOOTABLE");
    if (obj->flags & MF_NOSECTOR) cJSON_AddTrueToObject(flags, "MF_NOSECTOR");
    if (obj->flags & MF_NOBLOCKMAP) cJSON_AddTrueToObject(flags, "MF_NOBLOCKMAP");
    if (obj->flags & MF_AMBUSH) cJSON_AddTrueToObject(flags, "MF_AMBUSH");
    if (obj->flags & MF_JUSTHIT) cJSON_AddTrueToObject(flags, "MF_JUSTHIT");
    if (obj->flags & MF_JUSTATTACKED) cJSON_AddTrueToObject(flags, "MF_JUSTATTACKED");
    if (obj->flags & MF_SPAWNCEILING) cJSON_AddTrueToObject(flags, "MF_SPAWNCEILING");
    if (obj->flags & MF_NOGRAVITY) cJSON_AddTrueToObject(flags, "MF_NOGRAVITY");
    if (obj->flags & MF_DROPOFF) cJSON_AddTrueToObject(flags, "MF_DROPOFF");
    if (obj->flags & MF_PICKUP) cJSON_AddTrueToObject(flags, "MF_PICKUP");
    if (obj->flags & MF_NOCLIP) cJSON_AddTrueToObject(flags, "MF_NOCLIP");
    if (obj->flags & MF_SLIDE) cJSON_AddTrueToObject(flags, "MF_SLIDE");
    if (obj->flags & MF_FLOAT) cJSON_AddTrueToObject(flags, "MF_FLOAT");
    if (obj->flags & MF_TELEPORT) cJSON_AddTrueToObject(flags, "MF_TELEPORT");
    if (obj->flags & MF_MISSILE) cJSON_AddTrueToObject(flags, "MF_MISSILE");
    if (obj->flags & MF_DROPPED) cJSON_AddTrueToObject(flags, "MF_DROPPED");
    if (obj->flags & MF_SHADOW) cJSON_AddTrueToObject(flags, "MF_SHADOW");
    if (obj->flags & MF_NOBLOOD) cJSON_AddTrueToObject(flags, "MF_NOBLOOD");
    if (obj->flags & MF_CORPSE) cJSON_AddTrueToObject(flags, "MF_CORPSE");
    if (obj->flags & MF_INFLOAT) cJSON_AddTrueToObject(flags, "MF_INFLOAT");
    if (obj->flags & MF_COUNTKILL) cJSON_AddTrueToObject(flags, "MF_COUNTKILL");
    if (obj->flags & MF_COUNTITEM) cJSON_AddTrueToObject(flags, "MF_COUNTITEM");
    if (obj->flags & MF_SKULLFLY) cJSON_AddTrueToObject(flags, "MF_SKULLFLY");
    if (obj->flags & MF_NOTDMATCH) cJSON_AddTrueToObject(flags, "MF_NOTDMATCH");
    if (obj->flags & MF_TRANSLATION) cJSON_AddTrueToObject(flags, "MF_TRANSLATION");

    cJSON_AddItemToObject(root, "flags", flags);
    return root;
}

void API_SetHUDMessage(char *msg)
{
    snprintf(hud_message, sizeof(hud_message), "%s", msg != NULL ? msg : "");
    players[consoleplayer].message = hud_message;
}

void API_FlipFlag(int *flags, int mask, boolean on) {
    if (on)
    {
        *flags |= mask;
    }
    else
    {
        *flags &= ~mask;
    }
}
