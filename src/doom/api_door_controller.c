#include <stdio.h>
#include <string.h>

#include "api.h"
#include "api_cJSON.h"

#include "p_local.h"

// externally-defined game variables
extern player_t players[MAXPLAYERS];
extern int consoleplayer;

boolean IsLineADoor(line_t *line) 
{
    switch (line->special) 
    {
        case 1:        // Vertical Door
        case 26:      // Blue Door/Locked
        case 27:      // Yellow Door /Locked
        case 28:      // Red Door /Locked
        case 31:      // Manual door open
        case 32:      // Blue locked door open
        case 33:      // Red locked door open
        case 34:      // Yellow locked door open
        case 117:     // Blazing door raise
        case 118:     // Blazing door open
            return true;
        default:
            return false;
    }
}

static sector_t *DoorSector(line_t *line)
{
    if (line == NULL)
    {
        return NULL;
    }
    // sidenum[1] is -1 for a one-sided line; never index with it.
    if (line->sidenum[1] < 0 || line->sidenum[1] >= numsides)
    {
        return NULL;
    }
    return sides[line->sidenum[1]].sector;
}

cJSON* DescribeDoor(int id, line_t *line)
{
    sector_t *sec = DoorSector(line);
    cJSON *door = cJSON_CreateObject();
    cJSON_AddNumberToObject(door, "id", id);
    cJSON_AddNumberToObject(door, "specialType", line->special);
    cJSON_AddStringToObject(door, "state",
                            sec == NULL ? "unknown"
                                        : (sec->floorheight == sec->ceilingheight ? "closed" : "open"));
    char *key_color = "none";
    switch (line->special) {
        case 26:
        case 32:
            key_color = "blue";
            break;
        case 27:
        case 34:
            key_color = "yellow";
            break;
        case 28:
        case 33:
            key_color = "red";
            break;
    }
    cJSON_AddStringToObject(door, "keyRequired", key_color);
    cJSON *line_obj = cJSON_CreateObject();
    cJSON *v1 = cJSON_CreateObject();
    cJSON_AddNumberToObject(v1, "x", API_FixedToFloat(line->v1->x));
    cJSON_AddNumberToObject(v1, "y", API_FixedToFloat(line->v1->y));
    cJSON *v2 = cJSON_CreateObject();
    cJSON_AddNumberToObject(v2, "x", API_FixedToFloat(line->v2->x));
    cJSON_AddNumberToObject(v2, "y", API_FixedToFloat(line->v2->y));
    cJSON_AddItemToObject(line_obj, "v1", v1);
    cJSON_AddItemToObject(line_obj, "v2", v2);
    cJSON_AddItemToObject(door, "line", line_obj);
    return door;
}

// Controller methods

api_response_t API_GetDoors(int max_distance)
{
    mobj_t * player = players[consoleplayer].mo;
    int sectorIds[512];
    int sectorCount = 0;
    cJSON *root = cJSON_CreateArray();

    if (player == NULL)
    {
        cJSON_Delete(root);
        return API_CreateErrorResponse(409, "player has no active mobj");
    }

    for (int i = 0; i < numlines; i++)
    {
        line_t *line = &lines[i];
        sector_t *sec;

        if (!IsLineADoor(line))
        {
            continue;
        }

        sec = DoorSector(line);
        if (sec == NULL)
        {
            continue;
        }

        int seenSector = 0;
        for (int j = 0; j < sectorCount; j++)
        {
            if (sectorIds[j] == sec->id)
            {
                seenSector = 1;
                break;
            }
        }
        if (seenSector)
        {
            continue;
        }

        float dist = API_FixedToFloat(P_AproxDistance(player->x - line->v1->x, player->y - line->v1->y));
        if (max_distance == 0 || dist < max_distance)
        {
            cJSON *door = DescribeDoor(i, line);
            cJSON_AddNumberToObject(door, "distance", dist);
            cJSON_AddItemToArray(root, door);
        }
        if (sectorCount < (int)(sizeof(sectorIds) / sizeof(sectorIds[0])))
        {
            sectorIds[sectorCount++] = sec->id;
        }
    }
    api_response_t resp = {200, root};
    return resp;
}

api_response_t API_GetDoor(int id)
{
    if (!API_IndexInRange(id, numlines) || !IsLineADoor(&lines[id]))
    {
        return API_CreateErrorResponse(404, "door does not exist");
    }
    cJSON *door = DescribeDoor(id, &lines[id]);
    api_response_t resp = {200, door};
    return resp;
}

api_response_t API_PatchDoor(int id, cJSON *req)
{
    cJSON *state;
    sector_t *sec;

    if (M_CheckParm("-connect") > 0)
        return API_CreateErrorResponse(403, "clients may not patch doors");

    if (!API_IndexInRange(id, numlines) || !IsLineADoor(&lines[id]))
    {
        return API_CreateErrorResponse(404, "door does not exist");
    }

    state = cJSON_GetObjectItem(req, "state");
    if (state)
    {
        if (!cJSON_IsString(state))
        {
            return API_CreateErrorResponse(400, "state must be a string");
        }
        sec = DoorSector(&lines[id]);
        if (sec != NULL &&
            ((strcmp(state->valuestring, "open") == 0 && sec->floorheight == sec->ceilingheight) ||
             (strcmp(state->valuestring, "closed") == 0 && sec->floorheight != sec->ceilingheight)))
        {
            if (players[consoleplayer].mo == NULL)
            {
                return API_CreateErrorResponse(409, "player has no active mobj");
            }
            //EV_DoDoor(&lines[id], vld_open);
            EV_VerticalDoor(&lines[id], players[consoleplayer].mo);
        }
    }
    cJSON *door = DescribeDoor(id, &lines[id]);
    api_response_t resp = {200, door};
    return resp;
}

