#include "api_world_controller.h"
#include "g_game.h"
#include "p_local.h"
#include "v_video.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// externally-defined game variables
extern  skill_t         gameskill;
extern int              gameepisode;
extern int              gamemap;
extern player_t         players[MAXPLAYERS];
extern int consoleplayer;
static char *doom_skills[] =
{
    "I'm too young to die.", "Hey, not too rough.", "Hurt me plenty.",
    "Ultra-Violence.", "NIGHTMARE!",
};

cJSON *DescribeWorldState() 
{
    cJSON *root;
    sector_t *sector;
    char *iwadname;
    int iwadparm;
    
    root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "episode", gameepisode);
    cJSON_AddNumberToObject(root, "map", gamemap);
    
    iwadparm = M_CheckParmWithArgs("-iwad", 1);
    iwadname = myargv[iwadparm + 1];
    cJSON_AddStringToObject(root, "wad", iwadname);

    if (API_IndexInRange((int)gameskill, (int)(sizeof(doom_skills) / sizeof(doom_skills[0]))))
    {
        cJSON_AddStringToObject(root, "skill", doom_skills[gameskill]);
    }
    else
    {
        cJSON_AddStringToObject(root, "skill", "unknown");
    }

    if (players[consoleplayer].mo != NULL)
    {
        sector = players[consoleplayer].mo->subsector->sector;
        cJSON_AddStringToObject(root, "lights", sector->lightlevel == 75 ? "off" : "on");
    }
    else
    {
        cJSON_AddStringToObject(root, "lights", "unknown");
    }

    return root;
}


// Controller methods

api_response_t API_GetWorld() {
  return (api_response_t) { 200, DescribeWorldState() };
}

api_response_t API_PatchWorld(cJSON *req) 
{
    cJSON *val;
    boolean changed = false;
    int new_episode = gameepisode;
    int new_map = gamemap;
    int new_skill = (int)gameskill;

    if (M_CheckParm("-connect") > 0)
        return API_CreateErrorResponse(403, "clients may not patch the world");
      
    val = cJSON_GetObjectItem(req, "episode");
    if (val) 
    {
        if (cJSON_IsNumber(val))
        {
            if (val->valueint < 1 || val->valueint > 4)
            {
                return API_CreateErrorResponse(400, "episode must be between 1 and 4");
            }
            new_episode = val->valueint;
            changed = true;
        }
        else
        {
            return API_CreateErrorResponse(400, "episode must be integer");
        }
    }

    val = cJSON_GetObjectItem(req, "map");
    if (val) 
    {
        if (cJSON_IsNumber(val))
        {
            if (val->valueint < 1 || val->valueint > 9)
            {
                return API_CreateErrorResponse(400, "map must be between 1 and 9");
            }
            new_map = val->valueint;
            changed = true;
        }
        else
        {
            return API_CreateErrorResponse(400, "map must be integer");
        }
    }
  
    val = cJSON_GetObjectItem(req, "skill");
    if (val) 
    {
        if (cJSON_IsNumber(val))
        {
            if (val->valueint < sk_baby || val->valueint > sk_nightmare)
            {
                return API_CreateErrorResponse(400, "skill must be between 0 and 4");
            }
            new_skill = val->valueint;
            changed = true;
        }
        else
        {
            return API_CreateErrorResponse(400, "skill must be integer");
        }
    }

    if (changed) 
    {
        gameskill = new_skill;
        gameepisode = new_episode;
        gamemap = new_map;
        G_DeferedInitNew (gameskill, gameepisode, gamemap);
    }

    val = cJSON_GetObjectItem(req, "lights");
    if (val) 
    {
        if (cJSON_IsString(val))
        {
            if (players[consoleplayer].mo == NULL)
            {
                return API_CreateErrorResponse(409, "player has no active mobj");
            }
            if (strcmp(val->valuestring, "on") == 0) 
            {
                struct line_s *line = players[consoleplayer].mo->subsector->sector->lines[0];
                EV_LightTurnOn(line, 255);
            }
            else 
            {
                struct line_s *line = players[consoleplayer].mo->subsector->sector->lines[0];
                EV_LightTurnOn(line, 75);
            }
        }
        else
        {
            return API_CreateErrorResponse(400, "lights must be string: on|off");
        }
    }
  
    return (api_response_t) { 200, DescribeWorldState() };
}

static char screenshot_dir[512];

static const char *API_ScreenshotDir(void)
{
    const char *base;

    if (screenshot_dir[0] != '\0')
    {
        return screenshot_dir;
    }

    base = getenv("TMPDIR");
    if (base == NULL || *base == '\0')
    {
        base = "/tmp";
    }

    if (snprintf(screenshot_dir, sizeof(screenshot_dir), "%s/restfuldoom-XXXXXX", base)
            >= (int)sizeof(screenshot_dir))
    {
        screenshot_dir[0] = '\0';
        return NULL;
    }

    if (mkdtemp(screenshot_dir) == NULL)
    {
        screenshot_dir[0] = '\0';
        return NULL;
    }

    return screenshot_dir;
}

api_response_t API_GetWorldScreenshot()
{
    cJSON *body;
    const char *dir;
    char pcx_path[600];
    char pattern[600];

    dir = API_ScreenshotDir();
    if (dir == NULL)
    {
        return API_CreateErrorResponse(500, "screenshot directory unavailable");
    }

    snprintf(pcx_path, sizeof(pcx_path), "%s/DOOM00.pcx", dir);
    snprintf(pattern, sizeof(pattern), "%s/DOOM%%02i.%%s", dir);

    // Private, mkdtemp-created directory: no predictable /tmp path and no
    // symlink target for a local attacker to redirect the write.
    unlink(pcx_path);
    V_ScreenShot(pattern);
    body = cJSON_CreateObject();
    cJSON_AddStringToObject(body, "path", pcx_path);
    return (api_response_t) {200, body};
}
