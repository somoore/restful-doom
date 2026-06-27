#include "agent_bridge.h"

#include <stdio.h>
#include <string.h>

#include "d_player.h"
#include "doomdef.h"
#include "doomstat.h"
#include "g_game.h"
#include "info.h"
#include "p_local.h"
#include "tables.h"

extern int leveltime;

static boolean agent_bridge_initialized = false;

static int ClampInt(int value, int low, int high)
{
    if (value < low)
    {
        return low;
    }
    if (value > high)
    {
        return high;
    }
    return value;
}

static uint32_t AngleToDegrees(angle_t angle)
{
    return (uint32_t) (((uint64_t) angle * 360ULL) / ANG_MAX);
}

static uint32_t WeaponMask(const player_t *player)
{
    uint32_t mask = 0;

    for (int i = 0; i < NUMWEAPONS && i < 32; ++i)
    {
        if (player->weaponowned[i])
        {
            mask |= 1u << i;
        }
    }

    return mask;
}

static uint32_t KeyCardMask(const player_t *player)
{
    uint32_t mask = 0;

    for (int i = 0; i < NUMCARDS && i < 32; ++i)
    {
        if (player->cards[i])
        {
            mask |= 1u << i;
        }
    }

    return mask;
}

static void FillMobjState(agent_mobj_state_t *state, const mobj_t *obj, const mobj_t *player)
{
    memset(state, 0, sizeof(*state));

    if (obj == NULL)
    {
        return;
    }

    state->id = obj->id;
    state->x_fp = obj->x;
    state->y_fp = obj->y;
    state->z_fp = obj->z;
    state->angle_degrees = AngleToDegrees(obj->angle);
    state->height_fp = obj->height;
    state->health = obj->health;
    state->internal_type = obj->type;
    state->type_id = mobjinfo[obj->type].doomednum;
    state->flags = (uint32_t) obj->flags;
    state->attacking_id = obj->target != NULL ? obj->target->id : 0;

    if (player != NULL)
    {
        state->distance_fp = P_AproxDistance(player->x - obj->x, player->y - obj->y);
    }
}

static boolean IsUsefulAgentObject(const mobj_t *obj, const mobj_t *player)
{
    if (obj == NULL || obj == player)
    {
        return false;
    }

    return (obj->flags & (MF_COUNTKILL | MF_COUNTITEM | MF_SPECIAL | MF_SHOOTABLE | MF_MISSILE)) != 0;
}

static boolean IsLivingEnemy(const mobj_t *obj)
{
    return obj != NULL && obj->health > 0 && (obj->flags & MF_COUNTKILL) != 0;
}

static void FillPlayerState(agent_game_state_snapshot_t *snapshot)
{
    player_t *player;

    if (consoleplayer < 0 || consoleplayer >= MAXPLAYERS)
    {
        return;
    }

    player = &players[consoleplayer];
    if (!playeringame[consoleplayer] || player->mo == NULL)
    {
        return;
    }

    snapshot->player_active = 1;
    FillMobjState(&snapshot->player.object, player->mo, player->mo);
    snapshot->player.health = player->mo->health;
    snapshot->player.armor = player->armorpoints;
    snapshot->player.kills = player->killcount;
    snapshot->player.items = player->itemcount;
    snapshot->player.secrets = player->secretcount;
    snapshot->player.ready_weapon = player->readyweapon;
    snapshot->player.owned_weapons = WeaponMask(player);
    snapshot->player.ammo_bullets = player->ammo[am_clip];
    snapshot->player.ammo_shells = player->ammo[am_shell];
    snapshot->player.ammo_cells = player->ammo[am_cell];
    snapshot->player.ammo_rockets = player->ammo[am_misl];
    snapshot->player.key_cards = KeyCardMask(player);
    snapshot->player.cheat_flags = (uint32_t) player->cheats;
    snapshot->player.last_attacked_by = player->attacker != NULL ? player->attacker->id : 0;
}

static void FillThinkerStates(agent_game_state_snapshot_t *snapshot)
{
    thinker_t *thinker;
    mobj_t *player;

    if (consoleplayer < 0 || consoleplayer >= MAXPLAYERS)
    {
        player = NULL;
    }
    else
    {
        player = players[consoleplayer].mo;
    }

    for (thinker = thinkercap.next; thinker != &thinkercap; thinker = thinker->next)
    {
        mobj_t *obj;
        agent_mobj_state_t state;

        if (thinker->function.acp1 != (actionf_p1) P_MobjThinker)
        {
            continue;
        }

        obj = (mobj_t *) thinker;
        if (!IsUsefulAgentObject(obj, player))
        {
            continue;
        }

        FillMobjState(&state, obj, player);

        if (IsLivingEnemy(obj) && snapshot->enemy_count < AGENT_MAX_ENEMIES)
        {
            state.line_of_sight = player != NULL ? (uint32_t) P_CheckSight(player, obj) : 0;
            snapshot->enemies[snapshot->enemy_count++] = state;
        }

        if (snapshot->object_count < AGENT_MAX_OBJECTS)
        {
            snapshot->objects[snapshot->object_count++] = state;
        }
    }
}

void AgentBridge_Init(int port)
{
    if (agent_bridge_initialized)
    {
        return;
    }

    if (port < 0 || port > 65535)
    {
        fprintf(stderr, "AgentBridge_Init: invalid gRPC port %d\n", port);
        return;
    }

    if (restfuldoom_agent_init((uint16_t) port) != 0)
    {
        fprintf(stderr, "AgentBridge_Init: failed to start gRPC bridge\n");
        return;
    }

    agent_bridge_initialized = true;
    printf("AgentBridge_Init: Listening for gRPC agent connections on 0.0.0.0:%d\n",
           port == 0 ? 50051 : port);
}

void AgentBridge_AfterTic(int completed_tic)
{
    agent_game_state_snapshot_t snapshot;

    if (!agent_bridge_initialized)
    {
        return;
    }

    memset(&snapshot, 0, sizeof(snapshot));
    snapshot.tick = (uint64_t) completed_tic;
    snapshot.episode = gameepisode;
    snapshot.map = gamemap;
    snapshot.skill = gameskill;
    snapshot.level_time = leveltime;
    snapshot.total_kills = totalkills;
    snapshot.total_items = totalitems;
    snapshot.total_secrets = totalsecret;
    snapshot.gamestate = gamestate;

    FillPlayerState(&snapshot);
    FillThinkerStates(&snapshot);
    restfuldoom_agent_publish_state(&snapshot);
}

void AgentBridge_ApplyTiccmd(ticcmd_t *cmd)
{
    agent_player_action_t action;
    int actions_applied = 0;

    if (!agent_bridge_initialized || cmd == NULL)
    {
        return;
    }

    while (actions_applied < 16 && restfuldoom_agent_take_action(&action) == 1)
    {
        cmd->forwardmove = (signed char) ClampInt(
            cmd->forwardmove + action.forward_move,
            -50,
            50);
        cmd->sidemove = (signed char) ClampInt(
            cmd->sidemove + action.side_move,
            -50,
            50);
        cmd->angleturn = (short) ClampInt(
            cmd->angleturn + action.angle_turn,
            -32768,
            32767);
        cmd->buttons |= action.buttons;
        ++actions_applied;
    }
}
