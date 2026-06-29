#include "agent_bridge.h"

#include <stdio.h>
#include <string.h>

#include "d_player.h"
#include "doomdef.h"
#include "doomstat.h"
#include "g_game.h"
#include "info.h"
#include "m_random.h"
#include "p_local.h"
#include "r_main.h"
#include "r_state.h"
#include "tables.h"

static boolean agent_bridge_initialized = false;
static fixed_t agent_probe_distance;
static fixed_t agent_probe_height;
static int agent_probe_blocking_line_special;
static int agent_probe_block_distance_fp;
static boolean agent_probe_use_line_ahead;
static boolean agent_pending_start_active = false;
static agent_control_request_t agent_pending_start;
static boolean agent_pending_seed_active = false;
static agent_control_request_t agent_pending_seed;

#define AGENT_NAV_PROBE_DISTANCE (96 * FRACUNIT)

static boolean IsLivingEnemy(const mobj_t *obj);
static int ProgressionLinePriority(int special);
static void SnapshotDescription(char *out, size_t out_size,
                                const agent_control_request_t *request);
static const int agent_nav_direction_offsets[AGENT_MAX_NAV_DIRECTIONS] = {
    -90,
    -60,
    -30,
    -15,
    0,
    15,
    30,
    60,
    90,
};

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

static void SnapshotDescription(char *out, size_t out_size,
                                const agent_control_request_t *request)
{
    size_t index;

    if (out == NULL || out_size == 0)
    {
        return;
    }

    out[0] = '\0';
    if (request == NULL)
    {
        return;
    }

    for (index = 0;
         index + 1 < out_size && index < AGENT_CONTROL_SNAPSHOT_DESCRIPTION_BYTES;
         ++index)
    {
        if (request->snapshot_description[index] == '\0')
        {
            break;
        }
        out[index] = (char) request->snapshot_description[index];
    }
    out[index] = '\0';
}

static uint32_t AngleToDegrees(angle_t angle)
{
    return (uint32_t) (((uint64_t) angle * 360ULL) / ANG_MAX);
}

static angle_t DegreesToAngleOffset(int degrees)
{
    return (angle_t) (((int64_t) degrees * (int64_t) ANG_MAX) / 360);
}

static angle_t DegreesToAngle(int degrees)
{
    int normalized = degrees % 360;

    if (normalized < 0)
    {
        normalized += 360;
    }

    return (angle_t) (((uint64_t) normalized * (uint64_t) ANG_MAX) / 360ULL);
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

static boolean IsDamagingSectorSpecial(int special)
{
    return special == 4 || special == 5 || special == 7 || special == 11 || special == 16;
}

static int SectorDamagePer32Tics(int special)
{
    switch (special)
    {
        case 5:
            return 10;
        case 7:
            return 5;
        case 4:
        case 11:
        case 16:
            return 20;
        default:
            return 0;
    }
}

static boolean IsExitDamageSectorSpecial(int special)
{
    return special == 11;
}

static boolean IsWalkTriggerLineSpecial(int special)
{
    return special == 36 || special == 88;
}

static boolean IsExitLineSpecial(int special)
{
    return special == 11 || special == 51;
}

static int ProgressionLinePriority(int special)
{
    switch (special)
    {
        case 88:
            return 0;
        case 36:
            return 1;
        case 11:
        case 51:
            return 2;
        default:
            return -1;
    }
}

static boolean AgentBridge_ProbeLineTraverse(intercept_t *in)
{
    line_t *line;

    if (!in->isaline)
    {
        return true;
    }

    line = in->d.line;
    if (line->special)
    {
        agent_probe_use_line_ahead = true;
    }

    if (!line->backsector || (line->flags & ML_BLOCKING))
    {
        agent_probe_blocking_line_special = line->special;
        agent_probe_block_distance_fp = FixedMul(agent_probe_distance, in->frac);
        return false;
    }

    P_LineOpening(line);
    if (openrange < agent_probe_height)
    {
        agent_probe_blocking_line_special = line->special;
        agent_probe_block_distance_fp = FixedMul(agent_probe_distance, in->frac);
        return false;
    }

    return true;
}

static uint32_t AgentBridge_ProbeDirection(
    const mobj_t *obj,
    angle_t angle,
    fixed_t distance,
    int *blocking_line_special,
    int *block_distance_fp,
    boolean *use_line_ahead)
{
    int fine_angle;
    fixed_t x2;
    fixed_t y2;
    boolean clear;

    fine_angle = angle >> ANGLETOFINESHIFT;
    agent_probe_distance = distance;
    agent_probe_height = obj->height;
    agent_probe_blocking_line_special = 0;
    agent_probe_block_distance_fp = distance;
    agent_probe_use_line_ahead = false;

    x2 = obj->x + FixedMul(distance, finecosine[fine_angle]);
    y2 = obj->y + FixedMul(distance, finesine[fine_angle]);

    clear = P_PathTraverse(
        obj->x,
        obj->y,
        x2,
        y2,
        PT_ADDLINES | PT_EARLYOUT,
        AgentBridge_ProbeLineTraverse);

    if (blocking_line_special != NULL)
    {
        *blocking_line_special = agent_probe_blocking_line_special;
    }
    if (block_distance_fp != NULL)
    {
        *block_distance_fp = clear ? distance : agent_probe_block_distance_fp;
    }
    if (use_line_ahead != NULL)
    {
        *use_line_ahead = agent_probe_use_line_ahead;
    }

    return clear ? 1u : 0u;
}

static void FillSectorProbe(agent_sector_probe_t *probe, const mobj_t *obj)
{
    const sector_t *sector;

    if (probe == NULL || obj == NULL || obj->subsector == NULL)
    {
        return;
    }

    sector = obj->subsector->sector;
    if (sector == NULL)
    {
        return;
    }

    probe->sector_id = sector->id;
    probe->special = sector->special;
    probe->floor_height_fp = sector->floorheight;
    probe->ceiling_height_fp = sector->ceilingheight;
    probe->light_level = sector->lightlevel;
    probe->damaging = IsDamagingSectorSpecial(sector->special) ? 1u : 0u;
    probe->damage_per_32_tics = SectorDamagePer32Tics(sector->special);
    probe->exit_damage = IsExitDamageSectorSpecial(sector->special) ? 1u : 0u;
}

static void FillUseLine(
    agent_use_line_t *target,
    int line_id,
    const line_t *line,
    const mobj_t *obj,
    fixed_t midpoint_x,
    fixed_t midpoint_y,
    fixed_t nearest_x,
    fixed_t nearest_y,
    fixed_t distance,
    fixed_t nearest_distance)
{
    if (target == NULL || line == NULL || obj == NULL)
    {
        return;
    }

    target->line_id = line_id;
    target->midpoint_x_fp = midpoint_x;
    target->midpoint_y_fp = midpoint_y;
    target->midpoint_z_fp = obj->z;
    target->start_x_fp = line->v1->x;
    target->start_y_fp = line->v1->y;
    target->end_x_fp = line->v2->x;
    target->end_y_fp = line->v2->y;
    target->nearest_x_fp = nearest_x;
    target->nearest_y_fp = nearest_y;
    target->special = line->special;
    target->tag = line->tag;
    target->distance_fp = distance;
    target->nearest_distance_fp = nearest_distance;
}

static void FillNavigationProbe(agent_game_state_snapshot_t *snapshot, const mobj_t *obj)
{
    int front_special = 0;
    int front_distance = AGENT_NAV_PROBE_DISTANCE;
    boolean use_line_ahead = false;
    int best_route_priority = INT32_MAX;
    fixed_t best_route_distance = INT32_MAX;

    if (obj == NULL)
    {
        return;
    }

    FillSectorProbe(&snapshot->navigation.current_sector, obj);
    snapshot->navigation.probe_distance_fp = AGENT_NAV_PROBE_DISTANCE;
    snapshot->navigation.forward_open = AgentBridge_ProbeDirection(
        obj,
        obj->angle,
        AGENT_NAV_PROBE_DISTANCE,
        &front_special,
        &front_distance,
        &use_line_ahead);
    snapshot->navigation.back_open = AgentBridge_ProbeDirection(
        obj,
        obj->angle + ANG180,
        AGENT_NAV_PROBE_DISTANCE,
        NULL,
        NULL,
        NULL);
    snapshot->navigation.left_open = AgentBridge_ProbeDirection(
        obj,
        obj->angle + ANG90,
        AGENT_NAV_PROBE_DISTANCE,
        NULL,
        NULL,
        NULL);
    snapshot->navigation.right_open = AgentBridge_ProbeDirection(
        obj,
        obj->angle - ANG90,
        AGENT_NAV_PROBE_DISTANCE,
        NULL,
        NULL,
        NULL);
    snapshot->navigation.use_line_ahead = use_line_ahead ? 1u : 0u;
    snapshot->navigation.front_blocking_line_special = front_special;
    snapshot->navigation.front_block_distance_fp = front_distance;

    snapshot->navigation.direction_count = AGENT_MAX_NAV_DIRECTIONS;
    for (int i = 0; i < AGENT_MAX_NAV_DIRECTIONS; ++i)
    {
        int special = 0;
        int distance = AGENT_NAV_PROBE_DISTANCE;
        boolean use_line = false;
        const int offset = agent_nav_direction_offsets[i];

        snapshot->navigation.directions[i].angle_offset_degrees = offset;
        snapshot->navigation.directions[i].open = AgentBridge_ProbeDirection(
            obj,
            obj->angle + DegreesToAngleOffset(offset),
            AGENT_NAV_PROBE_DISTANCE,
            &special,
            &distance,
            &use_line);
        snapshot->navigation.directions[i].block_distance_fp = distance;
        snapshot->navigation.directions[i].blocking_line_special = special;
        snapshot->navigation.directions[i].use_line_ahead = use_line ? 1u : 0u;
    }

    snapshot->navigation.use_line_count = 0;
    for (int i = 0; i < numlines; ++i)
    {
        const line_t *line = &lines[i];
        fixed_t midpoint_x;
        fixed_t midpoint_y;
        fixed_t nearest_x;
        fixed_t nearest_y;
        fixed_t distance;
        fixed_t nearest_distance;
        double dx;
        double dy;
        double length_sq;
        double projection;
        int route_priority;
        agent_use_line_t candidate;
        uint32_t slot;

        if (line->special == 0)
        {
            continue;
        }

        midpoint_x = line->v1->x + ((line->v2->x - line->v1->x) / 2);
        midpoint_y = line->v1->y + ((line->v2->y - line->v1->y) / 2);
        distance = P_AproxDistance(obj->x - midpoint_x, obj->y - midpoint_y);
        dx = (double) (line->v2->x - line->v1->x);
        dy = (double) (line->v2->y - line->v1->y);
        length_sq = dx * dx + dy * dy;
        if (length_sq <= 1.0)
        {
            projection = 0.0;
        }
        else
        {
            projection = (((double) (obj->x - line->v1->x) * dx)
                          + ((double) (obj->y - line->v1->y) * dy))
                         / length_sq;
            if (projection < 0.0)
            {
                projection = 0.0;
            }
            else if (projection > 1.0)
            {
                projection = 1.0;
            }
        }
        nearest_x = line->v1->x + (fixed_t) (projection * dx);
        nearest_y = line->v1->y + (fixed_t) (projection * dy);
        nearest_distance = P_AproxDistance(obj->x - nearest_x, obj->y - nearest_y);
        memset(&candidate, 0, sizeof(candidate));
        FillUseLine(
            &candidate,
            i,
            line,
            obj,
            midpoint_x,
            midpoint_y,
            nearest_x,
            nearest_y,
            distance,
            nearest_distance);

        route_priority = ProgressionLinePriority(line->special);
        if (route_priority >= 0
            && (snapshot->navigation.has_route_waypoint == 0
                || route_priority < best_route_priority
                || (route_priority == best_route_priority
                    && nearest_distance < best_route_distance)))
        {
            snapshot->navigation.has_route_waypoint = 1u;
            snapshot->navigation.route_waypoint = candidate;
            snapshot->navigation.route_waypoint_priority = route_priority;
            snapshot->navigation.route_waypoint_exit = IsExitLineSpecial(line->special) ? 1u : 0u;
            snapshot->navigation.route_waypoint_walk_trigger =
                IsWalkTriggerLineSpecial(line->special) ? 1u : 0u;
            best_route_priority = route_priority;
            best_route_distance = nearest_distance;
        }

        if (snapshot->navigation.use_line_count < AGENT_MAX_USE_LINES)
        {
            slot = snapshot->navigation.use_line_count++;
        }
        else
        {
            int farthest = 0;
            for (int j = 1; j < AGENT_MAX_USE_LINES; ++j)
            {
                if (snapshot->navigation.use_lines[j].nearest_distance_fp
                    > snapshot->navigation.use_lines[farthest].nearest_distance_fp)
                {
                    farthest = j;
                }
            }
            if (nearest_distance >= snapshot->navigation.use_lines[farthest].nearest_distance_fp)
            {
                continue;
            }
            slot = (uint32_t) farthest;
        }

        snapshot->navigation.use_lines[slot] = candidate;
    }
}

static void FillCombatProbe(agent_game_state_snapshot_t *snapshot, mobj_t *obj)
{
    mobj_t *previous_linetarget;
    mobj_t *target;
    fixed_t slope;

    if (obj == NULL)
    {
        return;
    }

    previous_linetarget = linetarget;
    slope = P_AimLineAttack(obj, obj->angle, MISSILERANGE);
    target = linetarget;
    linetarget = previous_linetarget;

    snapshot->combat.range_fp = MISSILERANGE;
    snapshot->combat.aim_slope_fp = slope;

    if (target == NULL)
    {
        return;
    }

    snapshot->combat.has_shootable_target = 1u;
    snapshot->combat.target_is_enemy = IsLivingEnemy(target) ? 1u : 0u;
    snapshot->combat.target_id = target->id;
    snapshot->combat.target_health = target->health;
    snapshot->combat.target_distance_fp = P_AproxDistance(obj->x - target->x, obj->y - target->y);
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

static mobj_t *FindNearestLivingEnemy(const mobj_t *player)
{
    thinker_t *thinker;
    mobj_t *nearest = NULL;
    fixed_t nearest_distance = INT32_MAX;

    if (player == NULL)
    {
        return NULL;
    }

    for (thinker = thinkercap.next; thinker != &thinkercap; thinker = thinker->next)
    {
        mobj_t *obj;
        fixed_t distance;

        if (thinker->function.acp1 != (actionf_p1) P_MobjThinker)
        {
            continue;
        }

        obj = (mobj_t *) thinker;
        if (!IsLivingEnemy(obj))
        {
            continue;
        }

        distance = P_AproxDistance(player->x - obj->x, player->y - obj->y);
        if (nearest == NULL || distance < nearest_distance)
        {
            nearest = obj;
            nearest_distance = distance;
        }
    }

    return nearest;
}

static void ApplyPendingStart(void)
{
    player_t *player;
    mobj_t *mo;

    if (!agent_pending_start_active)
    {
        return;
    }

    if (gamestate != GS_LEVEL
        || consoleplayer < 0
        || consoleplayer >= MAXPLAYERS
        || !playeringame[consoleplayer])
    {
        return;
    }

    player = &players[consoleplayer];
    mo = player->mo;
    if (mo == NULL)
    {
        return;
    }

    if (agent_pending_start.episode != gameepisode
        || agent_pending_start.map != gamemap)
    {
        return;
    }

    if ((agent_pending_start.flags & AGENT_CONTROL_FLAG_START_POSITION) != 0)
    {
        if (P_TeleportMove(
                mo,
                (fixed_t) agent_pending_start.start_x_fp,
                (fixed_t) agent_pending_start.start_y_fp))
        {
            mo->z = mo->floorz;
            player->viewz = mo->z + player->viewheight;
            mo->momx = mo->momy = mo->momz = 0;
            mo->reactiontime = 0;
        }
    }

    if ((agent_pending_start.flags & AGENT_CONTROL_FLAG_FACE_NEAREST_ENEMY) != 0)
    {
        mobj_t *enemy = FindNearestLivingEnemy(mo);
        if (enemy != NULL)
        {
            mo->angle = R_PointToAngle2(mo->x, mo->y, enemy->x, enemy->y);
        }
    }
    else if ((agent_pending_start.flags & AGENT_CONTROL_FLAG_START_POSITION) != 0)
    {
        mo->angle = DegreesToAngle(agent_pending_start.start_angle_degrees);
    }

    if ((agent_pending_start.flags & AGENT_CONTROL_FLAG_APPLY_RESOURCES) != 0)
    {
        const int health = ClampInt(agent_pending_start.start_health, 1, 200);

        player->health = health;
        mo->health = health;
        player->armorpoints = ClampInt(agent_pending_start.start_armor, 0, 200);
        player->ammo[am_clip] = ClampInt(
            agent_pending_start.start_ammo_bullets,
            0,
            player->maxammo[am_clip]);
        player->ammo[am_shell] = ClampInt(
            agent_pending_start.start_ammo_shells,
            0,
            player->maxammo[am_shell]);
        player->ammo[am_cell] = ClampInt(
            agent_pending_start.start_ammo_cells,
            0,
            player->maxammo[am_cell]);
        player->ammo[am_misl] = ClampInt(
            agent_pending_start.start_ammo_rockets,
            0,
            player->maxammo[am_misl]);
    }

    agent_pending_start_active = false;
}

static void ApplyPendingSeed(void)
{
    if (!agent_pending_seed_active)
    {
        return;
    }

    if (gamestate != GS_LEVEL)
    {
        return;
    }

    if (agent_pending_seed.episode != gameepisode
        || agent_pending_seed.map != gamemap)
    {
        return;
    }

    M_SetRandomSeed((unsigned int) (agent_pending_seed.seed & 0xffffffffu));
    agent_pending_seed_active = false;
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
    FillNavigationProbe(snapshot, player->mo);
    FillCombatProbe(snapshot, player->mo);
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

    ApplyPendingSeed();
    ApplyPendingStart();

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

int AgentBridge_BeforeTic(void)
{
    agent_control_request_t request;
    agent_player_action_t discarded_action;
    char snapshot_description[AGENT_CONTROL_SNAPSHOT_DESCRIPTION_BYTES];
    int requests_applied = 0;
    int actions_discarded;
    int suppress_current_ticcmd = 0;

    if (!agent_bridge_initialized)
    {
        return 0;
    }

    while (requests_applied < 4 && restfuldoom_agent_take_control_request(&request) == 1)
    {
        switch (request.command)
        {
            case AGENT_CONTROL_RESET_EPISODE:
                actions_discarded = 0;
                while (actions_discarded < 256
                    && restfuldoom_agent_take_action(&discarded_action) == 1)
                {
                    ++actions_discarded;
                }
                agent_pending_start_active = false;
                agent_pending_seed = request;
                agent_pending_seed_active = true;
                G_DeferedInitNew((skill_t) ClampInt(request.skill, sk_baby, sk_nightmare),
                                 request.episode,
                                 request.map);
                suppress_current_ticcmd = 1;
                if ((request.flags
                     & (AGENT_CONTROL_FLAG_START_POSITION
                        | AGENT_CONTROL_FLAG_FACE_NEAREST_ENEMY
                        | AGENT_CONTROL_FLAG_APPLY_RESOURCES)) != 0)
                {
                    agent_pending_start = request;
                    agent_pending_start_active = true;
                }
                break;
            case AGENT_CONTROL_SAVE_SNAPSHOT:
                SnapshotDescription(
                    snapshot_description,
                    sizeof(snapshot_description),
                    &request);
                G_AgentSaveGame(
                    ClampInt(request.snapshot_slot, 0, 9),
                    snapshot_description);
                break;
            case AGENT_CONTROL_LOAD_SNAPSHOT:
                actions_discarded = 0;
                while (actions_discarded < 256
                    && restfuldoom_agent_take_action(&discarded_action) == 1)
                {
                    ++actions_discarded;
                }
                agent_pending_start_active = false;
                agent_pending_seed_active = false;
                G_AgentLoadGame(ClampInt(request.snapshot_slot, 0, 9));
                suppress_current_ticcmd = 1;
                break;
            default:
                break;
        }
        ++requests_applied;
    }

    return suppress_current_ticcmd;
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
