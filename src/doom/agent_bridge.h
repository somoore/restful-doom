#ifndef __AGENT_BRIDGE_H__
#define __AGENT_BRIDGE_H__

#include <stdint.h>

#include "../d_ticcmd.h"

#define AGENT_MAX_ENEMIES 128
#define AGENT_MAX_OBJECTS 256
#define AGENT_MAX_NAV_DIRECTIONS 9
#define AGENT_MAX_USE_LINES 32

typedef struct
{
    uint64_t tick;
    int8_t forward_move;
    int8_t side_move;
    int16_t angle_turn;
    uint8_t buttons;
    uint8_t reserved[3];
} agent_player_action_t;

#define AGENT_CONTROL_RESET_EPISODE 1u
#define AGENT_CONTROL_SAVE_SNAPSHOT 2u
#define AGENT_CONTROL_LOAD_SNAPSHOT 3u
#define AGENT_CONTROL_FLAG_START_POSITION 1u
#define AGENT_CONTROL_FLAG_FACE_NEAREST_ENEMY 2u
#define AGENT_CONTROL_FLAG_APPLY_RESOURCES 4u
#define AGENT_CONTROL_SNAPSHOT_DESCRIPTION_BYTES 24u

typedef struct
{
    uint32_t command;
    int32_t skill;
    int32_t episode;
    int32_t map;
    uint64_t seed;
    uint32_t flags;
    int32_t start_x_fp;
    int32_t start_y_fp;
    int32_t start_angle_degrees;
    int32_t start_health;
    int32_t start_armor;
    int32_t start_ammo_bullets;
    int32_t start_ammo_shells;
    int32_t start_ammo_cells;
    int32_t start_ammo_rockets;
    int32_t snapshot_slot;
    uint8_t snapshot_description[AGENT_CONTROL_SNAPSHOT_DESCRIPTION_BYTES];
    uint8_t reserved[4];
} agent_control_request_t;

typedef struct
{
    int32_t id;
    int32_t x_fp;
    int32_t y_fp;
    int32_t z_fp;
    uint32_t angle_degrees;
    int32_t height_fp;
    int32_t health;
    int32_t type_id;
    int32_t internal_type;
    uint32_t flags;
    int32_t attacking_id;
    int32_t distance_fp;
    uint32_t line_of_sight;
} agent_mobj_state_t;

typedef struct
{
    agent_mobj_state_t object;
    int32_t health;
    int32_t armor;
    int32_t kills;
    int32_t items;
    int32_t secrets;
    int32_t ready_weapon;
    uint32_t owned_weapons;
    int32_t ammo_bullets;
    int32_t ammo_shells;
    int32_t ammo_cells;
    int32_t ammo_rockets;
    uint32_t key_cards;
    uint32_t cheat_flags;
    int32_t last_attacked_by;
} agent_player_state_t;

typedef struct
{
    int32_t angle_offset_degrees;
    uint32_t open;
    int32_t block_distance_fp;
    int32_t blocking_line_special;
    uint32_t use_line_ahead;
} agent_direction_probe_t;

typedef struct
{
    int32_t line_id;
    int32_t midpoint_x_fp;
    int32_t midpoint_y_fp;
    int32_t midpoint_z_fp;
    int32_t start_x_fp;
    int32_t start_y_fp;
    int32_t end_x_fp;
    int32_t end_y_fp;
    int32_t nearest_x_fp;
    int32_t nearest_y_fp;
    int32_t special;
    int32_t tag;
    int32_t distance_fp;
    int32_t nearest_distance_fp;
} agent_use_line_t;

typedef struct
{
    int32_t sector_id;
    int32_t special;
    int32_t floor_height_fp;
    int32_t ceiling_height_fp;
    int32_t light_level;
    uint32_t damaging;
    int32_t damage_per_32_tics;
    uint32_t exit_damage;
} agent_sector_probe_t;

typedef struct
{
    uint32_t forward_open;
    uint32_t back_open;
    uint32_t left_open;
    uint32_t right_open;
    uint32_t use_line_ahead;
    int32_t front_blocking_line_special;
    int32_t front_block_distance_fp;
    int32_t probe_distance_fp;
    uint32_t direction_count;
    agent_direction_probe_t directions[AGENT_MAX_NAV_DIRECTIONS];
    uint32_t use_line_count;
    agent_use_line_t use_lines[AGENT_MAX_USE_LINES];
    agent_sector_probe_t current_sector;
    uint32_t has_route_waypoint;
    agent_use_line_t route_waypoint;
    int32_t route_waypoint_priority;
    uint32_t route_waypoint_exit;
    uint32_t route_waypoint_walk_trigger;
} agent_navigation_probe_t;

typedef struct
{
    uint32_t has_shootable_target;
    uint32_t target_is_enemy;
    int32_t target_id;
    int32_t target_health;
    int32_t target_distance_fp;
    int32_t aim_slope_fp;
    int32_t range_fp;
} agent_combat_probe_t;

typedef struct
{
    uint64_t tick;
    int32_t episode;
    int32_t map;
    int32_t skill;
    int32_t level_time;
    int32_t total_kills;
    int32_t total_items;
    int32_t total_secrets;
    int32_t gamestate;
    uint32_t player_active;
    agent_player_state_t player;
    agent_navigation_probe_t navigation;
    agent_combat_probe_t combat;
    uint32_t enemy_count;
    uint32_t object_count;
    agent_mobj_state_t enemies[AGENT_MAX_ENEMIES];
    agent_mobj_state_t objects[AGENT_MAX_OBJECTS];
} agent_game_state_snapshot_t;

int restfuldoom_agent_init(uint16_t port);
void restfuldoom_agent_publish_state(const agent_game_state_snapshot_t *snapshot);
int restfuldoom_agent_take_action(agent_player_action_t *out_action);
int restfuldoom_agent_take_control_request(agent_control_request_t *out_request);

void AgentBridge_Init(int port);
int AgentBridge_BeforeTic(void);
void AgentBridge_AfterTic(int completed_tic);
void AgentBridge_ApplyTiccmd(ticcmd_t *cmd);

#endif
