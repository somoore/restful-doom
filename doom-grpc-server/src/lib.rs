//! Provides the RESTful Doom gRPC bridge.
//!
//! The crate is compiled as a `staticlib` and linked into the C Doom binary.
//! Doom calls the exported C ABI from the game thread with copied snapshots.
//! The gRPC runtime runs on a background Tokio thread and communicates through
//! bounded channels so the 35 Hz simulation loop never blocks on network I/O.

// Rust guideline compliant 2026-02-21

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Mutex, OnceLock};
use std::thread;

use mimalloc::MiMalloc;
use prost::Message;
use tokio::runtime::Builder;
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::{event, Level};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Generated protobuf types.
pub mod proto {
    tonic::include_proto!("restfuldoom.v1");
}

use proto::doom_agent_server::{DoomAgent, DoomAgentServer};
use proto::{
    Ammo, CombatProbe, DirectionProbe, DoomKey, EnemyInfo, GameState, LevelInfo,
    LoadSnapshotRequest, MapObjectState, MouseInput, NavigationProbe, ObserveRequest, PlayerAction,
    PlayerActionType, PlayerState, RawTiccmd, ResetEpisodeRequest, ResetEpisodeResponse,
    RouteWaypoint, SaveSnapshotRequest, SectorProbe, SnapshotCommandResponse, StateDelta,
    UseLineInfo, Vec3Fixed,
};

const DEFAULT_GRPC_PORT: u16 = 50_051;
const ACTION_QUEUE_CAPACITY: usize = 256;
const CONTROL_QUEUE_CAPACITY: usize = 16;
const STATE_STREAM_CAPACITY: usize = 8;
const MAX_ACTION_DURATION_TICS: u32 = 35;
const DEFAULT_ACTION_AMOUNT: u32 = 10;
const MAX_TIC_MOVE: i32 = 50;
const MAX_TIC_TURN: i32 = i16::MAX as i32;
const TURN_AMOUNT_SCALE: i32 = 64;
const BT_ATTACK: u8 = 1;
const BT_USE: u8 = 2;
const BT_CHANGE: u8 = 4;
const BT_WEAPONSHIFT: u32 = 3;
const MAX_SNAPSHOT_SLOT: i32 = 9;
const SNAPSHOT_DESCRIPTION_BYTES: usize = 24;

static BRIDGE: OnceLock<Bridge> = OnceLock::new();

#[derive(Debug)]
struct Bridge {
    latest_state: watch::Sender<GameState>,
    action_rx: Mutex<mpsc::Receiver<AgentPlayerAction>>,
    control_rx: Mutex<mpsc::Receiver<AgentControlRequest>>,
}

#[derive(Clone, Debug)]
struct AgentRuntime {
    latest_state: watch::Sender<GameState>,
    action_tx: mpsc::Sender<AgentPlayerAction>,
    control_tx: mpsc::Sender<AgentControlRequest>,
}

#[tonic::async_trait]
impl DoomAgent for AgentRuntime {
    type GameSessionStream = ReceiverStream<Result<GameState, Status>>;
    type ObserveStream = ReceiverStream<Result<GameState, Status>>;

    /// Runs a bidirectional Doom agent stream.
    async fn game_session(
        &self,
        request: Request<tonic::Streaming<PlayerAction>>,
    ) -> Result<Response<Self::GameSessionStream>, Status> {
        let mut inbound = request.into_inner();
        let action_tx = self.action_tx.clone();

        tokio::spawn(async move {
            while let Ok(Some(action)) = inbound.message().await {
                for command in player_action_to_commands(&action) {
                    if action_tx.send(command).await.is_err() {
                        break;
                    }
                }
            }
        });

        Ok(Response::new(stream_states(
            self.latest_state.subscribe(),
            true,
        )))
    }

    /// Streams Doom observations without taking actions.
    async fn observe(
        &self,
        request: Request<ObserveRequest>,
    ) -> Result<Response<Self::ObserveStream>, Status> {
        let include_delta_state = request.into_inner().include_delta_state;
        Ok(Response::new(stream_states(
            self.latest_state.subscribe(),
            include_delta_state,
        )))
    }

    /// Queues a Doom episode reset for the simulation thread.
    async fn reset_episode(
        &self,
        request: Request<ResetEpisodeRequest>,
    ) -> Result<Response<ResetEpisodeResponse>, Status> {
        let request = request.into_inner();
        let control = control_request_from_proto(&request);
        self.queue_control(control, "episode reset")?;

        Ok(Response::new(ResetEpisodeResponse {
            accepted: true,
            message: "reset queued on Doom simulation thread".to_string(),
            skill: control.skill,
            episode: control.episode,
            map: control.map,
            seed: control.seed,
            seed_applied: false,
            start_queued: control.flags & AGENT_CONTROL_FLAG_START_POSITION != 0,
        }))
    }

    /// Queues a Doom savegame snapshot for the simulation thread.
    async fn save_snapshot(
        &self,
        request: Request<SaveSnapshotRequest>,
    ) -> Result<Response<SnapshotCommandResponse>, Status> {
        let control = save_snapshot_control_from_proto(&request.into_inner());
        self.queue_control(control, "snapshot save")?;
        Ok(Response::new(SnapshotCommandResponse {
            accepted: true,
            message: "snapshot save queued on Doom simulation thread".to_string(),
            slot: control.snapshot_slot,
            save_queued: true,
            load_queued: false,
        }))
    }

    /// Queues a Doom savegame restore for the simulation thread.
    async fn load_snapshot(
        &self,
        request: Request<LoadSnapshotRequest>,
    ) -> Result<Response<SnapshotCommandResponse>, Status> {
        let control = load_snapshot_control_from_proto(&request.into_inner());
        self.queue_control(control, "snapshot load")?;
        Ok(Response::new(SnapshotCommandResponse {
            accepted: true,
            message: "snapshot load queued on Doom simulation thread".to_string(),
            slot: control.snapshot_slot,
            save_queued: false,
            load_queued: true,
        }))
    }
}

impl AgentRuntime {
    /// Queues one control command for Doom's simulation thread.
    ///
    /// # Errors
    /// Returns a gRPC status when the bounded control queue is full or closed.
    fn queue_control(&self, control: AgentControlRequest, name: &str) -> Result<(), Status> {
        self.control_tx
            .try_send(control)
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => {
                    Status::resource_exhausted(format!("{name} queue is full"))
                }
                mpsc::error::TrySendError::Closed(_) => {
                    Status::unavailable(format!("{name} queue is closed"))
                }
            })
    }
}

fn stream_states(
    mut rx: watch::Receiver<GameState>,
    include_delta_state: bool,
) -> ReceiverStream<Result<GameState, Status>> {
    let (tx, output) = mpsc::channel(STATE_STREAM_CAPACITY);

    tokio::spawn(async move {
        let mut previous: Option<GameState> = None;
        loop {
            if rx.changed().await.is_err() {
                break;
            }

            let mut state = rx.borrow().clone();
            if include_delta_state {
                if let Some(last) = previous.as_ref() {
                    state.delta_state = encode_delta(last, &state);
                    state.has_delta_state = true;
                }
            }
            previous = Some(state.clone());

            if tx.send(Ok(state)).await.is_err() {
                break;
            }
        }
    });

    ReceiverStream::new(output)
}

fn encode_delta(previous: &GameState, current: &GameState) -> Vec<u8> {
    let mut prior_enemy_ids = HashMap::with_capacity(previous.enemies.len());
    for enemy in &previous.enemies {
        if let Some(object) = &enemy.object {
            prior_enemy_ids.insert(object.id, enemy);
        }
    }

    let mut changed_enemies = Vec::new();
    for enemy in &current.enemies {
        let changed = enemy
            .object
            .as_ref()
            .and_then(|object| prior_enemy_ids.remove(&object.id))
            != Some(enemy);
        if changed {
            changed_enemies.push(*enemy);
        }
    }

    let removed_enemy_ids = prior_enemy_ids.into_keys().collect();

    let mut prior_object_ids = HashMap::with_capacity(previous.objects.len());
    for object in &previous.objects {
        prior_object_ids.insert(object.id, object);
    }

    let mut changed_objects = Vec::new();
    for object in &current.objects {
        let changed = prior_object_ids.remove(&object.id) != Some(object);
        if changed {
            changed_objects.push(*object);
        }
    }

    let delta = StateDelta {
        base_tick: previous.tick,
        tick: current.tick,
        player_changed: previous.player != current.player,
        player: current.player,
        changed_enemies,
        removed_enemy_ids,
        changed_objects,
        removed_object_ids: prior_object_ids.into_keys().collect(),
        level: current.level,
        level_changed: previous.level != current.level,
        navigation: current.navigation.clone(),
        navigation_changed: previous.navigation != current.navigation,
        combat: current.combat,
        combat_changed: previous.combat != current.combat,
    };

    delta.encode_to_vec()
}

fn player_action_to_commands(action: &PlayerAction) -> VecDeque<AgentPlayerAction> {
    let duration = action
        .duration_tics
        .clamp(1, MAX_ACTION_DURATION_TICS)
        .max(1);
    let mut command = AgentPlayerAction {
        tick: action.tick,
        ..AgentPlayerAction::default()
    };

    apply_semantic_action(action.action(), action.amount, &mut command);

    if let Some(raw) = action.raw.as_ref() {
        apply_raw_ticcmd(raw, &mut command);
    }

    for key in &action.keys {
        if key.pressed {
            apply_key(key.key(), &mut command);
        }
    }

    if let Some(mouse) = action.mouse.as_ref() {
        apply_mouse(mouse, &mut command);
    }

    let mut commands = VecDeque::with_capacity(duration as usize);
    for _ in 0..duration {
        commands.push_back(command);
    }
    commands
}

fn apply_semantic_action(action: PlayerActionType, amount: u32, command: &mut AgentPlayerAction) {
    let amount = if amount == 0 {
        DEFAULT_ACTION_AMOUNT
    } else {
        amount
    };
    let movement = clamp_i32(amount as i32, -MAX_TIC_MOVE, MAX_TIC_MOVE);
    let turn = clamp_i32(
        (amount as i32) * TURN_AMOUNT_SCALE,
        -MAX_TIC_TURN,
        MAX_TIC_TURN,
    );

    match action {
        PlayerActionType::ActionForward => add_forward(command, movement),
        PlayerActionType::ActionBackward => add_forward(command, -movement),
        PlayerActionType::ActionTurnLeft => add_turn(command, turn),
        PlayerActionType::ActionTurnRight => add_turn(command, -turn),
        PlayerActionType::ActionStrafeLeft => add_side(command, -movement),
        PlayerActionType::ActionStrafeRight => add_side(command, movement),
        PlayerActionType::ActionShoot => command.buttons |= BT_ATTACK,
        PlayerActionType::ActionUse => command.buttons |= BT_USE,
        PlayerActionType::ActionSwitchWeapon => {
            let slot = amount.saturating_sub(1).min(7);
            command.buttons |= BT_CHANGE | ((slot << BT_WEAPONSHIFT) as u8);
        }
        PlayerActionType::ActionUnspecified => {}
    }
}

fn apply_raw_ticcmd(raw: &RawTiccmd, command: &mut AgentPlayerAction) {
    add_forward(command, raw.forward_move);
    add_side(command, raw.side_move);
    add_turn(command, raw.angle_turn);
    command.buttons |= raw.buttons as u8;
}

fn apply_key(key: DoomKey, command: &mut AgentPlayerAction) {
    match key {
        DoomKey::Forward => add_forward(command, 25),
        DoomKey::Backward => add_forward(command, -25),
        DoomKey::TurnLeft => add_turn(command, 640),
        DoomKey::TurnRight => add_turn(command, -640),
        DoomKey::StrafeLeft => add_side(command, -24),
        DoomKey::StrafeRight => add_side(command, 24),
        DoomKey::Shoot => command.buttons |= BT_ATTACK,
        DoomKey::Use => command.buttons |= BT_USE,
        DoomKey::Unspecified => {}
    }
}

fn apply_mouse(mouse: &MouseInput, command: &mut AgentPlayerAction) {
    add_turn(command, -mouse.turn.saturating_mul(8));
    add_forward(command, mouse.forward);
    command.buttons |= mouse.buttons as u8;
}

fn add_forward(command: &mut AgentPlayerAction, value: i32) {
    command.forward_move = clamp_i32(
        command.forward_move as i32 + value,
        -MAX_TIC_MOVE,
        MAX_TIC_MOVE,
    ) as i8;
}

fn add_side(command: &mut AgentPlayerAction, value: i32) {
    command.side_move = clamp_i32(
        command.side_move as i32 + value,
        -MAX_TIC_MOVE,
        MAX_TIC_MOVE,
    ) as i8;
}

fn add_turn(command: &mut AgentPlayerAction, value: i32) {
    command.angle_turn = clamp_i32(
        command.angle_turn as i32 + value,
        -MAX_TIC_TURN,
        MAX_TIC_TURN,
    ) as i16;
}

fn clamp_i32(value: i32, min: i32, max: i32) -> i32 {
    value.max(min).min(max)
}

fn control_request_from_proto(request: &ResetEpisodeRequest) -> AgentControlRequest {
    let skill = clamp_i32(request.skill, 0, 4);
    let episode = if request.episode <= 0 {
        1
    } else {
        request.episode
    };
    let map = if request.map <= 0 { 1 } else { request.map };
    let mut control = AgentControlRequest {
        command: AGENT_CONTROL_RESET_EPISODE,
        skill,
        episode,
        map,
        seed: request.seed,
        ..AgentControlRequest::default()
    };

    if let Some(start) = request.start.as_ref() {
        if let Some(position) = start.position.as_ref() {
            control.flags |= AGENT_CONTROL_FLAG_START_POSITION;
            control.start_x_fp = position.x_fp;
            control.start_y_fp = position.y_fp;
        }
        if start.face_nearest_enemy {
            control.flags |= AGENT_CONTROL_FLAG_FACE_NEAREST_ENEMY;
        }
        if start.apply_resources {
            control.flags |= AGENT_CONTROL_FLAG_APPLY_RESOURCES;
            control.start_health = start.health.max(1);
            control.start_armor = start.armor.max(0);
            if let Some(ammo) = start.ammo.as_ref() {
                control.start_ammo_bullets = ammo.bullets.max(0);
                control.start_ammo_shells = ammo.shells.max(0);
                control.start_ammo_cells = ammo.cells.max(0);
                control.start_ammo_rockets = ammo.rockets.max(0);
            }
        }
        if start.angle_degrees > 0 || start.face_nearest_enemy {
            control.start_angle_degrees = (start.angle_degrees % 360) as i32;
        }
    }

    control
}

fn save_snapshot_control_from_proto(request: &SaveSnapshotRequest) -> AgentControlRequest {
    let mut control = AgentControlRequest {
        command: AGENT_CONTROL_SAVE_SNAPSHOT,
        snapshot_slot: clamp_snapshot_slot(request.slot),
        ..AgentControlRequest::default()
    };
    copy_snapshot_description(&request.description, &mut control.snapshot_description);
    control
}

fn load_snapshot_control_from_proto(request: &LoadSnapshotRequest) -> AgentControlRequest {
    AgentControlRequest {
        command: AGENT_CONTROL_LOAD_SNAPSHOT,
        snapshot_slot: clamp_snapshot_slot(request.slot),
        ..AgentControlRequest::default()
    }
}

fn clamp_snapshot_slot(slot: i32) -> i32 {
    clamp_i32(slot, 0, MAX_SNAPSHOT_SLOT)
}

fn copy_snapshot_description(description: &str, out: &mut [u8; SNAPSHOT_DESCRIPTION_BYTES]) {
    let bytes = description.as_bytes();
    // Doom's save header reserves 24 bytes. Leave one byte for a C NUL.
    let copy_len = bytes
        .len()
        .min(SNAPSHOT_DESCRIPTION_BYTES.saturating_sub(1));
    out[..copy_len].copy_from_slice(&bytes[..copy_len]);
}

fn state_from_snapshot(snapshot: &AgentGameStateSnapshot) -> GameState {
    let enemy_count = (snapshot.enemy_count as usize).min(AGENT_MAX_ENEMIES);
    let object_count = (snapshot.object_count as usize).min(AGENT_MAX_OBJECTS);

    GameState {
        tick: snapshot.tick,
        player: Some(player_from_snapshot(
            &snapshot.player,
            snapshot.player_active != 0,
        )),
        enemies: snapshot.enemies[..enemy_count]
            .iter()
            .map(enemy_from_snapshot)
            .collect(),
        objects: snapshot.objects[..object_count]
            .iter()
            .map(object_from_snapshot)
            .collect(),
        level: Some(LevelInfo {
            episode: snapshot.episode,
            map: snapshot.map,
            skill: snapshot.skill,
            level_time: snapshot.level_time,
            total_kills: snapshot.total_kills,
            total_items: snapshot.total_items,
            total_secrets: snapshot.total_secrets,
            gamestate: snapshot.gamestate,
        }),
        delta_state: Vec::new(),
        has_delta_state: false,
        navigation: Some(navigation_from_snapshot(&snapshot.navigation)),
        combat: Some(combat_from_snapshot(&snapshot.combat)),
    }
}

fn player_from_snapshot(snapshot: &AgentPlayerState, active: bool) -> PlayerState {
    PlayerState {
        object: Some(object_from_snapshot(&snapshot.object)),
        health: snapshot.health,
        armor: snapshot.armor,
        kills: snapshot.kills,
        items: snapshot.items,
        secrets: snapshot.secrets,
        ready_weapon: snapshot.ready_weapon,
        owned_weapons: snapshot.owned_weapons,
        ammo: Some(Ammo {
            bullets: snapshot.ammo_bullets,
            shells: snapshot.ammo_shells,
            cells: snapshot.ammo_cells,
            rockets: snapshot.ammo_rockets,
        }),
        key_cards: snapshot.key_cards,
        cheat_flags: snapshot.cheat_flags,
        last_attacked_by: snapshot.last_attacked_by,
        active,
    }
}

fn enemy_from_snapshot(snapshot: &AgentMobjState) -> EnemyInfo {
    EnemyInfo {
        object: Some(object_from_snapshot(snapshot)),
        line_of_sight: snapshot.line_of_sight != 0,
        target_id: snapshot.attacking_id,
    }
}

fn object_from_snapshot(snapshot: &AgentMobjState) -> MapObjectState {
    MapObjectState {
        id: snapshot.id,
        position: Some(Vec3Fixed {
            x_fp: snapshot.x_fp,
            y_fp: snapshot.y_fp,
            z_fp: snapshot.z_fp,
        }),
        angle_degrees: snapshot.angle_degrees,
        height_fp: snapshot.height_fp,
        health: snapshot.health,
        type_id: snapshot.type_id,
        internal_type: snapshot.internal_type,
        flags: snapshot.flags,
        attacking_id: snapshot.attacking_id,
        distance_fp: snapshot.distance_fp,
    }
}

fn navigation_from_snapshot(snapshot: &AgentNavigationProbe) -> NavigationProbe {
    let direction_count = (snapshot.direction_count as usize).min(AGENT_MAX_NAV_DIRECTIONS);
    let use_line_count = (snapshot.use_line_count as usize).min(AGENT_MAX_USE_LINES);
    NavigationProbe {
        forward_open: snapshot.forward_open != 0,
        back_open: snapshot.back_open != 0,
        left_open: snapshot.left_open != 0,
        right_open: snapshot.right_open != 0,
        use_line_ahead: snapshot.use_line_ahead != 0,
        front_blocking_line_special: snapshot.front_blocking_line_special,
        front_block_distance_fp: snapshot.front_block_distance_fp,
        probe_distance_fp: snapshot.probe_distance_fp,
        direction_probes: snapshot.directions[..direction_count]
            .iter()
            .map(direction_probe_from_snapshot)
            .collect(),
        use_lines: snapshot.use_lines[..use_line_count]
            .iter()
            .map(use_line_from_snapshot)
            .collect(),
        current_sector: Some(sector_probe_from_snapshot(&snapshot.current_sector)),
        route_waypoint: (snapshot.has_route_waypoint != 0).then(|| RouteWaypoint {
            line: Some(use_line_from_snapshot(&snapshot.route_waypoint)),
            priority: snapshot.route_waypoint_priority,
            exit: snapshot.route_waypoint_exit != 0,
            walk_trigger: snapshot.route_waypoint_walk_trigger != 0,
        }),
    }
}

fn direction_probe_from_snapshot(snapshot: &AgentDirectionProbe) -> DirectionProbe {
    DirectionProbe {
        angle_offset_degrees: snapshot.angle_offset_degrees,
        open: snapshot.open != 0,
        block_distance_fp: snapshot.block_distance_fp,
        blocking_line_special: snapshot.blocking_line_special,
        use_line_ahead: snapshot.use_line_ahead != 0,
    }
}

fn use_line_from_snapshot(snapshot: &AgentUseLine) -> UseLineInfo {
    UseLineInfo {
        line_id: snapshot.line_id,
        midpoint: Some(Vec3Fixed {
            x_fp: snapshot.midpoint_x_fp,
            y_fp: snapshot.midpoint_y_fp,
            z_fp: snapshot.midpoint_z_fp,
        }),
        special: snapshot.special,
        tag: snapshot.tag,
        distance_fp: snapshot.distance_fp,
        start: Some(Vec3Fixed {
            x_fp: snapshot.start_x_fp,
            y_fp: snapshot.start_y_fp,
            z_fp: snapshot.midpoint_z_fp,
        }),
        end: Some(Vec3Fixed {
            x_fp: snapshot.end_x_fp,
            y_fp: snapshot.end_y_fp,
            z_fp: snapshot.midpoint_z_fp,
        }),
        nearest_point: Some(Vec3Fixed {
            x_fp: snapshot.nearest_x_fp,
            y_fp: snapshot.nearest_y_fp,
            z_fp: snapshot.midpoint_z_fp,
        }),
        nearest_distance_fp: snapshot.nearest_distance_fp,
    }
}

fn sector_probe_from_snapshot(snapshot: &AgentSectorProbe) -> SectorProbe {
    SectorProbe {
        sector_id: snapshot.sector_id,
        special: snapshot.special,
        floor_height_fp: snapshot.floor_height_fp,
        ceiling_height_fp: snapshot.ceiling_height_fp,
        light_level: snapshot.light_level,
        damaging: snapshot.damaging != 0,
        damage_per_32_tics: snapshot.damage_per_32_tics,
        exit_damage: snapshot.exit_damage != 0,
    }
}

fn combat_from_snapshot(snapshot: &AgentCombatProbe) -> CombatProbe {
    CombatProbe {
        has_shootable_target: snapshot.has_shootable_target != 0,
        target_id: snapshot.target_id,
        target_health: snapshot.target_health,
        target_distance_fp: snapshot.target_distance_fp,
        aim_slope_fp: snapshot.aim_slope_fp,
        range_fp: snapshot.range_fp,
        target_is_enemy: snapshot.target_is_enemy != 0,
    }
}

/// Initializes the gRPC bridge.
///
/// # Safety
///
/// This function is safe to call from C. It must be called at most once by the
/// embedding process. Later calls return success without replacing the runtime.
#[no_mangle]
pub extern "C" fn restfuldoom_agent_init(port: u16) -> i32 {
    if BRIDGE.get().is_some() {
        return 0;
    }

    let (state_tx, _state_rx) = watch::channel(GameState::default());
    let (action_tx, action_rx) = mpsc::channel(ACTION_QUEUE_CAPACITY);
    let (control_tx, control_rx) = mpsc::channel(CONTROL_QUEUE_CAPACITY);
    let bridge = Bridge {
        latest_state: state_tx.clone(),
        action_rx: Mutex::new(action_rx),
        control_rx: Mutex::new(control_rx),
    };

    if BRIDGE.set(bridge).is_err() {
        return 0;
    }

    let bind_port = if port == 0 { DEFAULT_GRPC_PORT } else { port };
    let runtime = AgentRuntime {
        latest_state: state_tx,
        action_tx,
        control_tx,
    };

    thread::Builder::new()
        .name("restfuldoom-grpc".to_string())
        .spawn(move || run_server(runtime, bind_port))
        .map_or(-1, |_| 0)
}

fn run_server(runtime: AgentRuntime, port: u16) {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    event!(
        name: "restfuldoom.grpc.start",
        Level::INFO,
        server.address = %addr,
        "starting Doom agent gRPC server on {{server.address}}",
    );

    let rt = match Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(error) => {
            event!(
                name: "restfuldoom.grpc.runtime_error",
                Level::ERROR,
                error.message = %error,
                "failed to start Tokio runtime: {{error.message}}",
            );
            return;
        }
    };

    rt.block_on(async move {
        if let Err(error) = tonic::transport::Server::builder()
            .add_service(DoomAgentServer::new(runtime))
            .serve(addr)
            .await
        {
            event!(
                name: "restfuldoom.grpc.server_error",
                Level::ERROR,
                error.message = %error,
                "Doom agent gRPC server stopped: {{error.message}}",
            );
        }
    });
}

/// Publishes a copied game-state snapshot from Doom.
///
/// # Safety
///
/// `snapshot` must point to an initialized `AgentGameStateSnapshot` for the
/// duration of this call. The function copies every value before returning.
#[no_mangle]
pub unsafe extern "C" fn restfuldoom_agent_publish_state(snapshot: *const AgentGameStateSnapshot) {
    let Some(bridge) = BRIDGE.get() else {
        return;
    };
    if snapshot.is_null() {
        return;
    }

    // SAFETY: The C caller promises `snapshot` points to an initialized value
    // for this call. We copy it immediately and never retain the pointer.
    let snapshot = unsafe { &*snapshot };
    let state = state_from_snapshot(snapshot);
    _ = bridge.latest_state.send(state);
}

/// Takes one queued action for the Doom game thread.
///
/// Returns `1` when an action was written and `0` otherwise.
///
/// # Safety
///
/// `out_action` must be either null or a valid writable pointer to an
/// `AgentPlayerAction`.
#[no_mangle]
pub unsafe extern "C" fn restfuldoom_agent_take_action(out_action: *mut AgentPlayerAction) -> i32 {
    let Some(bridge) = BRIDGE.get() else {
        return 0;
    };
    if out_action.is_null() {
        return 0;
    }

    let Ok(mut rx) = bridge.action_rx.lock() else {
        return 0;
    };
    match rx.try_recv() {
        Ok(action) => {
            // SAFETY: The C caller provides a valid writable `out_action`.
            unsafe {
                *out_action = action;
            }
            1
        }
        Err(_) => 0,
    }
}

/// Takes one queued control request for Doom's game thread.
///
/// Returns `1` when a request was written and `0` otherwise.
///
/// # Safety
///
/// `out_request` must be either null or a valid writable pointer to an
/// `AgentControlRequest`.
#[no_mangle]
pub unsafe extern "C" fn restfuldoom_agent_take_control_request(
    out_request: *mut AgentControlRequest,
) -> i32 {
    let Some(bridge) = BRIDGE.get() else {
        return 0;
    };
    if out_request.is_null() {
        return 0;
    }

    let Ok(mut rx) = bridge.control_rx.lock() else {
        return 0;
    };
    match rx.try_recv() {
        Ok(request) => {
            // SAFETY: The C caller provides a valid writable `out_request`.
            unsafe {
                *out_request = request;
            }
            1
        }
        Err(_) => 0,
    }
}

/// Fixed-size action returned to Doom's game thread.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AgentPlayerAction {
    pub tick: u64,
    pub forward_move: i8,
    pub side_move: i8,
    pub angle_turn: i16,
    pub buttons: u8,
    pub _reserved: [u8; 3],
}

const AGENT_CONTROL_RESET_EPISODE: u32 = 1;
const AGENT_CONTROL_SAVE_SNAPSHOT: u32 = 2;
const AGENT_CONTROL_LOAD_SNAPSHOT: u32 = 3;
const AGENT_CONTROL_FLAG_START_POSITION: u32 = 1;
const AGENT_CONTROL_FLAG_FACE_NEAREST_ENEMY: u32 = 2;
const AGENT_CONTROL_FLAG_APPLY_RESOURCES: u32 = 4;

/// Fixed-size control request returned to Doom's game thread.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AgentControlRequest {
    pub command: u32,
    pub skill: i32,
    pub episode: i32,
    pub map: i32,
    pub seed: u64,
    pub flags: u32,
    pub start_x_fp: i32,
    pub start_y_fp: i32,
    pub start_angle_degrees: i32,
    pub start_health: i32,
    pub start_armor: i32,
    pub start_ammo_bullets: i32,
    pub start_ammo_shells: i32,
    pub start_ammo_cells: i32,
    pub start_ammo_rockets: i32,
    pub snapshot_slot: i32,
    pub snapshot_description: [u8; SNAPSHOT_DESCRIPTION_BYTES],
    pub _reserved: [u8; 4],
}

/// Fixed-point map object copied from Doom.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AgentMobjState {
    pub id: i32,
    pub x_fp: i32,
    pub y_fp: i32,
    pub z_fp: i32,
    pub angle_degrees: u32,
    pub height_fp: i32,
    pub health: i32,
    pub type_id: i32,
    pub internal_type: i32,
    pub flags: u32,
    pub attacking_id: i32,
    pub distance_fp: i32,
    pub line_of_sight: u32,
}

/// Fixed-size player state copied from Doom.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AgentPlayerState {
    pub object: AgentMobjState,
    pub health: i32,
    pub armor: i32,
    pub kills: i32,
    pub items: i32,
    pub secrets: i32,
    pub ready_weapon: i32,
    pub owned_weapons: u32,
    pub ammo_bullets: i32,
    pub ammo_shells: i32,
    pub ammo_cells: i32,
    pub ammo_rockets: i32,
    pub key_cards: u32,
    pub cheat_flags: u32,
    pub last_attacked_by: i32,
}

/// One local movement ray copied from Doom.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AgentDirectionProbe {
    pub angle_offset_degrees: i32,
    pub open: u32,
    pub block_distance_fp: i32,
    pub blocking_line_special: i32,
    pub use_line_ahead: u32,
}

/// One nearby special/use line copied from Doom.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AgentUseLine {
    pub line_id: i32,
    pub midpoint_x_fp: i32,
    pub midpoint_y_fp: i32,
    pub midpoint_z_fp: i32,
    pub start_x_fp: i32,
    pub start_y_fp: i32,
    pub end_x_fp: i32,
    pub end_y_fp: i32,
    pub nearest_x_fp: i32,
    pub nearest_y_fp: i32,
    pub special: i32,
    pub tag: i32,
    pub distance_fp: i32,
    pub nearest_distance_fp: i32,
}

/// Current sector affordances copied from Doom.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AgentSectorProbe {
    pub sector_id: i32,
    pub special: i32,
    pub floor_height_fp: i32,
    pub ceiling_height_fp: i32,
    pub light_level: i32,
    pub damaging: u32,
    pub damage_per_32_tics: i32,
    pub exit_damage: u32,
}

/// Local movement affordances copied from Doom.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentNavigationProbe {
    pub forward_open: u32,
    pub back_open: u32,
    pub left_open: u32,
    pub right_open: u32,
    pub use_line_ahead: u32,
    pub front_blocking_line_special: i32,
    pub front_block_distance_fp: i32,
    pub probe_distance_fp: i32,
    pub direction_count: u32,
    pub directions: [AgentDirectionProbe; AGENT_MAX_NAV_DIRECTIONS],
    pub use_line_count: u32,
    pub use_lines: [AgentUseLine; AGENT_MAX_USE_LINES],
    pub current_sector: AgentSectorProbe,
    pub has_route_waypoint: u32,
    pub route_waypoint: AgentUseLine,
    pub route_waypoint_priority: i32,
    pub route_waypoint_exit: u32,
    pub route_waypoint_walk_trigger: u32,
}

impl Default for AgentNavigationProbe {
    fn default() -> Self {
        Self {
            forward_open: 0,
            back_open: 0,
            left_open: 0,
            right_open: 0,
            use_line_ahead: 0,
            front_blocking_line_special: 0,
            front_block_distance_fp: 0,
            probe_distance_fp: 0,
            direction_count: 0,
            directions: [AgentDirectionProbe::default(); AGENT_MAX_NAV_DIRECTIONS],
            use_line_count: 0,
            use_lines: [AgentUseLine::default(); AGENT_MAX_USE_LINES],
            current_sector: AgentSectorProbe::default(),
            has_route_waypoint: 0,
            route_waypoint: AgentUseLine::default(),
            route_waypoint_priority: 0,
            route_waypoint_exit: 0,
            route_waypoint_walk_trigger: 0,
        }
    }
}

/// Weapon-line combat affordances copied from Doom.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AgentCombatProbe {
    pub has_shootable_target: u32,
    pub target_is_enemy: u32,
    pub target_id: i32,
    pub target_health: i32,
    pub target_distance_fp: i32,
    pub aim_slope_fp: i32,
    pub range_fp: i32,
}

/// Maximum enemies copied per tic.
pub const AGENT_MAX_ENEMIES: usize = 128;

/// Maximum useful objects copied per tic.
pub const AGENT_MAX_OBJECTS: usize = 256;

/// Maximum local navigation rays copied per tic.
pub const AGENT_MAX_NAV_DIRECTIONS: usize = 9;

/// Maximum nearby special/use lines copied per tic.
pub const AGENT_MAX_USE_LINES: usize = 32;

/// Fixed-size game-state snapshot copied from Doom.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentGameStateSnapshot {
    pub tick: u64,
    pub episode: i32,
    pub map: i32,
    pub skill: i32,
    pub level_time: i32,
    pub total_kills: i32,
    pub total_items: i32,
    pub total_secrets: i32,
    pub gamestate: i32,
    pub player_active: u32,
    pub player: AgentPlayerState,
    pub navigation: AgentNavigationProbe,
    pub combat: AgentCombatProbe,
    pub enemy_count: u32,
    pub object_count: u32,
    pub enemies: [AgentMobjState; AGENT_MAX_ENEMIES],
    pub objects: [AgentMobjState; AGENT_MAX_OBJECTS],
}

impl Default for AgentGameStateSnapshot {
    fn default() -> Self {
        Self {
            tick: 0,
            episode: 0,
            map: 0,
            skill: 0,
            level_time: 0,
            total_kills: 0,
            total_items: 0,
            total_secrets: 0,
            gamestate: 0,
            player_active: 0,
            player: AgentPlayerState::default(),
            navigation: AgentNavigationProbe::default(),
            combat: AgentCombatProbe::default(),
            enemy_count: 0,
            object_count: 0,
            enemies: [AgentMobjState::default(); AGENT_MAX_ENEMIES],
            objects: [AgentMobjState::default(); AGENT_MAX_OBJECTS],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::doom_agent_client::DoomAgentClient;
    use proto::EpisodeStart;
    use tokio::time::{sleep, timeout, Duration};

    #[test]
    fn semantic_forward_uses_duration() {
        let commands = player_action_to_commands(&PlayerAction {
            action: PlayerActionType::ActionForward.into(),
            amount: 25,
            duration_tics: 3,
            ..PlayerAction::default()
        });

        assert_eq!(commands.len(), 3);
        assert!(commands.iter().all(|command| command.forward_move == 25));
    }

    #[test]
    fn switch_weapon_maps_to_doom_button_bits() {
        let command = player_action_to_commands(&PlayerAction {
            action: PlayerActionType::ActionSwitchWeapon.into(),
            amount: 3,
            ..PlayerAction::default()
        })
        .pop_front()
        .expect("command should exist");

        assert_eq!(command.buttons, BT_CHANGE | (2 << BT_WEAPONSHIFT) as u8);
    }

    #[test]
    fn reset_request_normalizes_training_defaults() {
        let request = ResetEpisodeRequest {
            skill: 99,
            episode: 0,
            map: 0,
            seed: 123,
            run_id: "ppo-smoke".to_string(),
            start: None,
        };

        let control = control_request_from_proto(&request);

        assert_eq!(control.command, AGENT_CONTROL_RESET_EPISODE);
        assert_eq!(control.skill, 4);
        assert_eq!(control.episode, 1);
        assert_eq!(control.map, 1);
        assert_eq!(control.seed, 123);
    }

    #[test]
    fn reset_request_carries_curriculum_start() {
        let request = ResetEpisodeRequest {
            skill: 2,
            episode: 1,
            map: 1,
            seed: 123,
            run_id: "combat-start".to_string(),
            start: Some(EpisodeStart {
                position: Some(Vec3Fixed {
                    x_fp: 1_024 * 65_536,
                    y_fp: -512 * 65_536,
                    z_fp: 0,
                }),
                angle_degrees: 450,
                face_nearest_enemy: true,
                health: 95,
                armor: 0,
                ammo: Some(Ammo {
                    bullets: 37,
                    ..Ammo::default()
                }),
                apply_resources: true,
            }),
        };

        let control = control_request_from_proto(&request);

        assert_eq!(
            control.flags,
            AGENT_CONTROL_FLAG_START_POSITION
                | AGENT_CONTROL_FLAG_FACE_NEAREST_ENEMY
                | AGENT_CONTROL_FLAG_APPLY_RESOURCES
        );
        assert_eq!(control.start_x_fp, 1_024 * 65_536);
        assert_eq!(control.start_y_fp, -512 * 65_536);
        assert_eq!(control.start_angle_degrees, 90);
        assert_eq!(control.start_health, 95);
        assert_eq!(control.start_ammo_bullets, 37);
    }

    #[test]
    fn snapshot_requests_normalize_fixed_size_control() {
        let save = save_snapshot_control_from_proto(&SaveSnapshotRequest {
            slot: 77,
            description: "abcdefghijklmnopqrstuvwxyz".to_string(),
            run_id: "capture".to_string(),
        });
        let load = load_snapshot_control_from_proto(&LoadSnapshotRequest {
            slot: -2,
            run_id: "restore".to_string(),
        });

        assert_eq!(save.command, AGENT_CONTROL_SAVE_SNAPSHOT);
        assert_eq!(save.snapshot_slot, 9);
        assert_eq!(save.snapshot_description[0], b'a');
        assert_eq!(save.snapshot_description[22], b'w');
        assert_eq!(save.snapshot_description[23], 0);
        assert_eq!(load.command, AGENT_CONTROL_LOAD_SNAPSHOT);
        assert_eq!(load.snapshot_slot, 0);
    }

    #[test]
    fn snapshot_conversion_bounds_enemies() {
        let mut snapshot = AgentGameStateSnapshot {
            tick: 42,
            episode: 1,
            map: 1,
            enemy_count: 1,
            ..AgentGameStateSnapshot::default()
        };
        snapshot.enemies[0].id = 7;
        snapshot.enemies[0].health = 20;
        snapshot.navigation.forward_open = 1;
        snapshot.navigation.probe_distance_fp = 96 * 65_536;
        snapshot.navigation.direction_count = 1;
        snapshot.navigation.directions[0] = AgentDirectionProbe {
            angle_offset_degrees: 30,
            open: 1,
            block_distance_fp: 96 * 65_536,
            blocking_line_special: 0,
            use_line_ahead: 0,
        };
        snapshot.navigation.use_line_count = 1;
        snapshot.navigation.use_lines[0] = AgentUseLine {
            line_id: 12,
            midpoint_x_fp: 128 * 65_536,
            midpoint_y_fp: -64 * 65_536,
            midpoint_z_fp: 0,
            start_x_fp: 128 * 65_536,
            start_y_fp: -128 * 65_536,
            end_x_fp: 128 * 65_536,
            end_y_fp: 0,
            nearest_x_fp: 128 * 65_536,
            nearest_y_fp: -32 * 65_536,
            special: 1,
            tag: 7,
            distance_fp: 96 * 65_536,
            nearest_distance_fp: 32 * 65_536,
        };
        snapshot.navigation.current_sector = AgentSectorProbe {
            sector_id: 4,
            special: 5,
            floor_height_fp: -24 * 65_536,
            ceiling_height_fp: 128 * 65_536,
            light_level: 160,
            damaging: 1,
            damage_per_32_tics: 10,
            exit_damage: 0,
        };
        snapshot.navigation.has_route_waypoint = 1;
        snapshot.navigation.route_waypoint = AgentUseLine {
            line_id: 88,
            midpoint_x_fp: 512 * 65_536,
            midpoint_y_fp: 64 * 65_536,
            midpoint_z_fp: 0,
            start_x_fp: 512 * 65_536,
            start_y_fp: 0,
            end_x_fp: 512 * 65_536,
            end_y_fp: 128 * 65_536,
            nearest_x_fp: 512 * 65_536,
            nearest_y_fp: 32 * 65_536,
            special: 88,
            tag: 1,
            distance_fp: 512 * 65_536,
            nearest_distance_fp: 384 * 65_536,
        };
        snapshot.navigation.route_waypoint_priority = 0;
        snapshot.navigation.route_waypoint_walk_trigger = 1;

        let state = state_from_snapshot(&snapshot);

        assert_eq!(state.tick, 42);
        assert_eq!(state.enemies.len(), 1);
        assert_eq!(
            state.enemies[0].object.as_ref().map(|object| object.id),
            Some(7)
        );
        assert_eq!(
            state
                .navigation
                .as_ref()
                .map(|navigation| navigation.forward_open),
            Some(true)
        );
        assert_eq!(
            state
                .navigation
                .as_ref()
                .and_then(|navigation| navigation.direction_probes.first())
                .map(|probe| (probe.angle_offset_degrees, probe.open)),
            Some((30, true))
        );
        assert_eq!(
            state
                .navigation
                .as_ref()
                .and_then(|navigation| navigation.use_lines.first())
                .map(|line| {
                    (
                        line.line_id,
                        line.special,
                        line.tag,
                        line.nearest_distance_fp / 65_536,
                    )
                }),
            Some((12, 1, 7, 32))
        );
        assert_eq!(
            state.navigation.as_ref().and_then(|navigation| {
                navigation
                    .current_sector
                    .as_ref()
                    .map(|sector| (sector.sector_id, sector.special, sector.damaging))
            }),
            Some((4, 5, true))
        );
        assert_eq!(
            state.navigation.as_ref().and_then(|navigation| {
                navigation.route_waypoint.as_ref().map(|waypoint| {
                    (
                        waypoint.line.as_ref().map(|line| line.line_id),
                        waypoint.priority,
                        waypoint.walk_trigger,
                    )
                })
            }),
            Some((Some(88), 0, true))
        );
    }

    #[test]
    fn delta_reports_removed_enemy() {
        let mut previous = GameState {
            tick: 1,
            enemies: vec![EnemyInfo {
                object: Some(MapObjectState {
                    id: 9,
                    ..MapObjectState::default()
                }),
                ..EnemyInfo::default()
            }],
            ..GameState::default()
        };
        previous.player = Some(PlayerState::default());

        let current = GameState {
            tick: 2,
            player: Some(PlayerState::default()),
            ..GameState::default()
        };

        let encoded = encode_delta(&previous, &current);
        let decoded = StateDelta::decode(encoded.as_slice()).expect("delta decodes");

        assert_eq!(decoded.removed_enemy_ids, vec![9]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn grpc_session_roundtrip() {
        let port = 50_071;
        assert_eq!(restfuldoom_agent_init(port), 0);
        sleep(Duration::from_millis(200)).await;

        let mut client = DoomAgentClient::connect(format!("http://127.0.0.1:{port}"))
            .await
            .expect("client connects");
        let (tx, rx) = mpsc::channel(4);
        let mut stream = client
            .game_session(ReceiverStream::new(rx))
            .await
            .expect("session starts")
            .into_inner();

        let mut snapshot = AgentGameStateSnapshot {
            tick: 99,
            episode: 1,
            map: 1,
            player_active: 1,
            ..AgentGameStateSnapshot::default()
        };
        snapshot.player.health = 100;

        // SAFETY: The pointer references an initialized stack snapshot for the
        // duration of the call.
        unsafe {
            restfuldoom_agent_publish_state(&snapshot);
        }

        let observed = timeout(Duration::from_secs(2), stream.message())
            .await
            .expect("state arrives")
            .expect("stream is healthy")
            .expect("state exists");
        assert_eq!(observed.tick, 99);

        let reset = client
            .reset_episode(ResetEpisodeRequest {
                skill: 9,
                episode: 0,
                map: 0,
                seed: 321,
                run_id: "roundtrip".to_string(),
                start: None,
            })
            .await
            .expect("reset queues")
            .into_inner();
        assert!(reset.accepted);
        assert!(!reset.seed_applied);
        assert_eq!(
            (reset.skill, reset.episode, reset.map, reset.seed),
            (4, 1, 1, 321)
        );

        let mut request = AgentControlRequest::default();
        for _ in 0..20 {
            // SAFETY: The output pointer references a valid stack variable.
            if unsafe { restfuldoom_agent_take_control_request(&mut request) } == 1 {
                break;
            }
            sleep(Duration::from_millis(25)).await;
        }
        assert_eq!(request.command, AGENT_CONTROL_RESET_EPISODE);
        assert_eq!(
            (request.skill, request.episode, request.map, request.seed),
            (4, 1, 1, 321)
        );

        let save = client
            .save_snapshot(SaveSnapshotRequest {
                slot: 4,
                description: "first contact".to_string(),
                run_id: "roundtrip".to_string(),
            })
            .await
            .expect("snapshot save queues")
            .into_inner();
        assert!(save.accepted);
        assert!(save.save_queued);
        assert_eq!(save.slot, 4);

        for _ in 0..20 {
            // SAFETY: The output pointer references a valid stack variable.
            if unsafe { restfuldoom_agent_take_control_request(&mut request) } == 1 {
                break;
            }
            sleep(Duration::from_millis(25)).await;
        }
        assert_eq!(request.command, AGENT_CONTROL_SAVE_SNAPSHOT);
        assert_eq!(request.snapshot_slot, 4);
        assert_eq!(&request.snapshot_description[..13], b"first contact");

        let load = client
            .load_snapshot(LoadSnapshotRequest {
                slot: 4,
                run_id: "roundtrip".to_string(),
            })
            .await
            .expect("snapshot load queues")
            .into_inner();
        assert!(load.accepted);
        assert!(load.load_queued);
        assert_eq!(load.slot, 4);

        for _ in 0..20 {
            // SAFETY: The output pointer references a valid stack variable.
            if unsafe { restfuldoom_agent_take_control_request(&mut request) } == 1 {
                break;
            }
            sleep(Duration::from_millis(25)).await;
        }
        assert_eq!(request.command, AGENT_CONTROL_LOAD_SNAPSHOT);
        assert_eq!(request.snapshot_slot, 4);

        tx.send(PlayerAction {
            tick: 100,
            action: PlayerActionType::ActionForward.into(),
            amount: 25,
            ..PlayerAction::default()
        })
        .await
        .expect("action sends");

        let mut action = AgentPlayerAction::default();
        for _ in 0..20 {
            // SAFETY: The output pointer references a valid stack variable.
            if unsafe { restfuldoom_agent_take_action(&mut action) } == 1 {
                break;
            }
            sleep(Duration::from_millis(25)).await;
        }

        assert_eq!(action.tick, 100);
        assert_eq!(action.forward_move, 25);
    }
}
