import numpy as np
from ursina import *
from ursina import *
# from ursina.prefabs.trail_renderer import TrailRenderer # REMOVED due to artifacts
import math
import json

# ==========================================
# CONFIGURATION
# ==========================================
with open("sim_v2_config.json", "r") as f:
    CONFIG = json.load(f)

DATA_FILE = "simulation_data_v2.npy"
MAX_POOL_SIZE = 1200 

# Type Handling
ID_SHAHED = 1
ID_DECOY = 2
ID_KALIBR = 3
ID_KINZHAL = 4
ID_PATRIOT_MSE = 30
ID_GEPARD = 31
ID_IRON_BEAM = 32
ID_EW_JAMMER = 33
ID_INTERCEPTOR_DRONE = 34
ID_OUTPOST = 35
ID_MISSILE_PAC3 = 40
ID_BULLET_35MM = 41

# Color Mapper
COLOR_MAP = {
    "white": color.white, "light_gray": color.light_gray, "gray": color.gray, "dark_gray": color.dark_gray,
    "black": color.black, "red": color.red, "green": color.green, "blue": color.blue,
    "cyan": color.cyan, "orange": color.orange, "yellow": color.yellow, "violet": color.violet,
    "magenta": color.magenta
}

def get_color(name_or_list):
    if isinstance(name_or_list, list): return color.rgba(*name_or_list)
    return COLOR_MAP.get(name_or_list, color.white)

# Build Type Map from Config
TYPE_MAP = {}
def add_to_map(json_key, default_model):
    if json_key in CONFIG['units']: data = CONFIG['units'][json_key]
    elif json_key in CONFIG['defenses']: data = CONFIG['defenses'][json_key]
    elif json_key in CONFIG['projectiles']: data = CONFIG['projectiles'][json_key]
    else: return

# Build Type Map from Config
def get_vec3(val):
    if isinstance(val, list): return tuple(val)
    return val

def get_color(name_or_list):
    if isinstance(name_or_list, list): return color.rgba(*name_or_list)
    if hasattr(color, name_or_list): return getattr(color, name_or_list)
    return color.white

TYPE_MAP = {}

# Units
for key, data in CONFIG['units'].items():
     uid = data['id']
     TYPE_MAP[uid] = {
         'model': 'diamond' if 'shahed' in key or 'kinzhal' in key else ('cube' if 'kalibr' in key else 'sphere'),
         'scale': get_vec3(data.get('scale', 1)),
         'color': get_color(data.get('color', 'white'))
     }

# Defenses
for key, data in CONFIG['defenses'].items():
     uid = data['id']
     TYPE_MAP[uid] = {
         'model': 'cube',
         'scale': get_vec3(data.get('scale', 1)),
         'color': get_color(data.get('color', 'blue'))
     }

# Projectiles (Manual Override or Config?)
TYPE_MAP[ID_MISSILE_PAC3] = {
    'model': 'diamond', 
    'scale': get_vec3(CONFIG['projectiles']['pac3_missile'].get('scale', 1)), 
    'color': get_color(CONFIG['projectiles']['pac3_missile'].get('color', 'green'))
}
TYPE_MAP[ID_BULLET_35MM] = {
    'model': 'cube',
    'scale': get_vec3(CONFIG['projectiles']['flak_tracer'].get('scale', 1)),
    'color': get_color(CONFIG['projectiles']['flak_tracer'].get('color', 'yellow'))
}
# Override Iron Beam model
TYPE_MAP[ID_IRON_BEAM]['model'] = 'sphere'

# ==========================================
# VISUALIZER ENGINE
# ==========================================
app = Ursina(title="Dark Winter V2: Advanced Simulation", vsync=False)

# Load Data
print(f"Loading {DATA_FILE}...")
try:
    history = np.load(DATA_FILE)
    FRAMES, ENTITIES, CHANNELS = history.shape
    print(f"Loaded: {FRAMES} frames, {ENTITIES} entities.")
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Run sim_kernel_v2.py first.")
    sys.exit()

# Scene Setup
# Custom Camera Controller
camera.position = (0, 8000, -5000) # Further back for better overview
camera.rotation_x = 60
camera.clip_plane_far = 50000 
CAMERA_SPEED = 2000 # m/s 

# Environment
Entity(model='sphere', scale=100000, color=color.black, double_sided=True, unlit=True)
AmbientLight(color=color.rgba(100, 100, 100, 0.1))

# Grid
def create_polar_grid():
    def create_circle_line(radius, color):
        segments = 128
        verts = []
        for i in range(segments + 1):
            angle = math.radians((i / segments) * 360)
            verts.append(Vec3(math.cos(angle) * radius, 0, math.sin(angle) * radius))
        Entity(model=Mesh(vertices=verts, mode='line', thickness=2), color=color, unlit=True)

    color_grid = color.rgba(255, 255, 255, 20)
    for r in [5000, 10000, 15000, 20000]: # 20km range
        create_circle_line(r, color_grid)
    
    ln = 25000
    Entity(model=Mesh(vertices=[Vec3(-ln,0,0), Vec3(ln,0,0)], mode='line', thickness=1), color=color_grid, unlit=True)
    Entity(model=Mesh(vertices=[Vec3(0,0,-ln), Vec3(0,0,ln)], mode='line', thickness=1), color=color_grid, unlit=True)
create_polar_grid()

# HUD
hud_panel = Entity(parent=camera.ui, model='quad', scale=(0.4, 0.2), position=(-0.65, 0.35), color=color.rgba(0, 0, 0, 150))
Text(text="[ HYBRID SIM V2 ]", position=(-0.83, 0.46), scale=1.2, color=color.cyan)
txt_status = Text(text="SYSTEM: ONLINE", position=(-0.83, 0.42), scale=1, color=color.white)
txt_stats = Text(text="BUDGET: $ -- | KILLS: 0", position=(-0.83, 0.38), scale=1, color=color.white)

# Game State (Replicates V1 Logic locally)
p_budget = 5000000 
p_kills = 0
p_game_over = False

# Object Pool
pool = []
for i in range(MAX_POOL_SIZE):
    e = Entity(enabled=False)
    # e.trail = TrailRenderer(parent=e, thickness=4, length=10, color=color.clear) # REMOVED
    
    # Beam Child (Iron Beam)
    e.beam = Entity(parent=e, model='cube', origin_z=-0.5, scale=(1,1,1), color=color.cyan, enabled=False, unlit=True)
    
    e.exhaust = Entity(parent=e, model='cube', origin_z=0.5, scale=(1,1,1), color=color.orange, enabled=False, unlit=True)
    
    # Explosion FX Pool (Simple Sphere Scaling)
    e.explosion = Entity(parent=scene, model='sphere', scale=1, color=color.orange, enabled=False, unlit=True)
    
    pool.append(e)

# Base Entity (The Target)
base_entity = Entity(model='cube', scale=(20, 60, 20), color=color.cyan, position=(0, 30, 0), shader=None)
Text(text="BASE", position=(0, 0.2), origin=(0, 0), scale=2, color=color.cyan, parent=base_entity)

# Game Over UI
txt_game_over = Text(text="BASE DESTROYED", origin=(0,0), scale=3, color=color.red, enabled=False)
slider = Slider(min=0, max=FRAMES-1, default=0, step=1, y=-0.45, scale=1.5, dynamic=True)
is_playing = True
playback_speed = 1.0
frame_accum = 0.0

# History for Velocity Calculation
# prev_positions[uid] = Vec3
prev_positions = {}

def update():
    global is_playing, frame_accum, playback_speed
    global p_budget, p_kills, p_game_over
    
    if p_game_over: return
    
    # Camera Movement (Arrow Keys)
    move_speed = CAMERA_SPEED * time.dt
    if held_keys['shift']: move_speed *= 2
    
    # Pan X/Z (Ground Plane)
    # Relative to camera rotation? No, user requested XY plane movement system.
    # Usually "Arrow Key XY Plane" means absolute world coordinates X and Z.
    if held_keys['up arrow']: camera.z += move_speed
    if held_keys['down arrow']: camera.z -= move_speed
    if held_keys['left arrow']: camera.x -= move_speed
    if held_keys['right arrow']: camera.x += move_speed
    
    # Zoom / Altitude (W/S or Scroll)
    if held_keys['w']: camera.y += move_speed
    if held_keys['s']: camera.y -= move_speed
    
    # Looking Around (Right Click)
    if mouse.right:
        camera.rotation_y += mouse.velocity[0] * 150
        camera.rotation_x -= mouse.velocity[1] * 150
    
    current_frame = int(slider.value)
    
    if is_playing:
        frame_accum += time.dt * 100 * playback_speed
        if frame_accum >= 1:
            step = int(frame_accum)
            current_frame += step
            frame_accum -= step
            if current_frame >= FRAMES: current_frame = 0
            slider.value = current_frame

    # Render
    data_frame = history[current_frame]
    
    pool_idx = 0
    
    # Fast Scan
    for i in range(ENTITIES):
        if pool_idx >= MAX_POOL_SIZE: break
        
        props = data_frame[i]
        active = props[0] > 0.5
        
        if active:
            e = pool[pool_idx]
            e.enabled = True
            
            # Pos
            e.position = Vec3(props[2], props[3], props[4])
            
            # Visuals
            type_id = int(props[1])
            if not hasattr(e, 'current_type') or e.current_type != type_id:
                if type_id in TYPE_MAP:
                    conf = TYPE_MAP[type_id]
                    e.model = conf['model']
                    e.scale = conf['scale']
                    e.color = conf['color']
                    e.current_type = type_id
            
            # Events
            evt = int(props[7])
            target_uid = int(props[8])
            
            # --- GEOMETRY TRACER LOGIC ---
            # Velocity Calculation
            uid = i # Index in this array logic is implicitly uid if data is ordered 0 to Max
            # history shape is (FRAMES, MAX_ENTITIES, 9). 
            # Yes, i is the entity UID for that slot.
            
            current_pos = e.position
            velocity = Vec3(0,0,1) # Default
            
            if i in prev_positions:
                velocity = current_pos - prev_positions[i]
                if velocity.length_squared() < 0.01:
                    velocity = Vec3(0,0,1) # Fallback to prevent look_at error
            
            # Store for next frame (Note: This logic runs every visual frame, logic steps might skip)
            # Actually we should store frame-to-frame from data, not visual update.
            # But visually this is fine for smoothing.
            prev_positions[i] = current_pos

            # 1. Beams (Iron Beam)
            if type_id == ID_IRON_BEAM:
                if target_uid != -1: # Firing
                     e.beam.enabled = True
                     # Look up target pos
                     t_row = history[current_frame][target_uid]
                     t_pos = Vec3(t_row[2], t_row[3], t_row[4])
                     dist = distance(e.position, t_pos)
                     
                     e.beam.enabled = False # User requested NO TARGETING LINES
                else:
                     e.beam.enabled = False
            else:
                 e.beam.enabled = False

            # 2. Jamming Effect
            if evt == 50: # JAMMED
                e.color = color.magenta
            
            # 3. Explosions & Events
            if evt == 99: # HIT / DIE
                 # Explosion Effect (Mimic V1)
                 active = False
                 e.explosion.enabled = True
                 e.explosion.position = e.position
                 e.explosion.scale = e.scale[0] 
                 e.explosion.animate_scale(e.scale[0] * 4, duration=0.2, curve=curve.out_expo)
                 e.explosion.color = color.rgba(255, 100, 0, 255)
                 e.explosion.animate_color(color.clear, duration=0.25)
                 invoke(setattr, e.explosion, 'enabled', False, delay=0.25)
                 e.beam.enabled = False
                 
                 # Score Update
                 p_kills += 1
                 # Add Value (Simple lookup)
                 val = 0
                 if type_id in TYPE_MAP: 
                     # Config doesn't store value in TYPEMAP, need to look up in CONFIG?
                     # Hack: standard reward
                     val = 1000 
                 p_budget += val
                 
            elif evt == 100: # BASE HIT
                 active = False
                 # Base Damage visual
                 base_entity.color = color.red
                 base_entity.shake(duration=0.5, magnitude=1)
                 txt_game_over.enabled = True
                 txt_status.text = "SYSTEM: CRITICAL FAILURE"
                 txt_status.color = color.red
                 p_game_over = True

            # Update HUD
            if i % 100 == 0: # Optimization
                txt_stats.text = f"BUDGET: ${p_budget:,} | KILLS: {p_kills}"

            # 4. EXHAUST PLUMES & TRACERS
            # Reset Visuals
            e.exhaust.enabled = False
            
            if active and evt != 99:
                 if type_id in [ID_MISSILE_PAC3, ID_KINZHAL, ID_KALIBR]: 
                     # Missile Exhaust
                     e.exhaust.enabled = True
                     e.exhaust.color = color.rgba(255, 100, 0, 150) # Orange Fire
                     # Scale relative to missile
                     exhaust_len = 50 if type_id == ID_KINZHAL else 30
                     e.exhaust.scale = (e.scale[0]*0.8, e.scale[1]*0.8, exhaust_len)
                     # Orientation is handled by parent `e` looking at velocity
                     if velocity.length_squared() > 1:
                        e.look_at(e.position + velocity)
                        
                 elif type_id == ID_BULLET_35MM:
                     # GEOMETRY TRACER
                     # Use the entity itself as the tracer
                     if velocity.length_squared() > 0.1:
                        e.look_at(e.position + velocity)
                        e.scale = (4, 4, 40) # 40m Solid Tracer
                         
                 elif type_id in [ID_SHAHED, ID_DECOY, ID_INTERCEPTOR_DRONE]:
                     # Orient Drones forward too
                     if velocity.length_squared() > 1:
                        e.look_at(e.position + velocity)

            
            pool_idx += 1
            
    # Hide Rest
    for i in range(pool_idx, MAX_POOL_SIZE):
        if pool[i].enabled:
            pool[i].enabled = False
        if pool[i].exhaust.enabled: pool[i].exhaust.enabled = False
        pool[i].beam.enabled = False

    # Stats
    txt_status.text = f"FRAME: {current_frame} | ENTITIES: {pool_idx}"

def input(key):
    global is_playing
    if key == 'space': is_playing = not is_playing
    if key == '.': slider.value += 10 # Scrub Forward
    if key == ',': slider.value -= 10 # Scrub Backward
    
    # Mouse Wheel Zoom
    if key == 'scroll up': camera.y -= 200
    if key == 'scroll down': camera.y += 200

if __name__ == '__main__':
    app.run()
