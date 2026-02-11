import numpy as np
from scipy.optimize import linear_sum_assignment
from ursina import *
import random
import math
from ursina.prefabs.trail_renderer import TrailRenderer
import json

# Load Configuration
with open("game_config.json", "r") as f:
    CONFIG = json.load(f)

# ==========================================
# PHYSICS KERNEL
# ==========================================
def predict_intercept(shooter_pos, target_pos, target_vel, projectile_speed):
    to_target = target_pos - shooter_pos
    target_speed_sq = target_vel.length_squared()
    proj_speed_sq = projectile_speed ** 2
    
    a = target_speed_sq - proj_speed_sq
    b = 2 * to_target.dot(target_vel)
    c = to_target.length_squared()
    
    if abs(a) < 1e-6:
        t = -c / (2*b) if b != 0 else -1
    else:
        delta = b**2 - 4*a*c
        if delta < 0: return None, None
        t1 = (-b + sqrt(delta)) / (2*a)
        t2 = (-b - sqrt(delta)) / (2*a)
        if t1 > 0 and t2 > 0: t = min(t1, t2)
        elif t1 > 0: t = t1
        elif t2 > 0: t = t2
        else: return None, None

    return target_pos + (target_vel * t), t

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
WINDOW_TITLE = "Counter-UAS Defense Simulation"
WINDOW_SIZE = (1600, 900) # Higher Res

# Game Balance
BUDGET_START = CONFIG['economy']['start_budget']
COST_LASER_FIRE = CONFIG['economy']['cost_laser_operation']
COST_MISSILE_FIRE = CONFIG['economy']['cost_missile_launch']
REWARD_DRONE_KILL = CONFIG['economy']['reward_kill']

# Colors
COLOR_BACKGROUND = color.hex(CONFIG['colors_hex']['background'])
COLOR_POLAR_GRID = color.hex(CONFIG['colors_hex']['grid'])
COLOR_HUD_BG = color.hex(CONFIG['colors_hex']['hud_bg'])
COLOR_TEXT = color.hex(CONFIG['colors_hex']['text'])
COLOR_RED_TEAM = color.hex(CONFIG['colors_hex']['red_team'])
COLOR_BLUE_TEAM = color.hex(CONFIG['colors_hex']['blue_team'])
COLOR_LASER = color.hex(CONFIG['colors_hex']['laser'])
COLOR_MISSILE = color.hex(CONFIG['colors_hex']['missile'])
COLOR_EXPLOSION = color.hex(CONFIG['colors_hex']['explosion'])

# ==========================================
# URSINA APP SETUP
# ==========================================
app = Ursina(title=CONFIG['window']['title'], size=(CONFIG['window']['width'], CONFIG['window']['height']))
window.color = color.black # Force black on window level
camera.bg_color = color.black # Force black on camera level
window.borderless = False
window.fullscreen = False

# Camera Setup (RTS Style)
camera.position = (0, 60, -60)
camera.rotation_x = 45
EditorCamera() 

# Force Dark Background for Mac (Giant Sphere)
Entity(model='sphere', scale=1000, color=color.black, double_sided=True, unlit=True)

# Lighting (Ambient mainly, as we use unlit neon materials)
AmbientLight(color=color.rgba(100, 100, 100, 0.1))

# Scene: Polar Grid (Procedural Lines)
def create_polar_grid():
    # Helper to create a circle line mesh
    def create_circle_line(radius, color):
        segments = 128 # High resolution
        verts = []
        for i in range(segments + 1):
            angle = math.radians((i / segments) * 360)
            verts.append(Vec3(math.cos(angle) * radius, 0, math.sin(angle) * radius))
        
        Entity(model=Mesh(vertices=verts, mode='line', thickness=2), color=color, unlit=True)

    # Concentric Circles
    for r in [50, 100, 150, 200, 250]:
        create_circle_line(r, COLOR_POLAR_GRID)
    
    # Radial Lines (Crosshairs) - Using Mesh lines for consistency
    line_length = 500
    Entity(model=Mesh(vertices=[Vec3(-line_length,0,0), Vec3(line_length,0,0)], mode='line', thickness=2), color=COLOR_POLAR_GRID, unlit=True)
    Entity(model=Mesh(vertices=[Vec3(0,0,-line_length), Vec3(0,0,line_length)], mode='line', thickness=2), color=COLOR_POLAR_GRID, unlit=True)
        
create_polar_grid()

# Scene: Central Asset (The thing to defend)
central_asset = Entity(model='cube', scale=(2, 4, 2), color=color.cyan, position=(0, 2, 0), shader=None)
Text(text="BASE", position=(0, 0.2), origin=(0, 0), scale=2, color=color.cyan, parent=central_asset)

# Global Game State
game_state = {
    'budget': BUDGET_START,
    'threats_neutralized': 0,
    'game_over': False,
    'total_threat_value_destroyed': 0,
    'total_ammo_cost': 0
}

# Removed Duplicates
# ==========================================
# UI: HEADS UP DISPLAY (HUD)
# ==========================================
# Dashboard Panel
hud_panel = Entity(parent=camera.ui, model='quad', scale=(0.4, 0.2), position=(-0.65, 0.35), color=COLOR_HUD_BG)

# Text Elements
ui_header = Text(text="[ SYSTEM STATUS: ONLINE ]", position=(-0.83, 0.46), scale=1.2, color=color.cyan)
ui_budget = Text(text=f"BUDGET:   ${game_state['budget']}", position=(-0.83, 0.42), scale=1, color=color.white)
ui_score  = Text(text=f"NEUTRALIZED: {game_state['threats_neutralized']}", position=(-0.83, 0.39), scale=1, color=color.white)
ui_wave   = Text(text="WAVE:     SCENARIO", position=(-0.83, 0.36), scale=1, color=color.white)
ui_ratio  = Text(text="EFFICIENCY: 0.00", position=(-0.83, 0.33), scale=1, color=color.yellow)
ui_wta    = Text(text="WTA LOGIC: ACTIVE (10Hz)", position=(-0.83, 0.30), scale=1, color=color.green)

# Ghost Lines Visuals
ghost_lines = Entity(model=Mesh(mode='line', thickness=1), color=color.rgba(255, 255, 255, 30), unlit=True)

game_over_text = Text(text="GAME OVER", position=(0, 0), origin=(0, 0), scale=4, color=color.red, enabled=False)
restart_text = Text(text="Press 'R' to Restart", position=(0, -0.1), origin=(0, 0), scale=2, color=color.white, enabled=False)

def update_ui():
    ui_budget.text = f"BUDGET:   ${int(game_state['budget']):,}"
    ui_score.text  = f"NEUTRALIZED: {game_state['threats_neutralized']}"
    # Calculate Ratio
    if game_state['total_ammo_cost'] > 0:
        ratio = game_state['total_threat_value_destroyed'] / game_state['total_ammo_cost']
        ui_ratio.text = f"EFFICIENCY: {ratio:.2f}"
    else:
        ui_ratio.text = "EFFICIENCY: 0.00"

def restart_game():
    # Reset State
    game_state['budget'] = BUDGET_START
    game_state['threats_neutralized'] = 0
    game_state['game_over'] = False
    game_state['total_threat_value_destroyed'] = 0
    game_state['total_ammo_cost'] = 0
    
    # Clear Entities
    for d in drones: destroy(d)
    for p in projectiles: destroy(p)
    drones.clear()
    projectiles.clear()
    
    # Reset Base
    central_asset.color = color.cyan
    game_over_text.enabled = False
    restart_text.enabled = False
    
    # Reset Scenario
    scenario_manager.current_event = 0
    scenario_manager.time_elapsed = 0
    
    update_ui()

def input(key):
    if key == 'r' and game_state['game_over']:
        restart_game()

# Lists to track entities for the Manager
drones = []
defenses = []
projectiles = []

# ==========================================
# CLASSES: RED TEAM (DRONES)
# ==========================================
# ==========================================
# CLASSES: RED TEAM (DRONES)
# ==========================================
class Drone(Entity):
    def __init__(self, position, target_pos=(0,0,0)):
        super().__init__(
            model='sphere',
            color=COLOR_RED_TEAM,
            position=position,
            scale=1,
            collider='box'
        )
        self.health = 100
        self.speed = 10
        self.value = 1000
        self.target_position = Vec3(target_pos)
        
        # Physics Mode: 'linear', 'noe', 'ballistic', 'jinking'
        self.physics_mode = 'linear' 
        
        # Specific Params
        self.jink_range = 200 # Distance to start jinking
        self.evasion_phase = random.uniform(0, 2 * math.pi)
        
        # For Ballistic
        self.start_pos = position
        self.ballistic_t = 0
        
        # Trail
        self.trail = TrailRenderer(parent=self, thickness=2, color=self.color, length=10)
        
        self.velocity = Vec3(0,0,0)  # Initialize velocity to prevent crash on first frame

    def update(self):
        if game_state['game_over']: return
        
        dist_to_target = distance(self.position, self.target_position)
        
        # 1. NAP-OF-THE-EARTH (NOE)
        if self.physics_mode == 'noe':
            # Fly at y=3, ignore hills (assuming flat ground for now)
            target_h = Vec3(self.target_position.x, 3, self.target_position.z)
            direction = (target_h - self.position).normalized()
            self.position += direction * self.speed * time.dt
            self.y = lerp(self.y, 3, time.dt * 5) # Snap to ground level

        # 2. BALLISTIC (Gravity Dive)
        elif self.physics_mode == 'ballistic':
            # Parabolic approximation or just linear dive from high altitude?
            # User said "pure ballistic dive". 
            # Simple implementation: steer towards target but keep high speed.
            direction = (self.target_position - self.position).normalized()
            self.position += direction * self.speed * time.dt
            self.rotation_x += time.dt * 10 # Spin or tilt?
            self.look_at(self.target_position)

        # 3. TERMINAL JINKING
        elif self.physics_mode == 'jinking':
            # Linear approach until close
            to_target = (self.target_position - self.position).normalized()
            
            if dist_to_target < self.jink_range:
                # High G Jinking
                t = time.time() * 10 # Fast frequency
                jink = Vec3(math.sin(t + self.evasion_phase), math.cos(t), 0) * 0.5
                move_dir = (to_target + jink).normalized()
            else:
                move_dir = to_target
                
            self.position += move_dir * self.speed * time.dt
            self.look_at(self.target_position)

        # 4. STANDARD LINEAR
        else:
            direction = (self.target_position - self.position).normalized()
            self.position += direction * self.speed * time.dt
            self.look_at(self.target_position)

        # Game Over Condition
        if dist_to_target < 3:
            print("Base Destroyed!")
            game_state['game_over'] = True
            central_asset.color = color.red

    def take_damage(self, amount):
        self.health -= amount
        self.shake(duration=0.1, magnitude=0.5)
        if self.health <= 0:
            self.die()

    def die(self):
        if self in drones:
            drones.remove(self)
        
        # Explosion Effect
        e = Entity(model='sphere', position=self.position, color=COLOR_EXPLOSION, scale=self.scale)
        e.animate_scale(self.scale*3, duration=0.2, curve=curve.out_expo)
        e.animate_color(color.clear, duration=0.2)
        destroy(e, delay=0.2)
        
        # Update Stats
        game_state['budget'] += self.value * 0.1 
        game_state['threats_neutralized'] += 1
        game_state['total_threat_value_destroyed'] += self.value
        update_ui()
        
        destroy(self)

# --- THREAT LIBRARY ---

class Shahed136(Drone): # The Swarm
    def __init__(self, position):
        super().__init__(position)
        self.model = 'diamond' # Tetrahedron-ish (Cone not standard)
        self.rotation_x = -90
        self.color = color.white
        self.scale = 0.8
        self.health = 50 # Buffed from 30
        self.speed = 12 # Slightly slower for visuals
        self.value = 20000 # $20k
        self.physics_mode = 'linear'
        self.trail.color = color.rgba(255, 255, 255, 100)
        
        # Config Overrides
        conf = CONFIG['units']['shahed']
        self.health = conf['health']
        self.speed = conf['speed']
        self.value = conf['value']
        self.scale = conf['scale']

class DJIMavic(Drone): # The Ghost
    def __init__(self, position):
        super().__init__(position)
        self.model = 'cube'
        self.scale = 0.3 # Tiny
        self.color = color.gray
        self.health = 10
        self.speed = 8 # Slow
        self.value = 1000 # $1k
        self.physics_mode = 'noe' # Nap of Earth
        self.trail.enabled = False # Hard to see
        
        # Config Overrides
        conf = CONFIG['units']['mavic']
        self.health = conf['health']
        self.speed = conf['speed']
        self.value = conf['value']
        self.scale = conf['scale']

class Kalibr(Drone): # The Hammer
    def __init__(self, position):
        super().__init__(position)
        self.model = 'cylinder'
        self.scale = (0.5, 0.5, 2)
        self.color = color.rgb(255, 50, 50) # Red
        self.health = 150
        self.speed = 45
        self.value = 6500000 # $6.5M
        self.physics_mode = 'jinking'
        self.jink_range = 250
        self.trail.color = color.red
        self.trail.thickness = 5
        
        # Config Overrides
        conf = CONFIG['units']['kalibr']
        self.health = conf['health']
        self.speed = conf['speed']
        self.value = conf['value']
        self.scale_z = conf['scale_z']
        self.jink_range = conf['jink_range']

class Kinzhal(Drone): # The Sprinter
    def __init__(self, position):
        super().__init__(position)
        self.model = 'diamond' # Aerodynamic
        self.scale = 1.0
        self.color = color.orange
        self.health = 300 # Tough
        self.speed = 80 # Hypersonic
        self.value = 10000000 # $10M
        self.physics_mode = 'ballistic'
        # Spawning logic handles the high altitude start
        self.trail.color = color.orange
        self.trail.thickness = 8

        # Apply Config Overrides
        unit_conf = CONFIG['units']['kinzhal']
        self.health = unit_conf['health']
        self.speed = unit_conf['speed']
        self.value = unit_conf['value']
        self.scale = unit_conf['scale']

# ==========================================
# CLASSES: BLUE TEAM (DEFENSES)
# ==========================================
class DefenseSystem(Entity):
    def __init__(self, position):
        super().__init__(
            model='cube',
            color=COLOR_BLUE_TEAM,
            position=position,
            scale=(1.5, 1, 1.5),
            collider='box'
        )
        self.cooldown_timer = 0
        self.cooldown = 1.0
        self.range = 50
        self.cost_per_shot = 0
        self.priority_weight = 1.0 # Multiplier for WTA

    def update(self):
        if self.cooldown_timer > 0:
            self.cooldown_timer -= time.dt

        self.cooldown_timer = 0
        self.cooldown = 1.0
        self.range = 50
        self.cost_per_shot = 0
        self.priority_weight = 1.0 # Multiplier for WTA
        self.name = "Generic Defense"

    def update(self):
        if self.cooldown_timer > 0:
            self.cooldown_timer -= time.dt

    def can_fire(self):
        return self.cooldown_timer <= 0 and game_state['budget'] >= self.cost_per_shot

    def fire(self, target_drone):
        game_state['budget'] -= self.cost_per_shot
        game_state['total_ammo_cost'] += self.cost_per_shot
        self.cooldown_timer = self.cooldown
        
        # Floating Text for Cost
        t = Text(text=f"-${self.cost_per_shot:,}", position=(0, 2, 0), origin=(0,0), scale=1.5, color=color.red, parent=self)
        t.animate_position(t.position + (0, 2, 0), duration=1.0)
        t.animate_color(color.clear, duration=1.0)
        destroy(t, delay=1.0)
        
        update_ui()
        # Look at target
        self.look_at(target_drone)

class DragonFireLaser(DefenseSystem): # The Cost Cutter
    def __init__(self, position):
        super().__init__(position)
        self.color = color.cyan
        self.cooldown = CONFIG['defenses']['dragonfire']['cooldown']
        self.range = CONFIG['defenses']['dragonfire']['range']
        self.cost_per_shot = CONFIG['defenses']['dragonfire']['cost']
        self.name = "DragonFire"
        self.current_target = None
        self.dwell_timer = 0.0
        self.required_dwell = CONFIG['defenses']['dragonfire']['required_dwell_time']
        
        # Visual
        self.turret = Entity(parent=self, model='cube', scale=(0.2, 0.2, 1), position=(0, 0.5, 0), color=color.cyan)
        self.beam = None

    def update(self):
        super().update()
        
        # Dwell Logic
        if self.current_target:
            if not self.current_target.enabled or distance(self.position, self.current_target.position) > self.range:
                self.stop_firing()
                return

            self.turret.look_at(self.current_target)
            self.dwell_timer += time.dt
            
            # Beam Visual
            if not self.beam:
                 self.beam = Entity(parent=self, model='cube', color=color.cyan, scale=(0.05, 0.05, 0), position=(0,0.5,0), unlit=True)
            
            dist = distance(self.position, self.current_target.position)
            self.beam.look_at(self.current_target)
            self.beam.scale_z = dist
            self.beam.position = self.turret.position + self.beam.forward * dist / 2
            
            # Cost per frame/tick? User said $10 per shot. Let's charge per kill or per second?
            # Let's charge per activation.
            
            if self.dwell_timer >= self.required_dwell:
                self.current_target.take_damage(1000) # Instakill small drones
                self.stop_firing()

    def stop_firing(self):
        self.current_target = None
        self.dwell_timer = 0
        if self.beam: destroy(self.beam)
        self.beam = None

    def fire(self, target_drone):
        if self.current_target == target_drone: return # Already cooking
        
        # Logic Check: DragonFire is for Mavic/Shahed. 
        # Can it hit missiles? Maybe, but let's focus it.
        
        game_state['budget'] -= self.cost_per_shot
        game_state['total_ammo_cost'] += self.cost_per_shot # Counts every frame active?
        self.current_target = target_drone
        self.dwell_timer = 0
        update_ui()
        
    def can_fire(self):
        return self.current_target is None # Only one target at a time

class PatriotBattery(DefenseSystem): # The Sniper
    def __init__(self, position):
        super().__init__(position)
        self.color = color.rgb(100, 100, 50) # Olive
        self.cooldown = 2.0 # Buffed: 5.0 -> 2.0s
        self.range = 400 # Extreme
        self.cost_per_shot = 5000 # Arcade Cost: $5k (was $3M)
        self.name = "Patriot PAC-3"
        
        # Config Overrides
        conf = CONFIG['defenses']['patriot']
        self.range = conf['range']
        self.cooldown = conf['cooldown']
        self.cost_per_shot = conf['cost']
        
        # Visuals
        self.turret = Entity(parent=self, model='cube', scale=(0.8, 0.8, 1.2), rotation_x=-45, color=color.rgb(150,150,100))

    def fire(self, target_drone):
        # Strict Logic: Only fire at High Value Targets
        if not isinstance(target_drone, (Kalibr, Kinzhal)):
            # print("Patriot: Target too cheap, holding fire.")
            return

        super().fire(target_drone)
        
        # Spawn Interceptor
        Projectile(self.position + (0, 2, 0), target_drone, damage=500, speed=60, color=color.yellow)

class GepardFlak(DefenseSystem): # The Shredder
    def __init__(self, position):
        super().__init__(position)
        self.color = color.gray
        self.color = color.gray
        self.cooldown = 0.08 # Balanced: 0.1 -> 0.05 -> 0.08
        self.range = 150
        self.cost_per_shot = 100 # Cheap ammo
        self.name = "Gepard"
        
        # Config Overrides
        conf = CONFIG['defenses']['gepard']
        self.range = conf['range']
        self.cooldown = conf['cooldown']
        self.cost_per_shot = conf['cost']
        
        self.turret = Entity(parent=self, model='cube', scale=(0.4, 0.4, 0.8), color=color.dark_gray)

    def fire(self, target_drone):
        # Logic: Ineffective vs Hypersonic (Kinzhal)
        if isinstance(target_drone, Kinzhal):
            return 

        super().fire(target_drone)
        
        # Burst Visuals (Tracers)
        for i in range(3):
            Projectile(self.position + (0, 1.5, 0) + Vec3(random.uniform(-0.5,0.5),0,0), target_drone, damage=15, speed=30, color=color.yellow)

class Projectile(Entity):
    def __init__(self, position, target, damage, speed=20, color=color.yellow):
        super().__init__(
            model='sphere',
            color=color,
            position=position,
            scale=0.2
        )
        self.target = target
        self.damage = damage
        self.speed = speed
        self.trail = TrailRenderer(parent=self, thickness=2, color=color, length=5)

    def update(self):
        try:
            if not self.target or not self.target.enabled:
                if self in projectiles: projectiles.remove(self)
                destroy(self)
                return
                
            # Proportional Navigation (Intercept Prediction)
            predicted_pos, t = predict_intercept(self.position, self.target.position, self.target.velocity, self.speed)
            target_point = predicted_pos if predicted_pos else self.target.position
            
            direction = (target_point - self.position).normalized()
            self.look_at(target_point) # Visual orientation
            self.position += direction * self.speed * time.dt
        except:
            if self in projectiles: projectiles.remove(self)
            destroy(self)
            return
        
        if distance(self.position, self.target.position) < 1:
            self.target.take_damage(self.damage)
            if self in projectiles: projectiles.remove(self)
            destroy(self)

# ==========================================
# WEAPON-TARGET ASSIGNMENT (WTA) LOGIC
# ==========================================
lock_lines = Entity(model=Mesh(mode='line', thickness=2), color=color.rgba(0, 255, 0, 100), unlit=True)

def run_wta_logic():
    """
    Solves the Weapon-Target Assignment problem using scipy.optimize.linear_sum_assignment.
    Cost Function considers:
    1. Distance (Closer is better/lower cost)
    2. Range Constraints (Out of range = Infinite Cost)
    3. Priority (Higher threat target = Lower cost to assign heavy weapons)
    """
    
    # 1. Identify Available Weapons (Ready to fire)
    available_defenses = [d for d in defenses if d.can_fire()]
    if not available_defenses or not drones:
        return

    # 2. Build Cost Matrix
    # Rows = Weapons, Cols = Targets (Drones)
    n_weapons = len(available_defenses)
    n_targets = len(drones)
    n_targets = len(drones)
    cost_matrix = np.zeros((n_weapons, n_targets))

    # Identify doomed targets (already have a missile incoming)
    doomed_targets = set()
    for p in projectiles:
        if p.target and p.target.enabled:
            doomed_targets.add(p.target)

    for i, weapon in enumerate(available_defenses):
        for j, drone in enumerate(drones):
            # Constraint: Already Doomed?
            if isinstance(weapon, PatriotBattery) and drone in doomed_targets:
                cost_matrix[i, j] = 1e8 
                continue

            dist = distance(weapon.position, drone.position)
            
            # Constraint: Range
            if dist > weapon.range:
                cost_matrix[i, j] = 1e9
                continue
            
            # 1. Logic Checks (Hard Constraints)
            if isinstance(weapon, PatriotBattery) and not isinstance(drone, (Kalibr, Kinzhal)):
                cost_matrix[i, j] = 1e9 # Block firing at small stuff
                continue

            if isinstance(weapon, GepardFlak) and isinstance(drone, Kinzhal):
                 cost_matrix[i, j] = 1e9 # Cannot hit hypersonic
                 continue

            # 2. Financial Logic (Soft Constraints)
            # "If Threat_Value < Interceptor_Cost, the assignment cost is penalized heavily"
            if drone.value < weapon.cost_per_shot:
                # Penalty proportional to waste
                waste_ratio = weapon.cost_per_shot / max(1, drone.value) # e.g. 3M / 1k = 3000
                financial_penalty = waste_ratio * 1000
            else:
                financial_penalty = 0

            norm_dist = dist / weapon.range
            priority_score = (drone.value / 100000) # Scale value
            
            # Total Cost
            cost = (norm_dist * 10) - priority_score + financial_penalty
            cost_matrix[i, j] = cost

    # 3. Solve Assignment Problem (Hungarian Algorithm)
    # Returns optimal row_ind (weapons) and col_ind (targets) to minimize total cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 4. Execute Orders & Visualize
    assignments = []
    lines_verts = [] # For lock lines
    
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1e8:
            # Safety Check: Ensure entities still exist
            if not available_defenses[r] or not drones[c]: continue
            if not available_defenses[r].enabled or not drones[c].enabled: continue
            
            assignments.append((available_defenses[r], drones[c]))
            
            # Add line visual (Safe Access)
            try:
                w_pos = available_defenses[r].position + Vec3(0,2,0)
                d_pos = drones[c].position
                lines_verts.append(w_pos)
                lines_verts.append(d_pos)
            except:
                continue
    
    # Update Lock Lines
    if lines_verts:
        lock_lines.model.vertices = lines_verts
        lock_lines.model.generate()
    else:
        lock_lines.model.vertices = []
        lock_lines.model.generate()
    
    # Firing Phase
    for weapon, target in assignments:
        try:
            if target and target.enabled: 
                 weapon.fire(target)
        except:
            continue

# ==========================================
# GAME MAP SETUP
# ==========================================
def setup_defenses():
    # Place defenses around the center
    
    # 2x DragonFire (Lasers) - Inner Layer
    defenses.append(DragonFireLaser(position=(5, 0, 0)))
    defenses.append(DragonFireLaser(position=(-5, 0, 0)))
    
    # 2x Gepard (Flak) - Mid Layer
    defenses.append(GepardFlak(position=(0, 0, 8)))
    defenses.append(GepardFlak(position=(0, 0, -8)))
    
    # 2x Patriot (Anti-Missile) - Outer Layer
    defenses.append(PatriotBattery(position=(15, 0, 15)))
    defenses.append(PatriotBattery(position=(-15, 0, -15)))
    
    # NEW: 2x Extra Gepards to handle saturation
    defenses.append(GepardFlak(position=(8, 0, 0)))
    defenses.append(GepardFlak(position=(-8, 0, 0)))

setup_defenses()

# ==========================================
# SCENARIO MANAGER: OPERATION DARK WINTER
# ==========================================
class ScenarioManager(Entity):
    def __init__(self):
        super().__init__()
        self.time_elapsed = 0
        self.timeline = CONFIG['scenario_timeline']
        # Convert timeline to tuple format if needed, but list of dicts is cleaner
        # Adapter to match old format Loop
        self.timeline_events = []
        for event in self.timeline:
            self.timeline_events.append((event['time'], event['type'], event['count'], event['desc']))
        self.current_event = 0
        self.wave_text = Text(text="", position=(0, 0.4), scale=2, color=color.red, origin=(0,0), enabled=False)

    def update(self):
        if game_state['game_over']: return
        
        self.time_elapsed += time.dt
        
        if self.current_event < len(self.timeline_events):
            t, type_code, count, desc = self.timeline_events[self.current_event]
            
            if self.time_elapsed >= t:
                self.execute_event(type_code, count, desc)
                self.current_event += 1

    def execute_event(self, type_code, count, desc):
        print(f"EVENT: {desc}")
        
        # UI Feedback
        self.wave_text.text = desc
        self.wave_text.enabled = True
        self.wave_text.animate_color(color.clear, duration=4.0)
        invoke(lambda: setattr(self.wave_text, 'enabled', False), delay=4.0)
        
        # Spawning Logic
        for i in range(count):
            # Spawn Logic based on type
            offset = Vec3(random.uniform(-10,10), 0, random.uniform(-10,10))
            
            if type_code == 'SHA': # Swarm
                 # Spawn in a flock at distance
                 angle = random.uniform(0, 360) 
                 # Tight group
                 spawn_pos = Vec3(math.cos(angle)*80, 20, math.sin(angle)*80) + offset
                 drones.append(Shahed136(spawn_pos))
                 
            elif type_code == 'MAV': # Ground Level
                 angle = random.uniform(0, 360) 
                 spawn_pos = Vec3(math.cos(angle)*60, 3, math.sin(angle)*60) + offset
                 drones.append(DJIMavic(spawn_pos))

            elif type_code == 'KIN': # High Altitude Dive
                 angle = random.uniform(0, 360) 
                 # Start very high (y=200)
                 spawn_pos = Vec3(math.cos(angle)*100, 200, math.sin(angle)*100)
                 drones.append(Kinzhal(spawn_pos))

            elif type_code == 'KAL': # Stand off distance
                 angle = random.uniform(0, 360) 
                 spawn_pos = Vec3(math.cos(angle)*120, 50, math.sin(angle)*120)
                 drones.append(Kalibr(spawn_pos))

scenario_manager = ScenarioManager()

# ==========================================
# VISUALIZATION UTILS
# ==========================================
def draw_ghost_lines():
    verts = []
    for d in drones:
        if not d.enabled: continue
        # Project future path (Linear assumption for visualization)
        start = d.position
        end = d.position + d.velocity * 3.0 # 3 seconds ahead
        verts.append(start)
        verts.append(end)
    
    ghost_lines.model.vertices = verts
    ghost_lines.model.generate()

# ==========================================
# MAIN UPDATE LOOP
# ==========================================
def update():
    if game_state['game_over']:
        if not game_over_text.enabled:
            game_over_text.enabled = True
            restart_text.enabled = True
        return

    # Scenario Manager runs via Entity.update automatic call
    
    # Run WTA Logic (The Brain)
    run_wta_logic()
    
    # Draw prediction lines
    draw_ghost_lines()

# ==========================================
# START
# ==========================================
camera.orthographic = False 

if __name__ == '__main__':
    app.run()
