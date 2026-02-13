import numpy as np
from scipy.optimize import linear_sum_assignment
from ursina import *
import random
import math
from ursina.prefabs.trail_renderer import TrailRenderer
import json

with open("reference_legacy/game_config.json", "r") as f:
    CONFIG = json.load(f)

def predict_intercept(shooter_pos, target_pos, target_vel, projectile_speed):
    """
    Calculates the interception point and time for a projectile to hit a moving target.
    Returns (intercept_position, time_to_intercept) or (None, None) if impossible.
    """
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

WINDOW_TITLE = "Counter-UAS Defense Simulation"
WINDOW_SIZE = (800, 450)

BUDGET_START = CONFIG['economy']['start_budget']
COST_LASER_FIRE = CONFIG['economy']['cost_laser_operation']
COST_MISSILE_FIRE = CONFIG['economy']['cost_missile_launch']
REWARD_DRONE_KILL = CONFIG['economy']['reward_kill']

COLOR_BACKGROUND = color.hex(CONFIG['colors_hex']['background'])
COLOR_POLAR_GRID = color.hex(CONFIG['colors_hex']['grid'])
COLOR_HUD_BG = color.hex(CONFIG['colors_hex']['hud_bg'])
COLOR_TEXT = color.hex(CONFIG['colors_hex']['text'])
COLOR_RED_TEAM = color.hex(CONFIG['colors_hex']['red_team'])
COLOR_BLUE_TEAM = color.hex(CONFIG['colors_hex']['blue_team'])
COLOR_LASER = color.hex(CONFIG['colors_hex']['laser'])
COLOR_MISSILE = color.hex(CONFIG['colors_hex']['missile'])
COLOR_EXPLOSION = color.hex(CONFIG['colors_hex']['explosion'])

from panda3d.core import loadPrcFileData
loadPrcFileData('', 'gl-version 2 1')
loadPrcFileData('', 'notify-level-display error')

app = Ursina(title=CONFIG['window']['title'], size=WINDOW_SIZE)
window.color = color.rgb(30, 30, 30)
camera.bg_color = color.rgb(30, 30, 30)
window.borderless = False
window.fullscreen = False

camera.position = (0, 60, -60)
camera.rotation_x = 45
EditorCamera()

DirectionalLight(direction=Vec3(0.5, -1, 0.5), color=color.white)

def create_polar_grid():
    """
    Generates the visual polar grid and crosshairs for the simulation environment.
    """
    def create_circle_line(radius, color):
        segments = 128
        verts = []
        for i in range(segments + 1):
            angle = math.radians((i / segments) * 360)
            verts.append(Vec3(math.cos(angle) * radius, 0, math.sin(angle) * radius))
        
        Entity(model=Mesh(vertices=verts, mode='line', thickness=2), color=color, unlit=True)

    for r in [50, 100, 150, 200, 250]:
        create_circle_line(r, color.white)
    
    line_length = 500
    Entity(model=Mesh(vertices=[Vec3(-line_length,0,0), Vec3(line_length,0,0)], mode='line', thickness=2), color=color.white, unlit=True)
    Entity(model=Mesh(vertices=[Vec3(0,0,-line_length), Vec3(0,0,line_length)], mode='line', thickness=2), color=color.white, unlit=True)
        
create_polar_grid()

central_asset = Entity(model='cube', scale=(2, 4, 2), color=color.cyan, position=(0, 2, 0), unlit=True)
# Text(text="BASE", position=(0, 0.2), origin=(0, 0), scale=2, color=color.cyan, parent=central_asset) # Disabled

game_state = {
    'budget': BUDGET_START,
    'threats_neutralized': 0,
    'game_over': False,
    'total_threat_value_destroyed': 0,
    'total_ammo_cost': 0
}

ghost_lines = Entity(model=Mesh(mode='line', thickness=1), color=color.rgba(255, 255, 255, 30), unlit=True)

# UI Elements are now disabled to prevent potential rendering issues
game_over_text = Text(text="GAME OVER", position=(0, 0), origin=(0, 0), scale=4, color=color.red, enabled=False)
restart_text = Text(text="Press 'R' to Restart", position=(0, -0.1), origin=(0, 0), scale=2, color=color.white, enabled=False)

def update_ui():
    """
    Updates the text elements of the Heads Up Display with current game state values.
    """
    pass # UI updates disabled
    # ui_budget.text = f"BUDGET:   ${int(game_state['budget']):,}"
    # ui_score.text  = f"NEUTRALIZED: {game_state['threats_neutralized']}"
    # if game_state['total_ammo_cost'] > 0:
    #     ratio = game_state['total_threat_value_destroyed'] / game_state['total_ammo_cost']
    #     ui_ratio.text = f"EFFICIENCY: {ratio:.2f}"
    # else:
    #     ui_ratio.text = "EFFICIENCY: 0.00"

def restart_game():
    """
    Resets the game state, clears entities, and restarts the scenario.
    """
    game_state['budget'] = BUDGET_START
    game_state['threats_neutralized'] = 0
    game_state['game_over'] = False
    game_state['total_threat_value_destroyed'] = 0
    game_state['total_ammo_cost'] = 0
    
    for d in drones: destroy(d)
    for p in projectiles: destroy(p)
    drones.clear()
    projectiles.clear()
    
    central_asset.color = color.cyan
    game_over_text.enabled = False
    restart_text.enabled = False
    
    scenario_manager.current_event = 0
    scenario_manager.time_elapsed = 0
    
    update_ui()

def input(key):
    """
    Handles user input events, specifically checking for the restart key.
    """
    if key == 'r' and game_state['game_over']:
        restart_game()

drones = []
defenses = []
projectiles = []

class Drone(Entity):
    """
    Base class for enemy drones, handling movement mechanics, damage, and death.
    """
    def __init__(self, position, target_pos=(0,0,0)):
        super().__init__(
            model='sphere',
            color=color.red,
            position=position,
            scale=1,
            collider='box',
            unlit=True
        )
        self.health = 100
        self.speed = 10
        self.value = 1000
        self.target_position = Vec3(target_pos)
        
        self.physics_mode = 'linear' 
        
        self.jink_range = 200 
        self.evasion_phase = random.uniform(0, 2 * math.pi)
        
        self.start_pos = position
        self.ballistic_t = 0
        
        # self.trail = TrailRenderer(parent=self, size=(1, 1), color=self.color, segments=16) # Disabled for shader compatibility
        
        self.velocity = Vec3(0,0,0)

    def update(self):
        """
        Updates drone position based on its physics mode and checks for base collision.
        """
        if game_state['game_over']: return
        
        dist_to_target = distance(self.position, self.target_position)
        
        if self.physics_mode == 'noe':
            target_h = Vec3(self.target_position.x, 3, self.target_position.z)
            direction = (target_h - self.position).normalized()
            self.position += direction * self.speed * time.dt
            self.y = lerp(self.y, 3, time.dt * 5) 

        elif self.physics_mode == 'ballistic':
            direction = (self.target_position - self.position).normalized()
            self.position += direction * self.speed * time.dt
            self.rotation_x += time.dt * 10
            self.look_at(self.target_position)

        elif self.physics_mode == 'jinking':
            to_target = (self.target_position - self.position).normalized()
            
            if dist_to_target < self.jink_range:
                t = time.time() * 10
                jink = Vec3(math.sin(t + self.evasion_phase), math.cos(t), 0) * 0.5
                move_dir = (to_target + jink).normalized()
            else:
                move_dir = to_target
                
            self.position += move_dir * self.speed * time.dt
            self.look_at(self.target_position)

        else:
            direction = (self.target_position - self.position).normalized()
            self.position += direction * self.speed * time.dt
            self.look_at(self.target_position)

        if dist_to_target < 3:
            print("Base Destroyed!")
            game_state['game_over'] = True
            central_asset.color = color.red

    def take_damage(self, amount):
        """
        Applies damage to the drone and triggers death sequence if health reaches zero.
        """
        self.health -= amount
        self.shake(duration=0.1, magnitude=0.5)
        if self.health <= 0:
            self.die()

    def die(self):
        """
        Handles drone destruction, explosion effects, and score updates.
        """
        if self in drones:
            drones.remove(self)
        
        e = Entity(model='sphere', position=self.position, color=COLOR_EXPLOSION, scale=self.scale)
        e.animate_scale(self.scale*3, duration=0.2, curve=curve.out_expo)
        e.animate_color(color.clear, duration=0.2)
        destroy(e, delay=0.2)
        
        game_state['budget'] += self.value * 0.1 
        game_state['threats_neutralized'] += 1
        game_state['total_threat_value_destroyed'] += self.value
        update_ui()
        
        destroy(self)

class Shahed136(Drone):
    """
    Represents the Shahed-136 drone type with specific stats and behavior.
    """
    def __init__(self, position):
        super().__init__(position)
        self.model = 'diamond'
        self.rotation_x = -90
        self.color = color.white
        self.scale = 0.8
        self.health = 50
        self.speed = 12
        self.value = 20000
        self.physics_mode = 'linear'
        # self.trail.color = color.rgba(255, 255, 255, 100) # Trail disabled
        
        conf = CONFIG['units']['shahed']
        self.health = conf['health']
        self.speed = conf['speed']
        self.value = conf['value']
        self.scale = conf['scale']

class DJIMavic(Drone):
    """
    Represents the DJI Mavic drone type with Nap-of-the-Earth flight behavior.
    """
    def __init__(self, position):
        super().__init__(position)
        self.model = 'cube'
        self.scale = 0.3
        self.color = color.gray
        self.health = 10
        self.speed = 8
        self.value = 1000
        self.physics_mode = 'noe'
        # self.trail.enabled = False # Trail disabled globally
        
        conf = CONFIG['units']['mavic']
        self.health = conf['health']
        self.speed = conf['speed']
        self.value = conf['value']
        self.scale = conf['scale']

class Kalibr(Drone):
    """
    Represents the Kalibr missile with jinking behavior.
    """
    def __init__(self, position):
        super().__init__(position)
        self.model = 'cylinder'
        self.scale = (0.5, 0.5, 2)
        self.color = color.rgb(255, 50, 50)
        self.health = 150
        self.speed = 45
        self.value = 6500000
        self.physics_mode = 'jinking'
        self.jink_range = 250
        # self.trail.color = color.red # Trail disabled
        
        conf = CONFIG['units']['kalibr']
        self.health = conf['health']
        self.speed = conf['speed']
        self.value = conf['value']
        self.scale_z = conf['scale_z']
        self.jink_range = conf['jink_range']

class Kinzhal(Drone):
    """
    Represents the Kinzhal hypersonic missile with ballistic flight behavior.
    """
    def __init__(self, position):
        super().__init__(position)
        self.model = 'diamond'
        self.scale = 1.0
        self.color = color.orange
        self.health = 300
        self.speed = 80
        self.value = 10000000
        self.physics_mode = 'ballistic'
        # self.trail.color = color.orange # Trail disabled

        unit_conf = CONFIG['units']['kinzhal']
        self.health = unit_conf['health']
        self.speed = unit_conf['speed']
        self.value = unit_conf['value']
        self.scale = unit_conf['scale']

class DefenseSystem(Entity):
    """
    Base class for defensive weapon systems.
    """
    def __init__(self, position):
        super().__init__(
            model='cube',
            color=color.yellow,
            position=position,
            scale=(1.5, 1, 1.5),
            collider='box',
            unlit=True
        )
        self.cooldown_timer = 0
        self.cooldown = 1.0
        self.range = 50
        self.cost_per_shot = 0
        self.priority_weight = 1.0
        self.name = "Generic Defense"

    def update(self):
        """
        Updates the cooldown timer for the defense system.
        """
        if self.cooldown_timer > 0:
            self.cooldown_timer -= time.dt

    def can_fire(self):
        """
        Checks if the system is ready to fire based on cooldown and budget.
        """
        return self.cooldown_timer <= 0 and game_state['budget'] >= self.cost_per_shot

    def fire(self, target_drone):
        """
        Executes the firing sequence, deducting cost and setting cooldown.
        """
        game_state['budget'] -= self.cost_per_shot
        game_state['total_ammo_cost'] += self.cost_per_shot
        self.cooldown_timer = self.cooldown
        
        t = Text(text=f"-${self.cost_per_shot:,}", position=(0, 2, 0), origin=(0,0), scale=1.5, color=color.red, parent=self)
        t.animate_position(t.position + (0, 2, 0), duration=1.0)
        t.animate_color(color.clear, duration=1.0)
        destroy(t, delay=1.0)
        
        update_ui()
        self.look_at(target_drone)

class DragonFireLaser(DefenseSystem):
    """
    Laser defense system that requires dwell time to destroy targets.
    """
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
        
        self.turret = Entity(parent=self, model='cube', scale=(0.2, 0.2, 1), position=(0, 0.5, 0), color=color.cyan, unlit=True)
        self.beam = None

    def update(self):
        """
        Updates the laser's dwell timer and visual beam effect.
        """
        super().update()
        
        if self.current_target:
            if not self.current_target.enabled or distance(self.position, self.current_target.position) > self.range:
                self.stop_firing()
                return

            self.turret.look_at(self.current_target)
            self.dwell_timer += time.dt
            
            if not self.beam:
                 self.beam = Entity(parent=self, model='cube', color=color.blue, scale=(0.05, 0.05, 0), position=(0,0.5,0), unlit=True)
            
            dist = distance(self.position, self.current_target.position)
            self.beam.look_at(self.current_target)
            self.beam.scale_z = dist
            self.beam.position = self.turret.position + self.beam.forward * dist / 2
            
            if self.dwell_timer >= self.required_dwell:
                self.current_target.take_damage(1000)
                self.stop_firing()

    def stop_firing(self):
        """
        Resets the laser firing state and removes the beam.
        """
        self.current_target = None
        self.dwell_timer = 0
        if self.beam: destroy(self.beam)
        self.beam = None

    def fire(self, target_drone):
        """
        Initiates the laser tracking on a new target.
        """
        if self.current_target == target_drone: return
        
        game_state['budget'] -= self.cost_per_shot
        game_state['total_ammo_cost'] += self.cost_per_shot
        self.current_target = target_drone
        self.dwell_timer = 0
        update_ui()
        
    def can_fire(self):
        """
        Checks if the laser is free to target a new drone.
        """
        return self.current_target is None

class PatriotBattery(DefenseSystem):
    """
    Long-range missile defense system for high-value targets.
    """
    def __init__(self, position):
        super().__init__(position)
        self.color = color.rgb(100, 100, 50)
        self.cooldown = 2.0
        self.range = 400
        self.cost_per_shot = 5000
        self.name = "Patriot PAC-3"
        
        conf = CONFIG['defenses']['patriot']
        self.range = conf['range']
        self.cooldown = conf['cooldown']
        self.cost_per_shot = conf['cost']
        
        self.turret = Entity(parent=self, model='cube', scale=(0.8, 0.8, 1.2), rotation_x=-45, color=color.rgb(150,150,100), unlit=True)

    def fire(self, target_drone):
        """
        Fires an interceptor missile if the target is a high-value threat.
        """
        if not isinstance(target_drone, (Kalibr, Kinzhal)):
            return 

        super().fire(target_drone)
        
        Projectile(self.position + (0, 2, 0), target_drone, damage=500, speed=60, color=color.blue)

class GepardFlak(DefenseSystem):
    """
    Rapid-fire anti-air gun system for short-range threats.
    """
    def __init__(self, position):
        super().__init__(position)
        self.color = color.gray
        self.color = color.gray
        self.cooldown = 0.08
        self.range = 150
        self.cost_per_shot = 100
        self.name = "Gepard"
        
        conf = CONFIG['defenses']['gepard']
        self.range = conf['range']
        self.cooldown = conf['cooldown']
        self.cost_per_shot = conf['cost']
        
        self.turret = Entity(parent=self, model='cube', scale=(0.4, 0.4, 0.8), color=color.dark_gray, unlit=True)

    def fire(self, target_drone):
        """
        Fires a burst of projectiles at the target, ineffective against hypersonic threats.
        """
        if isinstance(target_drone, Kinzhal):
            return 

        super().fire(target_drone)
        
        for i in range(3):
            Projectile(self.position + (0, 1.5, 0) + Vec3(random.uniform(-0.5,0.5),0,0), target_drone, damage=15, speed=30, color=color.blue)

class Projectile(Entity):
    """
    Represents a projectile or missile fired by a defense system.
    """
    def __init__(self, position, target, damage, speed=20, color=color.yellow):
        super().__init__(
            model='sphere',
            color=color.blue,
            position=position,
            scale=0.2,
            unlit=True
        )
        self.target = target
        self.damage = damage
        self.speed = speed
        # self.trail = TrailRenderer(parent=self, size=(1, 1), color=color, segments=16) # Disabled for shader compatibility

    def update(self):
        """
        Updates projectile position to intercept the target.
        """
        try:
            if not self.target or not self.target.enabled:
                if self in projectiles: projectiles.remove(self)
                destroy(self)
                return
                
            predicted_pos, t = predict_intercept(self.position, self.target.position, self.target.velocity, self.speed)
            target_point = predicted_pos if predicted_pos else self.target.position
            
            direction = (target_point - self.position).normalized()
            self.look_at(target_point)
            self.position += direction * self.speed * time.dt
        except:
            if self in projectiles: projectiles.remove(self)
            destroy(self)
            return
        
        if distance(self.position, self.target.position) < 1:
            self.target.take_damage(self.damage)
            if self in projectiles: projectiles.remove(self)
            destroy(self)

lock_lines = Entity(model=Mesh(mode='line', thickness=2), color=color.rgba(0, 255, 0, 100), unlit=True)

def run_wta_logic():
    """
    Solves the Weapon-Target Assignment problem to optimize defense allocation.
    Uses the Hungarian algorithm to minimize the total cost of assignments.
    """
    available_defenses = [d for d in defenses if d.can_fire()]
    if not available_defenses or not drones:
        return

    n_weapons = len(available_defenses)
    n_targets = len(drones)
    n_targets = len(drones)
    cost_matrix = np.zeros((n_weapons, n_targets))

    doomed_targets = set()
    for p in projectiles:
        if p.target and p.target.enabled:
            doomed_targets.add(p.target)

    for i, weapon in enumerate(available_defenses):
        for j, drone in enumerate(drones):
            if isinstance(weapon, PatriotBattery) and drone in doomed_targets:
                cost_matrix[i, j] = 1e8 
                continue

            dist = distance(weapon.position, drone.position)
            
            if dist > weapon.range:
                cost_matrix[i, j] = 1e9
                continue
            
            if isinstance(weapon, PatriotBattery) and not isinstance(drone, (Kalibr, Kinzhal)):
                cost_matrix[i, j] = 1e9
                continue

            if isinstance(weapon, GepardFlak) and isinstance(drone, Kinzhal):
                 cost_matrix[i, j] = 1e9
                 continue

            if drone.value < weapon.cost_per_shot:
                waste_ratio = weapon.cost_per_shot / max(1, drone.value)
                financial_penalty = waste_ratio * 1000
            else:
                financial_penalty = 0

            norm_dist = dist / weapon.range
            priority_score = (drone.value / 100000)
            
            cost = (norm_dist * 10) - priority_score + financial_penalty
            cost_matrix[i, j] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = []
    lines_verts = []
    
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1e8:
            if not available_defenses[r] or not drones[c]: continue
            if not available_defenses[r].enabled or not drones[c].enabled: continue
            
            assignments.append((available_defenses[r], drones[c]))
            
            try:
                w_pos = available_defenses[r].position + Vec3(0,2,0)
                d_pos = drones[c].position
                lines_verts.append(w_pos)
                lines_verts.append(d_pos)
            except:
                continue
    
    if lines_verts:
        lock_lines.model.vertices = lines_verts
        lock_lines.model.generate()
    else:
        lock_lines.model.vertices = []
        lock_lines.model.generate()
    
    for weapon, target in assignments:
        try:
            if target and target.enabled: 
                 weapon.fire(target)
        except:
            continue

def setup_defenses():
    """
    Initializes and places defense systems at strategic locations.
    """
    defenses.append(DragonFireLaser(position=(5, 0, 0)))
    defenses.append(DragonFireLaser(position=(-5, 0, 0)))
    
    defenses.append(GepardFlak(position=(0, 0, 8)))
    defenses.append(GepardFlak(position=(0, 0, -8)))
    
    defenses.append(PatriotBattery(position=(15, 0, 15)))
    defenses.append(PatriotBattery(position=(-15, 0, -15)))
    
    defenses.append(GepardFlak(position=(8, 0, 0)))
    defenses.append(GepardFlak(position=(-8, 0, 0)))

setup_defenses()

class ScenarioManager(Entity):
    """
    Manages the game scenario, spawning waves of drones according to a timeline.
    """
    def __init__(self):
        super().__init__()
        self.time_elapsed = 0
        self.timeline = CONFIG['scenario_timeline']
        self.current_event = 0
        self.wave_text = Text(text="", position=(0, 0.4), scale=2, color=color.red, origin=(0,0), enabled=False)

    def update(self):
        """
        Checks the timeline and triggers events when their time is reached.
        """
        if game_state['game_over']: return
        
        self.time_elapsed += time.dt
        
        if self.current_event < len(self.timeline):
            event = self.timeline[self.current_event]
            t = event['time']
            if self.time_elapsed >= t:
                self.execute_event(event['type'], event['count'], event['desc'])
                self.current_event += 1

    def execute_event(self, type_code, count, desc):
        """
        Spawns a specific number of drones of a given type with a description.
        """
        print(f"EVENT: {desc}")
        
        self.wave_text.text = desc
        self.wave_text.enabled = True
        self.wave_text.animate_color(color.clear, duration=4.0)
        invoke(lambda: setattr(self.wave_text, 'enabled', False), delay=4.0)
        
        for i in range(count):
            offset = Vec3(random.uniform(-10,10), 0, random.uniform(-10,10))
            
            if type_code == 'SHA':
                 angle = random.uniform(0, 360) 
                 spawn_pos = Vec3(math.cos(angle)*80, 20, math.sin(angle)*80) + offset
                 drones.append(Shahed136(spawn_pos))
                 
            elif type_code == 'MAV':
                 angle = random.uniform(0, 360) 
                 spawn_pos = Vec3(math.cos(angle)*60, 3, math.sin(angle)*60) + offset
                 drones.append(DJIMavic(spawn_pos))

            elif type_code == 'KIN':
                 angle = random.uniform(0, 360) 
                 spawn_pos = Vec3(math.cos(angle)*100, 200, math.sin(angle)*100)
                 drones.append(Kinzhal(spawn_pos))

            elif type_code == 'KAL':
                 angle = random.uniform(0, 360) 
                 spawn_pos = Vec3(math.cos(angle)*120, 50, math.sin(angle)*120)
                 drones.append(Kalibr(spawn_pos))

scenario_manager = ScenarioManager()

def draw_ghost_lines():
    """
    Visualizes projected paths of drones using ghost lines.
    """
    verts = []
    for d in drones:
        if not d.enabled: continue
        start = d.position
        end = d.position + d.velocity * 3.0 
        verts.append(start)
        verts.append(end)
    
    ghost_lines.model.vertices = verts
    ghost_lines.model.generate()

def update():
    """
    Main update loop for global game logic.
    """
    if game_state['game_over']:
        if not game_over_text.enabled:
            game_over_text.enabled = True
            restart_text.enabled = True
        return

    run_wta_logic()
    
    draw_ghost_lines()

camera.orthographic = False 

if __name__ == '__main__':
    app.run()