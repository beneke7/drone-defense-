import pygame
import numpy as np
from scipy.optimize import linear_sum_assignment
import random
import math
import json

with open("reference_legacy/game_config.json", "r") as f:
    CONFIG = json.load(f)

WINDOW_SIZE = (1000, 700)
FPS = 60

# 3D Camera settings (modifiable at runtime)
camera_distance = 200
camera_height = 100
camera_angle = 0  # Rotation angle around the base

BUDGET_START = CONFIG['economy']['start_budget']
COST_LASER_FIRE = CONFIG['economy']['cost_laser_operation']
COST_MISSILE_FIRE = CONFIG['economy']['cost_missile_launch']
REWARD_DRONE_KILL = CONFIG['economy']['reward_kill']

WHITE = (255, 255, 255)
GRAY = (40, 40, 40)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)
BLUE = (100, 150, 255)
CYAN = (0, 255, 255)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)

CENTER_X = WINDOW_SIZE[0] // 2
CENTER_Y = WINDOW_SIZE[1] // 2

game_state = {
    'budget': BUDGET_START,
    'threats_neutralized': 0,
    'game_over': False,
    'total_threat_value_destroyed': 0,
    'total_ammo_cost': 0
}

def world_to_screen_3d(x, y, z):
    """
    Convert 3D world coordinates to 2D screen coordinates using perspective projection.
    Returns (screen_x, screen_y, depth) where depth is used for z-sorting.
    """
    # Calculate camera position based on angle and distance
    cam_x = math.sin(math.radians(camera_angle)) * camera_distance
    cam_y = camera_height
    cam_z = -math.cos(math.radians(camera_angle)) * camera_distance
    
    # Translate to camera space
    rel_x = x - cam_x
    rel_y = y - cam_y
    rel_z = z - cam_z
    
    # Rotate to face center
    angle_rad = math.radians(-camera_angle)
    rotated_x = rel_x * math.cos(angle_rad) - rel_z * math.sin(angle_rad)
    rotated_z = rel_x * math.sin(angle_rad) + rel_z * math.cos(angle_rad)
    
    # Simple perspective projection
    if rotated_z < 1:
        rotated_z = 1
    
    fov = 500
    scale = fov / rotated_z
    screen_x = CENTER_X + int(rotated_x * scale)
    screen_y = CENTER_Y - int(rel_y * scale)
    
    return (screen_x, screen_y, rotated_z)

def predict_intercept(shooter_pos, target_pos, target_vel, projectile_speed):
    """
    Calculates the interception point and time for a projectile to hit a moving target.
    """
    to_target = target_pos - shooter_pos
    target_speed_sq = np.dot(target_vel, target_vel)
    proj_speed_sq = projectile_speed ** 2
    
    a = target_speed_sq - proj_speed_sq
    b = 2 * np.dot(to_target, target_vel)
    c = np.dot(to_target, to_target)
    
    if abs(a) < 1e-6:
        t = -c / (2*b) if b != 0 else -1
    else:
        delta = b**2 - 4*a*c
        if delta < 0: return None, None
        t1 = (-b + math.sqrt(delta)) / (2*a)
        t2 = (-b - math.sqrt(delta)) / (2*a)
        if t1 > 0 and t2 > 0: t = min(t1, t2)
        elif t1 > 0: t = t1
        elif t2 > 0: t = t2
        else: return None, None

    return target_pos + (target_vel * t), t

class Drone:
    """
    Base class for enemy drones.
    """
    def __init__(self, position, target_pos=np.array([0, 0, 0])):
        self.position = np.array(position, dtype=float)
        self.target_position = np.array(target_pos, dtype=float)
        self.velocity = np.array([0, 0, 0], dtype=float)
        self.health = 100
        self.speed = 10
        self.value = 1000
        self.physics_mode = 'linear'
        self.jink_range = 200
        self.evasion_phase = random.uniform(0, 2 * math.pi)
        self.color = RED
        self.shape = 'circle'
        self.size = 5
        self.alive = True
        
    def update(self, dt):
        """
        Updates drone position based on its physics mode.
        """
        if game_state['game_over']: return
        
        dist_to_target = np.linalg.norm(self.target_position - self.position)
        
        if self.physics_mode == 'noe':
            target_h = np.array([self.target_position[0], 3, self.target_position[2]])
            direction = target_h - self.position
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
            self.velocity = direction * self.speed
            self.position += self.velocity * dt
            self.position[1] = 3

        elif self.physics_mode == 'ballistic':
            direction = self.target_position - self.position
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
            self.velocity = direction * self.speed
            self.position += self.velocity * dt

        elif self.physics_mode == 'jinking':
            to_target = self.target_position - self.position
            to_target_norm = np.linalg.norm(to_target)
            if to_target_norm > 0:
                to_target = to_target / to_target_norm
            
            if dist_to_target < self.jink_range:
                import time
                t = time.time() * 10
                jink = np.array([math.sin(t + self.evasion_phase), math.cos(t), 0]) * 0.5
                move_dir = to_target + jink
                move_dir_norm = np.linalg.norm(move_dir)
                if move_dir_norm > 0:
                    move_dir = move_dir / move_dir_norm
            else:
                move_dir = to_target
                
            self.velocity = move_dir * self.speed
            self.position += self.velocity * dt

        else:
            direction = self.target_position - self.position
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
            self.velocity = direction * self.speed
            self.position += self.velocity * dt

        if dist_to_target < 10:
            print("Base Destroyed!")
            game_state['game_over'] = True

    def take_damage(self, amount):
        """
        Applies damage to the drone.
        """
        self.health -= amount
        if self.health <= 0:
            self.die()

    def die(self):
        """
        Handles drone destruction and score updates.
        """
        self.alive = False
        game_state['budget'] += self.value * 0.1
        game_state['threats_neutralized'] += 1
        game_state['total_threat_value_destroyed'] += self.value

    def draw(self, screen):
        """
        Renders the drone in 3D.
        """
        if not self.alive: return
        screen_x, screen_y, depth = world_to_screen_3d(self.position[0], self.position[1], self.position[2])
        
        # Size based on distance (perspective)
        fov = 500
        size = max(2, int(self.size * fov / depth))
        
        if self.shape == 'circle':
            pygame.draw.circle(screen, self.color, (screen_x, screen_y), size)
        elif self.shape == 'square':
            rect = pygame.Rect(screen_x - size, screen_y - size, size * 2, size * 2)
            pygame.draw.rect(screen, self.color, rect)
        elif self.shape == 'triangle':
            points = [
                (screen_x, screen_y - size),
                (screen_x - size, screen_y + size),
                (screen_x + size, screen_y + size)
            ]
            pygame.draw.polygon(screen, self.color, points)
        
        return depth

class DJIMavic(Drone):
    """
    Small reconnaissance drone with NOE flight.
    """
    def __init__(self, position):
        super().__init__(position)
        conf = CONFIG['units']['mavic']
        self.health = conf['health']
        self.speed = conf['speed']
        self.value = conf['value']
        self.physics_mode = 'noe'
        self.shape = 'square'
        self.size = 2
        self.color = RED

class Shahed136(Drone):
    """
    Loitering munition - main aerial threat.
    """
    def __init__(self, position):
        super().__init__(position)
        conf = CONFIG['units']['shahed']
        self.health = conf['health']
        self.speed = conf['speed']
        self.value = conf['value']
        self.physics_mode = 'linear'
        self.shape = 'triangle'
        self.size = 3
        self.color = RED

class Kalibr(Drone):
    """
    Cruise missile with terminal jinking.
    """
    def __init__(self, position):
        super().__init__(position)
        conf = CONFIG['units']['kalibr']
        self.health = conf['health']
        self.speed = conf['speed']
        self.value = conf['value']
        self.physics_mode = 'jinking'
        self.jink_range = conf['jink_range']
        self.shape = 'circle'
        self.size = 4
        self.color = RED

class Kinzhal(Drone):
    """
    Hypersonic ballistic missile.
    """
    def __init__(self, position):
        super().__init__(position)
        conf = CONFIG['units']['kinzhal']
        self.health = conf['health']
        self.speed = conf['speed']
        self.value = conf['value']
        self.physics_mode = 'ballistic'
        self.shape = 'triangle'
        self.size = 6
        self.color = RED

class DefenseSystem:
    """
    Base class for defensive weapons.
    """
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.cooldown_timer = 0
        self.cooldown = 1.0
        self.range = 50
        self.cost_per_shot = 0
        self.name = "Generic Defense"
        self.color = YELLOW
        self.size = 6

    def update(self, dt):
        """
        Updates cooldown timer.
        """
        if self.cooldown_timer > 0:
            self.cooldown_timer -= dt

    def can_fire(self):
        """
        Checks if system can fire.
        """
        return self.cooldown_timer <= 0 and game_state['budget'] >= self.cost_per_shot

    def fire(self, target_drone):
        """
        Deducts cost and sets cooldown.
        """
        game_state['budget'] -= self.cost_per_shot
        game_state['total_ammo_cost'] += self.cost_per_shot
        self.cooldown_timer = self.cooldown

    def draw(self, screen):
        """
        Renders defense system in 3D.
        """
        screen_x, screen_y, depth = world_to_screen_3d(self.position[0], self.position[1], self.position[2])
        fov = 500
        size = max(2, int(self.size * fov / depth * 0.25))  # 25% of original size
        
        # Draw defense as yellow cube
        rect = pygame.Rect(screen_x - size, screen_y - size, size * 2, size * 2)
        pygame.draw.rect(screen, self.color, rect)
        pygame.draw.rect(screen, WHITE, rect, 1)
        
        return depth

class DragonFireLaser(DefenseSystem):
    """
    Laser defense system.
    """
    def __init__(self, position):
        super().__init__(position)
        conf = CONFIG['defenses']['dragonfire']
        self.cooldown = conf['cooldown']
        self.range = conf['range']
        self.cost_per_shot = conf['cost']
        self.name = "DragonFire"
        self.current_target = None
        self.dwell_timer = 0.0
        self.required_dwell = conf['required_dwell_time']
        self.damage = conf['damage']
        self.color = RED

    def update(self, dt):
        """
        Updates laser dwell time.
        """
        super().update(dt)
        
        if self.current_target:
            if not self.current_target.alive or np.linalg.norm(self.position - self.current_target.position) > self.range:
                self.stop_firing()
                return

            self.dwell_timer += dt
            
            if self.dwell_timer >= self.required_dwell:
                self.current_target.take_damage(self.damage)
                self.stop_firing()

    def stop_firing(self):
        """
        Resets laser state.
        """
        self.current_target = None
        self.dwell_timer = 0

    def fire(self, target_drone):
        """
        Starts tracking target.
        """
        if self.current_target == target_drone: return
        
        game_state['budget'] -= self.cost_per_shot
        game_state['total_ammo_cost'] += self.cost_per_shot
        self.current_target = target_drone
        self.dwell_timer = 0

    def can_fire(self):
        """
        Laser can only track one target at a time.
        """
        return self.current_target is None

    def draw(self, screen):
        """
        Renders laser with beam in 3D.
        """
        depth = super().draw(screen)
        
        if self.current_target and self.current_target.alive:
            pos1 = world_to_screen_3d(self.position[0], self.position[1], self.position[2])
            pos2 = world_to_screen_3d(self.current_target.position[0], self.current_target.position[1], self.current_target.position[2])
            pygame.draw.line(screen, BLUE, (pos1[0], pos1[1]), (pos2[0], pos2[1]), 3)
        
        return depth

class PatriotBattery(DefenseSystem):
    """
    Long-range anti-missile system.
    """
    def __init__(self, position):
        super().__init__(position)
        conf = CONFIG['defenses']['patriot']
        self.range = conf['range']
        self.cooldown = conf['cooldown']
        self.cost_per_shot = conf['cost']
        self.missile_speed = conf['missile_speed']
        self.damage = conf['missile_damage']
        self.name = "Patriot PAC-3"

    def fire(self, target_drone):
        """
        Fires interceptor missile.
        """
        if not isinstance(target_drone, (Kalibr, Kinzhal)):
            return

        super().fire(target_drone)
        return Projectile(self.position.copy(), target_drone, damage=self.damage, speed=self.missile_speed, size_multiplier=5.0, guided=True)

class GepardFlak(DefenseSystem):
    """
    Rapid-fire anti-air gun.
    """
    def __init__(self, position):
        super().__init__(position)
        conf = CONFIG['defenses']['gepard']
        self.range = conf['range']
        self.cooldown = conf['cooldown']
        self.cost_per_shot = conf['cost']
        self.damage = conf['damage']
        self.max_spread = conf.get('max_spread', 0.1)
        self.name = "Gepard"

    def fire(self, target_drone):
        """
        Fires burst of projectiles.
        """
        if isinstance(target_drone, Kinzhal):
            return []

        super().fire(target_drone)
        
        projectiles = []
        burst_count = CONFIG['defenses']['gepard']['burst_count']
        
        # Calculate base error for this burst (wind, vibration, etc)
        # But user wants "incremental spread", so maybe spread increases with i?
        
        for i in range(burst_count):
            # Incremental spread: increases with each shot in the burst
            spread_factor = (i + 1) / burst_count * self.max_spread
            
            # Random vector in unit sphere
            u = np.random.randn(3)
            u /= np.linalg.norm(u)
            spread_error = u * spread_factor
            
            # Fire from same position, but with velocity error
            projectiles.append(Projectile(self.position.copy(), target_drone, damage=self.damage, speed=30, spread_error=spread_error, guided=False))
        return projectiles

class Projectile:
    """
    Missile or bullet projectile.
    """
    GRAVITY = 0.0

    def __init__(self, position, target, damage, speed=20, size_multiplier=1.0, spread_error=None, guided=True):
        self.position = np.array(position, dtype=float)
        self.target = target
        self.damage = damage
        self.max_speed = speed        # Target max speed
        self.guided = guided
        self.size_multiplier = size_multiplier
        self.alive = True
        self.color = BLUE
        self.lifetime = 5.0
        
        if self.guided:
            self.speed = 10.0          # Start slow for missiles
            self.acceleration = 120.0  # Accelerate towards max_speed
        else:
            self.speed = speed         # Start fast for bullets
            self.acceleration = 0.0

        # Calculate initial velocity vector
        if target:
            # Use max_speed for intercept prediction if guided
            pred_speed = self.max_speed if self.guided else self.speed
            predicted_pos, t_intercept = predict_intercept(self.position, target.position, target.velocity, pred_speed)
            
            target_point = predicted_pos if predicted_pos is not None else target.position
            
            direction = target_point - self.position
            norm = np.linalg.norm(direction)
            if norm > 0:
                self.velocity_dir = direction / norm
            else:
                self.velocity_dir = np.array([0, 0, 1], dtype=float)
                
            # Apply spread to the initial direction vector
            if spread_error is not None:
                self.velocity_dir = self.velocity_dir + spread_error
                # Renormalize
                norm = np.linalg.norm(self.velocity_dir)
                if norm > 0:
                    self.velocity_dir = self.velocity_dir / norm
        else:
             self.velocity_dir = np.array([0, 0, 1], dtype=float)

    def update(self, dt):
        """
        Updates projectile.
        """
        # Accelerate guided missiles
        if self.guided and self.speed < self.max_speed:
            self.speed += self.acceleration * dt
            if self.speed > self.max_speed:
                self.speed = self.max_speed

        if self.target and self.target.alive:
            dist = np.linalg.norm(self.target.position - self.position)
            
            if self.guided:
                # Guided behavior (Patriot) - constantly adjust course
                predicted_pos, _ = predict_intercept(self.position, self.target.position, self.target.velocity, self.max_speed)
                target_point = predicted_pos if predicted_pos is not None else self.target.position
                direction = target_point - self.position
                norm = np.linalg.norm(direction)
                if norm > 0:
                    self.velocity_dir = direction / norm
            
            # Hit check (using 3D distance)
            if dist < 10:
                self.target.take_damage(self.damage)
                self.alive = False
                return
        else:
            self.lifetime -= dt
            if self.lifetime <= 0:
                self.alive = False
        
        # Move along velocity vector
        self.position += self.velocity_dir * self.speed * dt
        
        if np.linalg.norm(self.position) > 2000:
            self.alive = False

    def draw(self, screen):
        """
        Renders projectile in 3D.
        """
        if not self.alive: return
        screen_x, screen_y, depth = world_to_screen_3d(self.position[0], self.position[1], self.position[2])
        fov = 500
        size = max(1, int(3 * fov / depth * 0.2 * self.size_multiplier))
        pygame.draw.circle(screen, self.color, (screen_x, screen_y), size)
        return depth

class ScenarioManager:
    """
    Manages timeline and spawning of threats.
    """
    def __init__(self):
        self.time_elapsed = 0
        self.timeline = CONFIG['scenario_timeline']
        self.current_event = 0

    def update(self, dt, drones):
        """
        Checks timeline and spawns drones.
        """
        if game_state['game_over']: return
        
        self.time_elapsed += dt
        
        if self.current_event < len(self.timeline):
            event = self.timeline[self.current_event]
            if self.time_elapsed >= event['time']:
                print(f"EVENT: {event['desc']}")
                self.execute_event(event['type'], event['count'], drones)
                self.current_event += 1

    def execute_event(self, type_code, count, drones):
        """
        Spawns drones based on event type.
        """
        for i in range(count):
            angle = random.uniform(0, 2 * math.pi)
            offset = [random.uniform(-10, 10), 0, random.uniform(-10, 10)]
            
            if type_code == 'SHA':
                dist = CONFIG['units']['shahed']['spawn_distance']
                spawn_pos = [math.cos(angle)*dist + offset[0], 20, math.sin(angle)*dist + offset[2]]
                drones.append(Shahed136(spawn_pos))
            elif type_code == 'MAV':
                dist = CONFIG['units']['mavic']['spawn_distance']
                spawn_pos = [math.cos(angle)*dist + offset[0], 3, math.sin(angle)*dist + offset[2]]
                drones.append(DJIMavic(spawn_pos))
            elif type_code == 'KIN':
                dist = CONFIG['units']['kinzhal']['spawn_distance']
                height = CONFIG['units']['kinzhal']['spawn_height']
                spawn_pos = [math.cos(angle)*dist, height, math.sin(angle)*dist]
                drones.append(Kinzhal(spawn_pos))
            elif type_code == 'KAL':
                dist = CONFIG['units']['kalibr']['spawn_distance']
                spawn_pos = [math.cos(angle)*dist + offset[0], 50, math.sin(angle)*dist + offset[2]]
                drones.append(Kalibr(spawn_pos))

def run_wta_logic(defenses, drones, projectiles):
    """
    Solves WTA problem using Hungarian algorithm.
    """
    available_defenses = [d for d in defenses if d.can_fire()]
    active_drones = [d for d in drones if d.alive]
    
    if not available_defenses or not active_drones:
        return []

    n_weapons = len(available_defenses)
    n_targets = len(active_drones)
    cost_matrix = np.zeros((n_weapons, n_targets))

    doomed_targets = set()
    for p in projectiles:
        if p.target and p.target.alive:
            doomed_targets.add(p.target)

    for i, weapon in enumerate(available_defenses):
        for j, drone in enumerate(active_drones):
            if isinstance(weapon, PatriotBattery) and drone in doomed_targets:
                cost_matrix[i, j] = 1e8
                continue

            dist = np.linalg.norm(weapon.position - drone.position)
            
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
    
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1e8:
            assignments.append((available_defenses[r], active_drones[c]))
    
    return assignments

def draw_grid_3d(screen):
    """
    Draws a 3D grid on the ground plane.
    """
    # Ground grid
    for r in [50, 100, 150, 200, 250]:
        points = []
        segments = 64
        for i in range(segments + 1):
            angle = (i / segments) * 2 * math.pi
            x = math.cos(angle) * r
            z = math.sin(angle) * r
            screen_pos = world_to_screen_3d(x, 0, z)
            points.append((screen_pos[0], screen_pos[1]))
        
        if len(points) > 1:
            pygame.draw.lines(screen, WHITE, False, points, 1)
    
    # Cross lines
    line_len = 300
    p1 = world_to_screen_3d(-line_len, 0, 0)
    p2 = world_to_screen_3d(line_len, 0, 0)
    pygame.draw.line(screen, WHITE, (p1[0], p1[1]), (p2[0], p2[1]), 1)
    
    p1 = world_to_screen_3d(0, 0, -line_len)
    p2 = world_to_screen_3d(0, 0, line_len)
    pygame.draw.line(screen, WHITE, (p1[0], p1[1]), (p2[0], p2[1]), 1)

def main():
    """
    Main game loop with 3D rendering.
    """
    global camera_angle, camera_distance, camera_height
    
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Counter-UAS Defense Simulation (3D)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    
    drones = []
    defenses = []
    projectiles = []
    
    defenses.append(DragonFireLaser([15, 0, 0]))
    defenses.append(DragonFireLaser([-15, 0, 0]))
    defenses.append(GepardFlak([0, 0, 25]))
    defenses.append(GepardFlak([0, 0, -25]))
    defenses.append(PatriotBattery([45, 0, 45]))
    defenses.append(PatriotBattery([-45, 0, -45]))
    defenses.append(GepardFlak([25, 0, 0]))
    defenses.append(GepardFlak([-25, 0, 0]))
    
    scenario_manager = ScenarioManager()
    
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game_state['game_over']:
                    game_state['budget'] = BUDGET_START
                    game_state['threats_neutralized'] = 0
                    game_state['game_over'] = False
                    game_state['total_threat_value_destroyed'] = 0
                    game_state['total_ammo_cost'] = 0
                    drones.clear()
                    projectiles.clear()
                    scenario_manager.current_event = 0
                    scenario_manager.time_elapsed = 0
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom with mouse wheel
                camera_distance = max(50, min(500, camera_distance - event.y * 20))
        
        # Camera controls
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            camera_angle -= 60 * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            camera_angle += 60 * dt
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            camera_height = min(250, camera_height + 50 * dt)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            camera_height = max(30, camera_height - 50 * dt)
        
        if not game_state['game_over']:
            scenario_manager.update(dt, drones)
            
            for drone in drones:
                drone.update(dt)
            drones = [d for d in drones if d.alive]
            
            for defense in defenses:
                defense.update(dt)
            
            for proj in projectiles:
                proj.update(dt)
            projectiles = [p for p in projectiles if p.alive]
            
            assignments = run_wta_logic(defenses, drones, projectiles)
            for weapon, target in assignments:
                result = weapon.fire(target)
                if result:
                    if isinstance(result, list):
                        projectiles.extend(result)
                    else:
                        projectiles.append(result)
        
        screen.fill(GRAY)
        
        # Draw 3D grid
        draw_grid_3d(screen)
        
        # Draw base
        base_pos = world_to_screen_3d(0, 0, 0)
        pygame.draw.rect(screen, CYAN, (base_pos[0] - 8, base_pos[1] - 8, 16, 16))
        
        # Collect all objects with depth for sorting
        render_queue = []
        
        for defense in defenses:
            depth = defense.draw(screen)
        
        for proj in projectiles:
            depth = proj.draw(screen)
        
        for drone in drones:
            depth = drone.draw(screen)
        
        # HUD
        budget_text = font.render(f"BUDGET: ${int(game_state['budget']):,}", True, WHITE)
        kills_text = font.render(f"KILLS: {game_state['threats_neutralized']}", True, WHITE)
        
        if game_state['total_ammo_cost'] > 0:
            ratio = game_state['total_threat_value_destroyed'] / game_state['total_ammo_cost']
            eff_text = font.render(f"EFFICIENCY: {ratio:.2f}", True, YELLOW)
        else:
            eff_text = font.render("EFFICIENCY: 0.00", True, YELLOW)
        
        screen.blit(budget_text, (10, 10))
        screen.blit(kills_text, (10, 40))
        screen.blit(eff_text, (10, 70))
        
        if game_state['game_over']:
            game_over_text = font.render("GAME OVER - Press R to Restart", True, RED)
            text_rect = game_over_text.get_rect(center=(WINDOW_SIZE[0]//2, WINDOW_SIZE[1]//2))
            screen.blit(game_over_text, text_rect)
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == '__main__':
    main()
