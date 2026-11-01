"""Hill-climb rover: drive a two-wheeled car over rolling hills.

Physics: Pymunk (2D rigid bodies, joints, and motors from the Chipmunk2D engine).
Drawing: pygame. No GPU needed.

Run:  python demo_rover.py
Headless check:  python demo_rover.py --selftest

Keys: Right / D = gas, Left / A = brake and reverse, R = reset, Esc = quit.
"""

import math
import os
import sys

import pygame
import pymunk

WIDTH, HEIGHT = 960, 540
FPS = 60
SUBSTEPS = 4
GRAVITY = (0.0, -900.0)

WHEEL_RADIUS = 22
WHEEL_RATE = 18.0  # rad/s at full throttle
MOTOR_FORCE = 30_000.0
CHASSIS_SIZE = (110, 24)

SEGMENT = 20
TERRAIN_LENGTH = 40_000
FLAT_START = 400


def terrain_height(x: float) -> float:
    """Flat start, then hills that get steeper the further you go."""
    if x < FLAT_START:
        return 80.0
    k = min(1.0, (x - FLAT_START) / 8000.0)
    return 80.0 + k * (45 * math.sin(x / 140.0) + 25 * math.sin(x / 53.0)) + 0.01 * (x - FLAT_START)


def build_world() -> tuple[pymunk.Space, pymunk.Body, list[pymunk.Body], list[pymunk.SimpleMotor]]:
    space = pymunk.Space()
    space.gravity = GRAVITY

    ground = space.static_body
    for x in range(-200, TERRAIN_LENGTH, SEGMENT):
        seg = pymunk.Segment(ground, (x, terrain_height(x)), (x + SEGMENT, terrain_height(x + SEGMENT)), 3)
        seg.friction = 1.5
        space.add(seg)

    start_x = 120
    start_y = terrain_height(start_x) + WHEEL_RADIUS + 30
    chassis = pymunk.Body(mass=6, moment=pymunk.moment_for_box(6, CHASSIS_SIZE))
    chassis.position = (start_x, start_y)
    box = pymunk.Poly.create_box(chassis, CHASSIS_SIZE)
    box.friction = 0.5
    cab = pymunk.Poly(chassis, [(-20, 12), (20, 12), (10, 34), (-12, 34)])
    cab.friction = 0.5
    space.add(chassis, box, cab)

    # The chassis and wheels share a group so they never collide with each other.
    shape_filter = pymunk.ShapeFilter(group=1)
    box.filter = cab.filter = shape_filter

    wheels, motors = [], []
    for offset in (-38, 38):
        wheel = pymunk.Body(mass=1.5, moment=pymunk.moment_for_circle(1.5, 0, WHEEL_RADIUS))
        wheel.position = (start_x + offset, start_y - 8)
        circle = pymunk.Circle(wheel, WHEEL_RADIUS)
        circle.friction = 1.8
        circle.filter = shape_filter
        # A damped spring on a groove joint gives the wheel some suspension travel.
        groove = pymunk.GrooveJoint(chassis, wheel, (offset, -4), (offset, -22), (0, 0))
        spring = pymunk.DampedSpring(chassis, wheel, (offset, -4), (0, 0), 18, 9000, 450)
        motor = pymunk.SimpleMotor(chassis, wheel, 0.0)
        motor.max_force = MOTOR_FORCE
        space.add(wheel, circle, groove, spring, motor)
        wheels.append(wheel)
        motors.append(motor)
    return space, chassis, wheels, motors


def drive(motors: list[pymunk.SimpleMotor], throttle: float) -> None:
    """throttle: +1 full forward, -1 full reverse, 0 coast."""
    for motor in motors:
        motor.rate = throttle * WHEEL_RATE
        motor.max_force = MOTOR_FORCE if throttle else 0.0


def step(space: pymunk.Space, dt: float = 1.0 / FPS) -> None:
    for _ in range(SUBSTEPS):
        space.step(dt / SUBSTEPS)


def flipped(chassis: pymunk.Body) -> bool:
    return math.cos(chassis.angle) < 0.0


def selftest() -> None:
    """Hold the gas with no window and report how far the rover gets in 20 seconds."""
    space, chassis, _, motors = build_world()
    drive(motors, 1.0)
    for second in range(1, 21):
        for _ in range(FPS):
            step(space)
        if second % 5 == 0:
            print(f"{second:2d}s  x={chassis.position.x:7.0f}  y={chassis.position.y:5.0f}  flipped={flipped(chassis)}")


def to_screen(point, camera_x: float) -> tuple[int, int]:
    return int(point[0] - camera_x + WIDTH * 0.35), int(HEIGHT - 60 - point[1] + 80)


def draw(screen, space, chassis, wheels, camera_x: float, font, best: float) -> None:
    screen.fill((135, 190, 235))
    left = int(camera_x - WIDTH * 0.35)
    xs = range(left - left % SEGMENT, left + WIDTH + SEGMENT, SEGMENT)
    ground = [to_screen((x, terrain_height(x)), camera_x) for x in xs]
    pygame.draw.polygon(screen, (84, 150, 70), ground + [(ground[-1][0], HEIGHT), (ground[0][0], HEIGHT)])
    pygame.draw.lines(screen, (50, 105, 45), False, ground, 4)

    for shape in chassis.shapes:
        pts = [to_screen(chassis.local_to_world(v), camera_x) for v in shape.get_vertices()]
        pygame.draw.polygon(screen, (205, 60, 50), pts)
        pygame.draw.polygon(screen, (90, 20, 20), pts, 2)
    for wheel in wheels:
        centre = to_screen(wheel.position, camera_x)
        pygame.draw.circle(screen, (35, 35, 35), centre, WHEEL_RADIUS)
        spoke = (centre[0] + int(WHEEL_RADIUS * math.cos(-wheel.angle)), centre[1] + int(WHEEL_RADIUS * math.sin(-wheel.angle)) * -1)
        pygame.draw.line(screen, (200, 200, 200), centre, spoke, 3)

    text = f"distance {max(0, chassis.position.x - 120) / 100:6.1f} m    best {best:6.1f} m    R = reset"
    screen.blit(font.render(text, True, (20, 20, 20)), (14, 12))


def play() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Hill-climb rover (Pymunk)")
    font = pygame.font.SysFont("consolas", 22)
    clock = pygame.time.Clock()

    space, chassis, wheels, motors = build_world()
    best = 0.0
    camera_x = chassis.position.x
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                space, chassis, wheels, motors = build_world()
                camera_x = chassis.position.x

        keys = pygame.key.get_pressed()
        throttle = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
        drive(motors, float(throttle))
        step(space)

        best = max(best, (chassis.position.x - 120) / 100)
        camera_x += (chassis.position.x - camera_x) * 0.12
        draw(screen, space, chassis, wheels, camera_x, font, best)
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        selftest()
    else:
        play()
