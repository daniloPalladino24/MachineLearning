import pygame
import numpy as np

# Constants for the game
CELL_SIZE = 50  # Reduced cell size

def init_graphics(grid_size):
    pygame.init()
    window_size = grid_size * CELL_SIZE
    window = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Cat Mouse Cheese Game")
    clock = pygame.time.Clock()
    return window, clock

def load_images():
    mouse_img = pygame.image.load("img/mouse.png")
    cat_img = pygame.image.load("img/cat.png")
    cheese_img = pygame.image.load("img/cheese.png")
    return mouse_img, cat_img, cheese_img

def draw_grid(window, grid_size):
    window_size = grid_size * CELL_SIZE
    for x in range(0, window_size, CELL_SIZE):
        for y in range(0, window_size, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(window, (200, 200, 200), rect, 1)

def draw_walls(window, walls, grid_size):
    for (i, j), is_wall in walls.items():
        if is_wall:
            x, y = j * CELL_SIZE, i * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(window, (0, 0, 0), rect)

def render_entities(window, mouse_pos, cat_pos, cheese_pos, walls, grid_size, mouse_img, cat_img, cheese_img):
    window.fill((255, 255, 255))  # Clear the screen
    draw_grid(window, grid_size)
    draw_walls(window, walls, grid_size)

    # Draw mouse
    mouse_rect = pygame.Rect(mouse_pos[1] * CELL_SIZE, mouse_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    window.blit(pygame.transform.scale(mouse_img, (CELL_SIZE, CELL_SIZE)), mouse_rect)

    # Draw cat
    cat_rect = pygame.Rect(cat_pos[1] * CELL_SIZE, cat_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    window.blit(pygame.transform.scale(cat_img, (CELL_SIZE, CELL_SIZE)), cat_rect)

    # Draw cheese
    cheese_rect = pygame.Rect(cheese_pos[1] * CELL_SIZE, cheese_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    window.blit(pygame.transform.scale(cheese_img, (CELL_SIZE, CELL_SIZE)), cheese_rect)

    pygame.display.flip()  # Update the display

def quit_graphics():
    pygame.quit()