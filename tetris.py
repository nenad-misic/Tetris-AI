import pygame
import random
import copy
import time
import math
import numpy

score = 0
R = 0

weights = [-1, 1, -1, -1, -1]
alpha = 0.01
gamma = 0.9
epsilon = 0.5

pygame.font.init()
s_width = 750
s_height = 650
play_width = 300
play_height = 600
block_size = 30
top_left_x = (s_width - play_width) // 2
top_left_y = (s_height - play_height) / 2

S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

shapes = [S, Z, I, O, J, L, T]
shapestr = ["S", "Z", "I", "O", "J", "L", "T"]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
shape_start = {
    "S": [[1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8]],
    "Z": [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
    "I": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8]],
    "O": [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
    "J": [[1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
    "L": [[1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
    "T": [[1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
}
next_n_shapes = []


class Node(object):
    def __init__(self, grid, x, rotation):
        self.grid = grid
        self.heuristic = heuristic(grid)
        self.params = get_params(grid)
        self.children = []
        self.parent = None
        self.x = x
        self.rotation = rotation
        self.lost = False

    def addChild(self, child):
        self.children.append(child)
        child.parent = self

class Piece(object):
    rows = 20  # y
    columns = 10  # x

    def __init__(self, column, row, shape):
        self.x = column
        self.y = row
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0  # number from 0-3

def weightsAtIteration(iteration):
    f = open('weights.txt')
    savedWeights = []
    for l in f:
        lsp = l.split(',')
        savedWeights.append([])
        for i in range(len(lsp)):
            savedWeights[len(savedWeights)-1].append(float(lsp[i]))
    return savedWeights[iteration]

    
def initializePiecesFromFile():
    global next_n_shapes
    f = open('testcase.txt')
    for l in f:
        next_n_shapes = list(map(lambda x: int(x), [char for char in l]))


def initializePieces(numOfInitializedPieces):
    global next_n_shapes,shapes
    next_n_shapes = list(map(lambda x: x%(len(shapes)), numpy.random.permutation(numOfInitializedPieces).tolist()))



def create_grid(locked_positions={}):
    grid = [[(0, 0, 0) for x in range(10)] for x in range(20)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_positions:
                c = locked_positions[(j, i)]
                grid[i][j] = c
    return grid


def convert_shape_format(shape):
    positions = []
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def valid_space(shape, grid):
    accepted_positions = [[(j, i) for j in range(10) if grid[i][j] == (0, 0, 0)] for i in range(20)]
    accepted_positions = [j for sub in accepted_positions for j in sub]
    formatted = convert_shape_format(shape)

    for pos in formatted:
        if pos not in accepted_positions:
            if pos[1] > -1:
                return False

    return True


def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False


def get_shape():
    global shapes, shape_colors, next_n_shapes
    if len(next_n_shapes) == 0:
        next_n_shapes = numpy.random.permutation(len(shapes)).tolist()
    i = next_n_shapes.pop(0)
    return (i, Piece(5, 0, shapes[i]))


def draw_grid(surface, row, col):
    sx = top_left_x
    sy = top_left_y
    for i in range(row):
        pygame.draw.line(surface, (128, 128, 128), (sx, sy + i * 30),
                         (sx + play_width, sy + i * 30))
        for j in range(col):
            pygame.draw.line(surface, (128, 128, 128), (sx + j * 30, sy),
                             (sx + j * 30, sy + play_height))


def clear_rows(grid, locked):
    inc = 0
    for i in range(len(grid) - 1, -1, -1):
        row = grid[i]
        if (0, 0, 0) not in row:
            inc += 1
            ind = i
            for j in range(len(row)):
                try:
                    del locked[(j, i)]
                except:
                    continue
    if inc > 0:
        for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                newKey = (x, y + inc)
                locked[newKey] = locked.pop(key)
    
    global score
    global R
    if inc == 1:
        score += 40
    elif inc == 2:
        score += 100
    elif inc == 3:
        score += 300
    elif inc >= 4:
        score += 1200
    

def draw_next_shape(shape, surface):
    global score
    global R
    font = pygame.font.SysFont('consolas', 20)
    font2 = pygame.font.SysFont('consolas', 12)
    label = font.render('Next Shape', 1, (255, 255, 255))
    label_score = font.render('Score: ' + str(score), 1, (255, 255, 255))
    label_r = font.render('Reward: ' + str(R), 1, (255, 255, 255))
    label_weights = font.render('Weights:', 1, (255, 255, 255))
    label_weights0 = font2.render('  Aggregate height: ' + str("{:+.4f}".format(weights[0])), 1, (255, 255, 255))
    label_weights1 = font2.render('  Complete lines:   ' + str("{:+.4f}".format(weights[1])), 1, (255, 255, 255))
    label_weights2 = font2.render('  Number of holes:  ' + str("{:+.4f}".format(weights[2])), 1, (255, 255, 255))
    label_weights3 = font2.render('  Bumpiness:        ' + str("{:+.4f}".format(weights[3])), 1, (255, 255, 255))
    label_weights4 = font2.render('  Max height:       ' + str("{:+.4f}".format(weights[4])), 1, (255, 255, 255))

    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height / 2 - 100
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, shape.color, (sx + j * 30, sy + i * 30, 30, 30), 0)

    surface.blit(label, (sx - 30, sy - 30))
    surface.blit(label_score, (sx - 30, sy - 120))
    surface.blit(label_r, (sx - 30, sy - 150))
    surface.blit(label_weights, (sx - 30, sy + 150))
    surface.blit(label_weights0, (sx - 30, sy + 180))
    surface.blit(label_weights1, (sx - 30, sy + 210))
    surface.blit(label_weights2, (sx - 30, sy + 240))
    surface.blit(label_weights3, (sx - 30, sy + 270))
    surface.blit(label_weights4, (sx - 30, sy + 300))

def draw_window(surface):
    surface.fill((0, 0, 0))

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j], (top_left_x + j * 30, top_left_y + i * 30, 30, 30), 0)

    draw_grid(surface, 20, 10)
    pygame.draw.rect(surface, (255, 0, 0), (top_left_x, top_left_y, play_width, play_height), 5)

def qlearning(grid, locked_positions, current_piece, current_piece_index, next_piece, next_piece_index, searchFunction):
    global epsilon
    x, rot = findMoveAndUpdateWeights(grid, locked_positions, current_piece, current_piece_index, next_piece, next_piece_index, searchFunction)
    if epsilon > 0.001:
        epsilon = epsilon * 0.99
    else:
        epsilon = 0
    return x, rot

def findMoveAndUpdateWeights(grid, locked_positions, current_piece, current_piece_index, next_piece, next_piece_index, searchFunction):
    global epsilon
    global score
    global weights
    global alpha
    global gamma
    global R
    if epsilon > random.random():
        rot = random.randint(0, len(shape_start[shapestr[current_piece_index]]) - 1)
        x = random.randint(0, len(shape_start[shapestr[current_piece_index]][rot]) - 1)
        x = shape_start[shapestr[current_piece_index]][rot][x]
        final_move_cpy = False
        current_piece_cpy = Piece(x, current_piece.y, current_piece.shape)
        current_piece_cpy.rotation = rot
        while not final_move_cpy:
            current_piece_cpy.y += 1
            if not (valid_space(current_piece_cpy, grid)) and current_piece_cpy.y > 0:
                current_piece_cpy.y -= 1
                final_move_cpy = True

            shape_pos_cpy = convert_shape_format(current_piece_cpy)

        locked_positions_cpy = copy.deepcopy(locked_positions)
        for pos in shape_pos_cpy:
            p = (pos[0], pos[1])
            locked_positions_cpy[p] = current_piece_cpy.color

        grid2 = create_grid(locked_positions_cpy)
        child = Node(grid2, current_piece_cpy.x, current_piece_cpy.rotation)
        params_new = child.params
        Q_new = child.heuristic
    else:
        x, rot, Q_new, params_new = searchFunction(grid, locked_positions, current_piece, current_piece_index, next_piece, next_piece_index)

    Q_old = heuristic(grid)
    params_old = get_params(grid)
    R = 5 * params_new[1]**2 - (params_new[0] - params_old[0])

    for i in range(0, len(weights)):
        weights[i] = weights[i] + alpha * weights[i] * (R - params_old[i] + gamma * params_new[i])

    # normalizacija tezina
    regularization_term = abs(sum(weights))
    for i in range(0, len(weights)):
        weights[i] = 100* weights[i] / regularization_term
        weights[i] = math.floor(1e4 * weights[i]) / 1e4

    return x, rot
    



def searchTreeForBestMove(grid, locked_positions, current_piece, current_piece_index, next_piece, next_piece_index, parentNode = None):
    rotacija = 0
    if parentNode is None:
        root = Node(grid, current_piece.x, current_piece.rotation)
    else:
        root = parentNode
    for lista in shape_start[shapestr[current_piece_index]]:
        for pozicija in range(len(lista)):

            final_move_cpy = False
            current_piece_cpy = Piece(lista[pozicija], current_piece.y, current_piece.shape)
            current_piece_cpy.rotation = rotacija
            while not final_move_cpy:
                current_piece_cpy.y += 1
                if not (valid_space(current_piece_cpy, grid)) and current_piece_cpy.y > 0:
                    current_piece_cpy.y -= 1
                    final_move_cpy = True

                shape_pos_cpy = convert_shape_format(current_piece_cpy)

            locked_positions_cpy = copy.deepcopy(locked_positions)
            for pos in shape_pos_cpy:
                p = (pos[0], pos[1])
                locked_positions_cpy[p] = current_piece_cpy.color

            grid2 = create_grid(locked_positions_cpy)
            child = Node(grid2, current_piece_cpy.x, current_piece_cpy.rotation)
            
            root.addChild(child)
            if parentNode is None:
                searchTreeForBestMove(grid2, locked_positions_cpy, next_piece, next_piece_index, None, 0, child)

        rotacija += 1

    if parentNode is None:
        maksHeuristic = -99999999999
        lost = False
        corresponding_params = (0,0,0,0,0)
        best = None
        for child1 in root.children:
            for child2 in child1.children:
                if child2.heuristic > maksHeuristic:
                    maksHeuristic = child2.heuristic
                    corresponding_params = child2.params
                    best = child1
        
        

        return best.x, best.rotation, maksHeuristic, corresponding_params

def searchTreeForBestMove_dumb(grid, locked_positions, current_piece, current_piece_index, next_piece, next_piece_index, parentNode = None):
    rotacija = 0
    if parentNode is None:
        root = Node(grid, current_piece.x, current_piece.rotation)
    else:
        root = parentNode
    for lista in shape_start[shapestr[current_piece_index]]:
        for pozicija in range(len(lista)):

            final_move_cpy = False
            current_piece_cpy = Piece(lista[pozicija], current_piece.y, current_piece.shape)
            current_piece_cpy.rotation = rotacija
            while not final_move_cpy:
                current_piece_cpy.y += 1
                if not (valid_space(current_piece_cpy, grid)) and current_piece_cpy.y > 0:
                    current_piece_cpy.y -= 1
                    final_move_cpy = True

                shape_pos_cpy = convert_shape_format(current_piece_cpy)

            locked_positions_cpy = copy.deepcopy(locked_positions)
            for pos in shape_pos_cpy:
                p = (pos[0], pos[1])
                locked_positions_cpy[p] = current_piece_cpy.color

            grid2 = create_grid(locked_positions_cpy)
            child = Node(grid2, current_piece_cpy.x, current_piece_cpy.rotation)
            root.addChild(child)
        rotacija += 1

    if parentNode is None:
        maksHeuristic = -99999999999
        lost = False
        corresponding_params = (0,0,0,0, 0)
        best = None
        for child1 in root.children:
            if child1.heuristic > maksHeuristic:
                maksHeuristic = child1.heuristic
                corresponding_params = child1.params
                best = child1
        
        

        return best.x, best.rotation, maksHeuristic, corresponding_params

def heuristic(current_grid):
    global weights
    return weights[0]*aggregate_height(current_grid) + weights[1]*complete_lines(current_grid) + weights[2]*holes(current_grid) + weights[3]*bumpiness(current_grid) + weights[4]*max_height(current_grid)

def get_params(current_grid):
    return aggregate_height(current_grid), complete_lines(current_grid), holes(current_grid), bumpiness(current_grid), max_height(current_grid)

def complete_lines(current_grid):
    rows = 0
    for row in current_grid:
        candidate = True
        for piece in row:
            if piece == (0,0,0):
                candidate = False
                break
        if candidate:
            rows += 1
    return rows

def aggregate_height(current_grid):
    heights = {}
    current_height = 20
    for row in current_grid:
        for indeks, piece in enumerate(row):
            if piece != (0,0,0):
                if not(indeks in heights):
                    heights[indeks] = current_height
        current_height -= 1
    return sum(heights.values())

def max_height(current_grid):
    heights = {}
    current_height = 20
    for row in current_grid:
        for indeks, piece in enumerate(row):
            if piece != (0,0,0):
                if not(indeks in heights):
                    heights[indeks] = current_height
        current_height -= 1
    return max(heights.values())

def holes(current_grid):
    populated_grid = {}
    holes = 0
    for rownum, row in enumerate(current_grid):
        for colnum, piece in enumerate(row):
            if piece == (0,0,0):
                if colnum in populated_grid:
                    holes += 1
            else:
                populated_grid[colnum] = True
    return holes

def bumpiness(current_grid):
    heights = {}
    columns = range(10)
    current_height = 20
    bumps = 0
    for row in current_grid:
        for indeks, piece in enumerate(row):
            if piece != (0, 0, 0):
                if not (indeks in heights):
                    heights[indeks] = current_height
        current_height -= 1

    for i in columns:
        if not(i in heights):
            heights[i] = 0
    for indeks in range(len(columns) - 1):
        bumps += abs(heights[columns[indeks]] - heights[columns[indeks + 1]])
    return bumps

def main_menu():
    run = True
    while run:
        win.fill((0, 0, 0))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                main()
    pygame.quit()


def main(searchFunction=searchTreeForBestMove, numOfInitializedPieces=999999):
    global grid
    global weights

    piecesToDrop = numOfInitializedPieces

    locked_positions = {}
    grid = create_grid(locked_positions)
    change_piece = False
    run = True
    indeksPiecea, current_piece = get_shape()
    indeksPieceaSledeceg, next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0

    while run:
        fall_speed = 1

        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        clock.tick()

        if fall_time >= fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not (valid_space(current_piece, grid)) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.display.quit()
                quit()

        shape_pos = convert_shape_format(current_piece)

        # add piece to the grid for drawing
        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                grid[y][x] = current_piece.color

        # IF PIECE HIT GROUND
        if change_piece:
            piecesToDrop-=1
            if(piecesToDrop==0):
                run=False
                print(score)
            for pos in shape_pos:
                p = (pos[0], pos[1])
                locked_positions[p] = current_piece.color
            current_piece = next_piece
            indeksPiecea = indeksPieceaSledeceg
            indeksPieceaSledeceg, next_piece = get_shape()

            current_piece.x, current_piece.rotation, maksHeuristic, corresponding_params = searchFunction(grid, locked_positions, current_piece, indeksPiecea, next_piece, indeksPieceaSledeceg)

            change_piece = False
            clear_rows(grid, locked_positions)

        draw_window(win)
        draw_next_shape(next_piece, win)
        pygame.display.update()

        # Check if user lost
        if check_lost(locked_positions):
            run = False
            if(piecesToDrop != 0):
                print("Pieces left to drop: " + str(piecesToDrop) + '/' + str(numOfInitializedPieces))
            print(score)

    pygame.display.update()
    pygame.time.delay(2000)

def main_ql(searchFunction=searchTreeForBestMove):
    global weights
    global score
    global grid
    global epsilon
    global R
    num_of_iterations = 25
    while(num_of_iterations > 0):
        score = 0
        locked_positions = {}
        grid = create_grid(locked_positions)

        change_piece = False
        run = True
        indeksPiecea, current_piece = get_shape()
        indeksPieceaSledeceg, next_piece = get_shape()
        clock = pygame.time.Clock()
        fall_time = 0

        while run:
            fall_speed = 1

            grid = create_grid(locked_positions)
            fall_time += clock.get_rawtime()
            clock.tick()

            if fall_time >= fall_speed:
                fall_time = 0
                current_piece.y += 1
                if not (valid_space(current_piece, grid)) and current_piece.y > 0:
                    current_piece.y -= 1
                    change_piece = True
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.display.quit()
                    quit()
            shape_pos = convert_shape_format(current_piece)

            # add piece to the grid for drawing
            for i in range(len(shape_pos)):
                x, y = shape_pos[i]
                if y > -1:
                    grid[y][x] = current_piece.color

            # IF PIECE HIT GROUND
            if change_piece:
                for pos in shape_pos:
                    p = (pos[0], pos[1])
                    locked_positions[p] = current_piece.color
                current_piece = next_piece
                indeksPiecea = indeksPieceaSledeceg
                indeksPieceaSledeceg, next_piece = get_shape()

                change_piece = False
                clear_rows(grid, locked_positions)
                current_piece.x, current_piece.rotation = qlearning(grid, locked_positions, current_piece, indeksPiecea, next_piece, indeksPieceaSledeceg, searchFunction)


            # Check if user lost
            if check_lost(locked_positions):
                run = False


            draw_window(win)
            draw_next_shape(next_piece, win)
            pygame.display.update()
            
        num_of_iterations-=1
    pygame.display.update()
    pygame.time.delay(2000)


def main_odbrana(algorithm, learn=False, searchFunction=searchTreeForBestMove, atIteration=0, numOfInitializedPieces=0):
    global weights
    
    if algorithm=="Greedy":
        initializePiecesFromFile()
        weights = [-0.510066, 0.760666, -0.35663, -0.184483, 0]
        main(searchFunction, numOfInitializedPieces)
    elif algorithm=="QLearning":
        if not learn:
            initializePiecesFromFile()
            weights = weightsAtIteration(atIteration)
            main(searchFunction, numOfInitializedPieces)
        else:
            weights = [-1, 1, -20, -1, -1]
            main_ql(searchFunction)
    else:
        print("Unknown algorithm: " + algorithm + ".\nPlease, input 'Greedy' or 'QLearning'")


win = pygame.display.set_mode((s_width, s_height))
pygame.display.set_caption('ORI - SW31/2016')

main_odbrana("QLearning", False, searchTreeForBestMove_dumb, 10, 100)
