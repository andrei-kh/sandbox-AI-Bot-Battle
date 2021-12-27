from time import time
from random import choice
from src.bot import IBot
from src.constants import MAX_GAME_ITERATIONS
from src.geometry import Direction, Coordinate, directions
from src.snake import Snake
from sys import maxsize
from collections import deque


class Bot(IBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.brain = AlphaBeta()
        self.circler = Circler()
        self.circle_apple = False

    def chooseDirection(self, snake: Snake, opponent: Snake, mazeSize: Coordinate, apple: Coordinate) -> Direction:
        self.full_reset(snake, opponent, mazeSize, apple)

        if self.circle_apple:
            circle_move = self.circler.circle_move()
            if circle_move is None:
                self.circle_apple = False
            else:
                return circle_move
        else:
            self.circle_apple = self.circler.circle_check()
            self.circler.reset_prev_apple()

        return self.brain.choose_direction()

    def full_reset(self, snake, opponent, mazeSize, apple):
        self.circler.reset(snake, opponent, mazeSize, apple)
        self.brain.reset(snake, opponent, mazeSize, apple)


class Score:
    INF = maxsize

    ATE_APPLE = 1250
    ENEMY_ATE_APPLE = -1000

    LOST_LEAD = -1250

    CANT_MOVE = -20000
    ENEMY_CANT_MOVE = 10000

    GOOD_COLLISION = 8500
    BAD_COLLISION = -20000
    DRAW_COLISION = -19900

    REACHED_CENTER = 80


class AlphaBeta:
    def __init__(self) -> None:
        self.game_iterations = 0
        self.start_time = None
        self.max_time = 0.85

        self.snake = None
        self.enemy = None
        self.mazeSize = None
        self.apple = None
        self.center = None

        self.max_depth = None
        self.best_move = None
        self.best_score = -Score.INF

        self.final_move = None

        self.apple_factor = -7
        self.reachable_tiles_factor = 50

    def run(self):
        depth = 2

        while True:
            self.max_depth = depth

            self.final_move = self.make_min_max_graph()

            depth += 2

    def choose_direction(self) -> Direction:
        try:
            self.run()
        except TimeoutError:
            # print("Max depth:", self.max_depth)
            # print(self.start_time, time())
            pass

        return self.final_move

    def reset(self, snake: Snake, enemy: Snake, mazeSize: Coordinate, apple: Coordinate):
        self.start_time = time()

        self.snake = snake
        self.enemy = enemy

        self.mazeSize = mazeSize
        self.final_move = self.get_valid_move(False, False)

        self.apple = apple
        self.center = Coordinate(mazeSize.x // 2, mazeSize.y // 2)
        self.best_move = None
        self.best_score = -Score.INF

        self.game_iterations += 1

    def make_min_max_graph(self):
        current_state = [self.snake.clone(), self.enemy.clone(), -1, -1, 0]

        self.min_max(current_state, current_state, self.max_depth, -maxsize, maxsize, True)

        if self.best_move is None:
            return self.get_valid_move(False, False)

        return self.best_move

    def check_time(self):
        if time() - self.start_time > self.max_time:
            raise TimeoutError

    def min_max(self, state, new_state, depth, alpha, beta, maximizing):
        self.check_time()

        if(depth % 2 == 0):
            state = new_state[:]

        snake, enemy, depth_snake_ate, depth_enemy_ate, _ = state

        if maximizing:
            head = snake.head
            moves = self.get_moves(state, maximizing)
        else:
            head = enemy.head
            moves = self.get_moves(state, maximizing)

        if depth == 0 or moves is None:
            score = self.get_score(state, maximizing, moves, depth)
            return score

        if maximizing:
            calc_score = -Score.INF
            for move in moves:
                new_head = head.moveTo(move)

                if new_head == enemy.body[-1] and distance(enemy.head, self.apple) == 1:
                    return Score.CANT_MOVE

                updated_state = new_state[:]

                new_snake = snake.clone()

                if new_head != self.apple:
                    new_snake.body.pop()
                elif depth_enemy_ate == -1:
                    updated_state[2] = depth // 2

                new_snake.body.insert(0, new_head)

                updated_state[0] = new_snake

                new_score = self.min_max(state, updated_state, depth - 1, alpha, beta, False)
                if new_score > calc_score:
                    calc_score = new_score
                    if depth == self.max_depth:
                        self.best_move = move
                        self.best_score = new_score

                alpha = max(alpha, new_score)
                if alpha >= beta:
                    break

            return calc_score
        else:
            calc_score = Score.INF
            for move in moves:
                updated_state = new_state[:]

                new_head = head.moveTo(move)

                new_snake = enemy.clone()

                if new_head != self.apple:
                    new_snake.body.pop()
                elif depth_snake_ate == -1:
                    updated_state[3] = (depth + 1) // 2

                new_snake.body.insert(0, new_head)

                updated_state[1] = new_snake
                if updated_state[3] == (depth + 1) // 2 or updated_state[2] == (depth + 1) // 2:
                    updated_state[4] = self.get_tile_percentage(updated_state[0], updated_state[1])

                new_score = self.min_max(state, updated_state, depth - 1, alpha, beta, True)

                if new_score < calc_score:
                    calc_score = new_score
                beta = min(beta, new_score)
                if alpha >= beta:
                    break

            return calc_score

    def get_score(self, state, maximizing, moves, depth) -> int:
        snake, enemy, depth_snake_ate, depth_enemy_ate, tile_precantage = state

        no_one_ate = depth_snake_ate == -1 and depth_enemy_ate == -1
        snake_head = snake.head
        enemy_head = enemy.head
        snake_size = len(snake.body)
        enemy_size = len(enemy.body)
        diff = snake_size - enemy_size

        score = 0

        score += self.lead_score(diff, depth_enemy_ate)

        score += self.score_for_going_somewhere(no_one_ate, depth_snake_ate,
                                                depth_enemy_ate, tile_precantage, snake_head)

        score += self.score_dead_end(maximizing, moves, depth)

        score += self.score_head_colision(snake_head, enemy_head, diff)

        score += self.score_colision(snake, enemy)

        return score

    def score_for_going_somewhere(self, no_one_ate, depth_snake_ate, depth_enemy_ate, tile_percentage, head):
        score = 0

        if no_one_ate:
            score += distance(head, self.apple) * self.apple_factor
        else:
            if depth_snake_ate == -1:
                score += Score.ENEMY_ATE_APPLE - depth_enemy_ate
                score += self.reachable_tiles_factor * tile_percentage
            else:
                score += Score.ATE_APPLE + depth_snake_ate

        return score

    def score_head_colision(self, snake_head, enemy_head, diff):
        score = 0

        if snake_head == enemy_head:
            if diff > 0:
                score += Score.GOOD_COLLISION
            elif diff == 0:
                if self.game_iterations + 20 > MAX_GAME_ITERATIONS:
                    score += Score.GOOD_COLLISION
                else:
                    score += Score.DRAW_COLISION
            else:
                score += Score.BAD_COLLISION

        return score

    def score_colision(self, snake, enemy):
        score = 0

        snake_head = snake.head
        enemy_head = enemy.head

        snake_no_head = snake.body[1:]
        enemy_no_head = enemy.body[1:]

        if snake_head in snake_no_head or snake_head in enemy_no_head:
            score += Score.CANT_MOVE

        if enemy_head in snake_no_head or enemy_head in enemy_no_head:
            score += Score.ENEMY_CANT_MOVE

        return score

    def score_dead_end(self, maximizing, moves, depth):
        score = 0

        if moves is None:
            if maximizing:
                score += Score.CANT_MOVE
            else:
                score += Score.ENEMY_CANT_MOVE

            score = score * depth + 1

        return score

    def lead_score(self, diff, depth_enemy_ate) -> int:
        score = 0

        if diff >= 1 and depth_enemy_ate > -1:
            score += Score.LOST_LEAD

        return score

    def get_tile_percentage(self, snake, enemy):
        snake_tiles, enemy_tiles = 0, 0
        for x in range(0, self.mazeSize.x):
            for y in range(0, self.mazeSize.y):
                tile = Coordinate(x, y)
                if tile not in snake.body and tile not in enemy.body:
                    snake_to_tile = distance(snake.head, tile)
                    enemy_to_tile = distance(enemy.head, tile)

                    if snake_to_tile < enemy_to_tile:
                        snake_tiles += 1
                    elif snake_to_tile > enemy_to_tile:
                        enemy_tiles += 1

        reachable_tiles = snake_tiles + enemy_tiles + 1

        return snake_tiles / reachable_tiles

    def get_moves(self, state, is_snake) -> list:
        if is_snake:
            snake, enemy, snake_ate, enemy_ate, _ = state
        else:
            enemy, snake, enemy_ate, snake_ate, _ = state

        head = snake.head

        moves = self.get_valid_moves(snake, enemy, snake_ate > -1, enemy_ate > -1)
        moves = sorted(moves, key=lambda move: distance(head.moveTo(move), self.apple))

        return moves if moves else None

    def move_is_valid(self, snake, enemy, move, snake_ate, enemy_ate):
        head = snake.head

        move_result = head.moveTo(move)

        if not move_result.inBounds(self.mazeSize):
            return False

        if move_result == enemy.body[-1] and not enemy_ate:
            return True

        if move_result == snake.body[-1] and not snake_ate:
            return True

        if move_result in snake.body or move_result in enemy.body:
            return False

        return True

    def get_valid_moves(self, snake, enemy, snake_ate, enemy_ate):
        valid_moves = [move for move in directions
                       if self.move_is_valid(snake, enemy, move, snake_ate, enemy_ate)]

        return valid_moves

    def get_valid_move(self, snake_ate, enemy_ate):
        valid_moves = self.get_valid_moves(self.snake, self.enemy, snake_ate, enemy_ate)

        if valid_moves:
            return choice(valid_moves)
        else:
            return choice(directions)


def distance(point1: Coordinate, point2: Coordinate) -> int:
    return abs(point1.x - point2.x) + abs(point1.y - point2.y)


class Circler:
    def __init__(self):
        self.snake = None
        self.enemy = None
        self.mazeSize = None
        self.apple = None
        self.prev_apple = None

    def reset(self, snake: Snake, enemy: Snake, mazeSize: Coordinate, apple: Coordinate):
        self.snake = snake
        self.enemy = enemy
        self.mazeSize = mazeSize
        self.apple = apple

    def reset_prev_apple(self):
        self.prev_apple = self.apple

    def bfs(self, start: Coordinate, end: list):
        queue = deque([start])
        visited = {start: True}
        parents = {}

        while queue:
            position = queue.popleft()

            for direction in directions:
                new_position = position.moveTo(direction)
                if visited.get(new_position, False):
                    continue

                if new_position == end:
                    parents[new_position] = position
                    return self.traverse_path(parents, start, new_position)

                if self.valid_for_circle(new_position):
                    visited[new_position] = True
                    parents[new_position] = position
                    queue.append(new_position)

    def traverse_path(self, parents, start, end):
        path = [end]

        while path[-1] != start:
            path.append(parents[path[-1]])

        path.reverse()
        return path

    def circle_check(self):
        if self.position_on_side(self.apple):
            return False

        snake_to_apple = self.distance_bfs(self.snake.head, self.apple)

        if snake_to_apple > 4:
            return False

        enemy_to_apple = self.distance_bfs(self.enemy.head, self.apple)

        snake_size = len(self.snake.body)
        enemy_size = len(self.enemy.body)

        if snake_size > 6 and snake_size > enemy_size and snake_to_apple < enemy_to_apple:
            return True

        return False

    def distance_bfs(self, start: Coordinate, end: Coordinate):
        try:
            return len(self.bfs(start, end)) - 1
        except TypeError:
            return maxsize

    def circle_move(self):
        if self.prev_apple != self.apple:
            return None

        apple_spots = self.get_apple_spots()

        if apple_spots:
            path = self.best_path(apple_spots)

            if path is None and len(apple_spots) == 1:
                path = self.bfs(self.snake.head, self.snake.body[-1])
                if len(path) > distance(self.snake.head, self.snake.body[-1]):
                    return None

            if len(path) - 1 > self.distance_bfs(self.enemy.head, path[-1]):
                return None
        else:
            path = self.bfs(self.snake.head, self.snake.body[-1])

        if path is None:
            return None

        return self.get_direction(self.snake.head, path[1])

    def best_path(self, apple_spots):
        sorted_apple_spots = sorted(apple_spots, key=lambda spot: self.distance_bfs(self.snake.head, spot))

        for spot in sorted_apple_spots:
            path = self.bfs(self.snake.head, spot)

            if path is None:
                continue

            positions = self.get_valid_for_circle_positions(spot)
            valid_positions = [position for position in positions if position not in path]

            if valid_positions:
                return path

    def get_direction(self, src: Coordinate, dst: Coordinate) -> Direction:
        for direction in directions:
            if src.moveTo(direction) == dst:
                return direction

    def position_on_x_side(self, position):
        return position.x == 0 or position.x == self.mazeSize.x - 1

    def position_on_y_side(self, position):
        return position.y == 0 or position.y == self.mazeSize.y - 1

    def position_on_side(self, position: Coordinate) -> bool:
        return self.position_on_x_side(position) or self.position_on_y_side(position)

    def position_on_corner(self, position: Coordinate) -> bool:
        return self.position_on_x_side(position) and self.position_on_y_side(position)

    def get_apple_spots(self):
        spots = self.get_valid_for_circle_positions(self.apple)
        return [spot for spot in spots if spot != self.snake.body[-1]]

    def get_valid_for_circle_positions(self, src):
        return [src.moveTo(move) for move in directions if self.valid_for_circle(src.moveTo(move))]

    def valid_position(self, position: Coordinate) -> bool:
        if not position.inBounds(self.mazeSize):
            return False

        if position == self.snake.body[-1]:
            return True

        if position == self.enemy.body[-1]:
            return True

        if position in self.snake.body:
            return False

        if position in self.enemy.body:
            return False

        return True

    def valid_for_circle(self, position: Coordinate) -> bool:
        return self.valid_position(position) and position != self.apple
