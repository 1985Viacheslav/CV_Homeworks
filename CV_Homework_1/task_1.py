import cv2
import numpy as np
from collections import deque

def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути через лабиринт в виде (x, y)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    height, width = binary.shape

    start = (0, np.where(binary[0] == 255)[0][0])  
    end = (height - 1, np.where(binary[-1] == 255)[0][0])  

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    queue = deque([start])
    visited = set([start])

    prev = {start: None}

    while queue:
        current = queue.popleft()

        if current == end:
            break

        for move in moves:
            neighbor = (current[0] + move[0], current[1] + move[1])

            if (0 <= neighbor[0] < height) and (0 <= neighbor[1] < width) and binary[neighbor] == 255 and neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                prev[neighbor] = current

    path = []
    if end in prev:
        step = end
        while step:
            path.append(step)
            step = prev[step]

    if path:
        path = path[::-1]
        x_coords, y_coords = zip(*path)
        return np.array(x_coords), np.array(y_coords)
    else:
        return None
