import cv2
import numpy as np
from itertools import permutations
import os
import time

def get_board(board_image):
    h, w, _ = board_image.shape
    hb_size = h//8
    wb_size = w//8
    board = [[] for _ in range(8)]
    for i in range(8):
        for j in range(8):
            block = board_image[i*hb_size:(i+1)*hb_size, j*wb_size:(j+1)*wb_size]
            hh, ww, _ = block.shape
            b,g,r = block[hh//2, ww//2]
            board[i].append(int((b,g,r) != (66, 32, 25)))
    # cv2.imshow('board', board)
    return np.array(board)

def get_figures(figures_image):
    h, w, _ = figures_image.shape
    figures_image = figures_image[int(h * 0.7):int(h * 0.82), :]
    mask = cv2.inRange(figures_image, np.array([140, 77, 49]) - 20, np.array([140, 77, 49]) + 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    closed = cv2.morphologyEx(cv2.bitwise_not(mask), cv2.MORPH_CLOSE, kernel)

    allContours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    figures = []
    if len(allContours) == 1:
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(closed))
        closed_copy = closed.copy()[y:y+h, x:x+w]
        h, w = closed_copy.shape
        hb = round(h/35)
        wb = round(w/35)
        hb_size = h//hb
        wb_size = w//wb
        figure = []
        for i in range(hb):
            figure.append([])
            for j in range(wb):
                mask = closed_copy[i*hb_size:(i+1)*hb_size, j*wb_size:(j+1)*wb_size]
                ones_count = cv2.countNonZero(mask)
                zeros_count = mask.size - ones_count
                figure[i].append(int(ones_count>zeros_count))
                
        x,y,w,h = cv2.boundingRect(allContours[0])
        if x<225:
            figures.append((0, np.array(figure, dtype=np.int64)))
            figures.append((1, None))
            figures.append((2, None))
        elif x<400:
            figures.append((0, None))
            figures.append((1, np.array(figure, dtype=np.int64)))
            figures.append((2, None))
        else:
            figures.append((0, None))
            figures.append((1, None))
            figures.append((2, np.array(figure, dtype=np.int64)))
    else:
        figure_cropped_images = [closed[:, :w//3], closed[:, w//3:w//3*2], closed[:, w//3*2:]]
        for figure_index, figure_image in enumerate(figure_cropped_images):
            allContours, _ = cv2.findContours(figure_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(allContours) != 0:
                allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[0]
                perimeter = cv2.arcLength(allContours, True)
            if len(allContours) == 0 or perimeter < 20:
                figures.append((figure_index, None))
                continue
            # ROIdimensions = cv2.approxPolyDP(allContours, 0.02*perimeter, True)
            # cv2.drawContours(figures_image, [ROIdimensions], -1, (0,255,0), 1, offset=(w//3*i, 0))
            # cv2.imshow(f'figure{i}', figure_image)

            x, y, w, h = cv2.boundingRect(cv2.findNonZero(figure_image))
            figure_image = figure_image[y:y+h, x:x+w]
            h, w = figure_image.shape
            hb = round(h/35)
            wb = round(w/35)
            hb_size = h//hb
            wb_size = w//wb
            figure = []
            for i in range(hb):
                figure.append([])
                for j in range(wb):
                    mask = figure_image[i*hb_size:(i+1)*hb_size, j*wb_size:(j+1)*wb_size]
                    ones_count = cv2.countNonZero(mask)
                    zeros_count = mask.size - ones_count
                    figure[i].append(int(ones_count>zeros_count))
            figures.append((figure_index, np.array(figure, dtype=np.int64)))
        # cv2.imshow('source', figures_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return figures

def can_place(board, figure, x, y):
    if figure is None:
        return True
    figure_h, figure_w = figure.shape
    board_h, board_w = board.shape

    if x + figure_h > board_h or y + figure_w > board_w:
        return False

    for i in range(figure_h):
        for j in range(figure_w):
            if figure[i, j] and board[x + i, y + j]:
                return False
    return True

def place_figure(board, figure, x, y, isLastFigure):
    new_board = board.copy()
    
    if figure is None:
        return new_board
    
    figure_h, figure_w = figure.shape

    for i in range(figure_h):
        for j in range(figure_w):
            if figure[i, j] == 1:
                new_board[x + i, y + j] = 1

    return clear_lines(new_board, isLastFigure)

def clear_lines(board, isLastFigure):
    global clearedLine
    new_board = board.copy()
    clearedLine = False
    
    for i in range(8):
        if np.all(board[i, :] == 1):
            if isLastFigure:
                clearedLine = True
            new_board[i, :] = 0

    for j in range(8):
        if np.all(board[:, j] == 1):
            if isLastFigure:
                clearedLine = True
            new_board[:, j] = 0

    return new_board

def solve(board, figures, path=[]):
    if not figures:
        return path
    for perm in permutations(figures):
        for (figure_index, figure) in perm:
            for x in range(7, -1, -1):
                for y in range(7, -1, -1):
                    if can_place(board, figure, x, y):
                        new_board = place_figure(board, figure, x, y, len(figures)==1)
                        solution = solve(new_board, [(i, p) for i, p in perm if i != figure_index], path + [(figure_index, x, y)])
                        if solution:
                            return solution
    return None

def take_screenshot():
    os.system(f"adb -s {device_id} exec-out screencap -p > screen.png")

def perform_swipe(x1, y1, x2, y2, duration=1000):
    os.system(f"adb -s {device_id} shell input swipe {x1} {y1} {x2} {y2} {duration}")

def perform_tap(x1, y1):
    os.system(f"adb -s {device_id} shell input tap {x1} {y1}")

coord = [
    (1.4, 8),
    (4.0, 8),
    (6.6, 8),
]

pixel_per_block = 58

devices = os.popen("adb devices").read().strip().split("\n")[1:]
devices = [line.split("\t")[0] for line in devices if "device" in line]

if not devices:
    print("Ошибка: не найдено подключенных устройств!")
    exit()

device_id = devices[0]

print(f"Используется устройство: '{device_id}'")

solution = True
clearedLine = False
block_app_top = True
while block_app_top:
    while solution:
        # 1. Делаем скриншот
        take_screenshot()
        image_path = "screen.png"
        image = cv2.imread(image_path)[:, 40:-40]

        board = get_board(image[384:-574, :])
        figures = get_figures(image)
        solution = solve(board, figures)

        if solution:
            for (figure_index, y, x) in solution:
                if figures[figure_index][1] is None: continue
                figure_h, figure_w = figures[figure_index][1].shape
                start_x, start_y = coord[figure_index]
                start_x -= figure_w/2
                start_y -= figure_h/2
                path_x = int((x-start_x)*pixel_per_block)
                path_y = int((y-start_y)*pixel_per_block) - pixel_per_block//5
                path_x += pixel_per_block//5*(-1 if path_x<0 else 1)
                perform_swipe(180*(figure_index+1), 1200, 180*(figure_index+1)+path_x, 1200+path_y, int((path_x**2+path_y**2)**0.5*1.6))
                time.sleep(0.3)
        else:
            print("Решения нет")
            perform_tap(360, 1075)
            break
        time.sleep(0.6 if clearedLine else 0.3)
        
    block_app_top = os.popen('adb -s 192.168.0.109:42039 shell "dumpsys activity | grep top-activity"').read().strip().find("com.block.juggle") != -1
    if block_app_top:
        print("block blast is top activity, trying again")
        solution = True